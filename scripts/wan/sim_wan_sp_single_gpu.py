#!/usr/bin/env python
"""Single-GPU simulator for the WAN SP all2all + per-rank attention workload.

Plays back what each rank would do in a `world_size`-way sequence-parallel run,
but executes the per-rank attention sequentially on one clean GPU. Useful when
you only have one free card and want to compare head-assignment strategies
(contiguous vs greedy from a logged density JSONL).

Because all ranks run the same kernel in the real cluster simultaneously, the
"parallel cost" we report is `max across ranks` of each iteration's per-rank
time — that is the bottleneck the cluster would actually see.
"""

import argparse
import gc
import json
import os
import statistics
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import torch


os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The sibling bench script holds helpers we want to reuse. scripts/ has no
# __init__.py, so add this directory to sys.path and import by module name.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from svg.kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton  # noqa: F401  (permute_tensor used via WanSPState)
from svg.kmeans_utils import (
    batch_kmeans_Euclid,  # noqa: F401  (used via WanSPState)
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,  # noqa: F401
)

from bench_wan_sp_all2all_attention import (
    WanSPState,
    compute_rank_heads,
    cuda_elapsed_ms,
    load_cost_model,
    record_cuda_region,
)
from profile_wan_maskgen_aware_cost import (
    auto_head_counts,
    auto_seq_lens,
    fit_affine,
    parse_int_list,
    profile_case,
    write_csv,
)


def default_cost_model_path(input_path: str) -> Path:
    path = Path(input_path)
    name = path.stem
    if name.startswith("attn_core_"):
        name = name[len("attn_core_") :]
    return path.with_name(f"maskgen_aware_cost_{name}.json")


def ensure_cost_model_cache(args: argparse.Namespace, device: torch.device) -> Path:
    model_path = Path(args.cost_model_json) if args.cost_model_json else default_cost_model_path(args.input)
    if model_path.exists() and not args.force_reprofile:
        print(f"[profile-cache] using existing cost model: {model_path}")
        return model_path

    print(f"[profile-cache] building cost model: {model_path}")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    cfg, original_heads, max_seq_len, _ = data["inputs"]["query"].shape
    min_seq_len = max(int(metadata["num_q_centroids"]), int(metadata["num_k_centroids"]), 1)
    seq_lens = (
        auto_seq_lens(max_seq_len, min_seq_len)
        if args.profile_seq_lens == "auto"
        else parse_int_list(args.profile_seq_lens, max_seq_len, min_value=min_seq_len)
    )
    head_counts = (
        auto_head_counts(original_heads)
        if args.profile_head_counts == "auto"
        else parse_int_list(args.profile_head_counts, original_heads)
    )

    profile_args = Namespace(
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        q_chunks=args.q_chunks,
    )
    rows = []
    q_chunk_rows = []
    for heads in head_counts:
        for seq_len in seq_lens:
            print(f"[profile-cache] run heads={heads} seq_len={seq_len}")
            row, chunk_rows = profile_case(data, heads, seq_len, profile_args, device)
            rows.append(row)
            q_chunk_rows.extend(chunk_rows)
            print(
                f"[profile-cache] mask={row['mask_mean_ms']:.3f} ms "
                f"attn={row['attention_mean_ms']:.3f} ms density={row['density_mean']:.6f}"
            )

    mask_fit = fit_affine(rows, "mask_feature_head_tokens", "mask_mean_ms")
    attention_fit = fit_affine(rows, "attention_feature_head_density_seq2", "attention_mean_ms")
    model = {
        "input": args.input,
        "metadata": {
            "layer_idx": metadata.get("layer_idx"),
            "linear_step": metadata.get("linear_step"),
            "timestep": metadata.get("timestep"),
            "cfg": int(cfg),
            "num_heads": int(original_heads),
            "seq_len": int(max_seq_len),
        },
        "formula": {
            "mask_ms": "intercept_ms + slope_ms_per_unit * heads * seq_len",
            "attention_ms": "intercept_ms + slope_ms_per_unit * heads * density * seq_len^2",
        },
        "mask_fit": mask_fit,
        "attention_fit": attention_fit,
        "samples": rows,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2))
    print(
        "[profile-cache] fit: "
        f"mask={mask_fit['intercept_ms']:.6f}+{mask_fit['slope_ms_per_unit']:.12f}*heads*seq_len "
        f"R2={mask_fit['r2']:.6f}; "
        f"attn={attention_fit['intercept_ms']:.6f}+{attention_fit['slope_ms_per_unit']:.12f}"
        f"*heads*density*seq_len^2 R2={attention_fit['r2']:.6f}"
    )
    print(f"[profile-cache] wrote cost model: {model_path}")

    if args.profile_csv:
        write_csv(Path(args.profile_csv), rows)
        print(f"[profile-cache] wrote profile CSV: {args.profile_csv}")
    if args.profile_q_chunk_csv:
        write_csv(Path(args.profile_q_chunk_csv), q_chunk_rows)
        print(f"[profile-cache] wrote q-chunk CSV: {args.profile_q_chunk_csv}")

    del data, rows, q_chunk_rows
    gc.collect()
    torch.cuda.empty_cache()
    return model_path


def run_rank_attention(
    full: Dict,
    head_indices: List[int],
    state_proto: Dict,
    device: torch.device,
) -> Dict:
    head_idx = torch.tensor(head_indices, device=device, dtype=torch.long)

    q_head = full["query"].index_select(1, head_idx).contiguous()
    k_head = full["key"].index_select(1, head_idx).contiguous()
    v_head = full["value"].index_select(1, head_idx).contiguous()

    state = WanSPState(
        num_q_centroids=state_proto["num_q_centroids"],
        num_k_centroids=state_proto["num_k_centroids"],
        top_p_kmeans=state_proto["top_p_kmeans"],
        min_kc_ratio=state_proto["min_kc_ratio"],
        kmeans_iter_step=state_proto["kmeans_iter_step"],
        q_centroids=full["q_cache"].index_select(0, head_idx).contiguous(),
        k_centroids=full["k_cache"].index_select(0, head_idx).contiguous(),
    )

    mask_out, mask_events = record_cuda_region(
        lambda: state.semantic_aware_permutation(q_head, k_head, v_head),
        "mask_semantic_aware_permutation",
    )
    q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, q_sorted_indices, qlabels, klabels = mask_out

    attn_out_permuted, attn_events = record_cuda_region(
        lambda: dynamic_block_sparse_fwd_flashinfer(
            q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False
        ),
        "dynamic_block_sparse_attention",
    )
    attn_out, inv_events = record_cuda_region(
        lambda: apply_inverse_permutation_triton(attn_out_permuted, q_sorted_indices, dim=2),
        "inverse_permute",
    )

    density_gpu = density_calculation(dyn_map, qc_sz, kc_sz).reshape(-1).detach().float()
    torch.cuda.synchronize(device)

    mask_ms = cuda_elapsed_ms(mask_events)
    attn_ms = cuda_elapsed_ms(attn_events)
    inv_ms = cuda_elapsed_ms(inv_events)

    out = {
        "heads": list(head_indices),
        "mask_ms": mask_ms,
        "attention_ms": attn_ms,
        "inverse_permute_ms": inv_ms,
        "total_ms": mask_ms + attn_ms + inv_ms,
        "density": density_gpu.cpu().tolist(),
    }

    del mask_out, q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, q_sorted_indices, qlabels, klabels
    del attn_out_permuted, attn_out, density_gpu, state, q_head, k_head, v_head
    return out


def simulate_strategy(
    full: Dict,
    state_proto: Dict,
    ranks_heads: List[List[int]],
    iters: int,
    warmup: int,
    device: torch.device,
    label: str,
) -> Dict:
    world_size = len(ranks_heads)

    for _ in range(warmup):
        for heads in ranks_heads:
            run_rank_attention(full, heads, state_proto, device)
        torch.cuda.empty_cache()

    per_rank_runs: List[List[Dict]] = [[] for _ in range(world_size)]
    for _ in range(iters):
        for r, heads in enumerate(ranks_heads):
            per_rank_runs[r].append(run_rank_attention(full, heads, state_proto, device))

    rank_summary = []
    for r, runs in enumerate(per_rank_runs):
        totals = [x["total_ms"] for x in runs]
        masks = [x["mask_ms"] for x in runs]
        attns = [x["attention_ms"] for x in runs]
        invs = [x["inverse_permute_ms"] for x in runs]
        rank_summary.append(
            {
                "rank": r,
                "heads": runs[0]["heads"],
                "total_ms_mean": statistics.mean(totals),
                "total_ms_median": statistics.median(totals),
                "mask_ms_mean": statistics.mean(masks),
                "attention_ms_mean": statistics.mean(attns),
                "inverse_permute_ms_mean": statistics.mean(invs),
                "density_sum": float(sum(runs[0]["density"])),
            }
        )

    parallel_cost_ms = statistics.mean(
        [max(per_rank_runs[r][i]["total_ms"] for r in range(world_size)) for i in range(iters)]
    )
    max_d = max(s["density_sum"] for s in rank_summary)
    min_d = min(s["density_sum"] for s in rank_summary)

    print(f"\n==== strategy: {label} ====")
    print(f"parallel cost (max across ranks, mean across {iters} iters): {parallel_cost_ms:.2f} ms")
    print(
        f"density per rank: {[round(s['density_sum'], 4) for s in rank_summary]} "
        f"max/min={max_d / min_d:.3f}"
    )
    hdr = (
        f"{'rank':>4}  {'heads':<30}  {'total_mean':>10}  "
        f"{'mask':>6}  {'attn':>6}  {'inv':>6}  {'dens_sum':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in rank_summary:
        print(
            f"{s['rank']:>4}  {str(s['heads']):<30}  "
            f"{s['total_ms_mean']:>10.2f}  "
            f"{s['mask_ms_mean']:>6.2f}  {s['attention_ms_mean']:>6.2f}  "
            f"{s['inverse_permute_ms_mean']:>6.2f}  {s['density_sum']:>8.4f}"
        )

    return {"parallel_cost_ms": parallel_cost_ms, "rank_summary": rank_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to SVG_WAN_ATTN_EXPORT_PATH .pt file.")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--density-log",
        default=None,
        help="JSONL with per-step density; required for --strategies greedy.",
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["contiguous", "greedy"],
        choices=["contiguous", "greedy", "greedy_unequal"],
        help="Which head-assignment strategies to simulate. "
        "'greedy_unequal' drops the equal-heads-per-rank constraint (LPT).",
    )
    parser.add_argument(
        "--min-heads-per-rank",
        type=int,
        default=1,
        help="For 'greedy_unequal': guarantee at least this many heads per rank.",
    )
    parser.add_argument(
        "--cost-model-json",
        default=None,
        help="Maskgen-aware cost model JSON. If omitted with --auto-profile-cache, defaults next to --input.",
    )
    parser.add_argument(
        "--auto-profile-cache",
        action="store_true",
        help="Before simulation, check cost-model JSON cache and build it if missing.",
    )
    parser.add_argument("--force-reprofile", action="store_true")
    parser.add_argument("--profile-seq-lens", default="auto")
    parser.add_argument("--profile-head-counts", default="auto")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--q-chunks", type=int, default=8)
    parser.add_argument("--profile-csv", default=None)
    parser.add_argument("--profile-q-chunk-csv", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cost_model = None
    if args.auto_profile_cache:
        model_path = ensure_cost_model_cache(args, device)
        args.cost_model_json = str(model_path)
    if args.cost_model_json:
        cost_model = load_cost_model(args.cost_model_json)

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    full = {
        "query": data["inputs"]["query"].to(device),
        "key": data["inputs"]["key"].to(device),
        "value": data["inputs"]["value"].to(device),
        "q_cache": data["centroid_cache"]["q_centroids"].to(device),
        "k_cache": data["centroid_cache"]["k_centroids"].to(device),
    }
    del data
    gc.collect()

    cfg, num_heads, seq_len, dim = full["query"].shape
    if cfg != 1:
        raise ValueError(f"cfg={cfg} not supported")
    if num_heads % args.world_size != 0:
        raise ValueError(f"num_heads {num_heads} not divisible by world_size {args.world_size}")

    state_proto = {
        "num_q_centroids": int(metadata["num_q_centroids"]),
        "num_k_centroids": int(metadata["num_k_centroids"]),
        "top_p_kmeans": float(metadata["top_p_kmeans"]),
        "min_kc_ratio": float(metadata["min_kc_ratio"]),
        "kmeans_iter_step": int(metadata["kmeans_iter_step"]),
    }

    print(
        f"capture: layer={metadata.get('layer_idx')} linear_step={metadata.get('linear_step')} "
        f"timestep={metadata.get('timestep')} heads={num_heads} seq_len={seq_len} "
        f"sim world_size={args.world_size}"
    )

    strategies: Dict[str, List[List[int]]] = {}
    for name in args.strategies:
        if name in ("greedy", "greedy_unequal") and args.density_log is None:
            raise SystemExit(f"'{name}' requires --density-log")
        ranks_heads, info = compute_rank_heads(
            args.density_log,
            metadata,
            num_heads,
            args.world_size,
            strategy=name,
            rank=0,
            min_heads_per_rank=args.min_heads_per_rank,
            cost_model=cost_model,
        )
        if name != "contiguous" and info is None:
            print(f"[warn] {name}: no prior density; skipping")
            continue
        strategies[name] = ranks_heads
        if info is not None:
            pl = info["predicted_loads"]
            line = (
                f"{name} prediction ({info.get('cost_model', 'density')}, prev density from layer={info['layer']} "
                f"timestep>{info['timestep']}): "
                f"loads={[round(x, 4) for x in pl]} max/min={max(pl) / min(pl):.3f} "
                f"heads/rank={info['heads_per_rank']}"
            )
            if info.get("predicted_contig") is not None:
                pc = info["predicted_contig"]
                line += (
                    f" | vs contiguous loads={[round(x, 4) for x in pc]} "
                    f"max/min={max(pc) / min(pc):.3f}"
                )
            print(line)

    results = {}
    for name, ranks_heads in strategies.items():
        results[name] = simulate_strategy(
            full,
            state_proto,
            ranks_heads,
            args.iters,
            args.warmup,
            device,
            name,
        )

    if len(results) >= 2:
        print("\n==== comparison (parallel cost = max across ranks) ====")
        for name, res in results.items():
            print(f"  {name:>12}: {res['parallel_cost_ms']:.2f} ms")
        names = list(results.keys())
        baseline = results[names[0]]["parallel_cost_ms"]
        for n in names[1:]:
            diff = results[n]["parallel_cost_ms"] - baseline
            pct = (diff / baseline * 100.0) if baseline > 0 else 0.0
            print(f"  delta {names[0]} -> {n}: {diff:+.2f} ms ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
