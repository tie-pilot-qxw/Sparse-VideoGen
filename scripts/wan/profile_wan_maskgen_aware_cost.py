#!/usr/bin/env python
import argparse
import csv
import gc
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svg.kernels.triton.permute import permute_tensor_by_labels_triton
from svg.kmeans_utils import (
    batch_kmeans_Euclid,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)


@dataclass
class WanCostProfileState:
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_step: int
    q_centroids: torch.Tensor
    k_centroids: torch.Tensor

    def semantic_aware_permutation(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, _ = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, _ = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        q_permuted, _ = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, qlabels


def parse_int_list(raw: str, max_value: int, *, min_value: int = 1) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < min_value:
            raise ValueError(f"value {value} < min_value {min_value}")
        if value <= max_value:
            values.append(value)
    if not values:
        raise ValueError(f"no values <= max_value {max_value} in {raw!r}")
    return sorted(set(values))


def auto_seq_lens(max_seq_len: int, min_seq_len: int) -> List[int]:
    candidates = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, max_seq_len]
    values = [s for s in candidates if min_seq_len <= s <= max_seq_len]
    if max_seq_len not in values:
        values.append(max_seq_len)
    return sorted(set(values))


def auto_head_counts(max_heads: int) -> List[int]:
    values = []
    head = 1
    while head < max_heads:
        values.append(head)
        head *= 2
    values.append(max_heads)
    return sorted(set(values))


def select_flat_head_cache(cache: torch.Tensor, cfg: int, heads: int) -> torch.Tensor:
    original_flat, clusters, dim = cache.shape
    if original_flat % cfg != 0:
        raise ValueError(f"centroid cache shape {tuple(cache.shape)} is not divisible by cfg={cfg}")
    original_heads = original_flat // cfg
    if heads > original_heads:
        raise ValueError(f"requested {heads} heads, but cache only has {original_heads}")
    return cache.view(cfg, original_heads, clusters, dim)[:, :heads].reshape(cfg * heads, clusters, dim).contiguous()


def percentile(values: List[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def cuda_timed(fn) -> Tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def q_sequence_chunk_density(
    dynamic_map: torch.Tensor,
    qlabels: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    seq_len: int,
    chunks: int,
) -> torch.Tensor:
    cfg, heads, _, _ = dynamic_map.shape
    qlabels = qlabels.view(cfg, heads, seq_len).long()
    kept_keys_per_qcluster = (dynamic_map.to(k_cluster_sizes.dtype) * k_cluster_sizes[:, :, None, :]).sum(dim=-1)
    total_keys = k_cluster_sizes.sum(dim=-1).clamp(min=1).to(kept_keys_per_qcluster.dtype)

    out = []
    for chunk_idx in range(chunks):
        start = seq_len * chunk_idx // chunks
        end = seq_len * (chunk_idx + 1) // chunks
        if end <= start:
            chunk_density = torch.zeros((cfg, heads), device=dynamic_map.device, dtype=kept_keys_per_qcluster.dtype)
        else:
            labels = qlabels[:, :, start:end]
            kept = torch.gather(kept_keys_per_qcluster, dim=-1, index=labels)
            chunk_density = kept.sum(dim=-1) / ((end - start) * total_keys)
        out.append(chunk_density)
    return torch.stack(out, dim=-1)


def make_state(metadata: Dict, q_cache: torch.Tensor, k_cache: torch.Tensor) -> WanCostProfileState:
    return WanCostProfileState(
        num_q_centroids=int(metadata["num_q_centroids"]),
        num_k_centroids=int(metadata["num_k_centroids"]),
        top_p_kmeans=float(metadata["top_p_kmeans"]),
        min_kc_ratio=float(metadata["min_kc_ratio"]),
        kmeans_iter_step=int(metadata["kmeans_iter_step"]),
        q_centroids=q_cache,
        k_centroids=k_cache,
    )


def profile_case(data: Dict, heads: int, seq_len: int, args: argparse.Namespace, device: torch.device) -> Tuple[Dict, List[Dict]]:
    metadata = data["metadata"]
    cfg = int(metadata["cfg"])
    query = data["inputs"]["query"][:, :heads, :seq_len, :].contiguous().to(device)
    key = data["inputs"]["key"][:, :heads, :seq_len, :].contiguous().to(device)
    value = data["inputs"]["value"][:, :heads, :seq_len, :].contiguous().to(device)
    q_cache = select_flat_head_cache(data["centroid_cache"]["q_centroids"], cfg, heads).to(device)
    k_cache = select_flat_head_cache(data["centroid_cache"]["k_centroids"], cfg, heads).to(device)

    for _ in range(args.warmup):
        state = make_state(metadata, q_cache, k_cache)
        outputs = state.semantic_aware_permutation(query, key, value)
        torch.cuda.synchronize(device)
        del outputs

    mask_times = []
    outputs = None
    for _ in range(args.iters):
        state = make_state(metadata, q_cache, k_cache)
        outputs, elapsed_ms = cuda_timed(lambda: state.semantic_aware_permutation(query, key, value))
        mask_times.append(elapsed_ms)

    q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, qlabels = outputs
    density = density_calculation(dyn_map, qc_sz, kc_sz).detach().float()
    density_mean = float(density.mean().item())

    for _ in range(args.warmup):
        attn_out = dynamic_block_sparse_fwd_flashinfer(q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False)
        torch.cuda.synchronize(device)
        del attn_out

    attention_times = []
    for _ in range(args.iters):
        attn_out, elapsed_ms = cuda_timed(
            lambda: dynamic_block_sparse_fwd_flashinfer(q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False)
        )
        attention_times.append(elapsed_ms)
        del attn_out

    q_chunk_density = q_sequence_chunk_density(dyn_map, qlabels, kc_sz, seq_len=seq_len, chunks=args.q_chunks)
    q_chunk_rows = []
    for head_idx, values in enumerate(q_chunk_density.reshape(heads, args.q_chunks).detach().float().cpu().tolist()):
        for chunk_idx, value in enumerate(values):
            start = seq_len * chunk_idx // args.q_chunks
            end = seq_len * (chunk_idx + 1) // args.q_chunks
            q_chunk_rows.append(
                {
                    "heads": heads,
                    "seq_len": seq_len,
                    "global_head": head_idx,
                    "q_chunk": chunk_idx,
                    "q_start": start,
                    "q_end": end,
                    "q_len": end - start,
                    "density": float(value),
                }
            )

    row = {
        "heads": heads,
        "seq_len": seq_len,
        "density_mean": density_mean,
        "mask_mean_ms": statistics.mean(mask_times),
        "mask_median_ms": statistics.median(mask_times),
        "mask_p90_ms": percentile(mask_times, 0.9),
        "attention_mean_ms": statistics.mean(attention_times),
        "attention_median_ms": statistics.median(attention_times),
        "attention_p90_ms": percentile(attention_times, 0.9),
        "mask_feature_head_tokens": heads * seq_len,
        "attention_feature_head_density_seq2": heads * density_mean * seq_len * seq_len,
    }

    del query, key, value, q_cache, k_cache, outputs, q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, qlabels
    gc.collect()
    torch.cuda.empty_cache()
    return row, q_chunk_rows


def fit_affine(rows: Iterable[Dict], x_key: str, y_key: str) -> Dict:
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    x = torch.tensor([[1.0, value] for value in xs], dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64).reshape(-1, 1)
    coeff = torch.linalg.lstsq(x, y).solution.reshape(-1)
    pred = (x @ coeff.reshape(-1, 1)).reshape(-1)
    y_mean = y.mean()
    ss_res = torch.sum((y.reshape(-1) - pred) ** 2)
    ss_tot = torch.sum((y.reshape(-1) - y_mean) ** 2)
    r2 = 1.0 if float(ss_tot) == 0.0 else 1.0 - float(ss_res / ss_tot)
    return {
        "intercept_ms": float(coeff[0]),
        "slope_ms_per_unit": float(coeff[1]),
        "r2": r2,
        "x_key": x_key,
        "y_key": y_key,
        "num_samples": len(xs),
    }


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Profile and fit a maskgen-aware Wan SVG2 cost model.")
    parser.add_argument("--input", required=True, help="Path to SVG_WAN_ATTN_EXPORT_PATH .pt file.")
    parser.add_argument("--seq-lens", default="auto", help="Comma list, or 'auto'.")
    parser.add_argument("--head-counts", default="auto", help="Comma list, or 'auto'.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--q-chunks", type=int, default=8)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--q-chunk-csv", default=None)
    parser.add_argument("--model-json", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")
    if args.q_chunks <= 0:
        raise ValueError("--q-chunks must be positive")

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    cfg, original_heads, max_seq_len, head_dim = data["inputs"]["query"].shape
    min_seq_len = max(int(metadata["num_q_centroids"]), int(metadata["num_k_centroids"]), 1)
    seq_lens = auto_seq_lens(max_seq_len, min_seq_len) if args.seq_lens == "auto" else parse_int_list(args.seq_lens, max_seq_len, min_value=min_seq_len)
    head_counts = auto_head_counts(original_heads) if args.head_counts == "auto" else parse_int_list(args.head_counts, original_heads)

    print(
        f"capture: layer={metadata.get('layer_idx')} linear_step={metadata.get('linear_step')} "
        f"timestep={metadata.get('timestep')} heads={original_heads} seq_len={max_seq_len} head_dim={head_dim}"
    )
    print(f"profile seq_lens={seq_lens} head_counts={head_counts}")

    rows = []
    q_chunk_rows = []
    for heads in head_counts:
        for seq_len in seq_lens:
            print(f"[run] heads={heads} seq_len={seq_len}")
            row, chunk_rows = profile_case(data, heads, seq_len, args, device)
            rows.append(row)
            q_chunk_rows.extend(chunk_rows)
            print(
                f"  mask={row['mask_mean_ms']:.3f} ms "
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
            "head_dim": int(head_dim),
        },
        "formula": {
            "mask_ms": "intercept_ms + slope_ms_per_unit * heads * seq_len",
            "attention_ms": "intercept_ms + slope_ms_per_unit * heads * density * seq_len^2",
        },
        "mask_fit": mask_fit,
        "attention_fit": attention_fit,
        "samples": rows,
    }

    print("\nfit:")
    print(
        "  mask_ms = "
        f"{mask_fit['intercept_ms']:.6f} + {mask_fit['slope_ms_per_unit']:.12f} * heads * seq_len "
        f"(R2={mask_fit['r2']:.6f})"
    )
    print(
        "  attention_ms = "
        f"{attention_fit['intercept_ms']:.6f} + {attention_fit['slope_ms_per_unit']:.12f} "
        f"* heads * density * seq_len^2 (R2={attention_fit['r2']:.6f})"
    )

    if args.csv:
        write_csv(Path(args.csv), rows)
        print(f"wrote CSV: {args.csv}")
    if args.q_chunk_csv:
        write_csv(Path(args.q_chunk_csv), q_chunk_rows)
        print(f"wrote q-chunk CSV: {args.q_chunk_csv}")
    if args.model_json:
        path = Path(args.model_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=2))
        print(f"wrote model JSON: {args.model_json}")


if __name__ == "__main__":
    main()
