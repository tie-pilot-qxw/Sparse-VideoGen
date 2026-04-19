#!/usr/bin/env python
import argparse
import csv
import gc
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch


os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svg.kernels.triton.permute import permute_tensor_by_labels_triton
from svg.kmeans_utils import batch_kmeans_Euclid, identify_dynamic_map


@dataclass
class WanSemanticPermutationState:
    layer_idx: int
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_step: int
    q_centroids: torch.Tensor
    k_centroids: torch.Tensor

    def kmeans_step(self, query: torch.Tensor, key: torch.Tensor):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )
        self.q_centroids = qcentroids
        self.k_centroids = kcentroids
        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def semantic_aware_permutation(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        cfg, num_heads, seq_len, dim = query.size()

        qlabels, qcentroids, qcluster_sizes, _, klabels, kcentroids, kcluster_sizes, _ = self.kmeans_step(query, key)

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

        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices


def parse_head_counts(raw: str, original_heads: int) -> List[int]:
    if raw == "auto":
        counts = []
        value = 1
        while value < original_heads:
            counts.append(value)
            value *= 2
        if original_heads not in counts:
            counts.append(original_heads)
        return counts

    counts = [int(item) for item in raw.split(",") if item.strip()]
    if not counts:
        raise ValueError("--head-counts cannot be empty")
    return counts


def select_heads(tensor: torch.Tensor, heads: int, *, allow_repeat: bool) -> torch.Tensor:
    original_heads = tensor.shape[1]
    if heads <= original_heads:
        return tensor[:, :heads].contiguous()
    if not allow_repeat:
        raise ValueError(f"Requested {heads} heads, but capture only has {original_heads}. Use --allow-repeat.")

    repeat_factor = (heads + original_heads - 1) // original_heads
    return tensor.repeat(1, repeat_factor, 1, 1)[:, :heads].contiguous()


def select_flat_head_cache(cache: torch.Tensor, cfg: int, heads: int, *, allow_repeat: bool) -> torch.Tensor:
    original_flat, clusters, dim = cache.shape
    if original_flat % cfg != 0:
        raise ValueError(f"Centroid cache shape {tuple(cache.shape)} is not divisible by cfg={cfg}")

    original_heads = original_flat // cfg
    cache = cache.view(cfg, original_heads, clusters, dim)
    if heads <= original_heads:
        return cache[:, :heads].reshape(cfg * heads, clusters, dim).contiguous()
    if not allow_repeat:
        raise ValueError(f"Requested {heads} heads, but centroid cache only has {original_heads}. Use --allow-repeat.")

    repeat_factor = (heads + original_heads - 1) // original_heads
    return cache.repeat(1, repeat_factor, 1, 1)[:, :heads].reshape(cfg * heads, clusters, dim).contiguous()


def percentile(values: List[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * p)
    return ordered[idx]


def synchronize_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(
    metadata: dict,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
):
    state = WanSemanticPermutationState(
        layer_idx=int(metadata["layer_idx"]),
        num_q_centroids=int(metadata["num_q_centroids"]),
        num_k_centroids=int(metadata["num_k_centroids"]),
        top_p_kmeans=float(metadata["top_p_kmeans"]),
        min_kc_ratio=float(metadata["min_kc_ratio"]),
        kmeans_iter_step=int(metadata["kmeans_iter_step"]),
        q_centroids=q_centroids,
        k_centroids=k_centroids,
    )
    return state.semantic_aware_permutation(query, key, value)


def benchmark_head_count(
    data: dict,
    heads: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    metadata = data["metadata"]
    cfg = int(metadata["cfg"])

    query = select_heads(data["inputs"]["query"], heads, allow_repeat=args.allow_repeat).to(device).contiguous()
    key = select_heads(data["inputs"]["key"], heads, allow_repeat=args.allow_repeat).to(device).contiguous()
    value = select_heads(data["inputs"]["value"], heads, allow_repeat=args.allow_repeat).to(device).contiguous()
    q_cache = select_flat_head_cache(
        data["centroid_cache"]["q_centroids"], cfg, heads, allow_repeat=args.allow_repeat
    ).to(device).contiguous()
    k_cache = select_flat_head_cache(
        data["centroid_cache"]["k_centroids"], cfg, heads, allow_repeat=args.allow_repeat
    ).to(device).contiguous()

    synchronize_if_cuda(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        outputs = run_once(metadata, query, key, value, q_cache.clone(), k_cache.clone())
        synchronize_if_cuda(device)
        del outputs

    timings_ms = []
    for _ in range(args.iters):
        q_centroids = q_cache.clone()
        k_centroids = k_cache.clone()
        synchronize_if_cuda(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = run_once(metadata, query, key, value, q_centroids, k_centroids)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
        else:
            import time

            start_time = time.perf_counter()
            outputs = run_once(metadata, query, key, value, q_centroids, k_centroids)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

        timings_ms.append(elapsed_ms)
        del outputs, q_centroids, k_centroids

    peak_gb = None
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    result = {
        "heads": heads,
        "effective_batch": cfg * heads,
        "seq_len": int(metadata["seq_len"]),
        "head_dim": int(metadata["head_dim"]),
        "iters": args.iters,
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p90_ms": percentile(timings_ms, 0.9),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "peak_mem_gb": peak_gb,
    }

    del query, key, value, q_cache, k_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_results(results: Iterable[dict]):
    rows = list(results)
    header = "heads  batch  mean_ms   median_ms p90_ms    min_ms    max_ms    peak_mem_gb"
    print(header)
    print("-" * len(header))
    for row in rows:
        peak = "n/a" if row["peak_mem_gb"] is None else f"{row['peak_mem_gb']:.2f}"
        print(
            f"{row['heads']:>5}  {row['effective_batch']:>5}  "
            f"{row['mean_ms']:>8.2f}  {row['median_ms']:>8.2f}  {row['p90_ms']:>8.2f}  "
            f"{row['min_ms']:>8.2f}  {row['max_ms']:>8.2f}  {peak:>11}"
        )


def write_csv(path: Path, results: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Wan SVG2 semantic_aware_permutation from an exported input.")
    parser.add_argument("--input", required=True, help="Path to semantic_perm_step*_layer*.pt")
    parser.add_argument("--head-counts", default="auto", help="Comma list such as 1,2,4,8,16,32,40, or 'auto'.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-repeat", action="store_true", help="Allow testing more heads than captured by repeating heads.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton permutation/kmeans path, but torch.cuda.is_available() is false.")

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    original_heads = int(metadata["num_heads"])
    head_counts = parse_head_counts(args.head_counts, original_heads)

    print(f"input: {args.input}")
    print(
        "capture: "
        f"linear_step={metadata.get('linear_step')} "
        f"layer_idx={metadata.get('layer_idx')} "
        f"timestep={metadata.get('timestep')} "
        f"original_heads={original_heads} "
        f"seq_len={metadata.get('seq_len')} "
        f"head_dim={metadata.get('head_dim')}"
    )
    print(
        "benchmark target: kmeans_step + identify_dynamic_map + "
        "permute_tensor_by_labels_triton(q/k/v)"
    )

    results = []
    for heads in head_counts:
        print(f"\n[run] heads={heads}")
        results.append(benchmark_head_count(data, heads, args, device))

    print("\nsummary:")
    print_results(results)

    if args.csv:
        write_csv(Path(args.csv), results)
        print(f"\nwrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
