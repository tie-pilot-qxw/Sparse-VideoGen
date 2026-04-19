#!/usr/bin/env python
import argparse
import csv
import gc
import json
import os
import statistics
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Sibling imports (scripts/wan/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg.kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from svg.kmeans_utils import (
    batch_kmeans_Euclid,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)


def pick_s_block(s_local: int, max_block: int = 128) -> int:
    """Largest power-of-2 <= min(max_block, s_local). TMA requires pow-2 block
    shapes; we pad the S dim upstream so s_local divisibility isn't required."""
    cap = min(max_block, max(1, s_local))
    b = 1
    while b * 2 <= cap:
        b *= 2
    return b


@dataclass
class WanSPState:
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

        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        # print(f"dyn_map shape {dynamic_map.shape}, qlabels shape {qlabels.shape}, klabels shape {qlabels.shape}, q_sorted_indices shape {q_sorted_indices.shape}")

        return (
            q_permuted,
            k_permuted,
            v_permuted,
            dynamic_map,
            q_cluster_sizes,
            k_cluster_sizes,
            q_sorted_indices,
            qlabels,
            klabels,
        )


CudaEventPair = Optional[Tuple[torch.cuda.Event, torch.cuda.Event]]


def record_cuda_region(fn, label: str = "timed_region", stream: Optional[torch.cuda.Stream] = None):
    if stream is None:
        stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    with torch.profiler.record_function(label):
        out = fn()
    end.record(stream)
    return out, (start, end)


def cuda_elapsed_ms(events: CudaEventPair) -> float:
    if events is None:
        return 0.0
    return events[0].elapsed_time(events[1])


def q_sequence_chunk_density(
    dynamic_map: torch.Tensor,
    qlabels: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    seq_len: int,
    chunks: int,
) -> torch.Tensor:
    """Density per head for contiguous chunks along the original q sequence dimension.

    For a q token assigned to q-cluster c, the kept key-token count is the sum of
    key cluster sizes selected by dynamic_map[c, :].  Chunk density is the
    average kept-key fraction over tokens in that q chunk.
    """
    cfg, heads, qc_num, _ = dynamic_map.shape
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


def all2all_sequence_to_heads(x_seq: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, CudaEventPair]:
    if world_size == 1:
        return x_seq.contiguous(), None

    batch, heads, seq_local, dim = x_seq.shape
    assert heads % world_size == 0
    heads_per_rank = heads // world_size
    send = x_seq.view(batch, world_size, heads_per_rank, seq_local, dim).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(send)

    _, events = record_cuda_region(lambda: dist.all_to_all_single(recv, send), "all2all_sequence_to_heads")
    x_head = recv.permute(1, 2, 0, 3, 4).reshape(batch, heads_per_rank, world_size * seq_local, dim).contiguous()
    return x_head, events


def all2all_heads_to_sequence(x_head: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, CudaEventPair]:
    if world_size == 1:
        return x_head.contiguous(), None

    batch, heads_per_rank, seq_len, dim = x_head.shape
    assert seq_len % world_size == 0
    seq_local = seq_len // world_size
    send = x_head.view(batch, heads_per_rank, world_size, seq_local, dim).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)

    _, events = record_cuda_region(lambda: dist.all_to_all_single(recv, send), "all2all_heads_to_sequence")
    x_seq = recv.permute(1, 0, 2, 3, 4).reshape(batch, world_size * heads_per_rank, seq_local, dim).contiguous()
    return x_seq, events


def start_qkv_allgather(local: Dict, world_size: int, device: torch.device):
    if world_size == 1:
        return None

    current_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device)
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    gathered = {}
    works = []

    with torch.cuda.stream(comm_stream), torch.profiler.record_function("overlap_qkv_allgather_start"):
        comm_stream.wait_stream(current_stream)
        comm_start.record(comm_stream)
        for name in ("query", "key", "value"):
            inp = local[f"{name}_seq"].contiguous().view(-1)
            out = torch.empty(inp.numel() * world_size, device=device, dtype=inp.dtype)
            work = dist.all_gather_into_tensor(out, inp, async_op=True)
            gathered[name] = out
            works.append(work)
        comm_end.record(comm_stream)

    return {
        "stream": comm_stream,
        "start": comm_start,
        "end": comm_end,
        "works": works,
        "gathered": gathered,
    }


def wait_qkv_allgather(handle: Optional[Dict], device: torch.device) -> CudaEventPair:
    if handle is None:
        return None

    current_stream = torch.cuda.current_stream(device)
    wait_start = torch.cuda.Event(enable_timing=True)
    wait_end = torch.cuda.Event(enable_timing=True)
    with torch.profiler.record_function("overlap_qkv_allgather_wait"):
        wait_start.record(current_stream)
        current_stream.wait_stream(handle["stream"])
        wait_end.record(current_stream)
    return wait_start, wait_end


def finish_qkv_allgather(handle: Optional[Dict]):
    if handle is None:
        return
    for work in handle["works"]:
        work.wait()
    del handle["gathered"]


def _flatten_density(nested) -> List[float]:
    out: List[float] = []
    if isinstance(nested, (list, tuple)):
        for x in nested:
            out.extend(_flatten_density(x))
    else:
        out.append(float(nested))
    return out


def load_density_log(path: str) -> Dict[int, List[Tuple[int, List[float]]]]:
    """Parse a density JSONL into {layer_idx: [(timestep, [density_per_head]), ...]} in denoising order (timestep desc)."""
    by_layer: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            layer = int(row["layer"])
            ts = int(row["timestep"])
            vals = _flatten_density(row["density"])
            by_layer[layer].append((ts, vals))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda r: -r[0])
    return by_layer


def pick_prev_density(
    log: Dict[int, List[Tuple[int, List[float]]]], layer: int, timestep: int
) -> Optional[List[float]]:
    """Return the density vector from the timestep immediately *before* `timestep` for this layer in denoising order.

    Denoising order is descending timesteps. The "previous" timestep is the smallest
    recorded timestep that is strictly larger than the current one.
    """
    rows = log.get(layer)
    if not rows:
        return None
    prev = None
    for ts, vals in rows:  # iterating desc
        if ts > timestep:
            prev = (ts, vals)
        else:
            break
    if prev is None:
        return None
    return prev[1]


def greedy_head_assignment(
    densities: Sequence[float], world_size: int
) -> Tuple[List[List[int]], List[float]]:
    """Assign heads to ranks greedily with equal heads-per-rank.

    Sort heads by density desc; place each onto the least-loaded rank that still
    has capacity (= n_heads / world_size). Returns (heads_per_rank, predicted_load).
    Within each rank the returned head-index list is sorted ascending.
    """
    n = len(densities)
    if n % world_size != 0:
        raise ValueError(f"n_heads={n} not divisible by world_size={world_size}")
    cap = n // world_size
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -densities[i])
    for head in order:
        best = -1
        for r in range(world_size):
            if len(assigned[r]) >= cap:
                continue
            if best == -1 or loads[r] < loads[best]:
                best = r
        assigned[best].append(head)
        loads[best] += densities[head]
    return [sorted(h) for h in assigned], loads


def load_cost_model(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def estimate_head_costs(densities: Sequence[float], seq_len: int, cost_model: Optional[Dict]) -> List[float]:
    if cost_model is None:
        return [float(d) for d in densities]
    mask_slope = float(cost_model["mask_fit"]["slope_ms_per_unit"])
    attention_slope = float(cost_model["attention_fit"]["slope_ms_per_unit"])
    return [
        mask_slope * seq_len + attention_slope * float(density) * seq_len * seq_len
        for density in densities
    ]


def greedy_lpt_assignment(
    densities: Sequence[float], world_size: int, min_heads_per_rank: int = 1
) -> Tuple[List[List[int]], List[float]]:
    """Longest-Processing-Time-first greedy, no equal-heads constraint.

    Sort heads by density desc; place each onto the least-loaded rank (no cap).
    `min_heads_per_rank` guarantees every rank gets at least that many heads
    (seeded with the largest densities in round-robin) so none stays idle.
    Returns (heads_per_rank, predicted_load); per-rank lists sorted ascending.
    """
    n = len(densities)
    if min_heads_per_rank * world_size > n:
        raise ValueError(
            f"min_heads_per_rank={min_heads_per_rank} * world_size={world_size} > n_heads={n}"
        )
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -densities[i])

    # Seed: give each rank `min_heads_per_rank` heads round-robin from the top.
    idx = 0
    for seed in range(min_heads_per_rank):
        for r in range(world_size):
            head = order[idx]
            assigned[r].append(head)
            loads[r] += densities[head]
            idx += 1

    # Remaining heads: pure LPT — least-loaded rank wins.
    for head in order[idx:]:
        best = min(range(world_size), key=lambda r: loads[r])
        assigned[best].append(head)
        loads[best] += densities[head]

    return [sorted(h) for h in assigned], loads


def pad_rank_heads_uniform(
    assigned: List[List[int]],
) -> Tuple[List[int], List[int], int]:
    """Pad per-rank head lists to a uniform length so the symmetric all2all works.

    Each rank's slot is padded by duplicating its first real head. Returns:
      - head_order_padded: flat list of length max_hpr * world_size (may contain duplicates)
      - real_heads_per_rank: how many heads at the start of each rank's slot are real
      - max_hpr: slot size per rank
    """
    world_size = len(assigned)
    max_hpr = max(len(a) for a in assigned)
    head_order: List[int] = []
    real_counts: List[int] = []
    for rank_heads in assigned:
        if not rank_heads:
            raise ValueError("cannot pad an empty rank slot; use min_heads_per_rank>=1")
        real_counts.append(len(rank_heads))
        head_order.extend(rank_heads)
        head_order.extend([rank_heads[0]] * (max_hpr - len(rank_heads)))
    return head_order, real_counts, max_hpr


def compute_rank_heads(
    log_path: Optional[str],
    metadata: Dict,
    num_heads: int,
    world_size: int,
    strategy: str,
    rank: int = 0,
    min_heads_per_rank: int = 1,
    cost_model: Optional[Dict] = None,
) -> Tuple[List[List[int]], Optional[Dict]]:
    """Return per-rank head lists for the requested strategy.

    strategy:
      - "contiguous": [r*hpr : (r+1)*hpr] for each r (requires num_heads % world_size == 0)
      - "greedy":     equal heads per rank, greedy by prev-step density
      - "greedy_unequal": LPT, variable heads per rank (>= min_heads_per_rank)

    Falls back to contiguous if the density log can't provide prev density.
    """
    heads_per_rank = num_heads // world_size

    def contiguous_split() -> List[List[int]]:
        if num_heads % world_size != 0:
            raise ValueError(
                f"contiguous split requires num_heads ({num_heads}) divisible by "
                f"world_size ({world_size})"
            )
        return [list(range(r * heads_per_rank, (r + 1) * heads_per_rank)) for r in range(world_size)]

    if strategy == "contiguous":
        return contiguous_split(), None
    if strategy not in ("greedy", "greedy_unequal"):
        raise ValueError(f"unknown strategy: {strategy}")

    if log_path is None:
        return contiguous_split(), None

    log = load_density_log(log_path)
    layer = int(metadata.get("layer_idx"))
    timestep = metadata.get("timestep")
    if timestep is None:
        timestep = metadata.get("linear_step")
    if timestep is None:
        if rank == 0:
            print("[greedy] metadata has no timestep/linear_step; falling back to contiguous")
        return contiguous_split(), None

    prev = pick_prev_density(log, layer, int(timestep))
    if prev is None:
        if rank == 0:
            print(
                f"[greedy] no prior density for layer={layer} before timestep={timestep}; "
                "falling back to contiguous"
            )
        return contiguous_split(), None
    if len(prev) != num_heads:
        if rank == 0:
            print(
                f"[greedy] prior density length {len(prev)} != num_heads {num_heads}; "
                "falling back to contiguous"
            )
        return contiguous_split(), None

    seq_len = int(metadata.get("seq_len"))
    costs = estimate_head_costs(prev, seq_len, cost_model)

    if strategy == "greedy":
        assigned, predicted_loads = greedy_head_assignment(costs, world_size)
    else:  # greedy_unequal
        assigned, predicted_loads = greedy_lpt_assignment(costs, world_size, min_heads_per_rank)

    predicted_contig = (
        [sum(costs[g * heads_per_rank : (g + 1) * heads_per_rank]) for g in range(world_size)]
        if num_heads % world_size == 0
        else None
    )

    info = {
        "strategy": strategy,
        "layer": layer,
        "timestep": int(timestep),
        "prev_density": prev,
        "head_costs": costs,
        "cost_model": "maskgen_aware" if cost_model is not None else "density",
        "assigned": assigned,
        "predicted_loads": predicted_loads,
        "predicted_contig": predicted_contig,
        "heads_per_rank": [len(a) for a in assigned],
    }
    return assigned, info


def compute_head_order(
    log_path: Optional[str],
    metadata: Dict,
    num_heads: int,
    world_size: int,
    rank: int,
    cost_model: Optional[Dict] = None,
) -> Tuple[List[int], Optional[Dict]]:
    """Decide the global head ordering used before the seq->head all2all.

    Returns (head_order, info). `head_order` is a permutation of range(num_heads)
    where slots [r*cap:(r+1)*cap] are the global heads assigned to rank r.
    `info` (rank-0-friendly) carries diagnostics; None when falling back.
    """
    heads_per_rank = num_heads // world_size
    contiguous = list(range(num_heads))
    if log_path is None:
        return contiguous, None

    log = load_density_log(log_path)
    layer = int(metadata.get("layer_idx"))
    timestep = metadata.get("timestep")
    if timestep is None:
        timestep = metadata.get("linear_step")
    if timestep is None:
        if rank == 0:
            print(f"[greedy] metadata has no timestep/linear_step; falling back to contiguous")
        return contiguous, None

    prev = pick_prev_density(log, layer, int(timestep))
    if prev is None:
        if rank == 0:
            print(
                f"[greedy] no prior density in log for layer={layer} "
                f"before timestep={timestep}; falling back to contiguous"
            )
        return contiguous, None
    if len(prev) != num_heads:
        if rank == 0:
            print(
                f"[greedy] prior density length {len(prev)} != num_heads {num_heads}; "
                "falling back to contiguous"
            )
        return contiguous, None

    seq_len = int(metadata.get("seq_len"))
    costs = estimate_head_costs(prev, seq_len, cost_model)
    assigned, predicted_loads = greedy_head_assignment(costs, world_size)
    head_order = [h for rank_heads in assigned for h in rank_heads]

    predicted_contig = [
        sum(costs[g * heads_per_rank : (g + 1) * heads_per_rank]) for g in range(world_size)
    ]

    info = {
        "layer": layer,
        "timestep": int(timestep),
        "prev_density": prev,
        "head_costs": costs,
        "cost_model": "maskgen_aware" if cost_model is not None else "density",
        "assigned": assigned,
        "predicted_loads": predicted_loads,
        "predicted_contig": predicted_contig,
    }
    return head_order, info


def load_local_shards(
    path: str,
    rank: int,
    world_size: int,
    device: torch.device,
    head_order: Optional[List[int]] = None,
    real_heads_per_rank: Optional[List[int]] = None,
    symm_a2a=None,
    h_idxs_r: Optional[List[int]] = None,
    max_heads_per_rank: Optional[int] = None,
):
    """Load the local (seq-sharded) QKV + centroid caches for this rank.

    `head_order`: optional flat list of global head indices. Its length must be a
    multiple of `world_size`; rank r gets slots `[r*hpr : (r+1)*hpr]` where
    `hpr = len(head_order) // world_size`. For `greedy_unequal`, the list may
    contain duplicates (pad slots filled by duplicating a real head) so that
    the symmetric all2all still works.

    `real_heads_per_rank`: when padding is used, how many of each rank's slots
    are real heads. Padded slots are ignored downstream (density / metrics).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    query = data["inputs"]["query"]
    key = data["inputs"]["key"]
    value = data["inputs"]["value"]
    q_cache = data["centroid_cache"]["q_centroids"]
    k_cache = data["centroid_cache"]["k_centroids"]

    cfg, num_heads_original, seq_len, dim = query.shape
    if cfg != 1:
        raise ValueError(f"Only cfg=1 is supported by this benchmark, got cfg={cfg}")
    if seq_len % world_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by world_size={world_size}")

    seq_local = seq_len // world_size
    s0, s1 = rank * seq_local, (rank + 1) * seq_local

    if symm_a2a is not None:
        # Asymmetric pull path: populate symm buffers with this rank's seq shard
        # for ALL heads (no permute — the kernel gathers by global index).
        if h_idxs_r is None:
            raise ValueError("symm_a2a path requires h_idxs_r")
        q_shard = query[:, :, s0:s1, :].to(device).contiguous()
        k_shard = key[:, :, s0:s1, :].to(device).contiguous()
        v_shard = value[:, :, s0:s1, :].to(device).contiguous()
        # Only the first `s_local` seq slots carry real data; the remaining
        # [s_local:s_local_padded] are garbage (TMA block-shape padding) and
        # will be pulled along with real data, then trimmed on the recv side.
        real_s = symm_a2a.s_local
        symm_a2a.q_symm[:, :, :real_s, :].copy_(q_shard)
        symm_a2a.k_symm[:, :, :real_s, :].copy_(k_shard)
        symm_a2a.v_symm[:, :, :real_s, :].copy_(v_shard)

        # Centroid caches stay local (never communicated): gather by h_idxs_r.
        h_idxs_long = torch.tensor(list(h_idxs_r), dtype=torch.long)
        q_cache_local = q_cache.index_select(0, h_idxs_long).contiguous().to(device)
        k_cache_local = k_cache.index_select(0, h_idxs_long).contiguous().to(device)

        max_hpr = int(max_heads_per_rank) if max_heads_per_rank else len(h_idxs_r)
        local = {
            "metadata": metadata,
            "symm_a2a": symm_a2a,
            "h_idxs_r": torch.tensor(list(h_idxs_r), dtype=torch.int32, device=device),
            "h_idxs_r_py": list(h_idxs_r),
            "q_cache": q_cache_local,
            "k_cache": k_cache_local,
            "head_start": int(h_idxs_r[0]) if len(h_idxs_r) else 0,
            "head_end": int(h_idxs_r[0]) + len(h_idxs_r) if len(h_idxs_r) else 0,
            # reverse all2all still goes through padded symmetric path, so we
            # need a uniform `heads_per_rank_padded` across ranks for it.
            "heads_per_rank_padded": max_hpr,
            "real_heads_count": len(h_idxs_r),
            "assigned_heads": list(h_idxs_r),
            "seq_start": s0,
            "seq_end": s1,
            "seq_total": seq_len,
        }
        del data, query, key, value, q_cache, k_cache
        gc.collect()
        return local

    if head_order is not None:
        if len(head_order) % world_size != 0:
            raise ValueError(
                f"head_order length {len(head_order)} not divisible by world_size {world_size}"
            )
        heads_per_rank = len(head_order) // world_size
        h0, h1 = rank * heads_per_rank, (rank + 1) * heads_per_rank
        if not all(0 <= h < num_heads_original for h in head_order):
            raise ValueError("head_order contains out-of-range head indices")
        perm = torch.tensor(head_order, dtype=torch.long)
        query = query.index_select(1, perm)
        key = key.index_select(1, perm)
        value = value.index_select(1, perm)
        q_cache = q_cache.index_select(0, perm)
        k_cache = k_cache.index_select(0, perm)

        if real_heads_per_rank is not None:
            real_count = int(real_heads_per_rank[rank])
            assigned_heads = list(head_order[h0 : h0 + real_count])
        else:
            real_count = heads_per_rank
            assigned_heads = list(head_order[h0:h1])
    else:
        if num_heads_original % world_size != 0:
            raise ValueError(
                f"num_heads={num_heads_original} must be divisible by world_size={world_size}"
            )
        heads_per_rank = num_heads_original // world_size
        h0, h1 = rank * heads_per_rank, (rank + 1) * heads_per_rank
        real_count = heads_per_rank
        assigned_heads = list(range(h0, h1))

    local = {
        "metadata": metadata,
        "query_seq": query[:, :, s0:s1, :].contiguous().to(device),
        "key_seq": key[:, :, s0:s1, :].contiguous().to(device),
        "value_seq": value[:, :, s0:s1, :].contiguous().to(device),
        "q_cache": q_cache[h0:h1].contiguous().to(device),
        "k_cache": k_cache[h0:h1].contiguous().to(device),
        "head_start": h0,
        "head_end": h1,
        "heads_per_rank_padded": heads_per_rank,
        "real_heads_count": real_count,
        "assigned_heads": assigned_heads,
        "seq_start": s0,
        "seq_end": s1,
    }

    del data, query, key, value, q_cache, k_cache
    gc.collect()
    return local


def run_iteration(local: Dict, args: argparse.Namespace, rank: int, world_size: int, device: torch.device):
    symm_a2a = local.get("symm_a2a")
    if symm_a2a is not None:
        # Asymmetric pull: no permute needed, kernel gathers by global head idx.
        # Low-sync path: Q/K/V are write-once (populated in load_local_shards
        # and guarded by a one-time setup barrier in main). Per-iter pulls run
        # barrier-free; visibility is already established.
        h_idxs_r = local["h_idxs_r"]
        num_sms = args.num_sms if args.num_sms > 0 else None

        def _pull(name: str):
            return symm_a2a.pull_seq_to_heads(
                name, h_idxs_r, num_sms=num_sms,
                pre_barrier=False, post_barrier=False,
            )

        q_head_full, q_a2a_events = record_cuda_region(
            lambda: _pull("q"), "all2all_sequence_to_heads",
        )
        k_head_full, k_a2a_events = record_cuda_region(
            lambda: _pull("k"), "all2all_sequence_to_heads",
        )
        v_head_full, v_a2a_events = record_cuda_region(
            lambda: _pull("v"), "all2all_sequence_to_heads",
        )
        # Pull output is [B, H_local, world * S_local_padded, D]. Trim per peer
        # back to the real S_local (when S_block had to pad for divisibility).
        seq_total = local["seq_total"]
        if q_head_full.shape[2] != seq_total:
            s_padded = q_head_full.shape[2] // world_size
            s_real = seq_total // world_size
            def _trim(x):
                x = x.view(x.shape[0], x.shape[1], world_size, s_padded, x.shape[3])
                x = x[:, :, :, :s_real, :].contiguous()
                return x.view(x.shape[0], x.shape[1], world_size * s_real, x.shape[4])
            q_head = _trim(q_head_full)
            k_head = _trim(k_head_full)
            v_head = _trim(v_head_full)
        else:
            q_head, k_head, v_head = q_head_full, k_head_full, v_head_full
        real_count = q_head.shape[1]
        # For the reverse (still padded symmetric all2all), pad attn_out to
        # a uniform width across ranks.
        heads_padded = int(local.get("heads_per_rank_padded", real_count))
    else:
        q_head, q_a2a_events = all2all_sequence_to_heads(local["query_seq"], world_size)
        k_head, k_a2a_events = all2all_sequence_to_heads(local["key_seq"], world_size)
        v_head, v_a2a_events = all2all_sequence_to_heads(local["value_seq"], world_size)

        # When padding was used to keep the all2all symmetric, slice off the pad
        # slots here so mask/kmeans/attention only see this rank's real heads.
        heads_padded = q_head.shape[1]
        real_count = int(local.get("real_heads_count", heads_padded))
        if real_count < heads_padded:
            q_head = q_head[:, :real_count, :, :].contiguous()
            k_head = k_head[:, :real_count, :, :].contiguous()
            v_head = v_head[:, :real_count, :, :].contiguous()

    metadata = local["metadata"]
    state = WanSPState(
        num_q_centroids=int(metadata["num_q_centroids"]),
        num_k_centroids=int(metadata["num_k_centroids"]),
        top_p_kmeans=float(metadata["top_p_kmeans"]),
        min_kc_ratio=float(metadata["min_kc_ratio"]),
        kmeans_iter_step=int(metadata["kmeans_iter_step"]),
        q_centroids=local["q_cache"][:real_count].clone(),
        k_centroids=local["k_cache"][:real_count].clone(),
    )

    overlap_handle = None
    if args.overlap_qkv_allgather_during_mask:
        overlap_handle = start_qkv_allgather(local, world_size, device)

    mask_outputs, mask_events = record_cuda_region(
        lambda: state.semantic_aware_permutation(q_head, k_head, v_head), "mask_semantic_aware_permutation"
    )
    qkv_allgather_wait_events = wait_qkv_allgather(overlap_handle, device)
    q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, q_sorted_indices, qlabels, klabels = mask_outputs

    attn_out_permuted, attention_events = record_cuda_region(
        lambda: dynamic_block_sparse_fwd_flashinfer(q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False),
        "dynamic_block_sparse_attention",
    )
    attn_out, inverse_permute_events = record_cuda_region(
        lambda: apply_inverse_permutation_triton(attn_out_permuted, q_sorted_indices, dim=2), "inverse_permute"
    )

    if symm_a2a is not None:
        # Reverse asymm pull: write attn_out into attn_symm (per-peer S padding),
        # kernel scatters each global head directly to its slot on every rank.
        B_, H_loc_, S_full_, D_ = attn_out.shape
        assert S_full_ == world_size * (S_full_ // world_size)
        s_real = S_full_ // world_size
        s_padded = symm_a2a.s_local_padded
        max_hpr = symm_a2a.max_h_per_rank
        attn_symm_view = symm_a2a.attn_symm.view(B_, max_hpr, world_size, s_padded, D_)
        attn_out_view = attn_out.reshape(B_, H_loc_, world_size, s_real, D_)
        attn_symm_view[:, :H_loc_, :, :s_real, :].copy_(attn_out_view)

        num_sms = args.num_sms if args.num_sms > 0 else None

        # One barrier sequences attn_symm write → rev pull. No post-barrier:
        # the `dist.barrier()` at the end of each outer iter guarantees peers
        # finish reading attn_symm before the next iter's write overwrites it.
        def _rev_pull():
            symm_a2a.barrier()
            return symm_a2a.pull_heads_to_seq(
                num_sms=num_sms, pre_barrier=False, post_barrier=False,
            )

        out_seq_full, all2all_out_events = record_cuda_region(
            _rev_pull, "all2all_heads_to_sequence",
        )
        # [B, H_total, s_padded, D] → trim padding to [B, H_total, s_real, D]
        if s_padded != s_real:
            out_seq = out_seq_full[:, :, :s_real, :].contiguous()
        else:
            out_seq = out_seq_full
    else:
        # Pad attn_out back to the symmetric head count before the reverse all2all.
        # The pad slots' outputs are junk; downstream is expected to ignore them.
        if real_count < heads_padded:
            pad = torch.zeros(
                attn_out.shape[0],
                heads_padded - real_count,
                attn_out.shape[2],
                attn_out.shape[3],
                device=attn_out.device,
                dtype=attn_out.dtype,
            )
            attn_out_for_a2a = torch.cat([attn_out, pad], dim=1).contiguous()
        else:
            attn_out_for_a2a = attn_out

        out_seq, all2all_out_events = all2all_heads_to_sequence(attn_out_for_a2a, world_size)
    density_gpu = density_calculation(dyn_map, qc_sz, kc_sz).reshape(-1).detach().float()
    q_chunk_density_gpu = q_sequence_chunk_density(
        dyn_map,
        qlabels,
        kc_sz,
        seq_len=q_head.shape[2],
        chunks=args.q_density_chunks,
    ).reshape(q_head.shape[1], args.q_density_chunks).detach().float()

    torch.cuda.synchronize(device)
    finish_qkv_allgather(overlap_handle)

    all2all_in_ms = cuda_elapsed_ms(q_a2a_events) + cuda_elapsed_ms(k_a2a_events) + cuda_elapsed_ms(v_a2a_events)
    mask_ms = cuda_elapsed_ms(mask_events)
    qkv_allgather_comm_ms = (
        cuda_elapsed_ms((overlap_handle["start"], overlap_handle["end"])) if overlap_handle is not None else 0.0
    )
    qkv_allgather_wait_ms = cuda_elapsed_ms(qkv_allgather_wait_events)
    attention_ms = cuda_elapsed_ms(attention_events)
    inverse_permute_ms = cuda_elapsed_ms(inverse_permute_events)
    all2all_out_ms = cuda_elapsed_ms(all2all_out_events)
    density = density_gpu.cpu()
    q_chunk_density = q_chunk_density_gpu.cpu()

    metrics = {
        "rank": rank,
        "head_start": local["head_start"],
        "head_end": local["head_end"],
        "seq_start": local["seq_start"],
        "seq_end": local["seq_end"],
        "all2all_in_ms": all2all_in_ms,
        "mask_ms": mask_ms,
        "qkv_allgather_comm_ms": qkv_allgather_comm_ms,
        "qkv_allgather_wait_ms": qkv_allgather_wait_ms,
        "attention_ms": attention_ms,
        "inverse_permute_ms": inverse_permute_ms,
        "all2all_out_ms": all2all_out_ms,
        "total_ms": all2all_in_ms + mask_ms + qkv_allgather_wait_ms + attention_ms + inverse_permute_ms + all2all_out_ms,
    }

    assigned_heads = local.get("assigned_heads") or list(range(local["head_start"], local["head_end"]))
    real_count = int(local.get("real_heads_count", len(assigned_heads)))
    head_density = [
        {"global_head": int(assigned_heads[i]), "rank": rank, "density": float(value)}
        for i, value in enumerate(density.tolist()[:real_count])
    ]

    q_chunk_density_rows = []
    for local_head_idx, values in enumerate(q_chunk_density.tolist()[:real_count]):
        global_head = int(assigned_heads[local_head_idx])
        for chunk_idx, value in enumerate(values):
            start = q_head.shape[2] * chunk_idx // args.q_density_chunks
            end = q_head.shape[2] * (chunk_idx + 1) // args.q_density_chunks
            q_chunk_density_rows.append(
                {
                    "global_head": global_head,
                    "rank": rank,
                    "q_chunk": chunk_idx,
                    "q_start": start,
                    "q_end": end,
                    "q_len": end - start,
                    "density": float(value),
                }
            )

    del q_head, k_head, v_head, mask_outputs, q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz
    del q_sorted_indices, qlabels, klabels, attn_out_permuted, attn_out, out_seq, density_gpu, q_chunk_density_gpu, state
    return metrics, head_density, q_chunk_density_rows


def summarize_rank_metrics(rows: List[Dict]) -> List[Dict]:
    by_rank: Dict[int, List[Dict]] = {}
    for row in rows:
        by_rank.setdefault(int(row["rank"]), []).append(row)

    summary = []
    for rank, rank_rows in sorted(by_rank.items()):
        out = {
            "rank": rank,
            "head_start": rank_rows[0]["head_start"],
            "head_end": rank_rows[0]["head_end"],
            "seq_start": rank_rows[0]["seq_start"],
            "seq_end": rank_rows[0]["seq_end"],
            "iters": len(rank_rows),
        }
        for key in [
            "all2all_in_ms",
            "mask_ms",
            "qkv_allgather_comm_ms",
            "qkv_allgather_wait_ms",
            "attention_ms",
            "inverse_permute_ms",
            "all2all_out_ms",
            "total_ms",
        ]:
            vals = [float(row[key]) for row in rank_rows]
            out[f"{key}_mean"] = statistics.mean(vals)
            out[f"{key}_median"] = statistics.median(vals)
        summary.append(out)
    return summary


def summarize_density(rows: List[Dict]) -> List[Dict]:
    by_head: Dict[int, List[Dict]] = {}
    for row in rows:
        by_head.setdefault(int(row["global_head"]), []).append(row)

    summary = []
    for head, head_rows in sorted(by_head.items()):
        vals = [float(row["density"]) for row in head_rows]
        summary.append(
            {
                "global_head": head,
                "rank": int(head_rows[0]["rank"]),
                "density_mean": statistics.mean(vals),
                "density_min": min(vals),
                "density_max": max(vals),
            }
        )
    return summary


def summarize_q_chunk_density(rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        key = (int(row["global_head"]), int(row["q_chunk"]))
        by_key.setdefault(key, []).append(row)

    summary = []
    for (head, chunk), chunk_rows in sorted(by_key.items()):
        vals = [float(row["density"]) for row in chunk_rows]
        first = chunk_rows[0]
        summary.append(
            {
                "global_head": head,
                "rank": int(first["rank"]),
                "q_chunk": chunk,
                "q_start": int(first["q_start"]),
                "q_end": int(first["q_end"]),
                "q_len": int(first["q_len"]),
                "density_mean": statistics.mean(vals),
                "density_min": min(vals),
                "density_max": max(vals),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rank_summary(rows: List[Dict]):
    header = (
        "rank heads      seq           a2a_in  mask     qkv_ag  ag_wait  attention inv_perm a2a_out  total"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>4} "
            f"{row['head_start']:>2}-{row['head_end'] - 1:<5} "
            f"{row['seq_start']:>6}-{row['seq_end'] - 1:<6} "
            f"{row['all2all_in_ms_mean']:>7.2f} "
            f"{row['mask_ms_mean']:>8.2f} "
            f"{row['qkv_allgather_comm_ms_mean']:>7.2f} "
            f"{row['qkv_allgather_wait_ms_mean']:>8.2f} "
            f"{row['attention_ms_mean']:>9.2f} "
            f"{row['inverse_permute_ms_mean']:>8.2f} "
            f"{row['all2all_out_ms_mean']:>7.2f} "
            f"{row['total_ms_mean']:>7.2f}"
        )


def print_density_summary(rows: List[Dict]):
    print("\nper-head density:")
    for row in rows:
        print(f"head {row['global_head']:>2} rank {row['rank']:>2}: {row['density_mean']:.6f}")


def print_q_chunk_density_summary(rows: List[Dict]):
    print("\nper-head q-chunk density:")
    for row in rows:
        print(
            f"head {row['global_head']:>2} rank {row['rank']:>2} "
            f"chunk {row['q_chunk']:>2} q[{row['q_start']},{row['q_end']}): "
            f"{row['density_mean']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Reproduce Wan SVG2 sequence-parallel all2all + per-rank head attention.")
    parser.add_argument("--input", required=True, help="Path to SVG_WAN_ATTN_EXPORT_PATH .pt file.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rank-csv", default=None)
    parser.add_argument("--density-csv", default=None)
    parser.add_argument("--q-chunk-density-csv", default=None)
    parser.add_argument(
        "--q-density-chunks",
        type=int,
        default=8,
        help="Number of contiguous chunks along q sequence for per-head chunk density reporting.",
    )
    parser.add_argument(
        "--overlap-qkv-allgather-during-mask",
        action="store_true",
        help="Start async all-gather of sequence-sharded Q/K/V on a separate CUDA stream while mask/permutation runs.",
    )
    parser.add_argument(
        "--density-log",
        default=None,
        help="Path to a density JSONL (layer/timestep/density). When set, heads are "
        "redistributed across ranks with greedy balancing using the density from "
        "the previous timestep of this layer (the only predictor available at runtime).",
    )
    parser.add_argument(
        "--balance",
        choices=["contiguous", "greedy", "greedy_unequal"],
        default="contiguous",
        help="How to assign heads to ranks. 'greedy'/'greedy_unequal' require --density-log. "
        "'greedy_unequal' drops the equal-heads constraint; pads shorter ranks by duplicating "
        "a real head so the symmetric all2all still works (v1 padding impl; padded slots "
        "waste compute but are masked from density reporting).",
    )
    parser.add_argument(
        "--min-heads-per-rank",
        type=int,
        default=1,
        help="For --balance greedy_unequal: guarantee each rank gets at least this many heads.",
    )
    parser.add_argument(
        "--cost-model-json",
        default=None,
        help="Optional model from profile_wan_maskgen_aware_cost.py. With --balance greedy, "
        "predicts load as mask_slope*seq_len + attn_slope*density*seq_len^2 instead of raw density.",
    )
    parser.add_argument(
        "--asymm-a2a",
        choices=["off", "pull_qkv"],
        default="off",
        help="Replace both forward (seq→heads for Q/K/V) and reverse "
        "(heads→seq for attn_out) all2alls with pull-based kernels over "
        "torch symmetric memory (Triton + TMA). Reverse scatters by global "
        "head index on the fly, so no pad+permute needed.",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=0,
        help="Persistent kernel grid size. 0 = device SM count.",
    )
    parser.add_argument(
        "--s-block",
        type=int,
        default=0,
        help="Kernel S_BLOCK. 0 = auto (largest divisor of S_local that is <=128).",
    )
    parser.add_argument("--profile-dir", default=None, help="Directory for per-rank torch profiler Chrome traces.")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-shapes", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    args = parser.parse_args()
    if args.q_density_chunks <= 0:
        raise SystemExit("--q-density-chunks must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.balance in ("greedy", "greedy_unequal") and args.density_log is None:
        raise SystemExit(f"--balance {args.balance} requires --density-log")
    cost_model = load_cost_model(args.cost_model_json)

    # Peek at metadata + head count to compute the greedy assignment (if enabled).
    head_order: Optional[List[int]] = None
    real_heads_per_rank: Optional[List[int]] = None
    greedy_info: Optional[Dict] = None
    assigned: Optional[List[List[int]]] = None
    if args.balance in ("greedy", "greedy_unequal"):
        peek = torch.load(args.input, map_location="cpu", weights_only=False)
        peek_metadata = peek["metadata"]
        num_heads = int(peek["inputs"]["query"].shape[1])
        del peek
        gc.collect()
        assigned, greedy_info = compute_rank_heads(
            args.density_log,
            peek_metadata,
            num_heads,
            world_size,
            strategy=args.balance,
            rank=rank,
            min_heads_per_rank=args.min_heads_per_rank,
            cost_model=cost_model,
        )
        if args.balance == "greedy_unequal":
            head_order, real_heads_per_rank, max_hpr = pad_rank_heads_uniform(assigned)
            if rank == 0 and greedy_info is not None:
                print(
                    f"[greedy_unequal] heads/rank={real_heads_per_rank} "
                    f"padded to {max_hpr} each (pads duplicate rank's first real head)"
                )
        else:
            # equal-split greedy: assigned already has uniform size
            head_order = [h for rank_heads in assigned for h in rank_heads]

    # Asymmetric all2all setup: allocate Q/K/V directly in symmetric memory and
    # skip the forward padded permute path entirely.
    symm_a2a = None
    h_idxs_r_asymm: Optional[List[int]] = None
    max_hpr_asymm: Optional[int] = None
    if args.asymm_a2a == "pull_qkv":
        from symm_a2a import SymmAsymA2A

        if assigned is None:
            # contiguous balance: generate trivial assignment
            peek = torch.load(args.input, map_location="cpu", weights_only=False)
            num_heads = int(peek["inputs"]["query"].shape[1])
            cfg, _, seq_len, dim = peek["inputs"]["query"].shape
            dtype = peek["inputs"]["query"].dtype
            del peek
            gc.collect()
            if num_heads % world_size != 0:
                raise SystemExit(
                    f"contiguous balance + asymm requires num_heads ({num_heads}) divisible by world_size"
                )
            hpr = num_heads // world_size
            assigned = [list(range(r * hpr, (r + 1) * hpr)) for r in range(world_size)]
        else:
            peek = torch.load(args.input, map_location="cpu", weights_only=False)
            cfg, num_heads, seq_len, dim = peek["inputs"]["query"].shape
            dtype = peek["inputs"]["query"].dtype
            del peek
            gc.collect()

        h_idxs_r_asymm = assigned[rank]
        max_hpr_asymm = max(len(h) for h in assigned)
        s_local_real = seq_len // world_size
        s_block = args.s_block if args.s_block > 0 else pick_s_block(s_local_real)
        if s_block <= 0 or (s_block & (s_block - 1)) != 0:
            raise SystemExit(
                f"--s-block must be a power of 2, got {s_block}"
            )

        symm_a2a = SymmAsymA2A(
            dist.group.WORLD,
            buffer_shape=(cfg, num_heads, s_local_real, dim),
            dtype=dtype,
            device=device,
            s_block=s_block,
            h_idxs_all=assigned,
        )
        if rank == 0:
            print(
                f"[asymm_a2a=pull_qkv] symm buffers [{cfg},{num_heads},{s_local_real},{dim}] "
                f"dtype={dtype} s_block={s_block} num_sms={args.num_sms or 'auto'}; "
                f"per-rank heads={[len(h) for h in assigned]} max_hpr={max_hpr_asymm}"
            )
        # Asymm path drives its own Q/K/V — don't also pad+permute in load_local_shards.
        head_order = None
        real_heads_per_rank = None

    local = load_local_shards(
        args.input, rank, world_size, device,
        head_order=head_order, real_heads_per_rank=real_heads_per_rank,
        symm_a2a=symm_a2a,
        h_idxs_r=h_idxs_r_asymm,
        max_heads_per_rank=max_hpr_asymm,
    )
    # One-time Q/K/V visibility barrier. With pre/post barriers removed from
    # per-iter pulls, every iter relies on this single symm-mem barrier to
    # make peers' one-shot Q/K/V writes visible. Q/K/V are never rewritten,
    # so this is sufficient for the whole run.
    if symm_a2a is not None:
        symm_a2a.barrier()
    dist.barrier()

    for _ in range(args.warmup):
        run_iteration(local, args, rank, world_size, device)
        dist.barrier()
        torch.cuda.empty_cache()

    profiler_ctx = nullcontext()
    if args.profile_dir:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_stack,
        )

    rank_rows = []
    density_rows = []
    q_chunk_density_rows = []
    with profiler_ctx as prof:
        for _ in range(args.iters):
            metrics, densities, q_chunk_densities = run_iteration(local, args, rank, world_size, device)
            rank_rows.append(metrics)
            density_rows.extend(densities)
            q_chunk_density_rows.extend(q_chunk_densities)
            dist.barrier()
            if args.profile_dir:
                prof.step()

    if args.profile_dir:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / f"rank{rank}.json"
        prof.export_chrome_trace(str(trace_path))

    gathered_rank_rows = [None for _ in range(world_size)] if rank == 0 else None
    gathered_density_rows = [None for _ in range(world_size)] if rank == 0 else None
    gathered_q_chunk_density_rows = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(rank_rows, gathered_rank_rows, dst=0)
    dist.gather_object(density_rows, gathered_density_rows, dst=0)
    dist.gather_object(q_chunk_density_rows, gathered_q_chunk_density_rows, dst=0)

    if rank == 0:
        flat_rank_rows = [row for rank_part in gathered_rank_rows for row in rank_part]
        flat_density_rows = [row for rank_part in gathered_density_rows for row in rank_part]
        flat_q_chunk_density_rows = [row for rank_part in gathered_q_chunk_density_rows for row in rank_part]
        rank_summary = summarize_rank_metrics(flat_rank_rows)
        density_summary = summarize_density(flat_density_rows)
        q_chunk_density_summary = summarize_q_chunk_density(flat_q_chunk_density_rows)

        metadata = local["metadata"]
        print(
            f"capture: layer={metadata.get('layer_idx')} linear_step={metadata.get('linear_step')} "
            f"timestep={metadata.get('timestep')} heads={metadata.get('num_heads')} "
            f"seq_len={metadata.get('seq_len')} world_size={world_size}"
        )
        if args.overlap_qkv_allgather_during_mask:
            print("overlap experiment: async Q/K/V all-gather launched during mask/permutation")
        print(f"balance strategy: {args.balance}")
        if greedy_info is not None:
            pl = greedy_info["predicted_loads"]
            pc = greedy_info["predicted_contig"]
            def _ratio(loads):
                lo = min(loads)
                return float("inf") if lo == 0 else max(loads) / lo
            print(
                f"greedy prediction ({greedy_info['cost_model']}, prev layer={greedy_info['layer']} "
                f"timestep>{greedy_info['timestep']}): "
                f"loads={[round(x, 4) for x in pl]} max/min={_ratio(pl):.3f} "
                f"vs contiguous={[round(x, 4) for x in pc]} max/min={_ratio(pc):.3f}"
            )
            # actual per-rank density sum observed this run
            actual = defaultdict(float)
            for row in flat_density_rows:
                actual[int(row["rank"])] += float(row["density"])
            # each rank logs densities per iteration, so average across iters
            iters = max(1, args.iters)
            actual_loads = [actual[r] / iters for r in range(world_size)]
            print(
                f"actual per-rank density (mean across {iters} iters): "
                f"{[round(x, 4) for x in actual_loads]} max/min={_ratio(actual_loads):.3f}"
            )
            for r, heads in enumerate(greedy_info["assigned"]):
                print(f"  rank {r} heads: {heads}")
        print_rank_summary(rank_summary)
        print_density_summary(density_summary)
        if args.q_chunk_density_csv:
            print_q_chunk_density_summary(q_chunk_density_summary)

        if args.rank_csv:
            write_csv(Path(args.rank_csv), rank_summary)
            print(f"\nwrote rank CSV: {args.rank_csv}")
        if args.density_csv:
            write_csv(Path(args.density_csv), density_summary)
            print(f"wrote density CSV: {args.density_csv}")
        if args.q_chunk_density_csv:
            write_csv(Path(args.q_chunk_density_csv), q_chunk_density_summary)
            print(f"wrote q-chunk density CSV: {args.q_chunk_density_csv}")
        if args.profile_dir:
            print(f"wrote profiler traces under: {args.profile_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
