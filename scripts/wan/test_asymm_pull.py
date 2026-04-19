"""Correctness harness for `SymmAsymA2A.pull_seq_to_heads`.

Launch with:

    torchrun --nproc_per_node=2 scripts/wan/test_asymm_pull.py
    torchrun --nproc_per_node=4 scripts/wan/test_asymm_pull.py --cases all

Fills each rank's symm buffer with a deterministic pattern
`rank * 1e6 + head * 1e3 + seq_idx` so that any miswired pull shows up as
a specific integer mismatch (printable, not a random float blob).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from symm_a2a import SymmAsymA2A


def _encode(rank: int, head: int, seq: int, H_TOTAL: int, S_LOCAL: int) -> int:
    """Unique integer encoding so we can tell mis-wired pulls apart. Fits in
    fp32 easily; caller must ensure it also fits in fp16/bf16 if used there."""
    return (rank * H_TOTAL + head) * S_LOCAL + seq


def _fill_symm(buf: torch.Tensor, rank: int, H_TOTAL: int, S_LOCAL: int) -> None:
    b, h, s, d = buf.shape
    heads = torch.arange(h, device=buf.device, dtype=torch.float32).view(1, h, 1, 1)
    seq = torch.arange(s, device=buf.device, dtype=torch.float32).view(1, 1, s, 1)
    val = (rank * H_TOTAL + heads) * S_LOCAL + seq
    buf.copy_(val.to(buf.dtype).expand(b, h, s, d).contiguous())


def _run_case(
    world_size: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    B: int,
    H_TOTAL: int,
    S_LOCAL: int,
    D: int,
    head_assignment,
    case_name: str,
    bw_warmup: int = 5,
    bw_iters: int = 50,
):
    h_idxs_r = torch.tensor(head_assignment[rank], dtype=torch.int32, device=device)
    a2a = SymmAsymA2A(
        dist.group.WORLD,
        buffer_shape=(B, H_TOTAL, S_LOCAL, D),
        dtype=dtype,
        device=device,
        s_block=min(128, S_LOCAL),
    )
    for name in ("q", "k", "v"):
        _fill_symm(a2a._symm[name], rank, H_TOTAL, S_LOCAL)

    dist.barrier()

    h_local = h_idxs_r.numel()
    # Build expected output: for each local head lh (→ global head gh),
    # and each peer, recv[b, lh, peer*S_LOCAL + s, :] == encode(peer, gh, s).
    expected = torch.empty((B, h_local, world_size * S_LOCAL, D), dtype=dtype, device=device)
    for lh in range(h_local):
        gh = int(head_assignment[rank][lh])
        for peer in range(world_size):
            seq = torch.arange(S_LOCAL, device=device, dtype=torch.float32)
            vals = (peer * H_TOTAL + gh) * S_LOCAL + seq  # [S_LOCAL]
            vals_bcast = vals.view(1, 1, S_LOCAL, 1).expand(B, 1, S_LOCAL, D).to(dtype)
            expected[:, lh : lh + 1, peer * S_LOCAL : (peer + 1) * S_LOCAL, :] = vals_bcast

    for name in ("q", "k", "v"):
        out = a2a.pull_seq_to_heads(name, h_idxs_r)

        diff = (out.float() - expected.float()).abs()
        max_diff = float(diff.max().item())
        ok = max_diff < 0.5
        if rank == 0:
            tag = f"[{case_name} | {name}] rank={rank} h_idxs={head_assignment[rank]}"
            if ok:
                print(f"PASS {tag}  max_diff={max_diff}")
            else:
                idx = diff.argmax()
                flat = diff.flatten()
                off = int(idx.item())
                total = flat.numel()
                # Decode (b, lh, s_global, d)
                _, h_local_dim, s_dim, d_dim = out.shape
                d_idx = off % d_dim
                s_idx = (off // d_dim) % s_dim
                lh_idx = (off // (d_dim * s_dim)) % h_local_dim
                got = float(out.float().flatten()[off].item())
                exp = float(expected.float().flatten()[off].item())
                print(
                    f"FAIL {tag}  max_diff={max_diff}  "
                    f"lh={lh_idx} s_global={s_idx} d={d_idx}  expected={exp} got={got}"
                )
        dist.barrier()

    # --- bandwidth: time the full pull_seq_to_heads (pre-barrier + kernel +
    # post-barrier) for Q only, averaged over bw_iters. Each rank reports its
    # own number since H_local (and so bytes pulled) differs per rank in
    # skewed assignments.
    h_local = h_idxs_r.numel()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    # S_LOCAL in this test is already a multiple of s_block, so no padding.
    bytes_remote = h_local * (world_size - 1) * B * S_LOCAL * D * elem_size
    bytes_total  = h_local *  world_size      * B * S_LOCAL * D * elem_size

    for _ in range(bw_warmup):
        a2a.pull_seq_to_heads("q", h_idxs_r)
    torch.cuda.synchronize(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bw_iters):
        a2a.pull_seq_to_heads("q", h_idxs_r)
    end.record()
    torch.cuda.synchronize(device)
    per_pull_ms = start.elapsed_time(end) / bw_iters

    bw_remote_gbs = bytes_remote / (per_pull_ms * 1e-3) / 1e9
    bw_total_gbs  = bytes_total  / (per_pull_ms * 1e-3) / 1e9

    # Gather to rank 0 for tidy printing.
    stats = torch.tensor([per_pull_ms, bw_remote_gbs, bw_total_gbs, float(h_local)],
                         device=device, dtype=torch.float64)
    gathered = [torch.empty_like(stats) for _ in range(world_size)] if rank == 0 else None
    dist.gather(stats, gathered, dst=0)
    if rank == 0:
        print(f"[{case_name} | bw] per-pull (Q), bytes include {world_size} peers "
              f"(1 local + {world_size-1} remote), S_LOCAL={S_LOCAL}, D={D}, dtype={dtype}:")
        print(f"  {'rank':>4}  {'H_loc':>5}  {'ms/pull':>8}  "
              f"{'remote GB/s':>12}  {'total GB/s':>11}")
        for r, row in enumerate(gathered):
            ms, rbw, tbw, hl = row.tolist()
            print(f"  {r:>4}  {int(hl):>5}  {ms:>8.3f}  {rbw:>12.1f}  {tbw:>11.1f}")
    dist.barrier()

    del a2a
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=["all", "equal", "skewed", "scattered"], default="all")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--size", choices=["small", "real"], default="small",
                        help="'small' (S_LOCAL=128) is enough for correctness but "
                             "too small for meaningful bandwidth; 'real' uses "
                             "S_LOCAL=8192, D=128 (close to WAN workload).")
    parser.add_argument("--bw-warmup", type=int, default=5)
    parser.add_argument("--bw-iters", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    H_TOTAL = 12
    if args.size == "small":
        B, D, S_LOCAL = 1, 128, 128  # 4 tiles per peer with s_block=128
    else:
        B, D, S_LOCAL = 1, 128, 8192  # realistic WAN-ish shard
        if dtype == torch.float16:
            raise SystemExit(
                "--size real + --dtype fp16: correctness encoder overflows fp16 "
                "(peak value ≈ 4e5). Use bf16 or fp32."
            )

    # Case 1: equal split (H_TOTAL=12, world=4 → 3 heads each).
    if args.cases in ("all", "equal"):
        if world_size == 4:
            assignment = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        elif world_size == 2:
            assignment = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
        else:
            raise SystemExit(f"test supports world_size ∈ {{2, 4}}, got {world_size}")
        _run_case(world_size, rank, device, dtype,
                  B=B, H_TOTAL=H_TOTAL, S_LOCAL=S_LOCAL, D=D,
                  bw_warmup=args.bw_warmup, bw_iters=args.bw_iters,
                  head_assignment=assignment, case_name="equal")

    # Case 2: skewed LPT-style assignment.
    if args.cases in ("all", "skewed"):
        if world_size == 4:
            # H_TOTAL=12, heads per rank = [1, 2, 4, 5]
            assignment = [[7], [0, 11], [1, 3, 5, 9], [2, 4, 6, 8, 10]]
        elif world_size == 2:
            assignment = [[0, 3, 7], [1, 2, 4, 5, 6, 8, 9, 10, 11]]
        _run_case(world_size, rank, device, dtype,
                  B=B, H_TOTAL=H_TOTAL, S_LOCAL=S_LOCAL, D=D,
                  bw_warmup=args.bw_warmup, bw_iters=args.bw_iters,
                  head_assignment=assignment, case_name="skewed")

    # Case 3: scattered (non-monotone) indices.
    if args.cases in ("all", "scattered"):
        if world_size == 4:
            assignment = [[11, 0, 5], [7, 2, 9], [1, 8, 4], [10, 3, 6]]
        elif world_size == 2:
            assignment = [[11, 0, 5, 7, 2, 9], [1, 8, 4, 10, 3, 6]]
        _run_case(world_size, rank, device, dtype,
                  B=B, H_TOTAL=H_TOTAL, S_LOCAL=S_LOCAL, D=D,
                  bw_warmup=args.bw_warmup, bw_iters=args.bw_iters,
                  head_assignment=assignment, case_name="scattered")

    dist.barrier()
    if rank == 0:
        print("\nAll cases complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
