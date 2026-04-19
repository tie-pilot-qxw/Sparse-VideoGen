# WAN SP Asymmetric all2all — pull kernel (symm_mem + Triton TMA)

Date: 2026-04-19
Status: Approved for v1 implementation (no commit yet per user request)

## Motivation

The current v1 of `greedy_unequal` in `bench_wan_sp_all2all_attention.py` keeps
`dist.all_to_all_single` working by padding shorter ranks (duplicating real
heads). This wastes NVLink bandwidth and, worse, inflates compute on the
padded rank because a hot head gets duplicated. We want an asymmetric all2all
that moves exactly the data each rank needs, eliminating pad-time waste.

Goal for v1: replace the forward `all2all_sequence_to_heads` (Q, K, V) with a
pull-based Triton kernel backed by `torch.distributed._symmetric_memory` and
TMA on H100 (sm_90). Reverse `all2all_heads_to_sequence` and any autograd
support are out of scope.

## Scope

In scope for v1:
- Forward direction: `seq → heads` for Q, K, V (three kernel launches sharing
  the same kernel body).
- Allocating Q, K, V on each rank directly from symmetric memory (option B
  from brainstorming).
- Pull model: each rank's kernel reads the head-slice it owns from every
  peer's symm buffer via TMA load, stores to a local recv buffer.
- Persistent Triton kernel with user-controllable SM budget.
- NVLink traffic spread via peer-swizzle (`actual_peer = (peer_axis + rank + 1) mod world`).
- A `--asymm-a2a {off, pull_qkv}` flag; off keeps the current padded path as
  a correctness reference.
- A `--validate-asymm` check that runs both paths and compares outputs.

Explicitly out of scope for v1:
- Reverse direction (`heads → seq`) — stays on padded symmetric all2all.
- K/V / attn_out stay on padded path via the same symm buffers (no pull
  for them yet, but they live in symm memory so the later extension is just
  two extra kernel launches).
- Autograd / backward.
- Double-buffered symm memory for async iteration overlap.
- Integration into the production model path.
- Changes to `sim_wan_sp_single_gpu.py` (single-GPU sim can't exercise symm
  memory; it keeps the padded path).

## Data flow

Per-rank Q (analogous for K, V):

```
Q ∈ symm_mem, shape [B, H_total, S/world, D]   (per-rank buffer, alloc at startup)
        │
        │ symm_handle.barrier(channel=0)                ← pre-barrier
        ▼
pull_kernel[(NUM_SMS,)](peer_ptrs, h_idxs_r, recv, ...)
        │ each CTA processes (batch, local_head, peer, s_tile) tiles
        │ peer_axis loop unrolled with tl.static_range(WORLD)
        │ peer = (peer_axis + RANK + 1) % WORLD         ← swizzle
        │ TMA load [1, S_BLOCK, D] from peer's symm buffer
        │ TMA store to local recv [b, local_head, peer*S_local + s_blk*S_BLOCK, :]
        ▼
q_head : [B, H_local_r, S_total, D]   (regular tensor, handed to kmeans)
        │
        │ symm_handle.barrier(channel=0)                ← post-barrier
        ▼
downstream (kmeans / attention) — no inverse_permute needed
```

## Components

### `scripts/wan/symm_a2a.py`

```python
class SymmAsymA2A:
    """Owns Q/K/V symm buffers + rendezvous handles for one process group."""
    def __init__(self, group, shape, dtype, device): ...
    def pull_seq_to_heads(self, name: str, h_idxs_r: torch.Tensor,
                          num_sms: int, out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    def q_symm / k_symm / v_symm : torch.Tensor   # publicly addressable for
                                                   # upstream to write into
```

Responsibilities:
- Allocate three symm tensors sized `[B, H_total, S/world_padded, D]` where
  `S/world_padded` rounds up `S/world` to a multiple of `S_BLOCK` (128).
- Run `rendezvous` and cache the `handle` + per-peer `buffer_ptrs` as a device
  `uint64` tensor.
- On `pull_seq_to_heads(name, h_idxs_r, num_sms)`: run pre-barrier → launch
  kernel → post-barrier → return recv.

### `scripts/wan/asymm_pull_kernel.py`

```python
@triton.jit
def asymm_pull_kernel(
    peer_ptrs,          # [WORLD] uint64
    h_idxs_r,           # [H_local] int32
    recv_ptr,           # [B, H_local, WORLD * S_local, D]
    B, H_total, S_local, D, H_local,
    WORLD: tl.constexpr, RANK: tl.constexpr,
    S_BLOCK: tl.constexpr, NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES = S_local // S_BLOCK
    T_per_peer = B * H_local * S_TILES

    recv_desc = tl.make_tensor_descriptor(
        recv_ptr,
        shape=[B, H_local, WORLD * S_local, D],
        strides=[H_local * WORLD * S_local * D, WORLD * S_local * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD
        src_ptr = tl.load(peer_ptrs + peer)
        src_desc = tl.make_tensor_descriptor(
            src_ptr,
            shape=[B, H_total, S_local, D],
            strides=[H_total * S_local * D, S_local * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )
        for tile in range(pid, T_per_peer, NUM_SMS):
            b  = tile // (H_local * S_TILES)
            rem = tile % (H_local * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES
            gh = tl.load(h_idxs_r + lh)
            data = src_desc.load([b, gh, s_blk_in_peer * S_BLOCK, 0])
            s_global = peer * S_local + s_blk_in_peer * S_BLOCK
            recv_desc.store([b, lh, s_global, 0], data)
```

Notes:
- `tl.static_range(WORLD)` unrolls the outer peer loop so each peer's
  descriptor is built once per CTA (not per tile).
- `D_BLOCK == D`; for D ∈ {64, 96, 128} a single TMA load covers the last dim.
- `S_local` is padded upstream to a multiple of `S_BLOCK` so the kernel needs
  no mask.

### `bench_wan_sp_all2all_attention.py` changes

- New CLI: `--asymm-a2a {off, pull_qkv}` (default off), `--num-sms N`
  (default = device SM count), `--validate-asymm` (default off).
- When asymm on: `load_local_shards` writes Q/K/V into the `SymmAsymA2A`'s
  symm buffers (no head permute for Q/K/V; the pull kernel handles
  gather-by-index).
- `run_iteration` forward: three `pull_seq_to_heads` calls replace the three
  `all2all_sequence_to_heads` calls.
- Reverse path unchanged (still padded heads → seq).
- `real_count` slicing after forward is unnecessary (recv buffer is already
  sized exactly `H_local_r`), but reverse pad-back still needed because
  reverse stays padded.

## Sync model

`symm_handle.barrier(channel=0)` before the kernel (ensures every peer's
symm buffer is populated and visible) and after the kernel (ensures no rank
overwrites its symm buffer while peers are still reading). Two host calls per
forward per tensor, ~2 μs each — negligible vs kernel time.

## Correctness validation

`--validate-asymm` runs both paths on the same inputs and asserts
`torch.equal(pad_path_output, asymm_path_output)` for each of Q, K, V
after mapping the padded layout's head axis through `h_idxs_r`. Pure data
movement, should match bit-exact.

Separately, `scripts/wan/test_asymm_pull.py` runs a minimal multi-rank
harness (torchrun, 2- or 4-rank) with synthetic inputs (each rank fills its
symm buffer with `rank*1e6 + head*1e3 + seq` so mismatches are obvious) and
varies:
- world_size ∈ {2, 4}
- H_local_r ∈ {equal split, skewed (e.g. [1, 1, 5, 5] with H_total=12)}
- head_index patterns ∈ {contiguous, scattered}

## Known limitations / v1 doesn't do

- Reverse direction unchanged.
- No double-buffering (post-barrier serializes iterations).
- Requires H100-class GPU (TMA + sm_90).
- `sim_wan_sp_single_gpu.py` keeps padded path; single-GPU sim doesn't
  exercise symm memory.
- `_symmetric_memory` is a private torch API; may break across torch
  versions.

## Rollout

1. Kernel + `SymmAsymA2A` class.
2. `test_asymm_pull.py` correctness harness.
3. Bench integration + `--validate-asymm`.
4. Profile with `ncu` for NVLink BW / TMA / SM utilization.
5. Measure end-to-end: asymm time vs padded × (real_heads / padded_heads).
