# WAN SP Asymm a2a — push-rev kernel + 2-GPU N-rank simulation

Date: 2026-04-20
Status: Approved for implementation (no commit until reviewed)

## Motivation

The current asymm a2a path keeps the rev direction (heads → seq) on a pull
kernel. Each rank writes attn_out to its own `attn_symm`, calls a barrier so
peers see the write, then everyone pulls. The barrier between local write and
peer read is a per-iteration sync that the prior commit (`1e01839`) labelled
"2 barriers per iter" but actually still requires (real layout: setup +
mid-iter + end-of-iter dist.barrier).

A push-based rev removes the mid-iter barrier outright: each rank writes
directly into peers' destination buffers, and peers see the data once the
push kernel retires. The only remaining barrier is at iteration end (to know
peers' pushes are done before the next iter's compute reads anything).

Separately, validating performance at sim_world ∈ {6, 12} is currently
blocked by hardware availability. With the sync model now light (2 barriers
total per iter), faking N-rank traffic on 2 GPUs is meaningful: each real GPU
plays one simulated rank, and remote pulls/pushes all route to the *other*
GPU's symmetric memory. Bandwidth is pessimistic (one NVLink carries
`(N-1)/N` of sim-direction traffic instead of spreading across `N-1` links),
but compute, kernel structure, and sync overhead remain faithful.

## Scope

In scope:
- New Triton kernel `asymm_push_heads_to_seq` mirroring the fwd-pull
  structure (TMA, peer-swizzle, persistent grid).
- `SymmAsymA2A` API surgery: replace `pull_heads_to_seq` with
  `push_heads_to_seq`; rename `attn_symm` → `recv_symm`; drop
  `peer_gh_table` / `peer_offsets` / `max_h_per_rank` (push doesn't need
  them).
- Bench `run_iteration` rev branch rewritten around push.
- Optional `sim_world` / `sim_rank` ctor args on `SymmAsymA2A` so a 2-rank
  symm-mem group can drive an `N`-length `peer_ptrs` table.
- Bench CLI flags `--sim-world N`, `--sim-ranks r0,r1` and an
  `effective_world` / `effective_rank` plumbing.
- Existing correctness harness `test_asymm_pull.py` updated: drop the
  reverse-direction pull test, add a reverse-direction push test (same
  encoding/decoding scheme).

Out of scope:
- Pull-rev kernel stays in the codebase as dead code? **No**: deleted, since
  the only consumer flips to push.
- Autograd / backward path.
- Single-GPU sim (`sim_wan_sp_single_gpu.py`) is unchanged.
- More than 2 sim_ranks per run (each `--sim-world` invocation samples
  exactly 2 sim ranks, one per real GPU).

## Push kernel: `asymm_push_heads_to_seq`

Mirror of `asymm_pull_seq_to_heads` with src/dst swapped.

```python
@triton.jit
def asymm_push_heads_to_seq_kernel(
    peer_ptrs,    # *uint64 [WORLD] — peers' recv_symm base pointers
    h_idxs_r,     # *int32  [H_LOCAL] — this rank's global head indices
    src_ptr,      # base of local src [B, H_LOCAL, WORLD * S_LOCAL, D]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    S_LOCAL: tl.constexpr,
    D: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    S_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES: tl.constexpr = S_LOCAL // S_BLOCK
    T_PER_PEER: tl.constexpr = B * H_LOCAL * S_TILES

    src_desc = tl.make_tensor_descriptor(
        src_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD
        dst_ptr_u64 = tl.load(peer_ptrs + peer)
        dst_ptr = dst_ptr_u64.to(tl.pointer_type(DTYPE))
        dst_desc = tl.make_tensor_descriptor(
            dst_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        for tile in range(pid, T_PER_PEER, NUM_SMS):
            b  = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            data = src_desc.load([b, lh, peer * S_LOCAL + s_blk * S_BLOCK, 0])
            dst_desc.store([b, gh, s_blk * S_BLOCK, 0], data)
```

Key invariants:
- `H_LOCAL` (this rank's heads) is `tl.constexpr`; varies per rank (each rank
  jit-compiles a specialized kernel).
- Different ranks write disjoint global-head rows into the same peer's
  `recv_symm`, so there is no intra-kernel race even though every rank
  pushes to every peer.
- `src` is a **local** tensor on this rank (no symm needed); only its
  pointer is used.
- `S_BLOCK` must divide `S_LOCAL`. `S_LOCAL` is the padded per-rank seq
  shard (rounded up to a multiple of `S_BLOCK`).

Python wrapper signature:

```python
def asymm_push_heads_to_seq(
    peer_ptrs: torch.Tensor,    # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,     # [H_LOCAL] int32 on device
    src: torch.Tensor,          # [B, H_LOCAL, WORLD * S_LOCAL, D]
    b: int, h_total: int, s_local: int, d: int,
    world_size: int, rank: int,
    s_block: int = 128, num_sms: int | None = None,
) -> None: ...
```

Same TMA-allocator setup as the pull kernels.

The `pull_heads_to_seq` kernel (and its Python wrapper) is removed.

## `SymmAsymA2A` changes

```python
class SymmAsymA2A:
    def __init__(
        self,
        group: dist.ProcessGroup | str,
        buffer_shape,            # (B, H_total, S_local, D) per-rank shard
        dtype: torch.dtype,
        device: torch.device,
        s_block: int = 128,
        h_idxs_r: Optional[Sequence[int]] = None,    # this rank's heads
        # — sim mode (optional) —
        sim_world: Optional[int] = None,
        sim_rank: Optional[int] = None,
    ):
```

Behavior changes vs current:

1. `attn_symm` → `recv_symm`, shape `[B, H_total, s_local_padded, D]` (same
   shape on every rank — no more `max_hpr` ragged padding because the push
   kernel writes by global head directly).
2. `peer_gh_table`, `peer_offsets`, `max_h_per_rank`, `attn_handle`,
   `attn_peer_ptrs`, `pull_heads_to_seq` are removed. Constructor takes
   `h_idxs_r` (this rank only) instead of `h_idxs_all` (per-peer).
3. New `push_heads_to_seq(src, num_sms=None, pre_barrier=False, post_barrier=True)`:
   - `src` is a local tensor of shape `[B, H_local, world*s_local_padded, D]`.
   - Launches the push kernel; `post_barrier=True` (default) issues
     `recv_handle.barrier(channel=0)` after the kernel so peers' pushed
     data is visible and the next iter is safe.
   - `pre_barrier` defaults to `False` since callers typically only need a
     trailing barrier — but the kwarg exists for symmetry with
     `pull_seq_to_heads`.

Sim-mode plumbing:

- If `sim_world` is set, `sim_rank` must be set too. Asserts
  `dist.get_world_size(group) == 2` (sim only works on a 2-rank symm group).
- `self.real_world`, `self.real_rank` — actual 2-rank values.
- `self.world_size = sim_world`, `self.rank = sim_rank` — kernel-facing.
- For each of `q_symm`, `k_symm`, `v_symm`, `recv_symm`, `peer_ptrs` becomes
  a length-`sim_world` `int64` tensor where slot `sim_rank` holds the local
  buffer pointer and every other slot holds the *other* real rank's buffer
  pointer (`real_buf_ptrs[1 - self.real_rank]`). All `sim_world - 1` remote
  slots therefore alias the same physical buffer, but the kernel doesn't
  care — it just reads/writes the addresses we hand it.
- All `barrier()` calls remain group-wide on the real 2-rank group.

Existing `pull_seq_to_heads` (Q/K/V fwd) is unchanged in semantics; it just
reads the new `peer_ptrs` table and `world_size` / `rank` constexprs.

## Bench changes (`bench_wan_sp_all2all_attention.py`)

CLI:
- `--sim-world N` (default 0 = off). Requires `--asymm-a2a pull_qkv` (sim
  only makes sense on the asymm path) and real `world_size == 2`.
- `--sim-ranks r0,r1` (default `0,N-1`). Two integers in `[0, N)`,
  comma-separated, must be distinct (asserted at startup). Real rank `i`
  plays sim rank `r_i`. Picking `[0, N-1]` by default samples the "first"
  and "last" head-assignment slots, which under LPT often see the most and
  least loaded shares.

Plumbing:
- After `dist.init_process_group`, compute:
  ```python
  if args.sim_world > 0:
      assert world_size == 2 and args.asymm_a2a == "pull_qkv"
      sim_world = args.sim_world
      sim_ranks = parse(args.sim_ranks) if args.sim_ranks else [0, sim_world - 1]
      effective_world = sim_world
      effective_rank = sim_ranks[rank]
  else:
      effective_world = world_size
      effective_rank = rank
  ```
- Every existing use of `world_size` for *logical* role decisions
  (head assignment, seq sharding, `load_local_shards` shard bounds, kernel
  `WORLD`/`RANK` constexprs) takes `effective_world` / `effective_rank`.
- Every use for *real* coordination (`dist.barrier`, `dist.gather_object`,
  `LOCAL_RANK`, `torch.cuda.set_device`, the `dist.barrier()` at end of
  outer iter) keeps `world_size` / `rank`.
- Metrics rows include both: `{"real_rank": rank, "sim_rank":
  effective_rank, ...}` so the output table prints
  `real=0/sim=0` and `real=1/sim=N-1`.
- On rank 0, before the warmup loop, print:
  `[sim] sim_world={N} sim_ranks={[r0,r1]} on 2 real GPUs:
   each NVLink carries (N-1)/N of one direction's sim traffic
   instead of spreading across (N-1) links;
   compute and sync model match real N-rank.`

Rev branch in `run_iteration`:

```python
if symm_a2a is not None:
    B_, H_loc_, S_full_, D_ = attn_out.shape
    s_real = S_full_ // effective_world
    s_padded = symm_a2a.s_local_padded

    # Local staging: pad each peer's S segment up to s_padded so TMA loads
    # from the src buffer line up with the push kernel's S_BLOCK tiles.
    if attn_src_padded is None or attn_src_padded.shape[2] != effective_world * s_padded:
        attn_src_padded = torch.empty(
            B_, H_loc_, effective_world * s_padded, D_,
            dtype=attn_out.dtype, device=attn_out.device,
        )
    attn_src_view = attn_src_padded.view(B_, H_loc_, effective_world, s_padded, D_)
    attn_out_view = attn_out.reshape(B_, H_loc_, effective_world, s_real, D_)
    attn_src_view[:, :, :, :s_real, :].copy_(attn_out_view)
    # pad region is left as garbage; receivers slice it away.

    num_sms = args.num_sms if args.num_sms > 0 else None
    out_seq_full, all2all_out_events = record_cuda_region(
        lambda: symm_a2a.push_heads_to_seq(
            attn_src_padded, num_sms=num_sms,
            pre_barrier=False, post_barrier=True,
        ),
        "all2all_heads_to_sequence",
    )
    if s_padded != s_real:
        out_seq = out_seq_full[:, :, :s_real, :].contiguous()
    else:
        out_seq = out_seq_full
```

`push_heads_to_seq` returns `self.recv_symm` (the local destination buffer);
caller trims its S padding back to `s_real`.

The `attn_src_padded` buffer can be cached on `local` to avoid re-allocating
each iter.

Setup-barrier call (already present) stays:
```python
if symm_a2a is not None:
    symm_a2a.barrier()
```

## Sync model

Per iter (asymm + push):
- 3 fwd pulls for Q/K/V: no barriers (Q/K/V are write-once, covered by the
  setup barrier).
- attention compute.
- local memcpy attn_out → attn_src_padded.
- push kernel + trailing `recv_handle.barrier()`.
- outer `dist.barrier()` at end of bench loop iter (already present).

Net symm barriers per iter: **1** (the trailing barrier inside push). Setup
barrier runs once. The mid-iter pre/post barriers around the rev pull
disappear with the kernel they used to guard.

## Correctness validation

`scripts/wan/test_asymm_pull.py`:
- Forward (Q/K/V seq → heads) test cases unchanged.
- Reverse cases switch from `pull_heads_to_seq` to `push_heads_to_seq`.
  Encoding stays the same idea: each rank fills its `attn_src_padded` with
  values that uniquely identify (rank, local_head, s_axis), then after push
  + barrier, every rank checks that `recv_symm` has the expected mosaic
  composed of all peers' contributions.
- Decoder formula (rank `r` checks its `recv_symm`):
  ```
  recv_symm[b, gh, s_axis, d] ==
      (owner * max_hpr_local + lh_in_owner) * (effective_world * s_padded)
      + r * s_padded + s_axis
  ```
  where `owner = peer that owns gh`, `lh_in_owner = position of gh in
  owner's h_idxs_r`. (Fp16 overflow guard kept for `--size real`.)
- Bandwidth-measurement loop kept; reports per-iter ms and remote/total
  GB/s on the push direction the same way the pull version does.

Sim-mode validation: a separate harness is **not** added for sim mode in v1
(the kernel and `SymmAsymA2A` correctness are already covered by
`test_asymm_pull.py` in non-sim mode). Sim mode is a perf tool; for
end-to-end smoke, run `bench_wan_sp_all2all_attention.py --sim-world 2
--sim-ranks 0,1 --asymm-a2a pull_qkv` (degenerates to a real 2-rank run,
should match a non-sim run within noise).

## Known limitations

- Sim bandwidth is pessimistic by ≈ `(N-1)×` vs a real N-GPU run: in real
  N-GPU each rank's remote traffic spreads across `N-1` NVLinks; in the sim
  the *entire* remote portion (`(N-1)/N` of per-rank data per direction)
  funnels through one NVLink pair. Latency / kernel time / SM-occupancy /
  sync overhead are faithful.
- Sim mode requires real `world_size == 2`. A 4-real-GPU sim
  ("2 sim ranks per real GPU" or similar topologies) is not designed here.
- `SymmAsymA2A` is single-iteration: post-barrier serializes iters. No
  double-buffering yet.
- Push kernel requires TMA store + sm_90 (H100). Same restriction as the
  pull kernels.

## Rollout

1. Push kernel + Python wrapper (in `asymm_pull_kernel.py` — keep filename;
   rename to `asymm_kernels.py` only if push lands cleanly first).
2. `SymmAsymA2A` rewrite (recv_symm, push_heads_to_seq, sim_world/sim_rank
   args).
3. `test_asymm_pull.py` updated reverse case — verify bit-exact + bandwidth.
4. Bench rewire: rev branch + setup barrier kept + sim-mode flags +
   effective_world/rank plumbing.
5. Smoke: 2-real-GPU run with `--sim-world 2` (degenerate) → equals non-sim
   2-rank within noise.
6. Stretch: 2-real-GPU run with `--sim-world 6` and `--sim-world 12` to get
   per-iter timings under the simulated head distributions.
