"""Triton pull kernel for WAN SP asymmetric seq→heads all2all.

Each rank runs this kernel to pull its own head-slice from every peer's
symmetric-memory send buffer. Persistent grid (one CTA per SM), peer loop
unrolled so the TMA source descriptor is rebuilt only at peer boundaries,
and actual peer index swizzled by rank so different ranks hit different
NVLinks in the same phase.

Source buffer (peer-side, symm memory): [B, H_total, S_local, D]
Destination buffer (local, regular):    [B, H_local, WORLD * S_local, D]

The kernel does NOT know how to sync — caller must sandwich it between
`symm.barrier()` calls so peers' writes are visible before loads and
local writes are complete before any peer overwrites its own buffer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_ALLOCATOR_REGISTERED = False


def _tma_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ensure_tma_allocator() -> None:
    global _ALLOCATOR_REGISTERED
    if _ALLOCATOR_REGISTERED:
        return
    triton.set_allocator(_tma_allocator)
    _ALLOCATOR_REGISTERED = True


@triton.jit
def asymm_pull_seq_to_heads_kernel(
    peer_ptrs,      # *uint64, [WORLD]
    h_idxs_r,       # *int32, [H_LOCAL]
    recv_ptr,       # fp16/bf16 base pointer of [B, H_LOCAL, WORLD*S_LOCAL, D]
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

    recv_desc = tl.make_tensor_descriptor(
        recv_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    # Outer peer loop fully unrolled: descriptor built once per peer, not per tile.
    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD  # swizzle: rank r starts at r+1
        src_ptr_u64 = tl.load(peer_ptrs + peer)
        src_ptr = src_ptr_u64.to(tl.pointer_type(DTYPE))
        src_desc = tl.make_tensor_descriptor(
            src_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        # Inner: CTA strides through tiles within this peer.
        for tile in range(pid, T_PER_PEER, NUM_SMS):
            b = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            data = src_desc.load([b, gh, s_blk_in_peer * S_BLOCK, 0])

            s_global = peer * S_LOCAL + s_blk_in_peer * S_BLOCK
            recv_desc.store([b, lh, s_global, 0], data)


_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def asymm_pull_seq_to_heads(
    peer_ptrs: torch.Tensor,    # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,     # [H_LOCAL] int32 on device
    recv: torch.Tensor,         # [B, H_LOCAL, WORLD*S_LOCAL, D]
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
    s_block: int = 128,
    num_sms: int | None = None,
) -> None:
    """Launch the pull kernel. Caller must barrier before and after."""
    ensure_tma_allocator()

    if recv.dtype not in _TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {recv.dtype}")
    dtype = _TRITON_DTYPE[recv.dtype]

    h_local = h_idxs_r.numel()
    if s_local % s_block != 0:
        raise ValueError(f"s_local={s_local} not divisible by s_block={s_block}")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(recv.device).multi_processor_count

    grid = (num_sms,)
    asymm_pull_seq_to_heads_kernel[grid](
        peer_ptrs,
        h_idxs_r,
        recv,
        B=b,
        H_TOTAL=h_total,
        S_LOCAL=s_local,
        D=d,
        H_LOCAL=h_local,
        WORLD=world_size,
        RANK=rank,
        S_BLOCK=s_block,
        NUM_SMS=num_sms,
        DTYPE=dtype,
    )
