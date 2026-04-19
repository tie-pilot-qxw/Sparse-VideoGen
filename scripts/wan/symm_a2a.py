"""Symmetric-memory wrapper for asymmetric seq→heads all2all (forward only).

Allocates Q/K/V in `torch.distributed._symmetric_memory` on construction and
runs `rendezvous` once so every rank knows every peer's base pointer. Call
`pull_seq_to_heads(name, h_idxs_r)` to pull this rank's head-slice from
every peer into a fresh recv buffer. Caller is responsible for writing Q/K/V
into the symm buffers before calling (the `.q_symm` / `.k_symm` / `.v_symm`
attrs expose them as regular tensors).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from asymm_pull_kernel import asymm_pull_seq_to_heads


class SymmAsymA2A:
    """Owns the per-rank symm buffers for Q/K/V and dispatches pull kernels.

    `buffer_shape` describes the per-rank shard in logical (unpadded) terms:
    `[B, H_total, S_local, D]`. Internally the S dim is rounded up to the next
    multiple of `s_block` (Triton TMA needs power-of-2 block shapes). Pad
    values are meaningless; the pull result is `[B, H_local, WORLD * S_local_padded, D]`
    and the caller trims down to the real `WORLD * S_local`.

    `s_block` must be a power of 2 (TMA block-shape constraint).
    """

    def __init__(
        self,
        group: dist.ProcessGroup | str,
        buffer_shape,
        dtype: torch.dtype,
        device: torch.device,
        s_block: int = 128,
    ):
        if isinstance(group, dist.ProcessGroup):
            self.group_name = group.group_name
            self.group = group
        else:
            self.group_name = group
            self.group = dist.distributed_c10d._resolve_process_group(group)

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = device
        self.dtype = dtype

        if s_block <= 0 or (s_block & (s_block - 1)) != 0:
            raise ValueError(f"s_block={s_block} must be a positive power of 2")
        self.s_block = s_block

        b, h_total, s_local, d = buffer_shape
        s_local_padded = ((s_local + s_block - 1) // s_block) * s_block
        if d & (d - 1) != 0:
            raise ValueError(f"d={d} must be a power of 2 for TMA block shape")
        self.b = b
        self.h_total = h_total
        self.s_local = s_local             # caller-facing logical size
        self.s_local_padded = s_local_padded
        self.d = d
        self.buffer_shape = (b, h_total, s_local_padded, d)

        self.q_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)
        self.k_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)
        self.v_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)

        self.q_handle = symm.rendezvous(self.q_symm, self.group_name)
        self.k_handle = symm.rendezvous(self.k_symm, self.group_name)
        self.v_handle = symm.rendezvous(self.v_symm, self.group_name)

        self._symm: Dict[str, torch.Tensor] = {
            "q": self.q_symm, "k": self.k_symm, "v": self.v_symm,
        }
        self._handle = {
            "q": self.q_handle, "k": self.k_handle, "v": self.v_handle,
        }
        # `handle.buffer_ptrs_dev` is a raw CUDA address (Python int), and
        # `handle.buffer_ptrs` is a host list of addresses. Triton needs a
        # torch tensor so it can pass a typed pointer into the kernel.
        self._peer_ptrs: Dict[str, torch.Tensor] = {
            name: torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=device)
            for name, h in self._handle.items()
        }

    def peer_ptrs(self, name: str) -> torch.Tensor:
        return self._peer_ptrs[name]

    def pull_seq_to_heads(
        self,
        name: str,
        h_idxs_r: torch.Tensor,
        num_sms: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pull this rank's heads (`h_idxs_r`, global head indices) from every
        peer's `{name}_symm` buffer. Returns `[B, H_local_r, WORLD*S_local, D]`.

        Pre-condition: caller has populated `self.{name}_symm` on every rank.
        """
        if name not in self._handle:
            raise KeyError(f"unknown buffer {name!r}; expected one of q/k/v")
        handle = self._handle[name]
        peer_ptrs = self._peer_ptrs[name]
        h_local = h_idxs_r.numel()

        if h_idxs_r.dtype != torch.int32:
            h_idxs_r = h_idxs_r.to(dtype=torch.int32)
        if h_idxs_r.device != self.device:
            h_idxs_r = h_idxs_r.to(self.device)

        if out is None:
            out = torch.empty(
                (self.b, h_local, self.world_size * self.s_local_padded, self.d),
                dtype=self.dtype, device=self.device,
            )

        # Pre-barrier: make all peers' writes visible.
        handle.barrier(channel=0)

        asymm_pull_seq_to_heads(
            peer_ptrs=peer_ptrs,
            h_idxs_r=h_idxs_r,
            recv=out,
            b=self.b,
            h_total=self.h_total,
            s_local=self.s_local_padded,
            d=self.d,
            world_size=self.world_size,
            rank=self.rank,
            s_block=self.s_block,
            num_sms=num_sms,
        )

        # Post-barrier: don't let anyone overwrite their symm buffer while we
        # (or other peers) are still reading.
        handle.barrier(channel=0)
        return out
