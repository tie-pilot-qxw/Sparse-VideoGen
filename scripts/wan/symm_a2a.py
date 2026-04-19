"""Symmetric-memory wrapper for asymmetric all2all (forward + reverse).

Forward (seq → heads) is always available. Pass `h_idxs_all` (global head
indices per rank) at construction to also enable reverse (heads → seq); this
extra setup allocates an `attn_symm` buffer and builds a small per-peer index
table so the reverse kernel can scatter by global head index on the fly.

Sync model for both directions: pre-barrier → kernel → post-barrier.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from asymm_pull_kernel import asymm_pull_heads_to_seq, asymm_pull_seq_to_heads


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
        h_idxs_all: Optional[Sequence[Sequence[int]]] = None,
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

        # Reverse direction is opt-in: only set up if the caller gave us the
        # full per-rank head assignment (we need it to route global heads).
        self.max_h_per_rank: Optional[int] = None
        self.attn_symm: Optional[torch.Tensor] = None
        self.attn_handle = None
        self.attn_peer_ptrs: Optional[torch.Tensor] = None
        self.peer_gh_table: Optional[torch.Tensor] = None
        self.peer_offsets: Optional[torch.Tensor] = None
        if h_idxs_all is not None:
            self._setup_reverse(h_idxs_all)

    def _setup_reverse(self, h_idxs_all: Sequence[Sequence[int]]) -> None:
        """Allocate `attn_symm` (padded to max heads per rank) and build the
        per-peer global-head table consumed by the reverse kernel.
        """
        if len(h_idxs_all) != self.world_size:
            raise ValueError(
                f"h_idxs_all has {len(h_idxs_all)} entries, expected {self.world_size}"
            )
        h_counts = [len(h) for h in h_idxs_all]
        total_assigned = sum(h_counts)
        if total_assigned != self.h_total:
            raise ValueError(
                f"h_idxs_all covers {total_assigned} heads, expected h_total={self.h_total}"
            )
        self.max_h_per_rank = max(h_counts)

        gh_flat: List[int] = []
        offsets: List[int] = [0]
        for peer_heads in h_idxs_all:
            for gh in peer_heads:
                if not (0 <= int(gh) < self.h_total):
                    raise ValueError(f"head index {gh} out of range [0, {self.h_total})")
                gh_flat.append(int(gh))
            offsets.append(offsets[-1] + len(peer_heads))

        self.peer_gh_table = torch.tensor(gh_flat, dtype=torch.int32, device=self.device)
        self.peer_offsets = torch.tensor(offsets, dtype=torch.int32, device=self.device)

        attn_shape = (self.b, self.max_h_per_rank, self.world_size * self.s_local_padded, self.d)
        self.attn_symm = symm.empty(*attn_shape, dtype=self.dtype, device=self.device)
        self.attn_handle = symm.rendezvous(self.attn_symm, self.group_name)
        self.attn_peer_ptrs = torch.tensor(
            list(self.attn_handle.buffer_ptrs), dtype=torch.int64, device=self.device
        )

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

    def pull_heads_to_seq(
        self,
        num_sms: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reverse pull: scatter this rank's sequence slice of every global
        head from the peer that owns it. Returns `[B, H_total, S_local_padded, D]`
        (caller trims to real S_local).

        Pre-condition: caller has written attn_out into `self.attn_symm`
        (shape `[B, max_H_per_rank, WORLD * S_local_padded, D]`) on every rank,
        with the per-peer S segment padded at its tail, i.e. real data lives
        at `attn_symm.view(B, max_H, WORLD, S_padded, D)[:, :H_local_r, :, :S_real, :]`.
        """
        if self.attn_symm is None:
            raise RuntimeError(
                "pull_heads_to_seq requires reverse setup; pass h_idxs_all to "
                "SymmAsymA2A(...)"
            )

        if out is None:
            out = torch.empty(
                (self.b, self.h_total, self.s_local_padded, self.d),
                dtype=self.dtype, device=self.device,
            )

        self.attn_handle.barrier(channel=0)

        asymm_pull_heads_to_seq(
            peer_ptrs=self.attn_peer_ptrs,
            peer_gh_table=self.peer_gh_table,
            peer_offsets=self.peer_offsets,
            recv=out,
            b=self.b,
            h_total=self.h_total,
            h_max_per_rank=self.max_h_per_rank,
            s_local=self.s_local_padded,
            d=self.d,
            world_size=self.world_size,
            rank=self.rank,
            s_block=self.s_block,
            num_sms=num_sms,
        )

        self.attn_handle.barrier(channel=0)
        return out
