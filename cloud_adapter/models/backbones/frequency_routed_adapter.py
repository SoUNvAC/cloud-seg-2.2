"""Frequency-routed cross-attention used by experiment 02.

The module keeps the public CloudAdapter tensor contract unchanged: transformer
tokens enter and leave as ``(N, B, D)``.  Spatial sizes are passed explicitly by
the DINOv2 wrapper, so rectangular inputs never rely on ``sqrt(N)``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    """Bias-free low-rank linear projection."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class FrequencyDecomposer(nn.Module):
    """Split a spatial feature map into spatial, low- and high-frequency views."""

    def __init__(self, cutoff_ratio: float = 0.3) -> None:
        super().__init__()
        if not 0.0 < cutoff_ratio <= 1.0:
            raise ValueError("cutoff_ratio must be in (0, 1]")
        # Persistent configuration makes adapter-only checkpoints self-describing.
        self.register_buffer(
            "cutoff_ratio", torch.tensor(float(cutoff_ratio)), persistent=True
        )
        # The dynamic mask is device/shape specific and should not enter checkpoints.
        self.register_buffer("_cached_mask", torch.empty(0), persistent=False)
        self._cached_shape: Tuple[int, int] | None = None

    def _build_mask(self, height: int, width: int, device: torch.device) -> Tensor:
        # Coordinates are normalized independently so the cutoff has the same
        # interpretation for square and rectangular feature maps.
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device))
        radius = torch.sqrt((fy[:, None] / 0.5) ** 2 + (fx[None, :] / 0.5) ** 2)
        return (radius <= self.cutoff_ratio.to(device=device)).to(torch.float32)[None, None]

    def _mask(self, context: Tensor) -> Tensor:
        height, width = context.shape[-2:]
        cache_invalid = (
            self._cached_shape != (height, width)
            or self._cached_mask.device != context.device
            or self._cached_mask.numel() == 0
        )
        if cache_invalid:
            self._cached_mask = self._build_mask(height, width, context.device)
            self._cached_shape = (height, width)
        return self._cached_mask

    def forward(self, context: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if context.ndim != 4:
            raise ValueError(
                f"context must have shape (B, C, H, W), got {tuple(context.shape)}"
            )
        original_dtype = context.dtype
        # FFT is intentionally FP32 under autocast to avoid unstable half-complex ops.
        with torch.autocast(device_type=context.device.type, enabled=False):
            spatial_fp32 = context.float()
            spectrum = torch.fft.fftshift(
                torch.fft.fft2(spatial_fp32, dim=(-2, -1), norm="ortho"),
                dim=(-2, -1),
            )
            low = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * self._mask(context), dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
            high = spatial_fp32 - low
        return context, low.to(original_dtype), high.to(original_dtype)


class SharedQueryMultiExpertCrossAttention(nn.Module):
    """Three attention experts with one Q projection and rank-budgeted K/V/O."""

    expert_names = ("spatial", "low", "high")

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        ranks: Sequence[int] = (8, 4, 4),
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        if len(ranks) != 3 or any(rank <= 0 for rank in ranks):
            raise ValueError("ranks must contain three positive integers")
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.to_q = LowRankLinear(query_dim, self.inner_dim, sum(ranks))
        self.to_k = nn.ModuleDict(
            {
                name: LowRankLinear(context_dim, self.inner_dim, rank)
                for name, rank in zip(self.expert_names, ranks)
            }
        )
        self.to_v = nn.ModuleDict(
            {
                name: LowRankLinear(context_dim, self.inner_dim, rank)
                for name, rank in zip(self.expert_names, ranks)
            }
        )
        self.to_out = nn.ModuleDict(
            {
                name: LowRankLinear(self.inner_dim, query_dim, rank)
                for name, rank in zip(self.expert_names, ranks)
            }
        )
        self.register_buffer(
            "rank_budget", torch.tensor(tuple(ranks), dtype=torch.int64), persistent=True
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, x: Tensor, contexts: Sequence[Tensor]) -> Tensor:
        if len(contexts) != 3:
            raise ValueError("exactly three contexts are required")
        q = self._split_heads(self.to_q(x))
        outputs = []
        for name, context in zip(self.expert_names, contexts):
            k = self._split_heads(self.to_k[name](context))
            v = self._split_heads(self.to_v[name](context))
            # SDPA selects Flash/Memory-Efficient Attention on supported CUDA
            # devices and avoids materializing a B*H*N*N probability tensor.
            delta = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            ).transpose(1, 2).contiguous()
            delta = delta.reshape(x.shape[0], x.shape[1], self.inner_dim)
            outputs.append(self.to_out[name](delta))
        return torch.stack(outputs, dim=2)  # B, N, 3, D


class TokenFrequencyRouter(nn.Module):
    """Generate fixed, image-wise, or token-wise expert weights."""

    valid_modes = {"fixed", "image", "token"}

    def __init__(
        self,
        token_dim: int,
        context_dim: int,
        router_dim: int = 8,
        temperature: float = 1.0,
        mode: str = "token",
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {sorted(self.valid_modes)}, got {mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.mode = mode
        self.register_buffer(
            "mode_id",
            torch.tensor({"fixed": 0, "image": 1, "token": 2}[mode]),
            persistent=True,
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.ctx_proj = nn.Linear(context_dim, router_dim)
        self.router = nn.Sequential(
            nn.Linear(token_dim + router_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 3),
        )
        self.register_buffer(
            "temperature", torch.tensor(float(temperature)), persistent=True
        )
        # Equal initial routing prevents an arbitrary expert advantage at step zero.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        if self.mode == "fixed":
            return x.new_full((batch, tokens, 3), 1.0 / 3.0)
        token_features = self.token_norm(x)
        if self.mode == "image":
            token_features = token_features.mean(dim=1, keepdim=True)
        context_global = self.ctx_proj(context.mean(dim=(-2, -1)))[:, None, :]
        context_global = context_global.expand(-1, token_features.shape[1], -1)
        logits = self.router(torch.cat((token_features, context_global), dim=-1))
        weights = torch.softmax(logits / self.temperature.to(logits.dtype), dim=-1)
        if self.mode == "image":
            weights = weights.expand(-1, tokens, -1)
        return weights


class FrequencyRoutedInteractiveModule(nn.Module):
    """Frequency decomposition, three attention experts, and dynamic routing."""

    def __init__(
        self,
        emd_dim: int = 1024,
        context_dim: int = 64,
        ranks: Sequence[int] = (8, 4, 4),
        cutoff_ratio: float = 0.3,
        router_dim: int = 8,
        temperature: float = 1.0,
        router_mode: str = "token",
        alpha_init: float = 0.1,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        self.decomposer = FrequencyDecomposer(cutoff_ratio=cutoff_ratio)
        self.attention = SharedQueryMultiExpertCrossAttention(
            query_dim=emd_dim,
            context_dim=context_dim,
            ranks=ranks,
            heads=heads,
            dim_head=dim_head,
        )
        self.router = TokenFrequencyRouter(
            token_dim=emd_dim,
            context_dim=context_dim,
            router_dim=router_dim,
            temperature=temperature,
            mode=router_mode,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_balance_loss: Tensor | None = None
        self.last_router_stats: Dict[str, Tensor | Tuple[int, int]] = {}

    @staticmethod
    def _select_context(cache: Tensor | Sequence[Tensor], index: int) -> Tensor:
        if isinstance(cache, (list, tuple)):
            if not cache:
                raise ValueError("cache list/tuple must not be empty")
            cache = cache[min(index, len(cache) - 1)]
        if not isinstance(cache, Tensor) or cache.ndim != 4:
            raise ValueError("cache must be a BCHW tensor or a non-empty sequence of them")
        return cache

    def forward(
        self,
        x: Tensor,
        cache: Tensor | Sequence[Tensor],
        index: int,
        spatial_shape: Tuple[int, int] | None = None,
        precomputed_branches: Sequence[Tensor] | None = None,
    ) -> Tensor:
        if spatial_shape is None:
            raise ValueError("spatial_shape=(H, W) is required for frequency routing")
        height, width = spatial_shape
        if height * width != x.shape[0]:
            raise ValueError(
                f"token count {x.shape[0]} does not match spatial shape {spatial_shape}"
            )
        context = self._select_context(cache, index)
        branches = (
            tuple(precomputed_branches)
            if precomputed_branches is not None
            else self.decomposer(context)
        )
        if len(branches) != 3:
            raise ValueError("precomputed_branches must contain spatial, low and high")
        flat_contexts = []
        for branch in branches:
            if branch.shape[-2:] != spatial_shape:
                branch = F.interpolate(
                    branch, size=spatial_shape, mode="bilinear", align_corners=False
                )
            flat_contexts.append(branch.flatten(2).transpose(1, 2))

        tokens = x.transpose(0, 1)
        expert_deltas = self.attention(tokens, flat_contexts)
        weights = self.router(tokens, context)
        delta = (expert_deltas * weights.unsqueeze(-1)).sum(dim=2)
        output = tokens + self.alpha * delta

        usage = weights.mean(dim=(0, 1))
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy.mean() / torch.log(weights.new_tensor(3.0))
        self.last_balance_loss = ((usage - (1.0 / 3.0)) ** 2).sum()
        self.last_router_stats = {
            "weights": weights.detach(),
            "usage": usage.detach(),
            "entropy": entropy.detach(),
            "token_variance": weights.var(dim=1, unbiased=False).mean(dim=0).detach(),
            "delta_norms": expert_deltas.norm(dim=-1).mean(dim=(0, 1)).detach(),
            "spatial_shape": spatial_shape,
        }
        return output.transpose(0, 1)
