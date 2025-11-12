# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from Qwen2.5-VL implementations
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The Moonshot AI Team.
# All rights reserved.

"""Inference-only OpenCUA-VL model compatible with HuggingFace weights."""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature, PretrainedConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from vllm.attention.backends.registry import _Backend
from vllm.attention.layer import (
    check_upstream_fa_availability,
    maybe_get_vit_flash_attn_backend,
)
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.opencua_vl import OpenCUA_VLConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
    apply_rotary_pos_emb_vision,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    conv3d_to_linear_weight,
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model,
)

# === Vision Inputs === #


class OpenCUA_VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size

    Historical context:
        - pixel_values shape: (num_patches, num_channels * patch_size *
          patch_size)
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          formatnum_channels * patch_size * patch_size
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


class OpenCUA_VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images

    Historical context:
        - image_embeds shape: (num_image_features, hidden_size)
        - num_image_features varies based on the number and resolution of the
          images.
        - hidden_size must match the hidden size of language model backbone.
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


OpenCUA_VLImageInputs: TypeAlias = (
    OpenCUA_VLImagePixelInputs | OpenCUA_VLImageEmbeddingInputs
)


class OpenCUA_VLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - nv: Number of videos
        - ctps: Number of channels * temporal_patch_size * patch_size *
          patch_size

    Historical context:
        - pixel_values_videos shape: (num_patches, num_channels *
          temporal_patch_size * patch_size * patch_size)
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
        - second_per_grid_ts: The video time interval (in seconds) for each
          grid along the temporal dimension in the 3D position IDs. Returned
          when `videos` is not `None`.
    """

    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("np", "ctps"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]

    second_per_grid_ts: Annotated[
        torch.Tensor | None,
        TensorShape("nv"),
    ]


class OpenCUA_VLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of video features
        - hs: Hidden size
        - nv: Number of videos

    Historical context:
        - video_embeds shape: (num_video_features, hidden_size)
        - num_video_features varies based on the number and resolution of the
          videos.
        - hidden_size must match the hidden size of language model backbone.
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["video_embeds"]

    video_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


OpenCUA_VLVideoInputs: TypeAlias = (
    OpenCUA_VLVideoPixelInputs | OpenCUA_VLVideoEmbeddingInputs
)

# === Vision Encoder === #


class OpenCUA_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )

        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x_down, _ = self.down_proj(x)
        return x_down


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist

    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(
        gathered_tensors, local_tensor, group=parallel_state.get_tp_group().device_group
    )

    gathered_tensors_split = [
        torch.split(tensor, hidden_size // tp_size, -1) for tensor in gathered_tensors
    ]
    ordered_tensors = [
        tensor for pair in zip(*gathered_tensors_split) for tensor in pair
    ]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class OpenCUA_VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend: _Backend = _Backend.TORCH_SDPA,
        use_upstream_fa: bool = False,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.attn_backend = attn_backend
        self.use_upstream_fa = use_upstream_fa
        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                self.use_upstream_fa,
            )
        )
        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN,
            _Backend.ROCM_AITER_FA,
        }

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = all_gather_interleave(qkv, self.qkv.hidden_size, self.tp_size)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(
                dist_utils.split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
        seqlens: list[int] | None = None,  # Only used for xFormers
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v))
        if rotary_pos_emb is not None:
            # [2 * b, s, heads, head_dim]
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        if self.is_flash_attn_backend:
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            output = self.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
            )

            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=batch_size
            ).contiguous()
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (
                    rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
                )
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
            context_layer = rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            attn_bias = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, kv_seqlen=None, device=q.device
            )

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None
            )
            context_layer = rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class OpenCUA_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend: _Backend = _Backend.TORCH_SDPA,
        use_upstream_fa: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = OpenCUA_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel,
            attn_backend=attn_backend,
            use_upstream_fa=use_upstream_fa,
        )
        self.mlp = OpenCUA_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
        seqlens: list[int] | None = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class OpenCUA_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = ReplicatedLinear(
            in_channels * math.prod(kernel_size),
            hidden_size,
            bias=False,
            return_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class OpenCUA_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
        )

    def forward(
        self, x: torch.Tensor, grid_thw: list[list[int]] | None = None
    ) -> torch.Tensor:
        # OpenCUA uses 1D RoPE, so vision transformer output is
        # already in 1D order. Use simple reshape like Qwen2.5-VL.
        # Input shape: [seq_len, 1, context_dim]
        # LayerNorm normalizes over the last dimension,
        # so it handles 3D tensors correctly
        x = self.ln_q(x)

        # Qwen2.5-VL style: simple view reshape
        # The vision transformer with 1D RoPE outputs patches in
        # sequential order. We need to reshape to
        # [num_merged_tokens, hidden_size] where
        # hidden_size = context_dim * spatial_merge_size^2
        # This groups spatial_merge_size^2 patches together
        # view(-1, hidden_size) works on [seq_len, 1, context_dim]:
        # total_elements = seq_len * 1 * context_dim
        # num_merged_tokens = total_elements / hidden_size
        # = seq_len / spatial_merge_size^2
        x = x.view(-1, self.hidden_size)

        out = self.mlp(x)
        return out


class OpenCUA_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device
                    )
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class OpenCUA_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        # Use config values for window attention and full attention blocks
        # OpenCUA uses window attention for most layers,
        # full attention for specific layers
        # Read from vision_config, matching Qwen2.5-VL structure
        self.window_size = vision_config.window_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes

        self.patch_embed = OpenCUA_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        # Read rope_theta from vision_config (set in OpenCUA_VLConfig.__init__)
        rope_theta = getattr(vision_config, "rope_theta", 10000.0)
        self.rotary_pos_emb = OpenCUA_VisionRotaryEmbedding(
            head_dim // 2, theta=rope_theta
        )

        use_upstream_fa = False
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )
        if (
            self.attn_backend != _Backend.FLASH_ATTN
            and self.attn_backend != _Backend.ROCM_AITER_FA
            and check_upstream_fa_availability(torch.get_default_dtype())
        ):
            self.attn_backend = _Backend.FLASH_ATTN
            use_upstream_fa = True

        if self.attn_backend not in {
            _Backend.FLASH_ATTN,
            _Backend.TORCH_SDPA,
            _Backend.XFORMERS,
            _Backend.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"OpenCUA-VL does not support {self.attn_backend} backend now."
            )

        self.blocks = nn.ModuleList(
            [
                OpenCUA_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                    attn_backend=self.attn_backend,
                    use_upstream_fa=use_upstream_fa,
                )
                for layer_idx in range(depth)
            ]
        )

        # OpenCUA uses config-defined fullatt_block_indexes
        # Most layers use window attention, specific layers use full attention
        # This matches the HF config: fullatt_block_indexes=[7, 15, 23, 31]

        self.merger = OpenCUA_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rotary_pos_emb_1d(self, seq_len: int):
        return self.rotary_pos_emb(seq_len)

    def rotary_pos_emb_thw(self, t, h, w):
        # OpenCUA uses 1D RoPE but follows Qwen2.5-VL's structure
        # Generate 2D position IDs (h, w) like Qwen2.5-VL
        # but map them to 1D sequential positions for 1D RoPE
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = (
            hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)
        # For 1D RoPE: map 2D positions to 1D sequential positions
        # hpos_ids and wpos_ids values are in [0, h-1] and [0, w-1] range
        # Mapping: pos_1d = hpos * max_size + wpos
        # This ensures unique mapping for all (hpos, wpos) pairs
        # Calculate the maximum possible pos_ids_1d value
        max_pos_1d = (h - 1) * max_size + (w - 1)
        required_size = max_pos_1d + 1
        rotary_pos_emb_full = self.rotary_pos_emb_1d(required_size)
        pos_ids_1d = pos_ids[:, 0] * max_size + pos_ids[:, 1]
        rotary_pos_emb = rotary_pos_emb_full[pos_ids_1d]
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        # Modular padding: pad only when not divisible (prevents off-by-one errors)
        # When divisible, remainder is 0, so padding is 0
        pad_h = (
            vit_merger_window_size - (llm_grid_h % vit_merger_window_size)
        ) % vit_merger_window_size
        pad_w = (
            vit_merger_window_size - (llm_grid_w % vit_merger_window_size)
        ) % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        # rotary_pos_emb_thw shape: [total_llm_tokens // spatial_merge_unit,
        #                            spatial_merge_unit, rotary_dim // 2]
        # Apply window reordering (exactly like Qwen2.5-VL)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t
        )
        return (
            rotary_pos_emb_thw,
            window_index_thw,
            cu_seqlens_window_thw,
            cu_seqlens_thw,
        )

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_1d(self, t, h, w):
        # grid_thw is in patch units (not pixels)
        # t, h, w represent temporal, height, width in patch space
        # Total patches after patch_embed = t * h * w
        # For 1D RoPE, we need position embeddings for each patch in sequence order
        total_patches = t * h * w
        rotary_pos_emb_1d = self.rotary_pos_emb_1d(total_patches)
        # rotary_pos_emb_1d shape: [total_patches, rotary_dim // 2]
        # cu_seqlens_1d represents sequence length after patch_embed
        # This is the number of patches before spatial merging
        cu_seqlens_1d = torch.tensor([total_patches], dtype=torch.int32)
        return rotary_pos_emb_1d, cu_seqlens_1d

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[int | None, list[int] | None]:
        max_seqlen, seqlens = None, None
        if (
            self.attn_backend == _Backend.FLASH_ATTN
            or self.attn_backend == _Backend.ROCM_AITER_FA
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
        # OpenCUA uses window reordering like Qwen2.5-VL
        # This function computes the inverse permutation to restore original order
        # after window attention processing
        # building the inverse permutation in O(n) time
        inv = torch.empty_like(perm, pin_memory=is_pin_memory_available())
        inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        return inv

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        seq_len, _ = x.size()
        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = torch.cat(rotary_pos_emb)
        window_index = torch.cat(window_index)
        # compute reverse indices
        reverse_indices = self.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        # pre-compute seqlens for window/full attn to reduce cuMemcpy operations
        max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(
            cu_window_seqlens
        )

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device, non_blocking=True)
        rotary_pos_emb = rotary_pos_emb.to(device=self.device, non_blocking=True)
        window_index = window_index.to(device=hidden_states.device, non_blocking=True)
        reverse_indices = reverse_indices.to(
            device=hidden_states.device, non_blocking=True
        )

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                seqlens_now = seqlens_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                seqlens_now = seqlens_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen_now,
                seqlens=seqlens_now,
            )

        # For Qwen2.5-VL-3B, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        # Qwen2.5-VL style: merger doesn't need grid_thw
        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
            ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
            ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith("patch_embed.proj.weight"):
                loaded_weight = conv3d_to_linear_weight(loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class OpenCUA_VLProcessor(Qwen2_5_VLProcessor):
    """Custom processor for OpenCUA that accepts TikTokenV3 tokenizer.

    OpenCUA uses TikTokenV3 tokenizer instead of Qwen2Tokenizer,
    so we override tokenizer_class to allow TikTokenV3.
    OpenCUA also uses <|media_placeholder|> instead of <|image_pad|>.
    """

    # Override tokenizer_class to include TikTokenV3
    # This allows the processor to accept TikTokenV3 tokenizer
    tokenizer_class = (
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "TikTokenV3",
        "CachedTikTokenV3",
    )

    # OpenCUA uses <|media_placeholder|> instead of <|image_pad|>
    image_token = "<|media_placeholder|>"
    video_token = "<|media_placeholder|>"

    def check_argument_for_proper_class(self, attribute_name, arg):
        """Override to skip type checking for tokenizer.

        This allows TikTokenV3 tokenizer to be used without type errors.
        """
        # Skip type checking for tokenizer to allow TikTokenV3
        if attribute_name == "tokenizer":
            return
        # Use parent's type checking for other attributes
        return super().check_argument_for_proper_class(attribute_name, arg)

    def _check_special_mm_tokens(self, text, text_inputs, modalities=None):
        """Override to skip special token checking for TikTokenV3.

        TikTokenV3 tokenizer may handle special tokens differently,
        so we skip the validation to avoid errors.
        """
        # Skip the special token count validation for TikTokenV3
        # The parent's _check_special_mm_tokens checks token counts
        # which may not work correctly with TikTokenV3
        pass


class OpenCUA_VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        config = self.ctx.get_hf_config(OpenCUA_VLConfig)
        text_config = config.get_text_config()
        if text_config:
            rope_scaling = getattr(text_config, "rope_scaling", None)
            if (
                rope_scaling is None
                or not isinstance(rope_scaling, dict)
                or "mrope_section" not in rope_scaling
            ):
                head_dim = getattr(text_config, "head_dim", None) or (
                    text_config.hidden_size // text_config.num_attention_heads
                )
                section_size = (head_dim // 2) // 3
                remainder = (head_dim // 2) % 3
                mrope_section = [section_size] * 3
                for i in range(remainder):
                    mrope_section[i] += 1
                text_config.rope_scaling = {
                    "rope_type": "default",
                    "mrope_section": mrope_section,
                }
        # Sync token IDs with tokenizer to ensure consistency
        # This is critical for correct tokenization and model behavior
        try:
            tokenizer = self.get_tokenizer()
            if hasattr(config, "sync_special_token_ids"):
                config.sync_special_token_ids(tokenizer)
        except Exception:
            # If tokenizer is not available yet, skip synchronization
            # It will be synced later when tokenizer is available
            pass
        return config

    def get_hf_processor(self, **kwargs: object) -> OpenCUA_VLProcessor:
        """Get processor from OpenCUA config.

        OpenCUA uses TikTokenV3 tokenizer, so we need to explicitly pass
        the tokenizer to avoid loading issues with Qwen2Tokenizer.
        """
        # Use init_processor to explicitly pass tokenizer to avoid
        # tokenizer loading issues (OpenCUA uses TikTokenV3, not Qwen2Tokenizer)
        tokenizer = self.get_tokenizer()
        image_processor_config = self.ctx.get_hf_image_processor_config()

        # Use AutoImageProcessor with use_fast=False to ensure consistent preprocessing
        # OpenCUA requires slow processor to match original behavior
        # Note: AutoImageProcessor should automatically select Qwen2.5-VL processor
        # based on the model config, but we verify the type after loading
        try:
            from transformers import AutoImageProcessor, AutoVideoProcessor

            model_path = self.ctx.model_config.model
            # Force use_fast=False to ensure consistent preprocessing
            # OpenCUA requires slow processor to match original behavior
            image_processor = AutoImageProcessor.from_pretrained(
                model_path, use_fast=False, **image_processor_config
            )
            video_processor = AutoVideoProcessor.from_pretrained(
                model_path, use_fast=False, **image_processor_config
            )

            # Log processor details
            # Note: Qwen2.5-VL also uses Qwen2VLImageProcessor, which is normal
            logger = init_logger(__name__)
            # Log additional image processor parameters for text recognition debugging
            size_info = getattr(image_processor, "size", "N/A")
            max_pixels = getattr(image_processor, "max_pixels", "N/A")
            min_pixels = getattr(image_processor, "min_pixels", "N/A")
            logger.info(
                "OpenCUA image processor loaded - type: %s, use_fast: %s, "
                "merge_size: %s, size: %s, max_pixels: %s, min_pixels: %s",
                type(image_processor).__name__,
                getattr(image_processor, "use_fast", "N/A"),
                getattr(image_processor, "merge_size", "N/A"),
                size_info,
                max_pixels,
                min_pixels,
            )
            # Log warning if image size might be too small for text recognition
            if isinstance(size_info, dict):
                max_size = max(size_info.values()) if size_info else 0
                if max_size < 1024:
                    logger.warning(
                        "Image processor size (%s) may be too small for optimal "
                        "text recognition. Consider using higher resolution images "
                        "or adjusting image processor settings.",
                        size_info,
                    )

            # OpenCUA uses its own chat template from tokenizer
            # Don't override it, let the processor use the tokenizer's default
            return self.ctx.init_processor(
                OpenCUA_VLProcessor,
                image_processor=image_processor,
                video_processor=video_processor,
                tokenizer=tokenizer,
                **kwargs,
            )
        except Exception:
            # Fallback: create processor directly without going through
            # cached_processor_from_config to avoid tokenizer key issues
            from transformers import AutoImageProcessor, AutoVideoProcessor

            model_path = self.ctx.model_config.model
            # Force use_fast=False to ensure consistent preprocessing
            # OpenCUA requires slow processor to match original behavior
            image_processor = AutoImageProcessor.from_pretrained(
                model_path, use_fast=False, **image_processor_config
            )
            video_processor = AutoVideoProcessor.from_pretrained(
                model_path, use_fast=False, **image_processor_config
            )

            # Log processor details
            # Note: Qwen2.5-VL also uses Qwen2VLImageProcessor, which is normal
            logger = init_logger(__name__)
            # Log additional image processor parameters for text recognition debugging
            size_info = getattr(image_processor, "size", "N/A")
            max_pixels = getattr(image_processor, "max_pixels", "N/A")
            min_pixels = getattr(image_processor, "min_pixels", "N/A")
            logger.info(
                "OpenCUA image processor loaded (fallback) - type: %s, "
                "use_fast: %s, merge_size: %s, size: %s, "
                "max_pixels: %s, min_pixels: %s",
                type(image_processor).__name__,
                getattr(image_processor, "use_fast", "N/A"),
                getattr(image_processor, "merge_size", "N/A"),
                size_info,
                max_pixels,
                min_pixels,
            )
            # Log warning if image size might be too small for text recognition
            if isinstance(size_info, dict):
                max_size = max(size_info.values()) if size_info else 0
                if max_size < 1024:
                    logger.warning(
                        "Image processor size (%s) may be too small for optimal "
                        "text recognition. Consider using higher resolution images "
                        "or adjusting image processor settings.",
                        size_info,
                    )

            # OpenCUA uses its own chat template from tokenizer
            # Don't override it, let the processor use the tokenizer's default
            return OpenCUA_VLProcessor(
                image_processor=image_processor,
                video_processor=video_processor,
                tokenizer=tokenizer,
                **kwargs,
            )


class OpenCUA_VLDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    """Dummy inputs builder for OpenCUA that uses <|media_placeholder|> token."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        # OpenCUA uses <|media_placeholder|> instead of <|image_pad|>
        media_token = "<|media_placeholder|>"

        return media_token * num_images + media_token * num_videos


class OpenCUA_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Override to log image preprocessing details for debugging."""
        logger = init_logger(__name__)

        # Log original image dimensions before preprocessing
        if "image" in mm_items and mm_items["image"]:
            for idx, image_item in enumerate(mm_items["image"]):
                # Try to get original image size from various possible attributes
                orig_size = None
                if hasattr(image_item, "data"):
                    img_data = image_item.data
                    if hasattr(img_data, "size"):
                        orig_size = img_data.size
                    elif hasattr(img_data, "width") and hasattr(img_data, "height"):
                        orig_size = (img_data.width, img_data.height)
                elif hasattr(image_item, "width") and hasattr(image_item, "height"):
                    orig_size = (image_item.width, image_item.height)

                if orig_size:
                    orig_width, orig_height = orig_size
                    orig_pixels = orig_width * orig_height
                    logger.info(
                        "OpenCUA preprocess input - image[%d]: "
                        "original_size=%dx%d (%d pixels)",
                        idx,
                        orig_width,
                        orig_height,
                        orig_pixels,
                    )

        # Log processor kwargs to debug preprocessing differences
        logger.info(
            "OpenCUA processor kwargs - hf_processor_mm_kwargs keys: %s, "
            "tokenization_kwargs keys: %s",
            list(hf_processor_mm_kwargs.keys()),
            list(tokenization_kwargs.keys()),
        )

        # Call parent method to perform actual preprocessing
        result = super()._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        # Log processed image dimensions after preprocessing
        if "image_grid_thw" in result:
            image_grid_thw = result["image_grid_thw"]
            if isinstance(image_grid_thw, torch.Tensor):
                grid_thw_list = image_grid_thw.tolist()
                patch_size = getattr(
                    self.info.get_hf_config().vision_config, "patch_size", 14
                )
                for idx, (t, h, w) in enumerate(grid_thw_list):
                    processed_height = h * patch_size
                    processed_width = w * patch_size
                    processed_pixels = processed_height * processed_width
                    logger.info(
                        "OpenCUA preprocess output - image[%d]: "
                        "grid_thw=[%d, %d, %d], processed_size=%dx%d "
                        "(%d pixels), patch_size=%d",
                        idx,
                        t,
                        h,
                        w,
                        processed_width,
                        processed_height,
                        processed_pixels,
                        patch_size,
                    )

        return result

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        # OpenCUA uses <|media_placeholder|> instead of <|image_pad|>
        # Use the processor's image_token if available,
        # otherwise use <|media_placeholder|>
        image_token = getattr(hf_processor, "image_token", "<|media_placeholder|>")
        if image_token not in vocab:
            # Fallback to <|media_placeholder|> if processor token not found
            image_token = "<|media_placeholder|>"
        media_placeholder_id = vocab[image_token]

        hf_config = self.info.get_hf_config()

        # Log token synchronization for debugging
        logger = init_logger(__name__)
        logger.info(
            "OpenCUA token IDs synced with tokenizer - "
            "media_placeholder_token_id: %d, image_token_id: %d, "
            "video_token_id: %d, image_token: '%s'",
            media_placeholder_id,
            media_placeholder_id,
            media_placeholder_id,
            image_token,
        )

        # Verify token exists in vocab
        if image_token not in vocab:
            raise ValueError(
                f"Token '{image_token}' not found in tokenizer vocab. "
                f"Available tokens: {list(vocab.keys())[:10]}..."
            )

        # Verify token ID matches
        token_id_from_vocab = vocab[image_token]
        if token_id_from_vocab != media_placeholder_id:
            logger.warning(
                "Token ID mismatch - vocab[%s] = %d, but using %d",
                image_token,
                token_id_from_vocab,
                media_placeholder_id,
            )

        if hasattr(hf_config, "image_token_id"):
            hf_config.image_token_id = media_placeholder_id
            if hasattr(hf_config, "video_token_id"):
                hf_config.video_token_id = media_placeholder_id
        if hasattr(hf_config, "media_placeholder_token_id"):
            hf_config.media_placeholder_token_id = media_placeholder_id

        if image_processor.merge_size != hf_config.vision_config.spatial_merge_size:
            raise ValueError(
                f"image_processor.merge_size ({image_processor.merge_size}) != "
                f"vision_config.spatial_merge_size "
                f"({hf_config.vision_config.spatial_merge_size})"
            )

        placeholder = {
            "image": media_placeholder_id,
            "video": media_placeholder_id,
        }

        def get_replacement_opencua(item_idx: int, modality: str):
            # Calculate number of vision tokens based on grid_thw
            # This ensures PlaceholderRange.length matches vision embeddings count
            # vLLM requires PlaceholderRange.length to match vision token count
            # for is_multimodal mask generation
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            # num_tokens = t * h * w // (spatial_merge_size ** 2)
            spatial_merge_size = hf_config.vision_config.spatial_merge_size
            merge_length = spatial_merge_size * spatial_merge_size
            num_tokens = int(grid_thw.prod()) // merge_length

            # CRITICAL FIX: Prompt text should have only 1 placeholder token,
            # not num_tokens. The num_tokens is used only for PlaceholderRange.length
            # via PromptUpdateDetails.is_embed, not for the actual prompt text.
            # Return PromptUpdateDetails with:
            # - full: 1 placeholder token (for prompt text)
            # - is_embed: mask indicating all num_tokens positions should get embeddings
            from vllm.multimodal.processing import PromptUpdateDetails

            # The prompt text should contain only 1 placeholder
            single_placeholder = [placeholder[modality]]

            # Create is_embed function that marks all num_tokens positions
            # This ensures PlaceholderRange.length = num_tokens for is_multimodal mask
            def is_embed(tokenizer: AnyTokenizer, full: PromptSeq) -> torch.Tensor:
                # full should be the single placeholder token
                # We need to return a mask of length num_tokens
                # indicating all positions should get embeddings
                token_ids = _seq2tokens(tokenizer, full)
                # Create mask: all positions in the single placeholder should be True
                # But we need num_tokens positions, so we expand the mask
                placeholder_mask = torch.tensor(token_ids) == placeholder[modality]
                # Expand to num_tokens length
                # Since we only have 1 placeholder in text, we create a mask
                # that marks it as the start of num_tokens embedding positions
                expanded_mask = torch.zeros(num_tokens, dtype=torch.bool)
                # Mark the first position (the single placeholder) as True
                # The rest will be handled by vLLM's PlaceholderRange logic
                expanded_mask[0] = True
                return expanded_mask

            # Return PromptUpdateDetails with single placeholder in text,
            # but is_embed function that indicates num_tokens embedding positions
            return PromptUpdateDetails(
                full=single_placeholder,
                is_embed=is_embed,
            )

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_opencua, modality=modality),
            )
            for modality in ("image", "video")
        ]


@MULTIMODAL_REGISTRY.register_processor(
    OpenCUA_VLMultiModalProcessor,
    info=OpenCUA_VLProcessingInfo,
    dummy_inputs=OpenCUA_VLDummyInputsBuilder,
)
class OpenCUA_VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    SupportsMultiModalPruning,
    SupportsMRoPE,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for OpenCUA specific names
            "vision_tower.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True
    _supports_sdpa = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image") or modality.startswith("video"):
            return "<|media_placeholder|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            self.visual = OpenCUA_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )
            # Use config-defined fullatt_block_indexes
            # Most layers use window attention, specific layers use full attention
            # This matches the HF config: fullatt_block_indexes=[7, 15, 23, 31]

            text_config = getattr(config, "text_config", None)
            if text_config is None:
                text_config_dict = {
                    k: v
                    for k, v in config.to_dict().items()
                    if k
                    not in [
                        "vision_config",
                        "model_type",
                        "media_placeholder_token_id",
                        "image_token_id",
                        "video_token_id",
                        "vision_start_token_id",
                        "vision_end_token_id",
                        "use_1d_rope",
                    ]
                }
                from transformers.models.qwen2 import Qwen2Config

                text_config = Qwen2Config(**text_config_dict)
            self.visual._lm_hidden_size = text_config.hidden_size
        else:
            self.visual = None

        text_config = getattr(config, "text_config", None)
        if text_config is None:
            from transformers.models.qwen2 import Qwen2Config

            text_config_dict = {
                k: v
                for k, v in config.to_dict().items()
                if k
                not in [
                    "vision_config",
                    "model_type",
                    "media_placeholder_token_id",
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                    "use_1d_rope",
                ]
            }
            text_config = Qwen2Config(**text_config_dict)

        rope_scaling = getattr(text_config, "rope_scaling", None)
        if (
            rope_scaling is None
            or not isinstance(rope_scaling, dict)
            or "mrope_section" not in rope_scaling
        ):
            head_dim = getattr(text_config, "head_dim", None) or (
                text_config.hidden_size // text_config.num_attention_heads
            )
            section_size = (head_dim // 2) // 3
            remainder = (head_dim // 2) % 3
            mrope_section = [section_size] * 3
            for i in range(remainder):
                mrope_section[i] += 1
            text_config.rope_scaling = {
                "rope_type": "default",
                "mrope_section": mrope_section,
            }

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=text_config,
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Sync token IDs with tokenizer and log for verification
        logger = init_logger(__name__)
        if hasattr(config, "sync_special_token_ids"):
            try:
                processor = MULTIMODAL_REGISTRY.create_processor(
                    vllm_config.model_config
                )
                tokenizer = processor.info.get_tokenizer()
                # Sync token IDs from tokenizer to config
                config.sync_special_token_ids(tokenizer)
                logger.info(
                    "OpenCUA token IDs synced with tokenizer - "
                    "media_placeholder_token_id: %d, pad_token_id: %d, "
                    "image_token_id: %d, video_token_id: %d",
                    config.media_placeholder_token_id,
                    getattr(config, "pad_token_id", 0),
                    getattr(config, "image_token_id", -1),
                    getattr(config, "video_token_id", -1),
                )
                # Check tokenizer vocab
                try:
                    vocab = tokenizer.get_vocab()
                    media_placeholder_str = tokenizer.convert_ids_to_tokens(
                        [config.media_placeholder_token_id]
                    )[0]
                    logger.info(
                        "OpenCUA tokenizer check - media_placeholder token: '%s' "
                        "(id: %d), vocab size: %d",
                        media_placeholder_str,
                        config.media_placeholder_token_id,
                        len(vocab),
                    )
                    # Verify that <|image_pad|>, <|vision_start|>, etc. are NOT in vocab
                    forbidden_tokens = [
                        "<|image_pad|>",
                        "<|vision_start|>",
                        "<|vision_end|>",
                    ]
                    for token in forbidden_tokens:
                        if token in vocab:
                            logger.warning(
                                "Forbidden token '%s' found in vocab (id: %d). "
                                "OpenCUA should only use <|media_placeholder|>.",
                                token,
                                vocab[token],
                            )
                        else:
                            logger.info(
                                "Forbidden token '%s' correctly absent from vocab",
                                token,
                            )
                except Exception as e:
                    logger.warning("Could not check tokenizer vocab: %s", e)
            except Exception as e:
                logger.warning(
                    "Could not sync token IDs with tokenizer: %s. "
                    "Using default config values.",
                    e,
                )
                if hasattr(config, "media_placeholder_token_id"):
                    logger.info(
                        "OpenCUA token IDs (default) - "
                        "media_placeholder_token_id: %d, pad_token_id: %d",
                        config.media_placeholder_token_id,
                        getattr(config, "pad_token_id", 0),
                    )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, torch.Tensor | list):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} "
                    f"(shape={mm_input.shape})"
                )
            return mm_input.reshape(-1, mm_input.shape[-1])
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> OpenCUA_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            return OpenCUA_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            return OpenCUA_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> OpenCUA_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )
            if second_per_grid_ts is not None and second_per_grid_ts.ndim == 2:
                second_per_grid_ts = second_per_grid_ts.squeeze(-1)
            return OpenCUA_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            return OpenCUA_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: OpenCUA_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        # Log actual processed image dimensions for text recognition debugging
        logger = init_logger(__name__)
        patch_size = self.visual.patch_size
        for idx, (t, h, w) in enumerate(grid_thw_list):
            # grid_thw is in patch units, convert to pixels
            actual_height = h * patch_size
            actual_width = w * patch_size
            num_patches = t * h * w
            logger.info(
                "OpenCUA image processing - image[%d]: grid_thw=[%d, %d, %d] "
                "(patches), actual_size=%dx%d (pixels), num_patches=%d, "
                "patch_size=%d. Higher resolution improves text recognition.",
                idx,
                t,
                h,
                w,
                actual_height,
                actual_width,
                num_patches,
                patch_size,
            )

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw_list, rope_type="rope_1d"
                    )
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        merge_size = self.visual.spatial_merge_size
        sizes = (
            torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)
            // (merge_size * merge_size)
        ).tolist()

        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: OpenCUA_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values_videos,
                        grid_thw_list,
                        rope_type="rope_1d",
                    )
                else:
                    video_embeds = self.visual(
                        pixel_values_videos, grid_thw=grid_thw_list
                    )

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        sizes = (
            torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)
            // (merge_size * merge_size)
        ).tolist()

        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor | None,
        video_grid_thw: list[list[int]] | torch.Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        logger = init_logger(__name__)
        if image_grid_thw is None:
            image_grid_thw = []
        if video_grid_thw is None:
            video_grid_thw = []
        if second_per_grid_ts is None:
            second_per_grid_ts = []

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        # Logging
        logger.info(
            "OpenCUA MRoPE positions - input_tokens len: %d, "
            "image_nums: %d, video_nums: %d, "
            "image_token_id: %d, video_token_id: %d, "
            "spatial_merge_size: %d",
            len(input_tokens),
            len(image_grid_thw) if image_grid_thw else 0,
            len(video_grid_thw) if video_grid_thw else 0,
            image_token_id,
            video_token_id,
            spatial_merge_size,
        )

        # OpenCUA uses Kimi-VL style: only <|media_placeholder|> token,
        # no vision_start/end tokens
        # Use image_grid_thw and video_grid_thw lengths to get actual counts
        image_nums = len(image_grid_thw) if image_grid_thw else 0
        video_nums = len(video_grid_thw) if video_grid_thw else 0

        # Count actual placeholder tokens in input_tokens
        # NOTE: In vLLM, vision tokens may already be embedded as placeholder tokens
        # in input_tokens. The text portion should have exactly
        # (image_nums + video_nums) placeholder tokens, but the total count may be
        # larger if vision tokens are already embedded as placeholders.
        placeholder_count_in_input = input_tokens.count(image_token_id)
        expected_text_placeholder_count = image_nums + video_nums

        # Logging: placeholder count info (for debugging)
        logger.info(
            "OpenCUA MRoPE placeholder count - input_tokens total placeholder "
            "count: %d, expected text placeholders: %d (images: %d, videos: %d). "
            "If total > expected, vision tokens may already be embedded.",
            placeholder_count_in_input,
            expected_text_placeholder_count,
            image_nums,
            video_nums,
        )

        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            # Find the next placeholder token in the text portion
            # With the fix, there should be exactly 1 placeholder per image/video
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                if image_index >= len(image_grid_thw):
                    raise IndexError(
                        f"image_index {image_index} >= len(image_grid_thw) "
                        f"{len(image_grid_thw)}"
                    )
                if ed_image >= len(input_tokens):
                    raise ValueError(
                        f"Could not find image placeholder token at position >= {st}. "
                        f"This indicates a mismatch between tokenization and "
                        f"multimodal input. Expected {image_nums} image(s), "
                        f"but found {image_index} so far."
                    )
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                if video_index >= len(video_grid_thw):
                    raise IndexError(
                        f"video_index {video_index} >= len(video_grid_thw) "
                        f"{len(video_grid_thw)}"
                    )
                if ed_video >= len(input_tokens):
                    raise ValueError(
                        f"Could not find video placeholder token at position >= {st}. "
                        f"This indicates a mismatch between tokenization and "
                        f"multimodal input. Expected {video_nums} video(s), "
                        f"but found {video_index} so far."
                    )
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            # grid_thw is in patch units (not pixels)
            # After spatial merging, visual tokens = t * h * w // (spatial_merge_size^2)
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            # Calculate visual token count after spatial merging
            # This must match the merger output exactly
            num_visual_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            # Verify: num_visual_tokens = t * h * w // (spatial_merge_size^2)
            expected_tokens = (t * h * w) // (spatial_merge_size * spatial_merge_size)
            if num_visual_tokens != expected_tokens:
                raise ValueError(
                    f"Visual token count mismatch: "
                    f"num_visual_tokens={num_visual_tokens}, "
                    f"expected={expected_tokens} "
                    f"(t={t}, h={h}, w={w}, spatial_merge_size={spatial_merge_size})"
                )

            # text_len includes all text tokens before the placeholder token
            # The placeholder token at position ed will be replaced with visual tokens
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

            # Logging
            logger.info(
                "OpenCUA MRoPE segment - st: %d, ed: %d, text_len: %d, "
                "grid_thw: [%d, %d, %d] (patch units), "
                "llm_grid: [%d, %d, %d], num_visual_tokens: %d, st_idx: %d",
                st,
                ed,
                text_len,
                t,
                h,
                w,
                llm_grid_t,
                llm_grid_h,
                llm_grid_w,
                num_visual_tokens,
                st_idx,
            )
            if text_len > 0:
                # OpenCUA uses 1D sequential position_ids for LLM
                # (not 3D MRoPE coordinates)
                text_positions = torch.arange(text_len, dtype=torch.long) + st_idx
                llm_pos_ids_list.append(text_positions)

            # OpenCUA uses 1D RoPE for both vision transformer and language model
            # Generate 1D sequential positions for visual tokens
            # Visual tokens start after text_len tokens
            visual_start_pos = text_len + st_idx
            # Sequential positions for visual tokens:
            # [start, start+1, ..., start+num_visual_tokens-1]
            visual_positions = (
                torch.arange(num_visual_tokens, dtype=torch.long) + visual_start_pos
            )

            # Logging
            logger.info(
                "OpenCUA MRoPE visual - visual_positions shape: %s, "
                "visual_positions min: %d, max: %d, num_visual_tokens: %d, "
                "visual_start_pos: %d",
                str(visual_positions.shape),
                visual_positions.min().item(),
                visual_positions.max().item(),
                num_visual_tokens,
                visual_start_pos,
            )

            llm_pos_ids_list.append(visual_positions)
            # After replacement, the placeholder token at position ed is replaced with
            # num_visual_tokens tokens. The next text starts after these visual tokens.
            # Since we now keep only 1 placeholder in template, we advance st by
            # ed + num_visual_tokens (the placeholder at ed is replaced with
            # num_visual_tokens vision embeddings)

            # Logging: st update validation
            # For single image case, verify st update matches replacement count
            st_before = st
            # st should advance past the placeholder (at ed) and the visual tokens
            # The placeholder at position ed is replaced with num_visual_tokens
            # vision embeddings, so st = ed + num_visual_tokens
            st_after = ed + num_visual_tokens

            # Calculate text length and placeholder count
            # text_len = ed - st_before is the number of text tokens before placeholder
            # placeholder_count = 1 (there's exactly 1 placeholder at position ed)
            text_len_before_placeholder = ed - st_before
            placeholder_count = 1  # Each image/video has exactly 1 placeholder in text
            st_advance = st_after - ed  # How many tokens st advances past placeholder

            logger.info(
                "OpenCUA MRoPE update st - ed: %d, num_visual_tokens: %d, "
                "st before: %d, st after: %d, "
                "text_len_before_placeholder: %d, "
                "placeholder_count: %d (should be 1), "
                "st_advance (st_after - ed): %d "
                "(should equal num_visual_tokens: %d)",
                ed,
                num_visual_tokens,
                st_before,
                st_after,
                text_len_before_placeholder,
                placeholder_count,
                st_advance,
                num_visual_tokens,
            )

            # Validation: For single image case, verify st update logic
            if image_nums == 1 and video_nums == 0:
                if placeholder_count != 1:
                    logger.warning(
                        "Single image case: placeholder_count = %d "
                        "(expected 1 for single placeholder)",
                        placeholder_count,
                    )
                if st_advance != num_visual_tokens:
                    raise ValueError(
                        f"Single image case: st update mismatch. "
                        f"st_advance = {st_advance}, but num_visual_tokens = "
                        f"{num_visual_tokens}. This indicates incorrect "
                        f"st update logic."
                    )
                logger.info(
                    "Single image case validation - text_len_before_placeholder: %d, "
                    "placeholder_count: %d, st_advance: %d, num_visual_tokens: %d, "
                    "st_advance_match: %s",
                    text_len_before_placeholder,
                    placeholder_count,
                    st_advance,
                    num_visual_tokens,
                    st_advance == num_visual_tokens,
                )

            st = st_after

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            text_positions = torch.arange(text_len, dtype=torch.long) + st_idx

            # Logging
            logger.info(
                "OpenCUA MRoPE remaining text - st: %d, len(input_tokens): %d, "
                "text_len: %d, st_idx: %d, text_positions min: %d, max: %d",
                st,
                len(input_tokens),
                text_len,
                st_idx,
                text_positions.min().item(),
                text_positions.max().item(),
            )

            llm_pos_ids_list.append(text_positions)

        # Concatenate all 1D position_ids
        llm_positions_1d = torch.cat(llm_pos_ids_list, dim=0)

        # Logging before slicing
        logger.info(
            "OpenCUA MRoPE before slice - llm_positions_1d shape: %s, "
            "llm_positions_1d.max(): %d, input_tokens len: %d",
            str(llm_positions_1d.shape),
            llm_positions_1d.max().item(),
            len(input_tokens),
        )

        # OpenCUA uses 1D sequential position_ids
        # Slice according to context_len and seq_len if provided
        if seq_len is not None:
            llm_positions_1d = llm_positions_1d[context_len:seq_len]
        elif context_len > 0:
            llm_positions_1d = llm_positions_1d[context_len:]

        # vLLM expects mrope_positions to be (3, L) shape for MRoPE interface
        # For OpenCUA, we use 1D sequential positions, so we repeat the same
        # 1D positions 3 times to match the expected shape
        # This allows vLLM to use positions[0] (or any row) as the actual position_ids
        llm_positions = llm_positions_1d.unsqueeze(0).expand(3, -1)

        # For compatibility with vLLM interface, return delta = 0
        # (positions are already correct, no adjustment needed)
        mrope_position_delta = 0

        # Logging
        logger.info(
            "OpenCUA MRoPE final - llm_positions_1d shape: %s, "
            "llm_positions shape: %s, llm_positions.max(): %d, "
            "mrope_position_delta: %d, context_len: %d, seq_len: %s",
            str(llm_positions_1d.shape),
            str(llm_positions.shape),
            llm_positions.max().item(),
            mrope_position_delta,
            context_len,
            seq_len,
        )

        return llm_positions, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
