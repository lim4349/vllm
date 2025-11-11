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

# Import Qwen2.5-VL components dynamically to avoid import issues
try:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLVisionConfig,
    )
except ImportError:
    # Fallback: create minimal classes
    from transformers import PretrainedConfig, ProcessorMixin

    class Qwen2_5_VLProcessor(ProcessorMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Qwen2_5_VLConfig(PretrainedConfig):
        model_type = "qwen2_5_vl"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Qwen2_5_VLVisionConfig(PretrainedConfig):
        model_type = "qwen2_5_vl_vision"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


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

logger = init_logger(__name__)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
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

        # args for get_window_index_thw
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = OpenCUA_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        # Use 1D RoPE instead of M-RoPE
        # Load rope_theta from vision_config if available, otherwise use default
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
        """Use 1D RoPE instead of 3D M-RoPE"""
        return self.rotary_pos_emb(seq_len)

    def get_window_index_1d(self, grid_t, grid_h, grid_w):
        """Window indexing for 1D RoPE with spatial locality preserved.

        Even though we use 1D RoPE, we still need to preserve spatial
        locality for window attention to work correctly. This is critical
        for vision transformer to extract spatial information from images.
        """
        # 1. Window size conversion: MUST be
        # (window_size // patch_size) // spatial_merge_size
        # Wrong order: window_size // spatial_merge_size // patch_size
        # breaks boundaries
        vit_merger_window_size = (
            self.window_size // self.patch_size
        ) // self.spatial_merge_size

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )

        # 2. Padding: use modular arithmetic for safety (remainder=0 -> pad=0)
        pad_h = (
            vit_merger_window_size - (llm_grid_h % vit_merger_window_size)
        ) % vit_merger_window_size
        pad_w = (
            vit_merger_window_size - (llm_grid_w % vit_merger_window_size)
        ) % vit_merger_window_size

        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)

        # 3. Index reshape order: row-major -> window grouping ->
        # window inner row/col
        # [t, H, W] -> pad -> [t, nH, win, nW, win] ->
        # permute(0,1,3,2,4) -> [t, nH*nW, win, win]
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

        # 4. cu_seqlens scale: window path uses seqlens.cumsum() * spatial_merge_unit
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_1d(self, t, h, w):
        """Get 1D RoPE embeddings instead of 3D M-RoPE"""
        window_index_1d, cu_seqlens_window_1d = self.get_window_index_1d(t, h, w)
        llm_grid_h = h // self.spatial_merge_size
        llm_grid_w = w // self.spatial_merge_size
        total_tokens = t * llm_grid_h * llm_grid_w

        actual_seq_len = total_tokens * self.spatial_merge_unit

        rotary_pos_emb_1d_full = self.rotary_pos_emb_1d(actual_seq_len)

        # Reshape to match Qwen2.5-VL shape:
        # [total_tokens, spatial_merge_unit, head_dim // 2]
        rotary_pos_emb_1d = rotary_pos_emb_1d_full.view(
            total_tokens, self.spatial_merge_unit, -1
        )

        rotary_pos_emb_1d = rotary_pos_emb_1d[window_index_1d, :, :]
        rotary_pos_emb_1d = rotary_pos_emb_1d.flatten(start_dim=0, end_dim=1)

        # cu_seqlens_1d must match Qwen2.5-VL's cu_seqlens_thw format:
        # This represents the number of patches before merge (h * w)
        # Note: h and w are already in patch units from grid_thw
        # This is used for full attention blocks and must be in patch units
        cu_seqlens_1d = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t
        )

        return (
            rotary_pos_emb_1d,
            window_index_1d,
            cu_seqlens_window_1d,
            cu_seqlens_1d,
        )

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
        for idx, (t, h, w) in enumerate(grid_thw):
            t, h, w = int(t), int(h), int(w)
            # 6. Processor/grid_thw unit verification: log to confirm patch units
            # grid_thw must be in patch units (not pixels)
            # If in pixels, values would be much larger and not divisible
            if h % self.spatial_merge_size != 0 or w % self.spatial_merge_size != 0:
                raise ValueError(
                    f"grid_thw[{idx}] values (h={h}, w={w}) must be in patch units, "
                    f"but are not divisible by "
                    f"spatial_merge_size={self.spatial_merge_size}. "
                    f"This suggests grid_thw is in pixels instead of patches, "
                    f"which will break token alignment and spatial structure."
                )
            # Log grid_thw values for verification (first item only to avoid spam)
            if idx == 0:
                logger.warning(
                    "OpenCUA grid_thw[0] verification: t=%d, h=%d, w=%d "
                    "(patch units), spatial_merge_size=%d, "
                    "llm_grid_h=%d, llm_grid_w=%d",
                    t,
                    h,
                    w,
                    self.spatial_merge_size,
                    h // self.spatial_merge_size,
                    w // self.spatial_merge_size,
                )
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_1d,
                window_index_1d,
                cu_seqlens_window_1d,
                cu_seqlens_1d,
            ) = self.get_rope_by_1d(t, h, w)

            window_index.append(window_index_1d + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_1d = cu_seqlens_window_1d + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_1d[-1]
            cu_window_seqlens.append(cu_seqlens_window_1d)

            rotary_pos_emb.append(rotary_pos_emb_1d)

            cu_seqlens.append(cu_seqlens_1d)

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

        # For OpenCUA-VL, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        # Qwen2.5-VL merger accepts [seq_len, 1, context_dim] directly
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


class OpenCUA_VLProcessingInfo(Qwen2VLProcessingInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for processor to avoid reloading
        self._cached_processor: Qwen2_5_VLProcessor | None = None

    def get_hf_config(self):
        # Try to get OpenCUA_VLConfig first
        try:
            config = self.ctx.get_hf_config(OpenCUA_VLConfig)
        except TypeError:
            # If the loaded config is OpenCUAConfig from the model repository,
            # load it directly and add vLLM-specific attributes if needed
            from transformers import AutoConfig

            model_path = self.ctx.model_config.model
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # If it's already OpenCUAConfig from HF, add vLLM-specific attributes
            if hasattr(config, "model_type") and config.model_type == "opencua":
                # Add vLLM-specific attributes if they don't exist
                if not hasattr(config, "image_token_id"):
                    config.image_token_id = getattr(
                        config, "media_placeholder_token_id", 151664
                    )
                if not hasattr(config, "video_token_id"):
                    config.video_token_id = getattr(
                        config, "media_placeholder_token_id", 151664
                    )
                if not hasattr(config, "vision_start_token_id"):
                    config.vision_start_token_id = 151661
                if not hasattr(config, "vision_end_token_id"):
                    config.vision_end_token_id = 151663
                if not hasattr(config, "use_1d_rope"):
                    config.use_1d_rope = True
            else:
                # Convert to OpenCUA_VLConfig if it's a different config
                config_dict = config.to_dict()
                config = OpenCUA_VLConfig.from_dict(config_dict)

        # Ensure text_config has rope_scaling with mrope_section for uses_mrope
        text_config = config.get_text_config()
        if text_config:
            rope_scaling = getattr(text_config, "rope_scaling", None)
            if (
                rope_scaling is None
                or not isinstance(rope_scaling, dict)
                or "mrope_section" not in rope_scaling
            ):
                # Calculate mrope_section for 1D RoPE
                head_dim = getattr(text_config, "head_dim", None)
                if head_dim is None:
                    head_dim = (
                        text_config.hidden_size // text_config.num_attention_heads
                    )
                rotary_dim = head_dim
                # Split rotary_dim // 2 equally across 3 dimensions for 1D RoPE
                section_size = (rotary_dim // 2) // 3
                remainder = (rotary_dim // 2) % 3
                mrope_section = [section_size] * 3
                # Distribute remainder to first dimensions
                for i in range(remainder):
                    mrope_section[i] += 1
                text_config.rope_scaling = {
                    "rope_type": "default",
                    "mrope_section": mrope_section,
                }
        return config

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        # If max_pixels is provided in kwargs, update the cached
        # processor's image_processor
        if self._cached_processor is not None:
            # If max_pixels is provided in kwargs, update the cached
            # processor's image_processor
            if "max_pixels" in kwargs and hasattr(
                self._cached_processor, "image_processor"
            ):
                current_max = getattr(
                    self._cached_processor.image_processor, "max_pixels", None
                )
                new_max = kwargs["max_pixels"]
                if current_max is None or current_max < new_max:
                    self._cached_processor.image_processor.max_pixels = new_max
            return self._cached_processor

        from transformers import AutoImageProcessor, AutoTokenizer

        # OpenCUA model path - use OpenCUA's own preprocessor/tokenizer
        model_path = self.ctx.model_config.model
        use_fast = kwargs.pop("use_fast", True)

        qwen2_vl_base = "Qwen/Qwen2.5-VL-7B-Instruct"

        # Priority 1: Try to load image processor from OpenCUA model path
        opencua_image_processor = None
        try:
            opencua_image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )
        except Exception as e:
            logger.warning(
                "Failed to load OpenCUA image processor from %s: %s. "
                "Will use Qwen2.5-VL processor.",
                model_path,
                e,
            )

        # Load Qwen2.5-VL processor (has min_pixels attribute required by vLLM)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            qwen2_vl_base,
            trust_remote_code=True,
            use_fast=use_fast,
            **kwargs,
        )

        # Replace image processor with OpenCUA's if available
        # OpenCUA image processor may not have min_pixels/max_pixels,
        # but we should use it anyway
        if opencua_image_processor is not None:
            # If OpenCUA image processor doesn't have min_pixels/max_pixels,
            # add them from Qwen2.5-VL processor
            qwen_image_processor = processor.image_processor

            # Log image processor parameters for verification
            logger.warning(
                "OpenCUA image processor loaded: size=%s, "
                "image_mean=%s, image_std=%s, interpolation=%s",
                getattr(opencua_image_processor, "size", None),
                getattr(opencua_image_processor, "image_mean", None),
                getattr(opencua_image_processor, "image_std", None),
                getattr(opencua_image_processor, "resample", None),
            )

            # Priority: 1) kwargs max_pixels, 2) Qwen2.5-VL processor max_pixels,
            # 3) OpenCUA's max_pixels
            target_max_pixels = None
            if "max_pixels" in kwargs:
                # Highest priority: use max_pixels from kwargs
                # (e.g., from mm_processor_kwargs)
                target_max_pixels = kwargs["max_pixels"]
            elif hasattr(qwen_image_processor, "max_pixels"):
                # Second priority: use Qwen2.5-VL processor's max_pixels
                target_max_pixels = qwen_image_processor.max_pixels

            if not hasattr(opencua_image_processor, "min_pixels") and hasattr(
                qwen_image_processor, "min_pixels"
            ):
                opencua_image_processor.min_pixels = qwen_image_processor.min_pixels

            if target_max_pixels is not None:
                if not hasattr(opencua_image_processor, "max_pixels"):
                    opencua_image_processor.max_pixels = target_max_pixels
                else:
                    current_max = opencua_image_processor.max_pixels
                    if current_max < target_max_pixels:
                        opencua_image_processor.max_pixels = target_max_pixels
            processor.image_processor = opencua_image_processor

        # Load OpenCUA tokenizer (highest priority)
        opencua_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=use_fast,
        )

        # Replace processor's tokenizer with OpenCUA tokenizer
        processor.tokenizer = opencua_tokenizer

        # Get OpenCUA config for token IDs
        hf_config = self.get_hf_config()

        vocab = opencua_tokenizer.get_vocab()
        if "<|media_placeholder|>" not in vocab:
            raise ValueError(
                "<|media_placeholder|> not found in OpenCUA tokenizer vocab. "
                "This is required for proper text recognition."
            )

        media_placeholder_id = vocab["<|media_placeholder|>"]
        eos_token_id = None
        if (
            hasattr(opencua_tokenizer, "eos_token_id")
            and opencua_tokenizer.eos_token_id is not None
        ):
            eos_token_id = opencua_tokenizer.eos_token_id
        elif hasattr(opencua_tokenizer, "eos_token") and opencua_tokenizer.eos_token:
            eos_token_id = opencua_tokenizer.convert_tokens_to_ids(
                opencua_tokenizer.eos_token
            )

        # Validate that image_token_id is not the same as EOS token ID
        if (
            hasattr(hf_config, "image_token_id")
            and eos_token_id is not None
            and hf_config.image_token_id == eos_token_id
        ):
            logger.error(
                "Config image_token_id (%s) is EOS token ID! "
                "This will cause serious issues. "
                "Expected <|media_placeholder|> ID: %s, "
                "EOS token ID: %s. "
                "Auto-correcting to <|media_placeholder|> ID.",
                hf_config.image_token_id,
                media_placeholder_id,
                eos_token_id,
            )

        # Validate that media_placeholder ID is not the same as EOS token ID
        if eos_token_id is not None and media_placeholder_id == eos_token_id:
            raise ValueError(
                f"<|media_placeholder|> token ID ({media_placeholder_id}) "
                f"is EOS token ID! "
                f"This is a critical configuration error. "
                f"EOS token ID: {eos_token_id}. "
                f"Please check the tokenizer configuration."
            )

        # Sync config with actual token IDs from tokenizer
        if hasattr(hf_config, "image_token_id"):
            if hf_config.image_token_id != media_placeholder_id:
                if (
                    eos_token_id is not None
                    and hf_config.image_token_id == eos_token_id
                ):
                    pass
                else:
                    logger.warning(
                        "Config image_token_id (%s) != "
                        "actual <|media_placeholder|> (%s). "
                        "Updating config.",
                        hf_config.image_token_id,
                        media_placeholder_id,
                    )
            hf_config.image_token_id = media_placeholder_id
            if hasattr(hf_config, "video_token_id"):
                hf_config.video_token_id = media_placeholder_id

        # Use the actual token ID (from vocab, not from config)
        image_token_id = media_placeholder_id
        video_token_id = media_placeholder_id

        processor.image_token = "<|media_placeholder|>"
        processor.video_token = "<|media_placeholder|>"

        # Verify processor is NOT using wrong token names
        if processor.image_token != "<|media_placeholder|>":
            raise ValueError(
                f"processor.image_token must be '<|media_placeholder|>', "
                f"but got '{processor.image_token}'. "
                f"This will break text recognition."
            )

        # Monkey patch _check_special_mm_tokens to use OpenCUA token IDs directly
        def patched_check_special_mm_tokens(text, text_inputs, modalities=None):
            """Patched version that uses OpenCUA config token IDs directly"""
            if modalities is None:
                modalities = ["image", "video"]

            # Get input_ids tensor/list
            input_ids = text_inputs["input_ids"]
            if hasattr(input_ids, "tolist"):
                input_ids_list = input_ids.tolist()
                if isinstance(input_ids_list[0], list):
                    input_ids_list = input_ids_list[0]
            else:
                input_ids_list = input_ids

            for modality in modalities:
                if modality == "image":
                    token_id = image_token_id
                elif modality == "video":
                    token_id = video_token_id
                else:
                    continue

                # Count occurrences in tokenized input_ids using config token ID
                ids_count = input_ids_list.count(token_id)

                if ids_count > 0:
                    pass

        # Apply the monkey patch
        processor._check_special_mm_tokens = patched_check_special_mm_tokens

        # Get OpenCUA chat template from OpenCUA tokenizer
        chat_template = None
        if (
            hasattr(opencua_tokenizer, "chat_template")
            and opencua_tokenizer.chat_template
        ):
            chat_template = opencua_tokenizer.chat_template
        elif hasattr(opencua_tokenizer, "get_chat_template"):
            try:
                chat_template = opencua_tokenizer.get_chat_template()
            except Exception as e:
                logger.warning(
                    "Failed to get chat_template from OpenCUA tokenizer: %s", e
                )

        # Set chat_template to processor
        if chat_template:
            # Log chat template for verification
            if isinstance(chat_template, str):
                logger.warning(
                    "OpenCUA chat_template loaded (length=%d, first_100_chars=%s)",
                    len(chat_template),
                    chat_template[:100] if len(chat_template) > 100 else chat_template,
                )
            else:
                logger.warning(
                    "OpenCUA chat_template loaded (type=%s)",
                    type(chat_template).__name__,
                )

            # Optionally inject system prompt from environment variable
            import os

            system_prompt_env = os.getenv("OPENCUA_SYSTEM_PROMPT")
            if system_prompt_env:
                # Wrap chat_template to inject system prompt
                original_template = chat_template

                def modified_chat_template(messages, tokenizer, **kwargs):
                    # Check if system message already exists
                    has_system = messages and messages[0].get("role") == "system"

                    if not has_system:
                        # Prepend system message
                        messages = [
                            {"role": "system", "content": system_prompt_env}
                        ] + messages

                    # Use original template
                    if callable(original_template):
                        return original_template(
                            messages, tokenizer=tokenizer, **kwargs
                        )
                    else:
                        # If it's a string (Jinja template),
                        # use tokenizer.apply_chat_template
                        return tokenizer.apply_chat_template(
                            messages,
                            tokenizer=tokenizer,
                            chat_template=original_template,
                            **kwargs,
                        )

                processor.chat_template = modified_chat_template
            else:
                processor.chat_template = chat_template

            # Also set chat_template on tokenizer for vLLM's resolve_hf_chat_template
            # This ensures the chat template is used even if processor is cached
            # Set on opencua_tokenizer (which is now processor.tokenizer)
            if hasattr(opencua_tokenizer, "chat_template"):
                if isinstance(chat_template, str):
                    opencua_tokenizer.chat_template = chat_template
                elif callable(chat_template):
                    # If it's a callable, wrap it for tokenizer
                    def tokenizer_chat_template(messages, tokenizer, **kwargs):
                        return chat_template(messages, tokenizer, **kwargs)

                    opencua_tokenizer.chat_template = tokenizer_chat_template

            # Also set on ctx.tokenizer if it's different from opencua_tokenizer
            # Note: ctx.tokenizer is frozen, so we can't modify it directly
            # But processor.tokenizer is set to opencua_tokenizer, and
            # resolve_hf_chat_template checks processor.chat_template first
            # If ctx.tokenizer is the same object, it will already have chat_template
            ctx_tokenizer = self.ctx.tokenizer
            if ctx_tokenizer is not opencua_tokenizer and hasattr(
                ctx_tokenizer, "chat_template"
            ):
                import contextlib

                if isinstance(chat_template, str):
                    with contextlib.suppress(AttributeError, TypeError):
                        ctx_tokenizer.chat_template = chat_template
                elif callable(chat_template):

                    def ctx_tokenizer_chat_template(messages, tokenizer, **kwargs):
                        return chat_template(messages, tokenizer, **kwargs)

                    with contextlib.suppress(AttributeError, TypeError):
                        ctx_tokenizer.chat_template = ctx_tokenizer_chat_template

            # Inject chat_template into vLLM's processor chat template cache
            # This ensures _try_get_processor_chat_template uses our modified template
            try:
                from vllm.entrypoints.chat_utils import _PROCESSOR_CHAT_TEMPLATES

                cache_key = (
                    opencua_tokenizer.name_or_path,
                    getattr(self.ctx.model_config, "trust_remote_code", False),
                )
                # Store the actual chat_template (string or callable)
                # _try_get_processor_chat_template will use processor.chat_template
                # which we've already set above, but we also cache it directly
                if isinstance(chat_template, str):
                    _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
                elif callable(chat_template):
                    # For callable templates, we need to store a reference
                    # The cache expects string templates, but we can store
                    # the callable and let processor.chat_template handle it
                    _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
                logger.warning(
                    "OpenCUA chat_template injected into vLLM cache: "
                    "cache_key=%s, type=%s",
                    cache_key,
                    type(chat_template).__name__,
                )
            except ImportError:
                logger.warning(
                    "Failed to import _PROCESSOR_CHAT_TEMPLATES, "
                    "chat_template cache injection skipped."
                )
        else:
            logger.warning("OpenCUA chat_template not found.")

        # Validate all required special tokens exist and are not UNK
        im_tokens = ["<|im_user|>", "<|im_assistant|>", "<|im_end|>"]
        media_tokens = [
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_end|>",
            "<|media_placeholder|>",
        ]
        all_required_tokens = im_tokens + media_tokens

        unk_token_id = (
            opencua_tokenizer.unk_token_id
            if hasattr(opencua_tokenizer, "unk_token_id")
            else None
        )

        for token_str in all_required_tokens:
            assert token_str in vocab, (
                f"Missing required token: {token_str}. "
                f"This will break text recognition."
            )
            token_id = vocab[token_str]
            if unk_token_id is not None:
                assert token_id != unk_token_id, (
                    f"Token {token_str} mapped to UNK (id={token_id}). "
                    f"This will break text recognition."
                )

        actual_media_placeholder_id = opencua_tokenizer.convert_tokens_to_ids(
            "<|media_placeholder|>"
        )
        assert actual_media_placeholder_id == media_placeholder_id, (
            f"processor.image_token_id mismatch: "
            f"vocab={media_placeholder_id}, "
            f"convert_tokens_to_ids={actual_media_placeholder_id}"
        )

        # Cache the processor to avoid reloading
        self._cached_processor = processor
        return processor


class OpenCUA_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Call the HF processor on the prompt text and associated multi-modal data.
        """
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # Use Qwen2.5-VL's approach: get token IDs from processor's
        # image_token/video_token strings
        # This matches what Qwen2VLDummyInputsBuilder generates
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        if hf_processor.image_token != "<|media_placeholder|>":
            raise ValueError(
                f"processor.image_token must be '<|media_placeholder|>', "
                f"but got '{hf_processor.image_token}'. "
                f"This will break text recognition. "
                f"Please ensure OpenCUA processor is correctly configured."
            )

        if hf_processor.image_token not in vocab:
            raise ValueError(
                f"processor.image_token '{hf_processor.image_token}' "
                f"not found in vocab. "
                f"This will break text recognition."
            )

        # Get actual token ID from vocab
        media_placeholder_id = vocab[hf_processor.image_token]

        # Get config for consistency check
        hf_config = self.info.get_hf_config()

        if hasattr(hf_config, "image_token_id"):
            hf_config.image_token_id = media_placeholder_id
            if hasattr(hf_config, "video_token_id"):
                hf_config.video_token_id = media_placeholder_id

        # Use the actual token ID from vocab (not from config)
        replacement_token_id = {
            "image": media_placeholder_id,
            "video": media_placeholder_id,  # OpenCUA may use same token
        }

        placeholder = {
            "image": media_placeholder_id,
            "video": media_placeholder_id,
        }

        merge_length = image_processor.merge_size**2
        hf_config = self.info.get_hf_config()
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        if image_processor.merge_size != spatial_merge_size:
            raise ValueError(
                f"CRITICAL: image_processor.merge_size "
                f"({image_processor.merge_size}) != "
                f"vision_config.spatial_merge_size ({spatial_merge_size}). "
                "This mismatch will cause incorrect visual token count "
                "calculation and break text/vision alignment. "
                "Please fix your configuration to match."
            )

        def get_replacement_opencua(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            # grid_thw shape: (t, h, w) - temporal, height, width in patches
            # num_tokens = (t×h×w) // merge_length
            grid_t, grid_h, grid_w = map(int, grid_thw)
            # 6. Processor/grid_thw unit verification: log to confirm patch units
            logger.warning(
                "OpenCUA %s item %d: grid_thw=[%d, %d, %d] (patch units), "
                "merge_size=%d, merge_length=%d",
                modality.upper(),
                item_idx,
                grid_t,
                grid_h,
                grid_w,
                image_processor.merge_size,
                merge_length,
            )
            total_patches = grid_t * grid_h * grid_w
            num_tokens = total_patches // merge_length

            assert num_tokens > 0, (
                f"Calculated {num_tokens} visual tokens for {modality} item {item_idx} "
                f"(grid_thw=[{grid_t}, {grid_h}, {grid_w}], "
                f"total_patches={total_patches}, merge_length={merge_length}). "
                f"This will break text recognition. "
                f"Check image processing configuration."
            )

            # Log placeholder count vs visual token count for validation
            # This is called during _bind_and_group_updates -> resolve(item_idx)
            logger.warning(
                "OpenCUA %s item %d: placeholder_count=1, visual_token_count=%d, "
                "grid_thw=[%d, %d, %d], merge_length=%d",
                modality.upper(),
                item_idx,
                num_tokens,
                grid_t,
                grid_h,
                grid_w,
                merge_length,
            )

            return [replacement_token_id[modality]] * num_tokens

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
    dummy_inputs=Qwen2VLDummyInputsBuilder,
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
        # OpenCUA uses <|media_placeholder|> instead of <|image_pad|>
        if modality.startswith("image"):
            return "<|media_placeholder|>"
        if modality.startswith("video"):
            return "<|media_placeholder|>"  # OpenCUA uses same token for video

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
            # Store language model hidden_size for debugging
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

        # Use text_config for language model initialization
        # (Qwen2ForCausalLM expects Qwen2Config, not OpenCUA_VLConfig)
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            # If text_config is not set, create one from the main config
            # Extract text model parameters from the main config
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

        # OpenCUA uses 1D RoPE, but implements SupportsMRoPE interface
        # Set rope_scaling to enable M-RoPE position calculation
        # (get_mrope_input_positions will use 1D sequential positions)
        # Check if rope_scaling needs to be set (even if text_config exists)
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if (
            rope_scaling is None
            or not isinstance(rope_scaling, dict)
            or "mrope_section" not in rope_scaling
        ):
            # Calculate mrope_section for 1D RoPE
            # For 1D RoPE, we split rotary_dim equally across T, H, W
            # This ensures MRotaryEmbedding is created and can handle 3D positions
            # with all dimensions having the same value (1D sequential positions)
            head_dim = getattr(text_config, "head_dim", None)
            if head_dim is None:
                head_dim = text_config.hidden_size // text_config.num_attention_heads
            rotary_dim = head_dim
            # Split rotary_dim // 2 equally across 3 dimensions for 1D RoPE
            section_size = (rotary_dim // 2) // 3
            remainder = (rotary_dim // 2) % 3
            mrope_section = [section_size] * 3
            # Distribute remainder to first dimensions
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
        """Run forward pass for OpenCUA-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch. **NOTE**: For OpenCUA-VL, positions are 1D instead of 3D.
        """

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
        """Get 1D RoPE input positions for OpenCUA-VL model.

        OpenCUA uses 1D RoPE instead of M-RoPE, so all position dimensions
        (T, H, W) are set to the same 1D sequential position value.
        """
        if image_grid_thw is None:
            image_grid_thw = []
        if video_grid_thw is None:
            video_grid_thw = []
        if second_per_grid_ts is None:
            second_per_grid_ts = []

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
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
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st
            num_visual_tokens = llm_grid_t * llm_grid_h * llm_grid_w

            # Calculate st_idx after text_len (match Qwen2.5-VL order)
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

            # Text tokens: 1D sequential positions (all dimensions same)
            # For 1D RoPE, all dimensions (T, H, W) use the same sequential position
            text_positions = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            llm_pos_ids_list.append(text_positions)

            # Visual tokens: 1D sequential positions (all dimensions same)
            # Position starts after text tokens (placeholder replaced by visual)
            # For 1D RoPE, all dimensions (T, H, W) use the same sequential position
            # Original OpenCUA: visual position starts from st_idx + text_len - 1
            # (placeholder position is replaced, so visual starts after it)
            visual_start_pos = st_idx + text_len - 1
            visual_positions = (
                torch.arange(num_visual_tokens).view(1, -1).expand(3, -1)
                + visual_start_pos
                + 1
            )
            llm_pos_ids_list.append(visual_positions)

            # Skip the placeholder token in input_tokens
            # (placeholder is replaced by num_visual_tokens visual embeddings
            # in actual sequence)
            st = ed + 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            text_positions = torch.arange(text_len) + st_idx
            llm_pos_ids_list.append(text_positions.view(1, -1).expand(3, -1))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Check critical vision components
        # Note: rotary_pos_emb is not a weight parameter (it's computed at runtime)
        critical_components = [
            "visual.patch_embed",
            "visual.blocks.0",
            "visual.merger",
        ]
        missing_components = []
        for component in critical_components:
            if not any(p.startswith(component) for p in loaded_params):
                missing_components.append(component)

        if missing_components:
            logger.warning(
                "OpenCUA missing critical vision components: %s. "
                "This may cause vision processing issues.",
                missing_components,
            )

        return loaded_params

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
