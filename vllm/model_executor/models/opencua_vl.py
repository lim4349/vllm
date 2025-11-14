# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from Qwen2.5-VL implementation
# OpenCUA-7B uses 1D RoPE instead of M-RoPE and does not use window attention

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, BatchFeature
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

from vllm.attention.backends.registry import _Backend
from vllm.attention.layer import maybe_get_vit_flash_attn_backend
from vllm.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_xformers_attn_wrapper,
)
from vllm.compilation.decorators import support_torch_compile
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
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.opencua_vl import OpenCUA_VLConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_vl import (
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
)

logger = init_logger(__name__)

# === Vision Inputs === #


class OpenCUAVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size
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


class OpenCUAVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images
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


OpenCUAVLImageInputs: TypeAlias = (
    OpenCUAVLImagePixelInputs | OpenCUAVLImageEmbeddingInputs
)


# === Vision Encoder === #


class OpenCUAVisionMLP(nn.Module):
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


class OpenCUAVisionAttention(nn.Module):
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
        attn_backend_override: _Backend | None = None,
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
                attn_backend_override=attn_backend_override,
            )
        )
        # On ROCm with FLASH_ATTN backend, upstream flash_attn is used
        from vllm.platforms import current_platform

        if current_platform.is_rocm() and self.attn_backend == _Backend.FLASH_ATTN:
            self.use_upstream_fa = True
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
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        seqlens: torch.Tensor,  # Only used for xFormers
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (einops.rearrange(x, "s b ... -> b s ...") for x in (q, k, v))
        if rotary_pos_emb is not None:
            # [2 * b, s, heads, head_dim]
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        if self.is_flash_attn_backend:
            context_layer = vit_flash_attn_wrapper(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen,
                batch_size,
                self.attn_backend == _Backend.ROCM_AITER_FA,
                self.use_upstream_fa,
            )
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            from vllm.platforms import current_platform

            # Never remove the next contiguous logic
            # Without it, hallucinations occur with the backend
            if current_platform.is_rocm():
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (
                    einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
                )
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
            context_layer = einops.rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()
        elif self.attn_backend == _Backend.XFORMERS:
            context_layer = vit_xformers_attn_wrapper(q, k, v, seqlens)

        output, _ = self.proj(context_layer)
        return output


class OpenCUAVisionBlock(nn.Module):
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
        attn_backend_override: _Backend | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = OpenCUAVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel,
            attn_backend=attn_backend,
            use_upstream_fa=use_upstream_fa,
            attn_backend_override=attn_backend_override,
        )
        self.mlp = OpenCUAVisionMLP(
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
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        seqlens: torch.Tensor,  # Only used for xFormers
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


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
    }
)
class OpenCUAVisionPatchEmbed(nn.Module):
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


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
    }
)
class OpenCUAVisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
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


class OpenCUAVisionRotaryEmbedding(nn.Module):
    """1D RoPE for OpenCUA - simplified from Qwen2.5-VL's M-RoPE"""

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
        """Return 1D RoPE embeddings for sequence length."""
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class OpenCUAVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: _Backend | None = None,
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

        # OpenCUA uses spatial merge but no window attention
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("OpenCUAVisionPatchEmbed"):
            self.patch_embed = OpenCUAVisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                hidden_size=self.hidden_size,
            )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        # Use 1D RoPE instead of M-RoPE
        self.rotary_pos_emb = OpenCUAVisionRotaryEmbedding(head_dim // 2)

        use_upstream_fa = False
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
                attn_backend_override=attn_backend_override,
            )
        )

        if self.attn_backend not in {
            _Backend.FLASH_ATTN,
            _Backend.TORCH_SDPA,
            _Backend.XFORMERS,
            _Backend.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"OpenCUA-VL does not support {self.attn_backend} backend now."
            )

        with set_model_tag("OpenCUAVisionBlock"):
            self.blocks = nn.ModuleList(
                [
                    OpenCUAVisionBlock(
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
                        attn_backend_override=attn_backend_override,
                    )
                    for layer_idx in range(depth)
                ]
            )

        with set_model_tag("OpenCUAVisionPatchMerger"):
            self.merger = OpenCUAVisionPatchMerger(
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

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_seqlen = torch.zeros([], device=cu_seqlens.device)
        seqlens = torch.zeros(1, device=cu_seqlens.device)
        if (
            self.attn_backend == _Backend.FLASH_ATTN
            or self.attn_backend == _Backend.ROCM_AITER_FA
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        return max_seqlen, seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        """
        Forward pass for OpenCUA Vision Transformer.
        Uses 1D RoPE and full attention (no window attention).
        """
        # patchify
        seq_len, _ = x.size()

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # Process each image/video in the batch
        # For OpenCUA, we use 1D RoPE with sequential position indices
        max_seq_len = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size
            total_seq_len = t * llm_h * llm_w * self.spatial_merge_unit
            max_seq_len = max(max_seq_len, total_seq_len)

        # Generate 1D RoPE embeddings for the maximum sequence length
        # OpenCUA uses simple 1D position indices (0, 1, 2, ...)
        rotary_pos_emb_full = self.rotary_pos_emb(max_seq_len)

        # Build cumulative sequence lengths for each media item
        cu_seqlens_list: list = [torch.tensor([0], dtype=torch.int32)]
        current_pos = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size
            total_seq_len = t * llm_h * llm_w * self.spatial_merge_unit
            current_pos += total_seq_len
            cu_seqlens_list.append(torch.tensor([current_pos], dtype=torch.int32))

        cu_seqlens = torch.cat(cu_seqlens_list)

        # Pre-compute seqlens for attention backends
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        rotary_pos_emb_full = rotary_pos_emb_full.to(
            device=hidden_states.device, non_blocking=True
        )

        # Reshape for spatial merge
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        # Apply all blocks with full attention (no window attention)
        # For each batch item, use the appropriate slice of rotary_pos_emb
        for blk in self.blocks:
            # Use the full rotary_pos_emb - it will be sliced per batch item
            # by the attention mechanism based on cu_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb_full,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )

        # For OpenCUA, float16 may overflow at last block for long sequences
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        hidden_states = self.merger(hidden_states)
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


class OpenCUAVLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OpenCUA_VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        """
        Load Qwen2_5_VLProcessor for OpenCUA model.
        OpenCUA uses Qwen2.5-VL processor structure but with TikTokenV3 tokenizer.
        """
        # Get already initialized tokenizer (TikTokenV3)
        tokenizer = self.get_tokenizer()

        # Get image processor
        image_processor = self.get_image_processor(**kwargs)

        # Initialize processor with tokenizer and image_processor
        # This avoids Qwen2Tokenizer loading issues
        return self.ctx.init_processor(
            Qwen2_5_VLProcessor,
            tokenizer=tokenizer,
            image_processor=image_processor,
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object):
        """
        Load image processor directly from OpenCUA model using AutoImageProcessor.
        This ensures we get the image processor even if AutoProcessor fails.
        """
        model_path = self.ctx.model_config.model
        # Load image processor directly from model path, same as HF
        image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )
        return image_processor


class OpenCUAVLDummyInputsBuilder(BaseDummyInputsBuilder[OpenCUAVLProcessingInfo]):
    """Dummy inputs builder for OpenCUA model."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Get dummy text for OpenCUA model.
        Uses image token directly from config or processor.
        """
        num_images = mm_counts.get("image", 0)

        # Try to get image token from processor, fallback to config
        try:
            hf_processor = self.info.get_hf_processor()
            # Check if processor has image_token attribute
            if hasattr(hf_processor, "image_token"):
                image_token: str = hf_processor.image_token
            else:
                # Fallback: use tokenizer to get image token
                tokenizer = self.info.get_tokenizer()
                # OpenCUA uses <|media_placeholder|> or similar
                # Try common image token names
                for token_name in [
                    "<|media_placeholder|>",
                    "<|image_pad|>",
                    "<|image|>",
                ]:
                    if token_name in tokenizer.get_vocab():
                        image_token = token_name
                        break
                else:
                    # Default fallback
                    image_token = "<|media_placeholder|>"
        except Exception:
            # Final fallback
            image_token = "<|media_placeholder|>"

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        """Get dummy multimodal data for OpenCUA model."""
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class OpenCUAVLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)

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
        config = self.info.get_hf_config()

        # Get image token - try processor first, then config, then tokenizer vocab
        if hasattr(hf_processor, "image_token"):
            image_token_str = hf_processor.image_token
        elif hasattr(config, "image_token_id"):
            # Use image_token_id from config to find token string
            image_token_id = config.image_token_id
            # Reverse lookup in vocab
            image_token_str = None
            for token, token_id in vocab.items():
                if token_id == image_token_id:
                    image_token_str = token
                    break
            if image_token_str is None:
                # Fallback: try common token names
                for token_name in [
                    "<|media_placeholder|>",
                    "<|image_pad|>",
                    "<|image|>",
                ]:
                    if token_name in vocab:
                        image_token_str = token_name
                        break
        else:
            # Final fallback: try common token names
            image_token_str = None
            for token_name in [
                "<|media_placeholder|>",
                "<|image_pad|>",
                "<|image|>",
            ]:
                if token_name in vocab:
                    image_token_str = token_name
                    break

        if image_token_str is None or image_token_str not in vocab:
            raise ValueError(
                "Could not find image token in processor, config, or tokenizer vocab"
            )

        placeholder = {
            "image": vocab[image_token_str],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_opencua(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length

            return [placeholder[modality]] * num_tokens

        # Only process modalities that exist in mm_items
        # out_mm_kwargs should be populated by this point from HF processor
        prompt_updates = []
        if "image" in mm_items:
            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=[placeholder["image"]],
                    replacement=partial(get_replacement_opencua, modality="image"),
                )
            )

        return prompt_updates


@MULTIMODAL_REGISTRY.register_processor(
    OpenCUAVLMultiModalProcessor,
    info=OpenCUAVLProcessingInfo,
    dummy_inputs=OpenCUAVLDummyInputsBuilder,
)
class OpenCUA_VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
):
    merge_by_field_config = True

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
            # mapping for vision_tower (some checkpoints use this name)
            "vision_tower.": "visual.",
            "model.vision_tower.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        raise ValueError("Only image modality is supported for OpenCUA")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: OpenCUA_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt("image"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            self.visual = OpenCUAVisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        # Use text_config for language model initialization
        # OpenCUA_VLConfig is multimodal config, need text_config for Qwen2
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=config.text_config,
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

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> OpenCUAVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return OpenCUAVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return OpenCUAVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _process_image_input(
        self, image_input: OpenCUAVLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    # For data parallel, we need to handle it differently
                    # since OpenCUA doesn't use M-RoPE
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (
            torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)
            // (merge_size * merge_size)
        ).tolist()

        return image_embeds.split(sizes)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        image_embeddings = self._process_image_input(image_input)
        return tuple(image_embeddings)

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
                batch. For OpenCUA, this is 1D position ids (not M-RoPE).
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
