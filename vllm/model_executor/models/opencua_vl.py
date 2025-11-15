# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Qwen2.5-VL and OpenCUA implementation
# OpenCUA uses 1D RoPE instead of MRoPE, and uses Kimi-VL tokenizer/template

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenCUA_VLConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)


class OpenCUA_VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OpenCUA_VLConfig)

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        # Use Kimi-VL processor for tokenizer and template
        # OpenCUA model doesn't have a processor, so we load
        # Kimi-VL processor explicitly
        from transformers import AutoProcessor

        # Load Kimi-VL processor explicitly using a known Kimi-VL model name
        # This ensures we get the correct processor with tokenizer and
        # image_processor
        kimi_vl_model_name = "moonshotai/Kimi-VL-A3B-Instruct"

        return AutoProcessor.from_pretrained(
            kimi_vl_model_name,
            trust_remote_code=kwargs.pop("trust_remote_code", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        return hf_processor.image_processor

    def get_tokenizer(self):
        # Use Kimi-VL tokenizer
        hf_processor = self.get_hf_processor()
        return hf_processor.tokenizer

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        patch_size = hf_config.spatial_patch_size
        merge_size = hf_config.spatial_merge_size

        # Calculate tokens after patching and merging
        h_tokens = (image_height // patch_size) // merge_size
        w_tokens = (image_width // patch_size) // merge_size
        return h_tokens * w_tokens

    @property
    def image_token_id(self) -> int:
        return self.get_hf_config().image_token_id

    @property
    def video_token_id(self) -> int:
        return self.get_hf_config().video_token_id


class OpenCUA_VLDummyInputsBuilder(BaseDummyInputsBuilder[OpenCUA_VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        image_token = getattr(processor, "image_token", "<|media_placeholder|>")
        video_token = getattr(processor, "video_token", "<|media_placeholder|>")

        tokens = []
        for _ in range(num_images):
            tokens.append(image_token)
        for _ in range(num_videos):
            tokens.append(video_token)

        return "".join(tokens)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        result = {}
        if num_images > 0:
            result["image"] = self._get_dummy_images(
                width=1024,
                height=1024,
                num_images=num_images,
                overrides=image_overrides,
            )
        if num_videos > 0:
            result["video"] = self._get_dummy_videos(
                width=1024,
                height=1024,
                num_videos=num_videos,
                overrides=video_overrides,
            )
        return result


class OpenCUA_VLMultiModalProcessor(BaseMultiModalProcessor[OpenCUA_VLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Always return config for both image and video modalities
        # to match Qwen2.5-VL behavior and _get_prompt_updates
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)
        video_grid_sizes = video_grid_thw.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        merge_size = hf_config.spatial_merge_size

        def get_replacement(item_idx: int, modality: str):
            # Qwen2.5-VL style: directly access out_mm_kwargs[modality][item_idx]
            # This assumes the modality exists in out_mm_kwargs when called
            out_item = out_mm_kwargs[modality][item_idx]
            grid_key = f"{modality}_grid_thw"
            grid_thw = out_item[grid_key].data
            assert isinstance(grid_thw, torch.Tensor)

            # Calculate number of tokens after spatial merging
            num_tokens = int(grid_thw.prod()) // (merge_size * merge_size)
            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * num_tokens

        # Always return updates for both image and video modalities
        # to match expected behavior in _merge_mm_kwargs
        from functools import partial

        return [
            PromptReplacement(
                modality=modality,
                target=[image_token_id if modality == "image" else video_token_id],
                replacement=partial(get_replacement, modality=modality),
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
):
    """OpenCUA VL model based on Qwen2.5-VL but using 1D RoPE.

    Key differences from Qwen2.5-VL:
    - Uses 1D RoPE (not MRoPE), so positions are (seq_len,)
      not (3, seq_len) like MRoPE
    - Uses Kimi-VL tokenizer and chat template
    - Does not implement get_mrope_input_positions
    """

    merge_by_field_config = True
    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}

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
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|media_placeholder|>"
        if modality.startswith("video"):
            return "<|media_placeholder|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: OpenCUA_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            # Use Qwen2.5 vision transformer but with 1D RoPE
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        # For now, treat video same as image (can be extended later)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            # Use image input format for video (Qwen2.5 vision handles both)
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values_videos,
                image_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=video_embeds,
                image_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
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
                    # Use rope_3d for vision model (Qwen2.5 vision transformer)
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
                    )
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                # Process video same as image for now
                video_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
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
                batch. **NOTE**: OpenCUA uses 1D RoPE, so positions shape is
                (seq_len,), not (3, seq_len) like MRoPE.
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
