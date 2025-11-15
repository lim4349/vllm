# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from HuggingFace OpenCUA configuration

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)


class OpenCUA_VLConfig(PretrainedConfig):
    model_type = "opencua"

    def __init__(
        self,
        vision_config: dict | Qwen2_5_VLVisionConfig | None = None,
        text_config: dict | Qwen2Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 151664,
        pad_token_id: int = 0,
        # vLLM-specific
        image_token_id: int = 151664,
        video_token_id: int = 151664,
        vision_start_token_id: int = 151661,
        vision_end_token_id: int = 151663,
        use_1d_rope: bool = True,
        # 1D RoPE 하이퍼 (필요시 모델에서 사용)
        rope_base: int = 10000,
        rope_scale: float = 1.0,
        # OpenCUA alias (HF와 키가 다를 때 보전)
        spatial_patch_size: int | None = None,
        spatial_merge_size: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ):
        # Vision/Text config 기본값 안전화
        if isinstance(vision_config, dict):
            vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        if text_config is None:
            text_config = Qwen2Config()
        self.text_config = text_config
        
        # Preserve existing rope_scaling from HF config if present
        # OpenCUA uses 1D RoPE, which should be handled via use_1d_rope flag
        # and model implementation, not by forcing MRoPE (mrope_section) format
        # If rope_scaling is already set in the config, we respect it as-is

        # HF 표준
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # vLLM 확장
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.use_1d_rope = use_1d_rope
        self.rope_base = rope_base
        self.rope_scale = rope_scale

        # OpenCUA alias 보전 (vision_config에서 가져오거나 인자 우선)
        vc = self.vision_config
        self.spatial_patch_size = (
            spatial_patch_size
            if spatial_patch_size is not None
            else getattr(vc, "spatial_patch_size", getattr(vc, "patch_size", 14))
        )
        self.spatial_merge_size = (
            spatial_merge_size
            if spatial_merge_size is not None
            else getattr(vc, "spatial_merge_size", getattr(vc, "merge_size", 2))
        )
        self.max_pixels = (
            max_pixels if max_pixels is not None else getattr(vc, "max_pixels", None)
        )
        
        # Set rope_theta in vision_config if not already set
        # This ensures 1D RoPE uses the correct base frequency
        if not hasattr(vc, "rope_theta") or vc.rope_theta is None:
            vc.rope_theta = float(rope_base)

        # 간단 validation
        for k in [
            "media_placeholder_token_id",
            "image_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            v = getattr(self, k)
            if not (isinstance(v, int) and v >= 0):
                raise ValueError(f"{k} invalid: {v}")

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    def get_text_config(self) -> Qwen2Config:
        """Get the text config for this multimodal model."""
        return self.text_config

    # (선택) 런타임 토크나이저 동기화를 쉽게 하는 helper
    def sync_special_token_ids(self, tokenizer):
        get = tokenizer.convert_tokens_to_ids
        mp = get("<|media_placeholder|>")
        mb = get("<|media_begin|>")
        me = get("<|media_end|>")
        if mp is not None and mp >= 0:
            self.media_placeholder_token_id = mp
            self.image_token_id = mp
        if mb is not None and mb >= 0:
            self.vision_start_token_id = mb
        if me is not None and me >= 0:
            self.vision_end_token_id = me
        # pad가 정의돼 있으면 패드도 동기화
        if getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = tokenizer.pad_token_id
