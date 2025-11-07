# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from HuggingFace OpenCUA configuration

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)


class OpenCUA_VLConfig(PretrainedConfig):
    """Configuration class for OpenCUA-VL model.

    This configuration follows the HuggingFace OpenCUAConfig structure
    with additional vLLM-specific attributes for proper integration.

    Args:
        vision_config: Configuration for the vision model (Qwen2_5_VLVisionConfig)
        text_config: Configuration for the text model (Qwen2Config)
        ignore_index: The token ID to use for ignoring during loss calculation
        media_placeholder_token_id: The token ID for <|media_placeholder|>
        pad_token_id: The token ID to use for padding
        image_token_id: The token ID for image placeholder (vLLM-specific)
        video_token_id: The token ID for video placeholder (vLLM-specific)
        vision_start_token_id: The token ID for <|media_begin|> (vLLM-specific)
        vision_end_token_id: The token ID for <|media_end|> (vLLM-specific)
        use_1d_rope: Whether to use 1D RoPE instead of M-RoPE (vLLM-specific)
    """

    model_type = "opencua"

    def __init__(
        self,
        vision_config: dict | Qwen2_5_VLVisionConfig | None = None,
        text_config: dict | Qwen2Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 151664,
        pad_token_id: int = 0,
        # vLLM-specific additional attributes
        image_token_id: int = 151664,
        video_token_id: int = 151664,
        vision_start_token_id: int = 151661,
        vision_end_token_id: int = 151663,
        use_1d_rope: bool = True,
        **kwargs,
    ):
        # Initialize vision config (HF structure)
        if isinstance(vision_config, dict):
            vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        # Initialize text config (HF structure)
        if isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        # HF standard attributes
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # vLLM-specific additional attributes
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.use_1d_rope = use_1d_rope

        # Call parent constructor
        super().__init__(pad_token_id=pad_token_id, **kwargs)
