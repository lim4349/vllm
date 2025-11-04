# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Qwen2.5-VL and Kimi-VL configurations

from transformers.configuration_utils import PretrainedConfig

# Import Qwen2.5-VL configs dynamically to avoid import issues
try:
    from vllm.transformers_utils.configs.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
except ImportError:
    # Fallback: create minimal config classes
    from transformers.configuration_utils import PretrainedConfig
    
    class Qwen2_5_VLConfig(PretrainedConfig):
        model_type = "qwen2_5_vl"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class Qwen2_5_VLVisionConfig(PretrainedConfig):
        model_type = "qwen2_5_vl_vision"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class OpenCUA_VLConfig(Qwen2_5_VLConfig):
    """Configuration class for OpenCUA-VL model.
    
    This configuration is based on Qwen2.5-VL but uses Kimi-VL's tokenizer
    and chat template, and replaces M-RoPE with 1D RoPE.
    """
    model_type = "opencua_vl"
    
    def __init__(
        self,
        vision_config: dict | Qwen2_5_VLVisionConfig | None = None,
        text_config: dict | None = None,
        # OpenCUA tokenizer settings (from tokenizer_config.json)
        # CRITICAL: 실제 OpenCUA-7B tokenizer_config.json에서 확인된 값 사용
        # 151644 = [EOS] (EOS token)
        # 151664 = <|media_placeholder|> (실제 image/video placeholder)
        # 151661 = <|media_begin|>
        # 151663 = <|media_end|>
        media_placeholder_token_id: int = 151664,  # <|media_placeholder|>
        image_token_id: int = 151664,  # <|media_placeholder|> (EOS와 다름!)
        video_token_id: int = 151664,  # OpenCUA는 image/video 동일 토큰 사용
        vision_start_token_id: int = 151661,  # <|media_begin|>
        vision_end_token_id: int = 151663,  # <|media_end|>
        # Use 1D RoPE instead of M-RoPE
        use_1d_rope: bool = True,
        **kwargs,
    ):
        # Initialize with Qwen2.5-VL config
        super().__init__(**kwargs)
        
        # Override model type
        self.model_type = "opencua_vl"
        
        # Set Kimi-VL tokenizer IDs
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        
        # Use 1D RoPE instead of M-RoPE
        self.use_1d_rope = use_1d_rope
        
        # Override vision config if provided
        if vision_config is not None:
            if isinstance(vision_config, dict):
                self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
            else:
                self.vision_config = vision_config
        
        # Override text config if provided
        if text_config is not None:
            if isinstance(text_config, dict):
                # Use Qwen2 config for text
                try:
                    from transformers.models.qwen2 import Qwen2Config
                except ImportError:
                    # Fallback: use Qwen2_5_VLConfig's text_config if available
                    from transformers.configuration_utils import PretrainedConfig
                    
                    class Qwen2Config(PretrainedConfig):
                        model_type = "qwen2"
                        
                        def __init__(self, **kwargs):
                            super().__init__(**kwargs)
                    
                self.text_config = Qwen2Config(**text_config)
            else:
                self.text_config = text_config
