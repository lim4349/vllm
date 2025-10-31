# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example script to test OpenCUA-VL model with vLLM.

Usage:
    python test_opencua_vl.py --model xlangai/OpenCUA-32B
"""

import argparse
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams


def test_image_inference(model_path: str, image_path: str, dtype: str = "auto"):
    """Test OpenCUA-VL with a single image."""
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 10, "video": 1},
    )

    # Load image
    image = Image.open(image_path)
    print(f"Loaded image: {image_path}")

    # Create prompt with image placeholder
    prompt = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "이 이미지를 자세히 설명해주세요."
    )

    # Generate
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["<|im_end|>", "<|vision_end|>"],
    )

    print("Generating response...")
    outputs = llm.generate(
        [prompt],
        sampling_params,
        multi_modal_data={"image": [image]},
    )

    print("\n" + "=" * 50)
    print("Response:")
    print("=" * 50)
    for output in outputs:
        print(output.outputs[0].text)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test OpenCUA-VL with vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="xlangai/OpenCUA-32B",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Model dtype",
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return

    test_image_inference(args.model, args.image, args.dtype)


if __name__ == "__main__":
    main()

