#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCUA vLLM-ready multimodal helper
-----------------------------------
Drop-in utility to build correct inputs for OpenCUA(-7B) style models using
single media placeholder tokens and 1D RoPE position mapping.

Key guarantees:
- Exactly one <|media_placeholder|> token per media item in the prompt text.
- No pre-expansion of media tokens in the text prompt – expansion happens here only
  when computing position ids (MRoPE→1D positions).
- Stable media order via an explicit media queue (works even when image/video share
  the same placeholder token id).
- Processor singleton caching + RGB normalization for robustness.
- Extensive guard logs to diagnose mismatches at runtime.

This file does **not** send requests; it prepares tensors you can feed to your
model/vLLM engine. Integrate its functions into your serving path.
"""
from __future__ import annotations

import io
import math
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, PreTrainedTokenizerBase

# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

MEDIA_PLACEHOLDER = "<|media_placeholder|>"


@dataclass
class MediaItem:
    kind: str  # "image" or "video"
    source: str  # url or filepath
    grid_thw: Optional[Tuple[int, int, int]] = None  # (t, h, w) patches


# -----------------------------
# Image fetching / normalization
# -----------------------------

def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    if img.mode not in ("RGB", "L"):
        # Ensure RGB (avoid RGBA etc.)
        img = img.convert("RGB")
    return img


# -----------------------------
# Processor / Tokenizer singletons
# -----------------------------

@lru_cache(maxsize=1)
def get_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if MEDIA_PLACEHOLDER not in tok.get_vocab():
        # Some tokenizers add the special token via added_tokens; try to fetch its id.
        # If not found, raise early.
        try:
            _ = tok.convert_tokens_to_ids(MEDIA_PLACEHOLDER)
        except Exception as e:
            raise RuntimeError(
                f"Tokenizer for {model_name_or_path} lacks {MEDIA_PLACEHOLDER}: {e}"
            )
    return tok


@lru_cache(maxsize=1)
def get_image_processor(model_name_or_path: str):
    proc = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    # Expected attributes for Qwen2VL family
    merge_size = getattr(proc, "merge_size", None)
    size = getattr(proc, "size", None)
    max_pixels = getattr(proc, "max_pixels", None)
    min_pixels = getattr(proc, "min_pixels", None)
    LOGGER.info(
        "OpenCUA image processor loaded - type: %s, merge_size: %s, size: %s, max_pixels: %s, min_pixels: %s",
        type(proc).__name__, merge_size, size, max_pixels, min_pixels,
    )
    return proc


# -----------------------------
# Prompt building (OpenAI-style messages -> single placeholder per media)
# -----------------------------

def build_prompt_and_media(
    messages: Sequence[Dict[str, Any]],
) -> Tuple[str, List[MediaItem]]:
    """Build a simple user-only prompt with exactly one placeholder per media.

    Assumes messages like OpenAI Chat Completions content format.
    Returns (text_prompt, media_queue).
    """
    text_parts: List[str] = []
    media_queue: List[MediaItem] = []

    # We collect content in arrival order to preserve media order.
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "system", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            if role in ("system", "user"):
                text_parts.append(content)
            continue
        if isinstance(content, list):
            for seg in content:
                t = seg.get("type")
                if t == "text":
                    text_parts.append(seg.get("text", ""))
                elif t == "image_url":
                    url = seg.get("image_url", {}).get("url")
                    if url:
                        text_parts.append(MEDIA_PLACEHOLDER)
                        media_queue.append(MediaItem(kind="image", source=url))
                elif t == "video_url":
                    url = seg.get("video_url", {}).get("url")
                    if url:
                        text_parts.append(MEDIA_PLACEHOLDER)
                        media_queue.append(MediaItem(kind="video", source=url))
                else:
                    # ignore other types
                    pass

    prompt = "\n".join([p for p in text_parts if p])

    # Collapse any accidental duplicate placeholders (defense-in-depth)
    prompt = re.sub(rf"({re.escape(MEDIA_PLACEHOLDER)})+", MEDIA_PLACEHOLDER, prompt)

    return prompt, media_queue


def tokenize_with_guard(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    expected_media_count: int,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = enc["input_ids"]  # (1, L)
    placeholder_id = tokenizer.convert_tokens_to_ids(MEDIA_PLACEHOLDER)
    count = int((input_ids == placeholder_id).sum().item())
    LOGGER.info(
        "OpenCUA GUARD - placeholder count after tokenization: %d (expected %d)",
        count, expected_media_count,
    )
    if count != expected_media_count:
        raise RuntimeError(
            f"Placeholder count mismatch: got {count}, expected {expected_media_count}."
        )
    return enc


# -----------------------------
# Preprocess media with processor (images only here; videos can be extended)
# -----------------------------

def preprocess_media(
    model_name_or_path: str,
    media_queue: List[MediaItem],
    patch_size: int,
) -> Dict[str, Any]:
    images: List[Image.Image] = []
    grid_thws: List[Tuple[int, int, int]] = []

    for idx, item in enumerate(media_queue):
        if item.kind != "image":
            raise NotImplementedError("Video support not implemented in this helper.")
        img = _fetch_image(item.source)
        images.append(img)

    proc = get_image_processor(model_name_or_path)
    # Let the processor compute the final tensor sizes; trust_remote_code=True ensures Qwen2VL path
    out = proc(images=images, return_tensors="pt")
    pixel_values: torch.Tensor = out.get("pixel_values")  # (B, C, H, W)

    # Compute grid_thw from pixel_values spatial dims and known patch_size
    B, C, H, W = pixel_values.shape
    if H % patch_size or W % patch_size:
        raise RuntimeError(
            f"Processed size not divisible by patch_size={patch_size}: {(H, W)}"
        )
    h_patches = H // patch_size
    w_patches = W // patch_size
    for _ in range(B):
        grid_thws.append((1, int(h_patches), int(w_patches)))

    # Log a sample
    LOGGER.info(
        "OpenCUA preprocess output - image[0]: grid_thw=%s, processed_size=%dx%d (pixels), patch_size=%d",
        grid_thws[0], W, H, patch_size,
    )

    return {
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
    }


# -----------------------------
# 1D RoPE / MRoPE position computation
# -----------------------------

def _num_visual_tokens_from_grid(grid_thw: Tuple[int, int, int], merge_size: int) -> int:
    t, h, w = grid_thw
    if (h % merge_size) or (w % merge_size):
        raise RuntimeError(
            f"Grid {grid_thw} not divisible by merge_size={merge_size}"
        )
    return int(t * (h // merge_size) * (w // merge_size))


def compute_positions_1d(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,  # (1, L_text)
    media_queue: List[MediaItem],
    grid_thws: List[Tuple[int, int, int]],
    merge_size: int,
) -> torch.Tensor:
    """Compute 1D positions that interleave text and visual tokens at each placeholder.

    Returns position_ids shaped (3, L_total) as many OpenCUA/Qwen-vl stacks expect.
    """
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    ids = input_ids[0]  # (L,)
    L_text = ids.numel()

    placeholder_id = tokenizer.convert_tokens_to_ids(MEDIA_PLACEHOLDER)

    # Collect placeholder indices in order of appearance in text
    ph_indices: List[int] = [i for i, t in enumerate(ids.tolist()) if t == placeholder_id]
    if len(ph_indices) != len(media_queue):
        raise RuntimeError(
            f"Text has {len(ph_indices)} placeholders but media_queue has {len(media_queue)} items."
        )

    # We will assign positions as a simple increasing 1D range, inserting visual spans
    # at each placeholder location in order.
    pos_list: List[int] = []
    cursor = 0
    total_visual = 0
    for media_idx, ph in enumerate(ph_indices):
        # emit text positions up to (but excluding) placeholder
        while cursor < ph:
            pos_list.append(len(pos_list))
            cursor += 1
        # emit one position for the placeholder token itself
        pos_list.append(len(pos_list))
        cursor += 1

        # now emit visual positions block for this media
        grid = grid_thws[media_idx]
        nv = _num_visual_tokens_from_grid(grid, merge_size)
        total_visual += nv
        start = len(pos_list)
        pos_list.extend(range(start, start + nv))

    # emit remaining text after the last placeholder
    while cursor < L_text:
        pos_list.append(len(pos_list))
        cursor += 1

    pos_1d = torch.tensor(pos_list, dtype=torch.long)  # (L_total,)
    # repeat 3 times to get (3, L_total)
    pos_3 = pos_1d.unsqueeze(0).repeat(3, 1)

    LOGGER.info(
        "OpenCUA MRoPE final - llm_positions_1d shape: %s, llm_positions shape: %s, text_len: %d, total_visual_tokens: %d",
        tuple(pos_1d.shape), tuple(pos_3.shape), L_text, total_visual,
    )

    # Sanity: last position should be len-1
    assert pos_1d[-1].item() == pos_1d.numel() - 1
    return pos_3


# -----------------------------
# Public API
# -----------------------------

def build_opencua_inputs(
    model_name_or_path: str,
    messages: Sequence[Dict[str, Any]],
    *,
    patch_size: int = 14,
    merge_size: int = 2,
) -> Dict[str, Any]:
    """High-level helper that prepares everything you need.

    Returns dict with keys:
      - input_ids (1, L_text)
      - attention_mask (1, L_text)
      - pixel_values (B, C, H, W)
      - grid_thws (list of (t,h,w))
      - position_ids (3, L_total)
      - media_count (int)
    """
    # 1) Build prompt & media queue
    prompt, media_queue = build_prompt_and_media(messages)
    LOGGER.info(
        "OpenCUA preprocess input - media_count=%d, prompt_preview=%r",
        len(media_queue), (prompt[:120] + "...") if len(prompt) > 120 else prompt,
    )

    # 2) Tokenize with guards
    tokenizer = get_tokenizer(model_name_or_path)
    enc = tokenize_with_guard(tokenizer, prompt, expected_media_count=len(media_queue))
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    # 3) Preprocess media (images)
    if media_queue:
        media_out = preprocess_media(
            model_name_or_path=model_name_or_path,
            media_queue=media_queue,
            patch_size=patch_size,
        )
        pixel_values = media_out["pixel_values"]
        grid_thws = media_out["grid_thws"]
    else:
        pixel_values = None
        grid_thws = []

    # 4) Compute positions (text + visual)
    if media_queue:
        position_ids = compute_positions_1d(
            tokenizer=tokenizer,
            input_ids=input_ids,
            media_queue=media_queue,
            grid_thws=grid_thws,
            merge_size=merge_size,
        )
    else:
        # text-only; simple monotonic positions
        L = input_ids.shape[1]
        pos_1d = torch.arange(L, dtype=torch.long)
        position_ids = pos_1d.unsqueeze(0).repeat(3, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "position_ids": position_ids,
        "media_count": len(media_queue),
    }


# -----------------------------
# Example (manual test)
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MODEL = "xlangai/OpenCUA-7B"  # or local path

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지를 설명해주세요."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/368px-Google_2015_logo.svg.png"
                    },
                },
            ],
        }
    ]

    out = build_opencua_inputs(
        model_name_or_path=MODEL,
        messages=messages,
        patch_size=14,
        merge_size=2,
    )
    LOGGER.info(
        "Final tensors: input_ids=%s, position_ids=%s, pixel_values=%s, media=%d",
        tuple(out["input_ids"].shape),
        tuple(out["position_ids"].shape),
        None if out["pixel_values"] is None else tuple(out["pixel_values"].shape),
        out["media_count"],
    )
