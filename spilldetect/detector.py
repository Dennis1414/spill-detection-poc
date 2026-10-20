"""Spill classification with Claude Opus 5.5 vision.

One image in, one validated SpillResult out. Uses structured outputs so the
model can only answer in the schema, and the server-side refusal fallback so
a safety-classifier decline is retried on another model inside the same call.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from .images import encode_image, estimate_image_tokens

MODEL_ID = "claude-opus-5-5"

# Pricing for cost reporting only ($ per 1M tokens, Claude API first-party rates):
# (input, output, cache read, cache write)
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

SYSTEM_PROMPT = """You are the vision module of an autonomous floor-cleaning robot that operates in shopping malls, supermarkets, lobbies and similar indoor commercial spaces.

Your job: look at a single camera frame and decide whether there is a LIQUID SPILL on the floor that the robot should clean or flag.

What counts as a spill (positive):
- Puddles, drips, splashes or smears of any liquid on the floor: water, soft drinks, coffee, juice, milk, oil, sauces, cleaning fluid, melted ice cream, vomit, etc.
- Wet patches that are clearly from a liquid, including partially dried sticky residue with a visible outline.
- Spilled food that is liquid or semi-liquid (soup, sauce, yoghurt).

What does NOT count (negative):
- Reflections of lights, windows or ceiling on polished, glossy or wet-look flooring.
- Floor patterns, dark tiles, grout lines, stains that are part of the material, worn patches, scuff marks.
- Shadows, including hard shadows from people, carts or furniture.
- Dry debris: wrappers, paper, crumbs, dust, leaves, hair. Report these as debris, not as a spill.
- Rugs, mats, tape, floor signage, wet-floor caution signs (the sign alone is not a spill unless liquid is visible).
- Liquid that is NOT on the floor (in cups, bottles, fountains, aquariums, on tables).
- Freshly mopped floors that are uniformly damp with no distinct puddle.

Rules:
- Be precise, not eager. A false alarm sends the robot across the mall for nothing; a miss leaves a slip hazard. Weigh both.
- confidence is your probability (0 to 1) that spill_detected is correct.
- If spill_detected is false, set spill_type to "none", size to "none", and bbox to null.
- bbox is [x, y, width, height] as fractions of the image (0 to 1), covering the main spill area. Give one box for the largest spill.
- Keep reasoning to one or two short sentences describing the visual evidence."""

USER_PROMPT = "Analyse this floor camera frame for liquid spills and answer in the required JSON format."

PAIR_SYSTEM_ADDENDUM = """

Fixed-camera mode. You receive TWO images from the SAME fixed security camera: first the REFERENCE frame
(the floor as it looked earlier, known to be clean) and second the CURRENT frame. Both may be a crop of the
same region. Decide whether a liquid spill has APPEARED on the floor in the current frame that was not in the
reference. Anything already present in the reference (floor pattern, stains, reflections, fixed shadows,
signage, furniture) is not a spill. People, trolleys, bags and their moving shadows are changes but not spills.
A spill is a new patch on the floor with liquid characteristics: wet sheen, darkening, drips, splash shape.
bbox refers to the CURRENT frame (fractions of the image you were given).
When a third image is given it is the same camera a few seconds AFTER the current frame: a spill is still
there and unchanged in it, while a person's shadow, a trolley or a dropped-then-picked-up item has moved or gone."""

PAIR_USER_PROMPT = ("The first image is the clean reference frame, the second is the current frame from the "
                    "same camera. Has a liquid spill appeared on the floor? Answer in the required JSON format.")
TRIPLE_USER_PROMPT = ("The first image is the clean reference frame, the second is the current frame, the third is "
                      "the same camera a few seconds later. Has a liquid spill appeared on the floor in the current "
                      "frame? Use the third image to rule out shadows and moving objects. Answer in the required JSON format.")

SpillType = Literal["water", "oil", "food_or_drink", "other_liquid", "unknown", "none"]
SizeType = Literal["small", "medium", "large", "none"]


class SpillResult(BaseModel):
    spill_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    spill_type: SpillType
    size: SizeType
    bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    debris_present: bool = False
    floor_visible: bool = True
    reasoning: str


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "spill_detected": {"type": "boolean"},
        "confidence": {"type": "number"},
        "spill_type": {
            "type": "string",
            "enum": ["water", "oil", "food_or_drink", "other_liquid", "unknown", "none"],
        },
        "size": {"type": "string", "enum": ["small", "medium", "large", "none"]},
        # The API's structured-output grammar does not support minItems/maxItems > 1,
        # so the 4-element rule is enforced by SpillResult below instead.
        "bbox": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "null"},
            ]
        },
        "debris_present": {"type": "boolean"},
        "floor_visible": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "spill_detected", "confidence", "spill_type", "size",
        "bbox", "debris_present", "floor_visible", "reasoning",
    ],
    "additionalProperties": False,
}


@dataclass
class Usage:
    model: str = MODEL_ID
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, usage) -> None:
        self.requests += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        p_in, p_out, p_cr, p_cw = PRICING.get(self.model, PRICING[MODEL_ID])
        return (
            self.input_tokens * p_in
            + self.output_tokens * p_out
            + self.cache_read_tokens * p_cr
            + self.cache_write_tokens * p_cw
        ) / 1_000_000


@dataclass
class Detection:
    path: str
    result: Optional[SpillResult]
    error: Optional[str]
    latency_s: float
    image_size: tuple[int, int]
    served_by: str
    raw_text: str = ""


class SpillDetector:
    def __init__(
        self,
        model: str = MODEL_ID,
        effort: str = "low",
        client: Optional[anthropic.Anthropic] = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic(max_retries=max_retries)
        self.usage = Usage(model=model)

    def classify(self, image_path: str | Path, crop: tuple[float, float, float, float] | None = None) -> Detection:
        """Single-frame mode (robot camera)."""
        data, media_type, size = encode_image(image_path, crop)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
            {"type": "text", "text": USER_PROMPT},
        ]
        return self._ask(str(image_path), SYSTEM_PROMPT, content, size)

    def classify_pair(
        self,
        reference_path: str | Path,
        current_path: str | Path,
        crop: tuple[float, float, float, float] | None = None,
        later_path: str | Path | None = None,
    ) -> Detection:
        """Fixed-camera mode: clean reference + current frame (+ optional frame a few seconds later), same crop."""
        ref, mt1, _ = encode_image(reference_path, crop)
        cur, mt2, size = encode_image(current_path, crop)
        content = [
            {"type": "text", "text": "Reference frame (earlier, clean):"},
            {"type": "image", "source": {"type": "base64", "media_type": mt1, "data": ref}},
            {"type": "text", "text": "Current frame:"},
            {"type": "image", "source": {"type": "base64", "media_type": mt2, "data": cur}},
        ]
        if later_path is not None:
            lat, mt3, _ = encode_image(later_path, crop)
            content += [
                {"type": "text", "text": "Same camera a few seconds later:"},
                {"type": "image", "source": {"type": "base64", "media_type": mt3, "data": lat}},
                {"type": "text", "text": TRIPLE_USER_PROMPT},
            ]
        else:
            content.append({"type": "text", "text": PAIR_USER_PROMPT})
        return self._ask(str(current_path), SYSTEM_PROMPT + PAIR_SYSTEM_ADDENDUM, content, size)

    def _ask(self, path: str, system: str, content: list, size: tuple[int, int]) -> Detection:
        image_path = path
        t0 = time.perf_counter()
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as e:
            return Detection(str(image_path), None, f"{type(e).__name__}: {e}", time.perf_counter() - t0, size, "")

        latency = time.perf_counter() - t0
        self.usage.add(response.usage)
        served_by = getattr(response, "model", self.model)

        if response.stop_reason == "refusal":
            cat = getattr(getattr(response, "stop_details", None), "category", None)
            return Detection(str(image_path), None, f"refusal ({cat})", latency, size, served_by)
        if response.stop_reason == "max_tokens":
            return Detection(str(image_path), None, "max_tokens hit", latency, size, served_by)

        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            result = SpillResult.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            return Detection(str(image_path), None, f"bad output: {e}", latency, size, served_by, text)
        return Detection(str(image_path), result, None, latency, size, served_by, text)

    def estimate_tokens(self, image_path: str | Path) -> int:
        _, _, size = encode_image(image_path)
        return estimate_image_tokens(size) + 600
