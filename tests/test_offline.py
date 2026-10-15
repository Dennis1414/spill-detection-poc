"""Offline checks: no API key needed. Run with: python -m pytest tests -q  (or python tests/test_offline.py)"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from spilldetect import SpillDetector, SpillResult, YoloDataset, encode_image  # noqa: E402
from spilldetect.images import load_resized  # noqa: E402


def make_floor_image(path: Path, w: int, h: int, puddle: bool) -> None:
    img = Image.new("RGB", (w, h), (205, 200, 190))
    px = img.load()
    for y in range(0, h, 64):
        for x in range(w):
            px[x, y] = (170, 165, 155)
    if puddle:
        cx, cy, r = w // 2, h // 2, min(w, h) // 6
        for y in range(cy - r, cy + r):
            for x in range(cx - r, cx + r):
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    px[x, y] = (120, 110, 90)
    img.save(path)


def test_resize_caps_pixels():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "big.png"
        make_floor_image(p, 4000, 3000, True)
        img = load_resized(p)
        assert max(img.size) <= 1568
        assert img.size[0] * img.size[1] <= 1_150_000
        data, mt, size = encode_image(p)
        assert mt == "image/jpeg" and len(data) > 1000 and size == img.size


def test_yolo_dataset_ground_truth():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "data.yaml").write_text("names: ['floor', 'spill']\nnc: 2\n")
        for split in ("train", "valid"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        make_floor_image(root / "train/images/a.jpg", 320, 240, True)
        (root / "train/labels/a.txt").write_text("0 0.5 0.5 1 1\n1 0.5 0.5 0.3 0.3\n")
        make_floor_image(root / "train/images/b.jpg", 320, 240, False)
        (root / "train/labels/b.txt").write_text("0 0.5 0.5 1 1\n")
        make_floor_image(root / "valid/images/c.jpg", 320, 240, False)  # no label file

        ds = YoloDataset(root)
        assert ds.spill_class_ids == {1}
        s = {x.image.name: x for x in ds.samples()}
        assert s["a.jpg"].has_spill is True and s["a.jpg"].spill_boxes == [[0.5, 0.5, 0.3, 0.3]]
        assert s["b.jpg"].has_spill is False
        assert s["c.jpg"].has_spill is None
        assert [x.image.name for x in ds.samples("valid")] == ["c.jpg"]


class _FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=None,
            model="claude-opus-5-5",
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=90,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def _fake_client(text, stop_reason="end_turn"):
    m = _FakeMessages(text, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=m)), m


def test_detector_parses_structured_output():
    payload = {"spill_detected": True, "confidence": 0.91, "spill_type": "water", "size": "medium",
               "bbox": [0.3, 0.4, 0.3, 0.25], "debris_present": False, "floor_visible": True,
               "reasoning": "Dark irregular wet patch with a defined edge on the tile."}
    client, m = _fake_client(json.dumps(payload))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jpg"
        make_floor_image(p, 640, 480, True)
        det = SpillDetector(client=client)
        d = det.classify(p)
    assert d.error is None and isinstance(d.result, SpillResult)
    assert d.result.spill_detected and d.result.bbox == [0.3, 0.4, 0.3, 0.25]
    assert det.usage.requests == 1 and det.usage.cost_usd > 0
    kw = m.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["messages"][0]["content"][0]["type"] == "image"


def test_detector_handles_refusal_and_bad_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jpg"
        make_floor_image(p, 320, 240, False)
        client, _ = _fake_client("", stop_reason="refusal")
        assert SpillDetector(client=client).classify(p).error.startswith("refusal")
        client, _ = _fake_client("{not json")
        assert SpillDetector(client=client).classify(p).error.startswith("bad output")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
