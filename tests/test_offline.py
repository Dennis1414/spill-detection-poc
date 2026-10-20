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


def test_pair_mode_sends_two_images_and_crop_keeps_resolution():
    payload = {"spill_detected": True, "confidence": 0.9, "spill_type": "water", "size": "small",
               "bbox": [0.4, 0.4, 0.2, 0.2], "debris_present": False, "floor_visible": True, "reasoning": "x"}
    client, m = _fake_client(json.dumps(payload))
    with tempfile.TemporaryDirectory() as d:
        ref, cur = Path(d) / "ref.jpg", Path(d) / "cur.jpg"
        make_floor_image(ref, 1920, 1080, False)
        make_floor_image(cur, 1920, 1080, True)
        det = SpillDetector(client=client)
        r = det.classify_pair(ref, cur, crop=(0.25, 0.25, 0.25, 0.25))
    assert r.error is None and r.result.spill_detected
    content = m.calls[0]["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 2
    assert "Fixed-camera mode" in m.calls[0]["system"][0]["text"]
    assert r.image_size == (480, 270)  # 25% crop of 1920x1080 kept at native resolution


def test_candidate_finder_and_synth_spill():
    from spilldetect.cctv import propose_candidates, box_to_frame
    from spilldetect.synth import composite_spill
    rng = np.random.default_rng(1)
    ref = np.full((540, 960, 3), 190, np.uint8) + rng.integers(0, 6, (540, 960, 3), np.uint8)
    cur, gt = composite_spill(ref, (0.3, 0.75), 0.08, "coffee", seed=3)
    assert 0.05 < gt[2] < 0.2 and 0.6 < gt[1] < 0.9
    cands = propose_candidates(ref, cur)
    assert cands, "the composited spill should register as a change"
    cx, cy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
    assert any(c[0] <= cx <= c[0] + c[2] and c[1] <= cy <= c[1] + c[3] for c in cands)
    assert propose_candidates(ref, ref) == []
    fb = box_to_frame([0.5, 0.5, 0.2, 0.2], (0.1, 0.2, 0.4, 0.4))
    assert [round(v, 3) for v in fb] == [0.3, 0.4, 0.08, 0.08]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
