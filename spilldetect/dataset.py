"""Loader for YOLO-format datasets (what Roboflow Universe exports as "YOLOv8").

Layout:
    root/
      data.yaml            # names: [...]  (class names)
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt
      test/images/*.jpg    test/labels/*.txt

Each label line: <class_id> <cx> <cy> <w> <h>   (normalised 0..1)
We reduce this to a per-image boolean: does the image contain at least one
box whose class is a spill class? That is the ground truth for the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image: Path
    has_spill: Optional[bool]       # None when no label file exists
    spill_boxes: list[list[float]]  # [cx, cy, w, h] normalised
    split: str


class YoloDataset:
    def __init__(self, root: str | Path, spill_classes: Optional[list[str]] = None):
        self.root = Path(root)
        cfg_path = self.root / "data.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no data.yaml in {self.root}")
        cfg = yaml.safe_load(cfg_path.read_text())
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        self.class_names: list[str] = list(names or [])
        if spill_classes is None:
            keywords = ("spill", "liquid", "water", "puddle", "oil", "wet")
            spill_classes = [n for n in self.class_names if any(k in n.lower() for k in keywords)]
        self.spill_class_ids = {i for i, n in enumerate(self.class_names) if n in spill_classes}
        if not self.spill_class_ids:
            raise ValueError(
                f"none of {spill_classes} found in dataset classes {self.class_names}; "
                "pass spill_classes explicitly"
            )

    def splits(self) -> list[str]:
        return [s for s in ("train", "valid", "test") if (self.root / s / "images").is_dir()]

    def samples(self, split: Optional[str] = None) -> Iterator[Sample]:
        for s in ([split] if split else self.splits()):
            img_dir = self.root / s / "images"
            lbl_dir = self.root / s / "labels"
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    yield Sample(img, None, [], s)
                    continue
                boxes = []
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    if int(parts[0]) in self.spill_class_ids:
                        boxes.append([float(p) for p in parts[1:5]])
                yield Sample(img, bool(boxes), boxes, s)


def folder_samples(folder: str | Path) -> Iterator[Sample]:
    """Unlabelled images in a plain folder (e.g. frames you shot yourself)."""
    folder = Path(folder)
    for img in sorted(folder.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            yield Sample(img, None, [], folder.name)
