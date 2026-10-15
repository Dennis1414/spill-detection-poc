"""Run the detector over a dataset or a folder of frames, save every result, report accuracy.

Usage:
    # Roboflow YOLO export (has data.yaml) -> accuracy against the labels
    python scripts/run_eval.py data/raw/spill_detection --split valid --limit 200

    # Any folder of images (your own frames) -> labels only, no accuracy
    python scripts/run_eval.py data/frames/balcony_test --out data/results/balcony

Outputs in --out (default data/results/<name>):
    results.jsonl   one line per image: path, ground truth, model output, usage
    results.csv     same, flat, for a spreadsheet
    summary.json    accuracy / precision / recall / F1 / confusion matrix / cost
    errors.txt      false positives and false negatives with the model's reasoning

The run is resumable: images already in results.jsonl are skipped, so you can
stop and restart, or raise --limit later.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spilldetect import MODEL_ID, SpillDetector, YoloDataset  # noqa: E402
from spilldetect.dataset import Sample, folder_samples  # noqa: E402


def load_done(jsonl: Path) -> dict[str, dict]:
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error"):
                    continue  # errored images are retried on the next run
                done[rec["path"]] = rec
    return done


def summarise(records: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    errors = 0
    unlabeled = 0
    for r in records:
        if r.get("error"):
            errors += 1
            continue
        gt = r.get("ground_truth")
        pred = r["result"]["spill_detected"]
        if gt is None:
            unlabeled += 1
            continue
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1
    labeled = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {
        "images": len(records),
        "labeled": labeled,
        "unlabeled": unlabeled,
        "api_errors": errors,
        "accuracy": (tp + tn) / labeled if labeled else None,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "predicted_positive": sum(1 for r in records if not r.get("error") and r["result"]["spill_detected"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YOLO dataset root (with data.yaml) or a folder of images")
    ap.add_argument("--split", default=None, help="train / valid / test (YOLO datasets only)")
    ap.add_argument("--spill-classes", nargs="*", default=None,
                    help="class names that count as a spill (default: any class containing 'spill' or 'liquid')")
    ap.add_argument("--limit", type=int, default=0, help="max images to process (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="random sample instead of the first N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID, help="e.g. claude-opus-5-5 (default), claude-haiku-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=4, help="parallel API requests")
    ap.add_argument("--out", default=None, help="output folder (default data/results/<source name>)")
    ap.add_argument("--dry-run", action="store_true", help="count images and estimate cost, no API calls")
    args = ap.parse_args()

    src = Path(args.source)
    if (src / "data.yaml").exists():
        ds = YoloDataset(src, args.spill_classes)
        print(f"dataset classes: {ds.class_names}; spill class ids: {sorted(ds.spill_class_ids)}")
        samples = list(ds.samples(args.split))
    else:
        samples = list(folder_samples(src))
    if not samples:
        sys.exit(f"no images found under {src}")

    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]

    out = Path(args.out) if args.out else Path("data/results") / (
        src.name + (f"_{args.split}" if args.split else "") + ("" if args.model == MODEL_ID else f"_{args.model}")
    )
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "results.jsonl"
    done = load_done(jsonl)
    todo = [s for s in samples if str(s.image) not in done]

    n_pos = sum(1 for s in samples if s.has_spill)
    n_neg = sum(1 for s in samples if s.has_spill is False)
    print(f"{len(samples)} images selected ({n_pos} spill / {n_neg} no-spill / "
          f"{len(samples) - n_pos - n_neg} unlabeled); {len(done)} already done, {len(todo)} to run")

    det = SpillDetector(model=args.model, effort=args.effort)
    if args.dry_run:
        from spilldetect.detector import PRICING
        p_in, p_out = PRICING.get(args.model, PRICING[MODEL_ID])[:2]
        est = sum(det.estimate_tokens(s.image) for s in todo[:50])
        per_img = est / max(1, min(50, len(todo)))
        per_cost = per_img * p_in / 1e6 + 150 * p_out / 1e6
        print(f"~{per_img:.0f} input tokens/image on {args.model} -> ~${per_cost:.4f}/image, "
              f"~${len(todo) * per_cost:.2f} for {len(todo)} images")
        return

    def work(s: Sample) -> dict:
        d = det.classify(s.image)
        return {
            "path": str(s.image),
            "split": s.split,
            "ground_truth": s.has_spill,
            "gt_boxes": s.spill_boxes,
            "result": d.result.model_dump() if d.result else None,
            "error": d.error,
            "raw_text": d.raw_text if d.error else "",
            "latency_s": round(d.latency_s, 2),
            "image_size": list(d.image_size),
            "served_by": d.served_by,
        }

    t0 = time.time()
    records = list(done.values())
    with jsonl.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gt = rec["ground_truth"]
            pred = rec["result"]["spill_detected"] if rec["result"] else "ERR"
            mark = "" if gt is None or pred == "ERR" else ("ok" if gt == pred else "MISS")
            print(f"[{i}/{len(todo)}] gt={gt} pred={pred} {mark} {Path(rec['path']).name}"
                  + (f"  ({rec['error']})" if rec["error"] else ""))

    summary = summarise(records)
    u = det.usage
    summary["this_run"] = {
        "images": len(todo),
        "seconds": round(time.time() - t0, 1),
        "requests": u.requests,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": u.cache_read_tokens,
        "est_cost_usd": round(u.cost_usd, 4),
        "effort": args.effort,
        "model": det.model,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "ground_truth", "spill_detected", "confidence", "spill_type", "size",
                    "bbox", "debris_present", "reasoning", "error", "latency_s"])
        for r in records:
            res = r["result"] or {}
            w.writerow([r["path"], r["split"], r["ground_truth"], res.get("spill_detected"), res.get("confidence"),
                        res.get("spill_type"), res.get("size"), res.get("bbox"), res.get("debris_present"),
                        res.get("reasoning"), r["error"], r["latency_s"]])

    with (out / "errors.txt").open("w", encoding="utf-8") as f:
        for r in records:
            if r["error"] or r["ground_truth"] is None:
                continue
            pred = r["result"]["spill_detected"]
            if pred != r["ground_truth"]:
                kind = "FALSE POSITIVE" if pred else "FALSE NEGATIVE"
                f.write(f"{kind}  conf={r['result']['confidence']:.2f}  {r['path']}\n   {r['result']['reasoning']}\n\n")

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/results.jsonl, results.csv, summary.json, errors.txt")


if __name__ == "__main__":
    main()
