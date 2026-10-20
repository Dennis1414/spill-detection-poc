"""Evaluate the fixed-camera pipeline on before/after pairs.

    python scripts/run_cctv_eval.py                    # candidates + crops (the real pipeline)
    python scripts/run_cctv_eval.py --mode full --limit 40   # whole frame, no crops, for comparison

Per pair:
  1. propose_candidates(reference, current) -> up to N changed regions (normalised boxes)
  2. for each candidate, classify_pair(ref, cur, crop) -> SpillResult
  3. pair verdict = any candidate says spill; bbox mapped back to the full frame
A pair with no candidates is "clear" at zero cost. Results: data/results/cctv_<mode>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spilldetect import MODEL_ID, SpillDetector  # noqa: E402
from spilldetect.cctv import box_to_frame, contains, propose_candidates  # noqa: E402


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/cctv_pairs/pairs.jsonl")
    ap.add_argument("--mode", choices=["crops", "full"], default="crops")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--no-persistence", action="store_true", help="ignore the 'later' frame (no persistence filter)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        pairs = pairs[: args.limit]
    out = Path(args.out or f"data/results/cctv_{args.mode}" + ("_nopersist" if args.no_persistence else "")
               + ("" if args.model == MODEL_ID else f"_{args.model}"))
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "results.jsonl"
    done = {}
    if jsonl.exists():
        for l in jsonl.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if not r.get("error"):
                    done[r["id"]] = r
    todo = [p for p in pairs if p["id"] not in done]
    print(f"{len(pairs)} pairs, {len(done)} done, {len(todo)} to run, mode={args.mode}, model={args.model}")

    det = SpillDetector(model=args.model, effort=args.effort)

    def work(p: dict) -> dict:
        rec = dict(p)
        rec.update({"mode": args.mode, "candidates": [], "calls": [], "pred": False, "pred_bbox": None,
                    "confidence": 0.0, "reasoning": "", "error": None, "latency_s": 0.0})
        t0 = time.perf_counter()
        if args.mode == "full":
            crops = [None]
        else:
            ref, cur = cv2.imread(p["ref"]), cv2.imread(p["cur"])
            later = cv2.imread(p["later"]) if p.get("later") and not args.no_persistence else None
            crops = propose_candidates(ref, cur, later, max_candidates=args.max_candidates)
            rec["candidates"] = crops
            if not crops:
                rec["reasoning"] = "no change detected against the reference; not sent to the model"
                rec["latency_s"] = round(time.perf_counter() - t0, 2)
                return rec
        best = None
        for crop in crops:
            d = det.classify_pair(p["ref"], p["cur"], crop=crop)
            if d.error:
                rec["calls"].append({"crop": crop, "error": d.error})
                rec["error"] = d.error
                continue
            x = d.result
            fb = box_to_frame(x.bbox, crop) if (x.bbox and crop) else x.bbox
            rec["calls"].append({"crop": crop, "spill": x.spill_detected, "confidence": x.confidence,
                                 "bbox_frame": fb, "reasoning": x.reasoning, "type": x.spill_type})
            if x.spill_detected and (best is None or x.confidence > best["confidence"]):
                best = rec["calls"][-1]
        if best:
            rec.update({"pred": True, "pred_bbox": best["bbox_frame"], "confidence": best["confidence"],
                        "reasoning": best["reasoning"]})
        elif rec["calls"]:
            top = max((c for c in rec["calls"] if "spill" in c), key=lambda c: c["confidence"], default=None)
            if top:
                rec.update({"confidence": top["confidence"], "reasoning": top["reasoning"]})
        if rec["calls"] and all("error" in c for c in rec["calls"]):
            pass
        else:
            rec["error"] = None
        rec["latency_s"] = round(time.perf_counter() - t0, 2)
        return rec

    records = list(done.values())
    with jsonl.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            records.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            mark = "ok" if r["pred"] == r["has_spill"] else "MISS"
            print(f"[{i}/{len(todo)}] gt={r['has_spill']!s:5} pred={r['pred']!s:5} cands={len(r['candidates'])} "
                  f"calls={len(r['calls'])} {mark} {r['id']}" + (f" ({r['error']})" if r["error"] else ""))

    # summary
    tp = sum(1 for r in records if r["has_spill"] and r["pred"])
    fn = sum(1 for r in records if r["has_spill"] and not r["pred"])
    fp = sum(1 for r in records if not r["has_spill"] and r["pred"])
    tn = sum(1 for r in records if not r["has_spill"] and not r["pred"])
    located = sum(1 for r in records if r["has_spill"] and r["pred"] and r["pred_bbox"] and r["gt_bbox"]
                  and (iou(r["pred_bbox"], r["gt_bbox"]) > 0.1 or contains(tuple(r["pred_bbox"]),
                       (r["gt_bbox"][0] + r["gt_bbox"][2] / 2, r["gt_bbox"][1] + r["gt_bbox"][3] / 2))))
    cand_hit = sum(1 for r in records if r["has_spill"] and r["mode"] == "crops" and any(
        contains(tuple(c), (r["gt_bbox"][0] + r["gt_bbox"][2] / 2, r["gt_bbox"][1] + r["gt_bbox"][3] / 2))
        for c in r["candidates"]))
    n_pos = tp + fn
    calls = sum(len(r["calls"]) for r in records)
    skipped = sum(1 for r in records if r["mode"] == "crops" and not r["candidates"])
    u = det.usage
    summary = {
        "pairs": len(records), "mode": args.mode, "model": det.model,
        "accuracy": (tp + tn) / len(records) if records else None,
        "recall": tp / n_pos if n_pos else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "spill_located_correctly": f"{located} of {tp}",
        "candidate_finder_covered_spill": f"{cand_hit} of {n_pos}" if args.mode == "crops" else None,
        "pairs_skipped_no_change": skipped,
        "model_calls": calls,
        "this_run": {"pairs": len(todo), "requests": u.requests, "input_tokens": u.input_tokens,
                     "output_tokens": u.output_tokens, "cache_read_tokens": u.cache_read_tokens,
                     "est_cost_usd": round(u.cost_usd, 4), "effort": args.effort},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "errors.txt").open("w", encoding="utf-8") as f:
        for r in records:
            if r["pred"] != r["has_spill"]:
                f.write(f"{'FALSE POSITIVE' if r['pred'] else 'FALSE NEGATIVE'}  conf={r['confidence']:.2f}  "
                        f"{r['id']}  kind={r.get('kind')} width={r.get('spill_width_frac')}\n   {r['reasoning']}\n\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
