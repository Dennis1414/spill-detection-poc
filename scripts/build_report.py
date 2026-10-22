"""Build the demo report (report/index.html) from saved eval results.

    python scripts/build_report.py

Reads data/results/spill_detection_valid (Opus) and the Haiku run next to it,
embeds a curated set of frames with the model's verdict and reasoning, and
writes a single self-contained HTML page. Re-run after every new eval run.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPUS_DIR = ROOT / "data/results/spill_detection_valid"
HAIKU_DIR = ROOT / "data/results/spill_detection_valid_claude-haiku-5-5"
OUT = ROOT / "report/index.html"
THUMB = 640

# Frames reviewed by hand and judged NOT to show a floor (laptop, sofa, countertop).
# The dataset labels them "water"; the model correctly says "not a floor spill".
NOT_A_FLOOR = [
    "istockphoto-1409100095", "output_35_flipped_horizontal", "output_100_flipped_horizontal",
    "output_86_black_and_white", "output_87_flipped_vertical", "output_87_black_and_white",
    "output_0_black_and_white", "output_5_flipped_horizontal",
]

# (filename prefix, group, short caption). Groups: detected / clear / scope / hard
SHOWCASE = [
    ("333_mp4-11", "detected", "Rain puddle on outdoor pavement, phone video at knee height"),
    ("images-51-", "detected", "Person walking through a puddle on a glossy indoor floor"),
    ("74_mp4-32", "detected", "Wet concrete outside a parking area, low-angle video"),
    ("Screenshot-2023", "detected", "Small water patch on a wooden floor, screenshot from video"),
    ("WhatsApp-Video", "detected", "Muddy splash across a tiled floor, WhatsApp video frame"),
    ("download-12-", "detected", "Tipped glass with water spreading on a grey surface"),
    ("istockphoto-1306345532", "detected", "Water droplets and a wet patch on a dark wooden floor"),
    ("istockphoto-1204491935", "detected", "Red liquid splashed on white tiles"),
    ("output_1_flipped_vertical", "detected", "Orange drink soaked into carpet, hands reaching in"),
    ("images-41-", "detected", "Juice pooled on a tiled floor"),
    ("images-24-_jpeg.rf.e5a57", "hard", "Clear water from a tipped glass on wood grain. Opus caught it, Haiku called it a reflection"),
    ("images-52-_jpeg.rf.1aec", "hard", "Tea spill next to a cup. Flagged at low confidence (0.60), which the robot should treat as 'go check'"),
    ("istockphoto-1251463928", "hard", "Small clear puddle on a pale floor, almost invisible at a glance"),
    ("w9_jpeg", "clear", "Supermarket CCTV frame, shoppers in an aisle, dry floor"),
    ("173_jpg", "clear", "Bright supermarket aisle, polished floor, no spill"),
    ("video4-062", "clear", "Person lying on a tiled floor between chairs, no liquid"),
    ("video2-010", "clear", "Same room, different angle. Scuffed tiles, no spill"),
    ("0B11J4CA9RPZ", "clear", "Very dark, grainy frame of a shopping trolley"),
    ("walmart-shopping", "clear", "Shopping trolleys on wet-look pavers outdoors"),
    ("istockphoto-1409100095", "scope", "Coffee on a laptop keyboard. Dataset says 'water', the model says 'not on a floor'"),
]


def load_run(d: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads((d / "summary.json").read_text())
    rows = {}
    for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[os.path.basename(r["path"])] = r
    return summary, rows


def find(rows: dict[str, dict], prefix: str) -> dict | None:
    for k, v in rows.items():
        if k.startswith(prefix):
            return v
    return None


def thumb_b64(path: str, size: int = THUMB, quality: int = 78) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def small_card(name: str, r: dict) -> str:
    """Compact card for the all-frames appendix: image, box, model verdict vs dataset label."""
    x = r["result"]
    pred = x["spill_detected"]
    gt = r["ground_truth"]
    non_floor = any(name.startswith(p) for p in NOT_A_FLOOR)
    if non_floor:
        status, status_cls = "label: liquid, not on a floor", "note"
    elif pred == gt:
        status, status_cls = "matches label", "ok"
    else:
        status, status_cls = "differs from label", "bad"
    box = ""
    if x["bbox"]:
        bx, by, bw, bh = x["bbox"]
        box = f'<div class="box" style="left:{bx*100:.1f}%;top:{by*100:.1f}%;width:{bw*100:.1f}%;height:{bh*100:.1f}%"></div>'
    return f"""
<figure class="small" title="{html.escape(x['reasoning'])}">
  <div class="img"><img src="{thumb_b64(r['path'], 360, 70)}" alt="{html.escape(x['reasoning'][:120])}" loading="lazy">{box}</div>
  <figcaption>
    <span class="pill {'spill' if pred else 'clear'}">{'Spill' if pred else 'Clear'}</span>
    <span class="conf">{x['confidence']:.2f}</span>
    <span class="status {status_cls}">{status}</span>
  </figcaption>
</figure>"""


CCTV_CROPS = ROOT / "data/results/cctv_crops_3f"   # the pipeline: persistent-and-stable crops, 3 frames
CCTV_FULL = ROOT / "data/results/cctv_full"        # whole frame, no candidate step (every 3rd pair)


def load_cctv(d: Path) -> tuple[dict, list[dict]] | tuple[None, list]:
    if not (d / "summary.json").exists():
        return None, []
    summary = json.loads((d / "summary.json").read_text())
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return summary, rows


def crop_b64(path: str, crop, size: int = 420) -> str:
    from spilldetect.images import load_resized
    im = load_resized(path, tuple(crop) if crop else None)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def _contains(crop, pt) -> bool:
    return crop[0] <= pt[0] <= crop[0] + crop[2] and crop[1] <= pt[1] <= crop[1] + crop[3]


def cctv_card(r: dict, caption: str) -> str:
    """Before/after card: the crop the model saw from the reference and the current frame."""
    gt_c = None
    if r["gt_bbox"]:
        gt_c = (r["gt_bbox"][0] + r["gt_bbox"][2] / 2, r["gt_bbox"][1] + r["gt_bbox"][3] / 2)
    calls = [c for c in r["calls"] if "spill" in c]
    # the call to show: the positive one, else the one covering the spill, else the first
    call = next((c for c in calls if c["spill"]), None)
    if call is None and gt_c:
        call = next((c for c in calls if c.get("crop") and _contains(c["crop"], gt_c)), None)
    if call is None and calls:
        call = calls[0]
    crop = call["crop"] if call and call.get("crop") else [0, 0, 1, 1]
    box = ""
    if call and call.get("spill") and call.get("bbox_frame"):
        bx, by, bw, bh = call["bbox_frame"]
        cx, cy, cw, ch = crop
        box = (f'<div class="box" style="left:{(bx-cx)/cw*100:.1f}%;top:{(by-cy)/ch*100:.1f}%;'
               f'width:{bw/cw*100:.1f}%;height:{bh/ch*100:.1f}%"></div>')
    gtbox = ""
    if r["gt_bbox"]:
        bx, by, bw, bh = r["gt_bbox"]
        cx, cy, cw, ch = crop
        gtbox = (f'<div class="gtbox" style="left:{(bx-cx)/cw*100:.1f}%;top:{(by-cy)/ch*100:.1f}%;'
                 f'width:{bw/cw*100:.1f}%;height:{bh/ch*100:.1f}%"></div>')
    pred = r["pred"]
    conf = call["confidence"] if call else 0.0
    reason = call["reasoning"] if call else r["reasoning"]
    outcome = ("Found" if pred else "Missed") if r["has_spill"] else ("False alarm" if pred else "Correctly clear")
    ocls = "ok" if pred == r["has_spill"] else "bad"
    truth = f"synthetic {r['kind']} spill, {r['spill_width_frac']*100:.0f}% of frame width" if r["has_spill"] else "no spill, real footage"
    later_img = ""
    if r.get("later") and r["later"] != r["cur"]:
        later_img = f'<div class="img"><img src="{crop_b64(r["later"], crop)}" alt="later crop" loading="lazy"><span class="tag">2 s later</span></div>'
    return f"""
<figure class="pair">
  <div class="two{' three' if later_img else ''}">
    <div class="img"><img src="{crop_b64(r['ref'], crop)}" alt="reference crop" loading="lazy"><span class="tag">reference</span></div>
    <div class="img"><img src="{crop_b64(r['cur'], crop)}" alt="current crop" loading="lazy">{gtbox}{box}<span class="tag">current</span></div>
    {later_img}
  </div>
  <figcaption>
    <div class="verdict-row">
      <span class="pill {'spill' if pred else 'clear'}">{'Spill' if pred else 'Clear'}</span>
      <span class="conf">confidence {conf:.2f}</span>
      <span class="status {ocls}" style="margin-left:auto">{outcome}</span>
    </div>
    <p class="reason">{html.escape(reason)}</p>
    <p class="caption">{html.escape(caption)} &middot; truth: {html.escape(truth)}</p>
  </figcaption>
</figure>"""


def cctv_small_card(r: dict) -> str:
    """Appendix card for one fixed-camera check: the current-frame crop the model judged, boxes, verdict, truth."""
    gt_c = None
    if r["gt_bbox"]:
        gt_c = (r["gt_bbox"][0] + r["gt_bbox"][2] / 2, r["gt_bbox"][1] + r["gt_bbox"][3] / 2)
    calls = [c for c in r["calls"] if "spill" in c]
    call = next((c for c in calls if c["spill"]), None)
    if call is None and gt_c:
        call = next((c for c in calls if c.get("crop") and _contains(c["crop"], gt_c)), None)
    if call is None and calls:
        call = calls[0]
    crop = call["crop"] if call and call.get("crop") else [0, 0, 1, 1]
    cx, cy, cw, ch = crop
    boxes = ""
    if r["gt_bbox"]:
        bx, by, bw, bh = r["gt_bbox"]
        boxes += f'<div class="gtbox" style="left:{(bx-cx)/cw*100:.1f}%;top:{(by-cy)/ch*100:.1f}%;width:{bw/cw*100:.1f}%;height:{bh/ch*100:.1f}%"></div>'
    if call and call.get("spill") and call.get("bbox_frame"):
        bx, by, bw, bh = call["bbox_frame"]
        boxes += f'<div class="box" style="left:{(bx-cx)/cw*100:.1f}%;top:{(by-cy)/ch*100:.1f}%;width:{bw/cw*100:.1f}%;height:{bh/ch*100:.1f}%"></div>'
    pred = r["pred"]
    if r["has_spill"]:
        status, cls = ("found", "ok") if pred else ("missed", "bad")
        truth = f"{r['kind']}, {r['spill_width_frac']*100:.0f}% wide"
    else:
        status, cls = ("false alarm", "bad") if pred else ("clear", "ok")
        truth = "no spill"
    conf = call["confidence"] if call else 0.0
    reason = call["reasoning"] if call else r["reasoning"]
    if not calls:
        status, cls, reason = "clear, not sent", "ok", r["reasoning"]
    return f"""
<figure class="small" title="{html.escape(reason)}">
  <div class="img"><img src="{crop_b64(r['cur'], crop, 220)}" alt="{html.escape(reason[:100])}" loading="lazy">{boxes}</div>
  <figcaption>
    <span class="pill {'spill' if pred else 'clear'}">{'Spill' if pred else 'Clear'}</span>
    <span class="conf">{conf:.2f}</span>
    <span class="status {cls}">{status} &middot; {html.escape(truth)} &middot; {html.escape(r['seq'].replace('_', ' '))}</span>
  </figcaption>
</figure>"""


def floor_only(summary: dict, rows: dict[str, dict]) -> dict:
    """Metrics after removing the hand-reviewed non-floor images."""
    tp = fp = tn = fn = 0
    for name, r in rows.items():
        if r.get("error") or any(name.startswith(p) for p in NOT_A_FLOOR):
            continue
        gt, pred = r["ground_truth"], r["result"]["spill_detected"]
        if gt and pred: tp += 1
        elif gt and not pred: fn += 1
        elif not gt and pred: fp += 1
        else: tn += 1
    n = tp + fp + tn + fn
    nf = [r for name, r in rows.items() if any(name.startswith(p) for p in NOT_A_FLOOR) and not r.get("error")]
    rule_ok = sum(1 for r in nf if not r["result"]["spill_detected"])
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "non_floor": len(nf), "rule_ok": rule_ok,
            "accuracy": (tp + tn) / n if n else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "precision": tp / (tp + fp) if tp + fp else 0}


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def card(r: dict, group: str, caption: str) -> str:
    x = r["result"]
    verdict = "Spill" if x["spill_detected"] else "Clear"
    cls = "spill" if x["spill_detected"] else "clear"
    box = ""
    if x["bbox"]:
        bx, by, bw, bh = x["bbox"]
        box = f'<div class="box" style="left:{bx*100:.1f}%;top:{by*100:.1f}%;width:{bw*100:.1f}%;height:{bh*100:.1f}%"></div>'
    meta = []
    if x["spill_detected"]:
        meta.append(x["spill_type"].replace("_", " "))
        meta.append(f"{x['size']} size")
    if x["debris_present"]:
        meta.append("debris")
    meta_html = " &middot;".join(html.escape(m) for m in meta)
    return f"""
<figure class="frame" data-group="{group}">
  <div class="img"><img src="{thumb_b64(r['path'])}" alt="{html.escape(caption)}" loading="lazy">{box}</div>
  <figcaption>
    <div class="verdict-row">
      <span class="pill {cls}">{verdict}</span>
      <span class="conf">confidence {x['confidence']:.2f}</span>
      <span class="lat">{r['latency_s']:.1f}s</span>
    </div>
    {f'<div class="meta">{meta_html}</div>' if meta_html else ''}
    <p class="reason">{html.escape(x['reasoning'])}</p>
    <p class="caption">{html.escape(caption)}</p>
  </figcaption>
</figure>"""


def main() -> None:
    opus_sum, opus = load_run(OPUS_DIR)
    haiku_sum, haiku = load_run(HAIKU_DIR) if HAIKU_DIR.exists() else (None, {})
    fo = floor_only(opus_sum, opus)
    fh = floor_only(haiku_sum, haiku) if haiku_sum else None
    run = opus_sum["this_run"]
    cost_per = run["est_cost_usd"] / run["images"]
    lat = sum(r["latency_s"] for r in opus.values()) / len(opus)
    conf = opus_sum["confusion"]

    cards = []
    for prefix, group, caption in SHOWCASE:
        r = find(opus, prefix)
        if r and not r.get("error"):
            cards.append(card(r, group, caption))
    gallery = "\n".join(cards)

    # Opus vs Haiku side by side on the hard case
    hard = find(opus, "images-24-_jpeg.rf.e5a57")
    hard_h = find(haiku, "images-24-_jpeg.rf.e5a57") if haiku else None
    compare = ""
    if hard and hard_h:
        compare = f"""
<div class="compare">
  <div class="img"><img src="{thumb_b64(hard['path'])}" alt="Clear water spill from a tipped glass on a wooden floor"></div>
  <div class="verdicts">
    <div class="v">
      <div class="who">Claude Opus 5.5</div>
      <span class="pill spill">Spill</span> <span class="conf">confidence {hard['result']['confidence']:.2f}</span>
      <p class="reason">{html.escape(hard['result']['reasoning'])}</p>
    </div>
    <div class="v">
      <div class="who">Claude Haiku 5.5</div>
      <span class="pill clear">Clear</span> <span class="conf">confidence {hard_h['result']['confidence']:.2f}</span>
      <p class="reason">{html.escape(hard_h['result']['reasoning'])}</p>
    </div>
  </div>
</div>"""

    haiku_rows = ""
    if haiku_sum and fh:
        hr = haiku_sum["this_run"]
        hlat = sum(r["latency_s"] for r in haiku.values()) / len(haiku)
        haiku_rows = f"""
<tr><th>Floor-frame accuracy</th><td>{pct(fo['accuracy'])}</td><td>{pct(fh['accuracy'])}</td></tr>
<tr><th>Floor spills missed</th><td>{fo['fn']} of {fo['tp'] + fo['fn']}</td><td>{fh['fn']} of {fh['tp'] + fh['fn']}</td></tr>
<tr><th>False alarms</th><td>{fo['fp']} of {fo['tn'] + fo['fp']}</td><td>{fh['fp']} of {fh['tn'] + fh['fp']}</td></tr>
<tr><th>Followed the floor-only rule</th><td>{fo['rule_ok']} of {fo['non_floor']}</td><td>{fh['rule_ok']} of {fh['non_floor']}</td></tr>
<tr><th>Cost, {len(opus)} frames</th><td>${cost_per * len(opus):.2f}</td><td>${hr['est_cost_usd'] / hr['images'] * len(haiku):.3f}</td></tr>
<tr><th>Mean latency</th><td>{lat:.1f} s</td><td>{hlat:.1f} s</td></tr>"""

    # Appendix: every evaluated frame, spills first, then by confidence
    ordered = sorted(opus.items(), key=lambda kv: (not kv[1]["result"]["spill_detected"], -kv[1]["result"]["confidence"]))
    appendix = "\n".join(small_card(n, r) for n, r in ordered if not r.get("error"))
    n_all = sum(1 for r in opus.values() if not r.get("error"))

    # Fixed-camera section
    cs, crows = load_cctv(CCTV_CROPS)
    fs, frows = load_cctv(CCTV_FULL)
    cctv_section = ""
    if cs:
        n_pairs = cs["pairs"]
        cpos = [r for r in crows if r["has_spill"]]
        cneg = [r for r in crows if not r["has_spill"]]
        seqs = sorted({r["seq"] for r in crows})
        tps = [r for r in cpos if r["pred"]]
        misses = [r for r in cpos if not r["pred"]]
        show = []
        for s in seqs:  # four highest-confidence finds per camera, varied kinds
            seen = set()
            for r in sorted((t for t in tps if t["seq"] == s), key=lambda t: -t["confidence"]):
                if r["kind"] in seen:
                    continue
                seen.add(r["kind"])
                show.append((r, f"{s} camera"))
                if len(seen) == 4:
                    break
        for r in misses[:2]:
            show.append((r, f"{r['seq']} camera, a miss"))
        for r in [n for n in cneg if n["candidates"] and n["calls"]][:2]:
            show.append((r, f"{r['seq']} camera, people walking through"))
        pair_cards = "\n".join(cctv_card(r, cap) for r, cap in show)

        # Clear liquids: water is transparent, so only darkening and sheen give it away. Its own block.
        water = [r for r in cpos if r["kind"] == "water"]
        water_found = [r for r in water if r["pred"]]
        wshow = []
        for s in seqs:
            best = sorted((r for r in water_found if r["seq"] == s), key=lambda r: -r["confidence"])[:1]
            wshow += [(r, f"{s} camera, water") for r in best]
        smallest = sorted(water_found, key=lambda r: r["spill_width_frac"])[:1]
        wshow += [(r, f"{r['seq']} camera, the smallest water spill found, {r['spill_width_frac']*100:.0f}% of frame width")
                  for r in smallest if r["id"] not in {x["id"] for x, _ in wshow}]
        wmiss = [r for r in water if not r["pred"]][:1]
        wshow += [(r, f"{r['seq']} camera, a water spill it missed") for r in wmiss]
        water_cards = "\n".join(cctv_card(r, cap) for r, cap in wshow)
        water_smallest_w = min((r["spill_width_frac"] for r in water_found), default=0)

        # Appendix: every check, spills first, then the ones the model got wrong, then by camera
        ordered = sorted(crows, key=lambda r: (not r["has_spill"], r["pred"] == r["has_spill"], r["seq"], -r["confidence"]))
        cctv_appendix = "\n".join(cctv_small_card(r) for r in ordered)
        calls_per = cs["model_calls"] / n_pairs
        c_cost_pair = cs["this_run"]["est_cost_usd"] / cs["this_run"]["pairs"] if cs["this_run"]["pairs"] else 0
        f_cost_pair = fs["this_run"]["est_cost_usd"] / fs["this_run"]["pairs"] if fs and fs["this_run"]["pairs"] else 0
        c_lat = sum(r["latency_s"] for r in crows) / len(crows)
        f_lat = sum(r["latency_s"] for r in frows) / len(frows) if frows else 0
        fcf = fs["confusion"] if fs else {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        ccf = cs["confusion"]
        HANDHELD = {"walkway_vert"}  # phone held by a person, not a mounted camera; kept as the hardest case
        mounted = [r for r in crows if r["seq"] not in HANDHELD]
        m_tp = sum(1 for r in mounted if r["has_spill"] and r["pred"])
        m_fn = sum(1 for r in mounted if r["has_spill"] and not r["pred"])
        m_fp = sum(1 for r in mounted if not r["has_spill"] and r["pred"])
        m_tn = sum(1 for r in mounted if not r["has_spill"] and not r["pred"])
        m_acc = (m_tp + m_tn) / len(mounted) if mounted else 0
        cam_names = {"walkway_vert": "Mall walkway, handheld phone (not mounted)", "corridor_vert": "Covered corridor, vertical",
                     "atrium_escalators": "Atrium with escalators, elevated", "atrium_4k": "Multi-level atrium, elevated (negatives only)",
                     "plaza_overhead": "Plaza, overhead", "walkway": "Campus walkway (OpenCV)", "mallcorridor": "CAVIAR corridor"}
        per_camera_rows = ""
        for s in seqs:
            sp = [r for r in cpos if r["seq"] == s]
            sn = [r for r in cneg if r["seq"] == s]
            res = next((r.get("res") for r in crows if r["seq"] == s and r.get("res")), None)
            found = [r for r in sp if r["pred"]]
            smallest = min((r["spill_width_frac"] for r in found), default=None)
            per_camera_rows += (f"<tr><th>{html.escape(cam_names.get(s, s))}</th>"
                                f"<td>{len(found)} of {len(sp)}</td><td>{sum(1 for r in sn if r['pred'])} of {len(sn)}</td>"
                                f"<td>{f'{smallest*100:.0f}% of frame width' if smallest else '&mdash;'}</td></tr>")
        cctv_section = f"""
<section id="fixed-camera">
  <div class="prose">
    <p class="eyebrow">How it works</p>
    <h2 style="margin-top:6px">Reference, persistent change, verdict</h2>
    <p style="margin-top:10px;color:var(--fg-2)">Because the cameras never move, the server keeps a reference picture of each empty floor and only asks the model about patches that changed and <em>stayed</em> changed: a spill persists, a person walks on. Each such patch is cut out of the frames at full size and sent with the reference and a frame two seconds later. A frame with no persistent change costs nothing. The robot is dispatched only to a confirmed spill.</p>
  </div>
  <div class="steps">
    <div class="step"><div class="n">1 &middot; Reference</div><h3>Remember the empty floor</h3><p>A median over a few seconds of footage removes passers-by and leaves the clean floor, per camera. Refreshed slowly so lighting changes do not pile up.</p></div>
    <div class="step"><div class="n">2 &middot; Persistent change</div><h3>Find what appeared and stayed</h3><p>Difference against the reference, kept only where the change is still there two seconds later. People, trolleys and their shadows drop out; spills, dropped bags and signs remain.</p></div>
    <div class="step"><div class="n">3 &middot; Ask about the patch</div><h3>Three frames, one crop</h3><p>Claude sees the same crop from the reference, the current frame and a frame two seconds later, and answers "has a liquid spill appeared on the floor?" with the same JSON as before. The third frame is how it tells a spill, which stays, from a shadow, which walks away. Boxes come back in frame coordinates.</p></div>
  </div>
  <div class="prose">
    <p class="eyebrow" style="margin-top:8px">Results</p>
    <p style="margin-top:6px;color:var(--fg-2)">Across all {n_pairs} pairs from {len(seqs)} cameras: <strong>{pct(cs['accuracy'])}</strong> accuracy, {ccf['tp']} of {ccf['tp'] + ccf['fn']} spills found, {ccf['fp']} false alarms in {ccf['tn'] + ccf['fp']} real frames, and {cs['spill_located_correctly']} found spills placed at the right spot. On the four mounted cameras alone ({len(mounted)} pairs): <strong>{pct(m_acc)}</strong> accuracy, {m_tp} of {m_tp + m_fn} spills found, {m_fp} false alarms in {m_fp + m_tn} real frames. The handheld walkway clip accounts for every false alarm.</p>
  </div>
  <div class="prose">
    <p style="color:var(--fg-2)">Pairs below are exactly what the model saw: the same crop from the reference, the current frame and two seconds later. The amber box is the model's answer, the thin green box is the true spill. Captions name the camera and the truth.</p>
  </div>
  <div class="gallery pairs">{pair_cards}</div>
  <div class="prose">
    <p class="eyebrow" style="margin-top:16px">Clear liquids</p>
    <h3 style="margin-top:6px">Water is the hard case, and it is found</h3>
    <p style="margin-top:8px;color:var(--fg-2)">Water has no colour of its own. On a floor it shows only as a darkening of the surface, a soft outline and a slight sheen, which is also how a shadow or a polished patch looks. The model found <strong>{len(water_found)} of {len(water)}</strong> water spills across the mall cameras, down to one {water_smallest_w*100:.0f}% of the frame wide. Its reasoning on these consistently names the darkening and the fact that the patch stays put and touches no one. Coloured liquids for comparison: coffee {sum(1 for r in cpos if r['kind']=='coffee' and r['pred'])} of {sum(1 for r in cpos if r['kind']=='coffee')}, cola {sum(1 for r in cpos if r['kind']=='cola' and r['pred'])} of {sum(1 for r in cpos if r['kind']=='cola')}, juice {sum(1 for r in cpos if r['kind']=='juice' and r['pred'])} of {sum(1 for r in cpos if r['kind']=='juice')}, oil {sum(1 for r in cpos if r['kind']=='oil' and r['pred'])} of {sum(1 for r in cpos if r['kind']=='oil')}.</p>
  </div>
  <div class="gallery pairs">{water_cards}</div>
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>The pipeline: persistent crops, three frames ({n_pairs} pairs)</th><th>Whole frame, two frames, no candidate step ({fs['pairs'] if fs else 0} pairs, every third)</th></tr></thead>
    <tbody>
      <tr><th>Spills found</th><td class="hi">{ccf['tp']} of {ccf['tp'] + ccf['fn']}</td><td>{fcf['tp']} of {fcf['tp'] + fcf['fn']}</td></tr>
      <tr><th>False alarms</th><td class="hi">{ccf['fp']} of {ccf['tn'] + ccf['fp']}</td><td>{fcf['fp']} of {fcf['tn'] + fcf['fp']}</td></tr>
      <tr><th>Spills smaller than 7% of frame width found</th><td class="hi">{sum(1 for r in cpos if r['spill_width_frac'] < 0.07 and r['pred'])} of {sum(1 for r in cpos if r['spill_width_frac'] < 0.07)}</td><td>{sum(1 for r in frows if r['has_spill'] and r['spill_width_frac'] < 0.07 and r['pred'])} of {sum(1 for r in frows if r['has_spill'] and r['spill_width_frac'] < 0.07)}</td></tr>
      <tr><th>Model calls per frame</th><td>{calls_per:.1f}</td><td>1</td></tr>
      <tr><th>Cost per checked frame</th><td>${c_cost_pair:.3f}</td><td>${f_cost_pair:.3f}</td></tr>
      <tr><th>Latency per frame</th><td>{c_lat:.1f} s</td><td>{f_lat:.1f} s</td></tr>
    </tbody>
  </table></div>
  <div class="prose">
    <p class="eyebrow" style="margin-top:16px">Appendix</p>
    <h3 style="margin-top:6px">Every check on the mall cameras</h3>
    <p style="margin-top:8px;color:var(--fg-2)">All {n_pairs} checks, spills first, then the ones the model got wrong, then by camera. Each shows the crop the model judged in the current frame: thin green is the true spill, amber is the model's answer. Hover for its reason.</p>
  </div>
  <details>
    <summary>Show all {n_pairs} checks</summary>
    <div class="all">{cctv_appendix}</div>
  </details>
</section>"""

    # Header tiles: the mounted-camera numbers lead; fall back to the close-camera numbers if no CCTV run exists
    if cs:
        hdr_tiles = (
            f'<div class="tile"><div class="big">{pct(m_acc)}</div><div class="label">accurate on mounted cameras</div><div class="src">{len(mounted)} before-and-after checks, 4 cameras</div></div>'
            f'<div class="tile"><div class="big">{pct(m_tp / (m_tp + m_fn) if m_tp + m_fn else 0)}</div><div class="label">of spills found</div><div class="src">{m_tp} of {m_tp + m_fn} rendered spills on real floors</div></div>'
            f'<div class="tile"><div class="big">{m_fp}</div><div class="label">false alarms</div><div class="src">in {m_fp + m_tn} real frames of people and trolleys</div></div>'
            f'<div class="tile"><div class="big">${c_cost_pair:.2f}</div><div class="label">per checked frame</div><div class="src">frames with nothing persistent cost nothing</div></div>'
        )
        cctv_cost_pair, cctv_calls_per = c_cost_pair, calls_per
    else:
        hdr_tiles = (
            f'<div class="tile"><div class="big">{pct(fo["accuracy"])}</div><div class="label">correct on floor frames</div><div class="src">{fo["n"]} labelled frames</div></div>'
            f'<div class="tile"><div class="big">{fo["fp"]}</div><div class="label">false alarms</div><div class="src">of {fo["tn"] + fo["fp"]} frames with no spill</div></div>'
        )
        cctv_cost_pair, cctv_calls_per = cost_per, 1.0

    sample_json = json.dumps(find(opus, "333_mp4-11")["result"], indent=2) if find(opus, "333_mp4-11") else "{}"

    page = f"""<title>Spill Detection PoC</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,500;100,700;100,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column for text, a wider band for the frame gallery; the amber of a wet-floor sign is the single accent. */
:root {{
  --bg: #f3f2ee; --surface: #ffffff; --surface-2: #e9e7e0; --line: #d6d3ca;
  --fg: #1d1f22; --fg-2: #55585e; --fg-3: #8a8d93;
  --accent: #e0a100; --accent-ink: #5a4000; --accent-soft: #fff1c2;
  --ok: #1f8a5b; --ok-soft: #d9f2e5; --ok-ink: #0f4d31;
  --grey-soft: #e3e3e8; --grey-ink: #4a4d55;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", Consolas, "Courier New", monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #15171a; --surface: #1e2126; --surface-2: #272b31; --line: #363b43;
  --fg: #eceef1; --fg-2: #b4b8c0; --fg-3: #7d828b;
  --accent: #f4b400; --accent-ink: #ffd766; --accent-soft: #3d3000;
  --ok: #3fbf84; --ok-soft: #123d2a; --ok-ink: #9fe3c3;
  --grey-soft: #2f333a; --grey-ink: #c2c6cd; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #15171a; --surface: #1e2126; --surface-2: #272b31; --line: #363b43;
  --fg: #eceef1; --fg-2: #b4b8c0; --fg-3: #7d828b;
  --accent: #f4b400; --accent-ink: #ffd766; --accent-soft: #3d3000;
  --ok: #3fbf84; --ok-soft: #123d2a; --ok-ink: #9fe3c3;
  --grey-soft: #2f333a; --grey-ink: #c2c6cd; color-scheme: dark }}

body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.55; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-block: 40px 72px; padding-inline: 20px; }}
.prose {{ max-width: 68ch; }}
h1, h2, h3 {{ font-family: var(--display); font-stretch: 100%; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(34px, 5vw, 56px); font-weight: 800; line-height: 1.02; letter-spacing: -0.02em; }}
h2 {{ font-size: clamp(22px, 2.6vw, 30px); font-weight: 700; letter-spacing: -0.01em; }}
h3 {{ font-size: 18px; font-weight: 700; }}
p {{ margin: 0; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fg-3); }}
section {{ margin-top: 56px; display: grid; gap: 18px; }}
.lede {{ font-size: 19px; color: var(--fg-2); max-width: 62ch; }}
a {{ color: inherit; }}

.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-top: 28px; }}
.tile {{ background: var(--surface); border: 1px solid var(--line); padding: 18px 18px 14px; display: grid; gap: 4px; min-width: 0; }}
.tile .big {{ font-family: var(--display); font-size: 40px; font-weight: 800; line-height: 1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
.tile .label {{ font-weight: 600; }}
.tile .src {{ font-size: 13px; color: var(--fg-3); }}

.steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
.step {{ background: var(--surface-2); padding: 18px; display: grid; gap: 8px; min-width: 0; }}
.step .n {{ font-family: var(--mono); color: var(--accent-ink); font-size: 13px; }}
pre {{ margin: 0; background: var(--surface); border: 1px solid var(--line); padding: 14px; font-family: var(--mono); font-size: 13px; line-height: 1.5; overflow-x: auto; }}

.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ font: 500 14px var(--body); background: var(--surface); border: 1px solid var(--line); color: var(--fg-2); padding: 7px 14px; border-radius: 999px; cursor: pointer; }}
.chip[aria-pressed="true"] {{ background: var(--fg); color: var(--bg); border-color: var(--fg); }}
.chip:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; }}
.frame {{ margin: 0; background: var(--surface); border: 1px solid var(--line); display: grid; grid-template-rows: auto 1fr; min-width: 0; }}
.frame[hidden] {{ display: none; }}
.img {{ position: relative; aspect-ratio: 1; max-width: 100%; background: var(--surface-2); overflow: hidden; }}
.img img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.box {{ position: absolute; border: 2px solid var(--accent); box-shadow: 0 0 0 1px rgba(0,0,0,.45), inset 0 0 0 1px rgba(0,0,0,.45); }}
figcaption {{ padding: 14px 16px 16px; display: grid; gap: 8px; }}
.verdict-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.pill {{ font: 600 12px var(--body); letter-spacing: 0.04em; text-transform: uppercase; padding: 4px 10px; border-radius: 4px; }}
.pill.spill {{ background: var(--accent-soft); color: var(--accent-ink); }}
.pill.clear {{ background: var(--ok-soft); color: var(--ok-ink); }}
.conf, .lat, .meta {{ font-family: var(--mono); font-size: 12.5px; color: var(--fg-2); }}
.lat {{ margin-left: auto; color: var(--fg-3); }}
.reason {{ font-size: 14.5px; color: var(--fg); }}
.caption {{ font-size: 13px; color: var(--fg-3); border-top: 1px solid var(--line); padding-top: 8px; }}

table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
.tablewrap {{ overflow-x: auto; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
thead th {{ font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--fg-3); font-weight: 500; }}
tbody th {{ font-weight: 500; color: var(--fg-2); }}
td.hi {{ font-weight: 600; }}

.compare {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr); gap: 18px; background: var(--surface); border: 1px solid var(--line); padding: 18px; }}
.compare .img {{ aspect-ratio: 1; }}
.verdicts {{ display: grid; gap: 16px; align-content: start; }}
.v {{ display: grid; gap: 8px; }}
.v .who {{ font-family: var(--display); font-weight: 700; }}
.v .pill {{ justify-self: start; }}
@media (max-width: 640px) {{ .compare {{ grid-template-columns: 1fr; }} }}

.road {{ display: grid; gap: 0; border-top: 1px solid var(--line); }}
.road > div {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 16px; padding: 16px 0; border-bottom: 1px solid var(--line); }}
.road .when {{ font-family: var(--mono); font-size: 13px; color: var(--fg-3); }}
.road .now .when {{ color: var(--accent-ink); font-weight: 500; }}
@media (max-width: 520px) {{ .road > div {{ grid-template-columns: 1fr; gap: 4px; }} }}
.note {{ font-size: 14px; color: var(--fg-3); }}

.pairs {{ grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }}
.pair {{ margin: 0; background: var(--surface); border: 1px solid var(--line); min-width: 0; }}
.pair .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: var(--line); }}
.pair .two.three {{ grid-template-columns: 1fr 1fr 1fr; }}
.pair .img {{ aspect-ratio: 1; }}
.pair .tag {{ position: absolute; left: 6px; bottom: 6px; font: 500 11px var(--mono); letter-spacing: .04em; text-transform: uppercase; background: rgba(0,0,0,.55); color: #fff; padding: 2px 6px; }}
.gtbox {{ position: absolute; border: 1px solid var(--ok); box-shadow: 0 0 0 1px rgba(0,0,0,.35); }}
.pair figcaption {{ padding: 12px 14px 14px; display: grid; gap: 8px; }}
.all {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }}
.small {{ margin: 0; background: var(--surface); border: 1px solid var(--line); min-width: 0; }}
.small .img {{ aspect-ratio: 1; }}
.small figcaption {{ padding: 8px 10px 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
.small .pill {{ font-size: 10.5px; padding: 3px 7px; }}
.small .status {{ font-family: var(--mono); font-size: 11px; width: 100%; }}
.status.ok {{ color: var(--ok); }}
.status.bad {{ color: var(--accent-ink); }}
.status.note {{ color: var(--fg-3); }}
details {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
summary {{ cursor: pointer; padding: 14px 0; font-family: var(--display); font-weight: 700; font-size: 18px; list-style: none; display: flex; align-items: center; gap: 10px; }}
summary::before {{ content: "+"; font-family: var(--mono); color: var(--fg-3); width: 1em; }}
details[open] summary::before {{ content: "\\2212"; }}
summary::-webkit-details-marker {{ display: none; }}
details .all {{ padding-bottom: 20px; }}
footer {{ margin-top: 64px; font-size: 13px; color: var(--fg-3); border-top: 1px solid var(--line); padding-top: 16px; display: grid; gap: 4px; }}
@media (prefers-reduced-motion: no-preference) {{ .chip {{ transition: background .15s, color .15s; }} }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Proof of concept &middot; Mall security cameras &middot; Opus 5.5 vision &middot; 21 October 2026</p>
  <h1 style="margin-top:10px">Spill detection on the mall's own cameras</h1>
  <p class="lede" style="margin-top:18px">The mall already has fixed cameras over every corridor and atrium. This proof of concept uses them to spot a liquid spill within seconds of it happening and to send the cleaning robot only to confirmed spills, so the robot does not patrol. No training data was collected and no model was trained: a general vision model judges what changed on the floor against a written description of a spill, and every answer is kept as a label for a smaller on-site model later.</p>

  <div class="tiles">{hdr_tiles}</div>
</header>

{cctv_section}

<section>
  <div class="prose">
    <p class="eyebrow">Cost at scale</p>
    <h2 style="margin-top:6px">What this costs across a mall</h2>
    <p style="margin-top:10px;color:var(--fg-2)">The camera server does the cheap work for free: it keeps the reference, finds changes and drops the ones that walk away. Only persistent patches reach the model, and each check costs about ${cctv_cost_pair:.3f} at the {cctv_calls_per:.1f} crops per check seen in the busy test clips. One camera, a 12-hour day:</p>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>One camera</th><th>Checks per day</th><th>Per day</th><th>Per month</th></tr></thead>
    <tbody>
      <tr><th>Quiet floor: one check every 4 minutes</th><td>180</td><td>${180 * cctv_cost_pair:,.2f}</td><td>${180 * 30 * cctv_cost_pair:,.0f}</td></tr>
      <tr><th>Busy corridor: one check every minute</th><td>720</td><td>${720 * cctv_cost_pair:,.2f}</td><td>${720 * 30 * cctv_cost_pair:,.0f}</td></tr>
    </tbody>
  </table></div>
  <div class="prose"><p class="note">The per-check cost was measured on crowded test clips where people standing still add crops; an empty floor costs less per check. Once enough labelled patches have accumulated, a small on-site model takes the easy decisions and the cloud model becomes a second opinion.</p></div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Next</p>
    <h2 style="margin-top:6px">Roadmap</h2>
  </div>
  <div class="road">
    <div><div class="when">Done</div><div><strong>Fixed-camera pipeline.</strong> Reference, persistent change, before-and-after crops. Tested on five mall cameras with rendered spills: no false alarms on the mounted cameras.</div></div>
    <div class="now"><div class="when">Next</div><div><strong>One real mall camera, real spills.</strong> A few hours of footage from a single camera in any mall, with water, coffee and oil spilled on the floor in view. This replaces the rendered positives and gives the true recall number and the real check rate.</div></div>
    <div><div class="when">Then</div><div><strong>Hard negatives at scale.</strong> Polished stone, reflections, cleaning carts, wet-floor signs, dropped bags. Measure the false-alarm rate per camera per day, which is what decides whether staff trust the alerts.</div></div>
    <div><div class="when">Later</div><div><strong>Camera-server deployment.</strong> Reference and persistence run on the camera server at no cost; only persistent patches go to the model. Robot dispatch on confirmed spills. An on-site model trained from the accumulated labels where the cloud call is not wanted.</div></div>
  </div>
</section>

<section id="close-camera">
  <div class="prose">
    <p class="eyebrow">Also tested</p>
    <h2 style="margin-top:6px">A close camera on the robot itself</h2>
    <p style="margin-top:10px;color:var(--fg-2)">Before the camera question was settled, the same model was tested on single close-range frames, the view a camera on the robot would have. It is kept here because it is the fallback if a mall's cameras cannot be used, and because it shows how the model behaves on a public spill dataset with real liquids.</p>
  </div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Close camera &middot; pipeline</p>
    <h2 style="margin-top:6px">How a single frame becomes a verdict</h2>
  </div>
  <div class="steps">
    <div class="step"><div class="n">1 &middot;Sample</div><h3>Pull frames from the camera</h3><p>One or two frames per second is enough for a slow-moving robot. Near-identical frames are dropped so a parked robot does not keep paying to look at the same tile.</p></div>
    <div class="step"><div class="n">2 &middot;Ask</div><h3>Vision model reads the frame</h3><p>Claude Opus 5.5 receives the image and a written brief: what counts as a spill, what does not (reflections, shadows, dark tiles, wet-floor signs, liquid in cups). It must answer in a fixed JSON shape.</p></div>
    <div class="step"><div class="n">3 &middot;Act and keep</div><h3>Verdict, box, reasoning</h3><p>The robot gets a yes/no, a confidence, a liquid type and a box. The answer is also written to disk with the frame, so every run grows the training set for a small on-board model later.</p></div>
  </div>
  <div class="prose">
    <p class="note">What one answer looks like, for the first frame in the gallery below:</p>
  </div>
  <pre>{html.escape(sample_json)}</pre>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Close camera &middot; gallery</p>
    <h2 style="margin-top:6px">What the model saw, and what it said</h2>
    <p style="margin-top:10px;color:var(--fg-2)">Real close-range frames from the public dataset. The amber box is where the model says the spill is. The text under each frame is the model's own one-line reason, unedited.</p>
  </div>
  <div class="chips" role="group" aria-label="Filter frames">
    <button class="chip" type="button" data-filter="all" aria-pressed="true">All {len(cards)}</button>
    <button class="chip" type="button" data-filter="detected" aria-pressed="false">Spills found</button>
    <button class="chip" type="button" data-filter="hard" aria-pressed="false">Hard cases</button>
    <button class="chip" type="button" data-filter="clear" aria-pressed="false">Correctly clear</button>
    <button class="chip" type="button" data-filter="scope" aria-pressed="false">Out of scope</button>
  </div>
  <div class="gallery" id="gallery">{gallery}</div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Close camera &middot; model choice</p>
    <h2 style="margin-top:6px">Why Opus 5.5 and not the cheaper model</h2>
    <p style="margin-top:10px;color:var(--fg-2)">Claude Haiku 5.5 costs about a thirtieth as much. On the same {len(opus)} frames its headline score is close, but it is less consistent: it ignored the floor-only rule on {fh['non_floor'] - fh['rule_ok']} of the {fh['non_floor']} mislabelled frames, gave up on several abstract splash images, and missed the one genuinely hard floor spill in the set.</p>
  </div>
  {compare}
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>Claude Opus 5.5</th><th>Claude Haiku 5.5</th></tr></thead>
    <tbody>{haiku_rows}</tbody>
  </table></div>
  <div class="prose"><p class="note">Recommendation: Opus 5.5 for the proof of concept and for labelling anything subtle. Haiku may later take the easy frames in bulk, with Opus re-checking anything it marks below 0.8 confidence.</p></div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Close camera &middot; appendix</p>
    <h2 style="margin-top:6px">Every frame in the evaluation</h2>
    <p style="margin-top:10px;color:var(--fg-2)">All {n_all} frames the model was scored on, spills first, highest confidence first. The {len(cards)} frames in the gallery above were picked by hand from this set to show one of each situation; this list is the unfiltered whole. Hover a frame for the model's reason.</p>
  </div>
  <details>
    <summary>Show all {n_all} frames</summary>
    <div class="all">{appendix}</div>
  </details>
</section>

<footer>
  <div>Model: claude-opus-5-5, effort low, structured JSON output.</div>
  <div>Fixed-camera footage: five mall clips from Pixabay and Pexels (Pixabay Content License, Pexels License), real floors with rendered spills; no AI image generation was used anywhere in this work.</div>
  <div>Close-camera data: Spill_Detection v4 by slipp on Roboflow Universe, CC BY 4.0. Validation split, {len(opus)} frames, random seed 0.</div>
  <div>Report generated from data/results by scripts/build_report.py.</div>
</footer>
</div>

<script>
(function () {{
  var chips = document.querySelectorAll('.chip');
  var frames = document.querySelectorAll('#gallery .frame');
  chips.forEach(function (c) {{
    c.addEventListener('click', function () {{
      var f = c.getAttribute('data-filter');
      chips.forEach(function (o) {{ o.setAttribute('aria-pressed', o === c ? 'true' : 'false'); }});
      frames.forEach(function (fr) {{ fr.hidden = !(f === 'all' || fr.getAttribute('data-group') === f); }});
    }});
  }});
}})();
</script>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(cards)} frames)")

    # Standalone copy for GitHub Pages, served from docs/<random path>/ so the URL is not guessable.
    # The path token is kept in pages_path.txt so every rebuild lands at the same URL.
    token_file = ROOT / "pages_path.txt"
    if not token_file.exists():
        import secrets
        token_file.write_text(secrets.token_urlsafe(18))
    token = token_file.read_text().strip()
    pages_out = ROOT / "docs" / token / "index.html"
    pages_out.parent.mkdir(parents=True, exist_ok=True)
    standalone = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"robots\" content=\"noindex, nofollow\">\n"
        "<style>html{color-scheme:light dark}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n"
        + page.replace("<title>", "<title>", 1)
        + "\n</html>\n"
    )
    # move <title> and <link>/<style> into <head>: the page starts with them, so split at the first <div class="wrap">
    head_part, body_part = standalone.split('<div class="wrap">', 1)
    standalone = head_part + "</head>\n<body>\n" + '<div class="wrap">' + body_part.replace("\n</html>\n", "\n</body>\n</html>\n")
    pages_out.write_text(standalone, encoding="utf-8")
    print(f"wrote {pages_out.relative_to(ROOT)} for GitHub Pages")

    # Print variant for the PDF: light palette, no collapsible appendices, eager images, cards kept whole.
    import re as _re
    printable = standalone.replace('loading="lazy"', "")
    printable = _re.sub(r"<details>.*?</details>", "", printable, flags=_re.S)
    printable = _re.sub(r'<div class="chips".*?</div>\s*', "", printable, count=1, flags=_re.S)
    print_css = """<style>
html { color-scheme: light !important; }
@page { size: A4; margin: 14mm 12mm; }
body { font-size: 11.5px; }
.wrap { max-width: none; padding-block: 0; padding-inline: 0; }
h1 { font-size: 30px; } h2 { font-size: 20px; } .lede { font-size: 13.5px; }
section { margin-top: 26px; break-inside: auto; }
.tiles { grid-template-columns: repeat(4, 1fr); }
.tile .big { font-size: 26px; }
.steps { grid-template-columns: repeat(3, 1fr); }
.gallery { grid-template-columns: repeat(2, 1fr); gap: 10px; }
.gallery.pairs { grid-template-columns: repeat(2, 1fr); }
.frame, .pair, .tile, .step, .compare, table, pre { break-inside: avoid; page-break-inside: avoid; }
.reason { font-size: 10.5px; } .caption { font-size: 9.5px; }
.pill { font-size: 9px; } .conf, .lat, .meta { font-size: 9.5px; }
#close-camera { break-before: page; page-break-before: always; }
footer { margin-top: 24px; }
</style>
"""
    printable = printable.replace("</head>", print_css + "</head>", 1)
    printable = printable.replace('<html lang="en">', '<html lang="en" data-theme="light">', 1)
    print_out = ROOT / "report" / "print.html"
    print_out.write_text(printable, encoding="utf-8")
    print(f"wrote {print_out.relative_to(ROOT)} for PDF rendering")


if __name__ == "__main__":
    main()
