# Spill Detection PoC (Claude Opus 5.5 vision)

Proof of concept for a mall cleaning robot: given a camera frame, decide whether there is a
liquid spill on the floor. No CV training. Claude Opus 5.5 classifies each frame through
structured outputs, and every answer is saved so the run doubles as a labeling pass for the
on-device model you will train later.

```
spilldetect/
  detector.py      SpillDetector: one image -> SpillResult (JSON schema enforced by the API)
  images.py        resize + base64 encode (keeps each image at <= 1.15 MP, the API's own cap)
  dataset.py       YOLO-format loader (Roboflow export) -> per-image spill / no-spill ground truth
scripts/
  run_eval.py      run over a dataset or folder, write results.jsonl/csv, summary.json, errors.txt
  classify_one.py  classify a few images and print the JSON
  sample_frames.py video -> frames at 1-2 fps, skipping near-duplicates
  make_test_images.py  synthetic floors for a first pipeline check
tests/test_offline.py  runs without an API key (fake client)
```

## 1. Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key in the shell you will run from (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Or persist it for your user account so every new terminal has it:

```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

Offline sanity check (no key needed):

```bash
python tests/test_offline.py
```

## 2. First live run (5 synthetic images, about $0.05)

```bash
python scripts/make_test_images.py data/frames/synthetic
python scripts/classify_one.py data/frames/synthetic/pos_water_puddle.jpg data/frames/synthetic/neg_glossy_reflection.jpg
```

## 3. Get the public dataset

The biggest public set is [Spill_Detection on Roboflow Universe](https://universe.roboflow.com/slipp/spill_detection)
(about 3,400 images, classes `floor` and `spill`). Roboflow requires a free account to download.

1. Open the project page, click **Download Dataset** (or the dataset version, then Download).
2. Format: **YOLOv8**. Choose "download zip to computer".
3. Unzip into `data/raw/spill_detection/` so you have `data/raw/spill_detection/data.yaml`
   plus `train/`, `valid/`, `test/` folders.

Other sets worth adding the same way (check each license page first):
[floor damage](https://universe.roboflow.com/l-n5dzz/floor-damage) (spill vs dirt, CC BY 4.0),
[powerbee spill](https://universe.roboflow.com/powerbee/spill),
[Infinity AI Spills Basic](https://astro.paperswithcode.com/dataset/infinity-spills-basic-dataset) (synthetic videos, run `sample_frames.py` on them).

If a dataset's spill class is not literally called "spill", pass it:
`--spill-classes liquid puddle`.

## 4. Baseline accuracy

Start small to check the prompt, then scale:

```bash
python scripts/run_eval.py data/raw/spill_detection --split valid --limit 100 --shuffle --dry-run
python scripts/run_eval.py data/raw/spill_detection --split valid --limit 100 --shuffle
```

Read `data/results/spill_detection_valid/errors.txt`: every false positive and false negative with
the model's one-line reasoning. That tells you whether to tighten the prompt in
`spilldetect/detector.py` (SYSTEM_PROMPT) before spending more.

Runs are resumable. Re-run the same command with a higher `--limit` and only new images are sent.

Useful flags: `--effort medium` (slower, usually a bit more accurate on subtle cases),
`--workers 8` (more parallel requests, watch your rate limit).

## 5. Your own footage

Shoot video from robot height (30-60 cm, tilted slightly down), then:

```bash
python scripts/sample_frames.py my_clips/ data/frames --fps 1
python scripts/run_eval.py data/frames/my_clip_name
```

No labels, so no accuracy number, but `results.csv` gives you the model's call on every frame and
you can review the positives quickly.

## 6. Demo report

```bash
python scripts/build_report.py
```

Writes two copies from the saved results:

- `report/index.html`: the body-only page published as a claude.ai artifact (private, access controlled).
- `docs/<random path>/index.html`: a standalone page for GitHub Pages. The random path lives in
  `pages_path.txt` so rebuilds keep the same URL. There is no index at the site root, and the page
  carries a `noindex` tag, but the repo is public, so treat the URL as obscure rather than private.

To update the public page: rebuild, then commit and push `docs/`. GitHub Pages redeploys in a minute or two.
Which frames appear in the gallery is set by the `SHOWCASE` list at the top of `scripts/build_report.py`;
the appendix always shows every frame in the latest Opus run.

## 7. Fixed security camera mode

The client may use the mall's existing CCTV instead of a robot camera. The pipeline for that:

1. **Reference frame**: per-pixel median of the first seconds of the clip (empty floor, passers-by removed).
2. **Candidates**: background difference against the reference, kept only where the change is still
   there a couple of seconds later (`spilldetect/cctv.py`). Spills persist, people walk on.
3. **Model**: each candidate is cropped at full resolution from both frames and sent to Claude in
   two-image mode (`SpillDetector.classify_pair`): "has a liquid spill appeared on the floor?"
4. A frame with no persistent change costs nothing.

Evaluation data: two public fixed-camera clips in `data/raw/cctv/` (OpenCV `vtest.avi`, a walkway;
CAVIAR `OneShopOneWait1cor.mpg`, a shopping-centre corridor, CC BY-SA) with synthetic spills composited
onto the real floor (`spilldetect/synth.py`), so every positive has an exact box and every negative
is real footage of people moving.

```bash
python scripts/build_cctv_pairs_video.py data/raw/cctv/vtest.avi --name walkway --pairs 30 --floor-poly "..."
python scripts/build_cctv_pairs_video.py data/raw/cctv/OneShopOneWait1cor.mpg --name mallcorridor --pairs 30 --floor-poly "..."
python scripts/run_cctv_eval.py --mode crops      # the real pipeline
python scripts/run_cctv_eval.py --mode full       # whole frame, no candidate step, for comparison
```

The `--floor-poly` is the floor zone in frame fractions; in a real install the integrator draws it once per camera.
Results land in `data/results/cctv_crops/` and `cctv_full/`.

## Cost

Roughly 1,000 to 1,650 input tokens per frame after resizing, plus ~100 to 200 output tokens.
On Opus 5.5 at `$4 / $20` per million tokens that is about `$0.01` per frame, so 1,000 frames is
around `$10`. The system prompt is marked for prompt caching. `summary.json` reports the real usage
and estimated cost of each run.

## What is built in

- Structured outputs: the API enforces the JSON schema, so no regex parsing of free text.
- Server-side refusal fallback (`fallbacks: "default"`): if a safety classifier declines a frame,
  the request is retried on another model inside the same call. `served_by` in the results shows
  which model answered.
- `stop_reason` handling: refusals and truncation are recorded as errors, never as "no spill".
- Thread pool with resume, so a 3,000-image run can be stopped and continued.

## Next steps after the baseline

1. Tune the prompt on the error list, re-run the same 100, compare `summary.json`.
2. Run Claude Haiku 5.5 on the same slice (`SpillDetector(model="claude-haiku-5-5")`) and compare.
   If it holds up, bulk labeling gets about 40x cheaper.
3. Use the saved `bbox` and `spill_detected` fields as labels to train a small on-device detector
   for the robot.
