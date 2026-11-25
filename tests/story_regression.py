"""
Story Mode regression script.

NOT a pytest/unittest suite — it hits the REAL, running /generate-story
endpoint (not the functions directly) and saves every scene's output image
to disk for visual comparison. There is no automated pass/fail on image
content (no CV classifier in this project) -- you (or Claude) look at the
saved images the same way every finding in this investigation was verified.

Covers the two scenarios from this investigation:
  - story_pose_3scene: 3 scenes (forest standing -> crouching in forest ->
    city street at night). Regresses the action/pose + location strength bug
    (frontend default silently overriding STORY_BASE_STRENGTH).
  - story_objectchar_5scene: 5 scenes (forest -> running -> beach at sunset
    -> sitting beside red vintage car -> talking with young man beside car).
    Regresses the object/secondary-character introduction issue, including
    the sequential-input-image compounding case (scene 4's input is scene
    3's real beach output, not a fresh image).

Usage:
    python tests/story_regression.py
    python tests/story_regression.py --base-url http://localhost:8000
    python tests/story_regression.py --story pose --seeds 1,7,42,100,2024
    STORY_REGRESSION_BASE_URL=http://localhost:8000 python tests/story_regression.py

--seeds runs every selected story once per seed (default: one fixed seed,
42). --story selects which scenario(s) to run: "pose", "objectchar", or
"all" (default).

Requires the backend server already running. Does NOT start/stop it, and
does not touch strength, IP-Adapter, or prompt-building logic anywhere.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

# Project root = parent of this file's directory (tests/ -> project root),
# computed relative to this script's own location, not hardcoded.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "regression")

DEFAULT_BASE_URL = os.environ.get("STORY_REGRESSION_BASE_URL", "http://localhost:8000")

# Fixed base_strength=0.90 (production default, validated in this
# investigation) for every run, so results are directly comparable across
# runs after a future change to strength, IP-Adapter, or prompt building.
# Default seed list covers a spread, not just the one seed (42) used in
# every earlier validation run, so single-seed luck can't hide drift.
BASE_STRENGTH = 0.90
DEFAULT_SEEDS = [1, 7, 42, 100, 2024]

STORY_POSE = {
    "name": "story_pose_3scene",
    "scenes": [
        {"prompt": "A young woman with short red hair, wearing a blue jacket, standing calmly in a forest."},
        {"prompt": "The same woman is crouching low and looking behind her, still in the forest."},
        {"prompt": "The same woman is standing calmly on a busy city street at night."},
    ],
}

STORY_OBJECT_CHAR = {
    "name": "story_objectchar_5scene",
    "scenes": [
        {"prompt": "A young woman with short red hair, wearing a blue jacket, standing calmly in a forest."},
        {"prompt": "The same woman is running quickly through the forest."},
        {"prompt": "The same woman is walking along a beach at sunset."},
        {"prompt": "The same woman is sitting beside a red vintage car on the beach."},
        {"prompt": "The same woman is talking with a young man beside the red vintage car."},
    ],
}

STORIES_BY_NAME = {"pose": STORY_POSE, "objectchar": STORY_OBJECT_CHAR}
STORIES = [STORY_POSE, STORY_OBJECT_CHAR]


def run_story(base_url: str, story: dict, run_dir: str, seed: int) -> bool:
    payload = {
        "scenes": story["scenes"],
        "global_negative": "(worst quality:2), (low quality:2), blurry, bad anatomy, watermark, text",
        "steps": 30,
        "guidance_scale": 9.0,
        "base_strength": BASE_STRENGTH,
        "width": 512,
        "height": 512,
        "seed": seed,
    }
    req = urllib.request.Request(
        f"{base_url}/generate-story",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[{story['name']} seed={seed}] POST {base_url}/generate-story "
          f"({len(story['scenes'])} scenes, base_strength={BASE_STRENGTH}) ...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"[{story['name']} seed={seed}] Could not reach the server at {base_url} — "
              f"is it running? ({e})", file=sys.stderr)
        return False
    except urllib.error.HTTPError as e:
        print(f"[{story['name']} seed={seed}] Server returned HTTP {e.code}: "
              f"{e.read()[:300]}", file=sys.stderr)
        return False
    elapsed = time.time() - t0

    story_dir = os.path.join(run_dir, story["name"], f"seed{seed}")
    os.makedirs(story_dir, exist_ok=True)
    for i, b64 in enumerate(body.get("scenes", [])):
        img_bytes = base64.b64decode(b64)
        out_path = os.path.join(story_dir, f"scene{i+1}.jpg")
        with open(out_path, "wb") as f:
            f.write(img_bytes)

    # /generate-story now returns HTTP 200 even when a scene fails midway
    # (it returns whatever scenes succeeded first, instead of raising) --
    # so HTTP success alone no longer means the STORY succeeded. Check the
    # body's own "success" flag explicitly, or a partial failure silently
    # gets reported as a pass here.
    if not body.get("success", True):
        print(f"[{story['name']} seed={seed}] FAILED at scene "
              f"{body.get('failed_scene')}/{body.get('requested_total')} "
              f"({len(body.get('scenes', []))} scene(s) saved to {story_dir} before "
              f"the failure, {elapsed:.1f}s) — {body.get('error')}", file=sys.stderr)
        return False

    print(f"[{story['name']} seed={seed}] OK — {len(body.get('scenes', []))} scenes saved to "
          f"{story_dir} ({elapsed:.1f}s, seed_used={body.get('seed_used')})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                     help=f"Backend base URL (default: {DEFAULT_BASE_URL}, "
                          f"or set STORY_REGRESSION_BASE_URL)")
    ap.add_argument("--story", default="all", choices=["pose", "objectchar", "all"],
                     help="Which scenario(s) to run (default: all)")
    ap.add_argument("--seeds", default=None,
                     help="Comma-separated seed list (default: one run with "
                          f"seed=42, unless --seeds is given, e.g. "
                          f"{','.join(str(s) for s in DEFAULT_SEEDS)})")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [42]
    stories = STORIES if args.story == "all" else [STORIES_BY_NAME[args.story]]

    run_dir = os.path.join(OUT_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Regression run -> {run_dir}  (seeds={seeds})")

    all_ok = True
    for story in stories:
        for seed in seeds:
            ok = run_story(args.base_url, story, run_dir, seed)
            all_ok = all_ok and ok

    if not all_ok:
        print("\nOne or more stories failed — see messages above.", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll stories generated. Visually compare {run_dir} against a previous "
          f"known-good run under {OUT_ROOT} before/after your change.")


if __name__ == "__main__":
    main()
