
import argparse
import os

import numpy as np
from PIL import Image
import imageio.v2 as imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FPS = 24
CROSSFADE_SECONDS = 0.6
ZOOM_START = 1.0
ZOOM_END = 1.12  # gentle 12% zoom over each clip


def resolve(path: str) -> str:
    """Resolve a path relative to the project root, unless already absolute."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_scene_images(input_dir: str):
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().startswith("scene") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise SystemExit(f"No scene*.jpg/png files found in {input_dir}")
    return [Image.open(os.path.join(input_dir, f)).convert("RGB") for f in files]


def ken_burns_frames(img: Image.Image, num_frames: int, out_size, pan_direction: str):
    w, h = img.size
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * t
        crop_w, crop_h = w / zoom, h / zoom
        max_dx, max_dy = w - crop_w, h - crop_h
        if pan_direction == "right":
            cx = max_dx * t
        elif pan_direction == "left":
            cx = max_dx * (1 - t)
        else:
            cx = max_dx / 2
        cy = max_dy / 2
        frame = img.resize(
            out_size,
            resample=Image.LANCZOS,
            box=(cx, cy, cx + crop_w, cy + crop_h),
        )
        yield np.asarray(frame)


def crossfade(frames_a_tail, frames_b_head):
    """Alpha-blend two equal-length frame lists (tail of clip A, head of clip B)."""
    n = len(frames_a_tail)
    out = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        blended = (frames_a_tail[i].astype(np.float32) * (1 - alpha)
                   + frames_b_head[i].astype(np.float32) * alpha)
        out.append(blended.astype(np.uint8))
    return out


def build_video(images, out_path: str, seconds_per_image: float):
    out_size = images[0].size  # native size of the input images, no forced resize
    n_fade = int(CROSSFADE_SECONDS * FPS)
    n_frames_per_image = int(seconds_per_image * FPS)
    if n_frames_per_image <= n_fade * 2:
        raise SystemExit(
            f"--seconds-per-image too short for the {CROSSFADE_SECONDS}s crossfade "
            f"(need > {2 * CROSSFADE_SECONDS}s per image)"
        )

    clips = [
        list(ken_burns_frames(img, n_frames_per_image, out_size,
                               "right" if idx % 2 == 0 else "left"))
        for idx, img in enumerate(images)
    ]

    final_frames = clips[0][:-n_fade]
    for idx in range(1, len(clips)):
        prev_tail = clips[idx - 1][-n_fade:]
        curr_head = clips[idx][:n_fade]
        final_frames.extend(crossfade(prev_tail, curr_head))
        final_frames.extend(clips[idx][n_fade:])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8)
    for frame in final_frames:
        writer.append_data(frame)
    writer.close()

    duration = len(final_frames) / FPS
    return duration, out_size, len(final_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True,
                     help="Directory with scene*.jpg images (relative to project root unless absolute)")
    ap.add_argument("--out", default="outputs/video_proto/baseline.mp4",
                     help="Output mp4 path (relative to project root unless absolute)")
    ap.add_argument("--seconds-per-image", type=float, default=4.2,
                     help="Seconds of zoom/pan per image before crossfading to the next "
                          "(default 4.2 -> ~12s total for 3 images with a 0.6s crossfade)")
    args = ap.parse_args()

    input_dir = resolve(args.input_dir)
    out_path = resolve(args.out)

    images = load_scene_images(input_dir)
    print(f"Loaded {len(images)} images from {input_dir}")

    duration, out_size, n_frames = build_video(images, out_path, args.seconds_per_image)
    print(f"Wrote {out_path}")
    print(f"Duration: {duration:.2f}s ({n_frames} frames @ {FPS}fps)")
    print(f"Resolution: {out_size[0]}x{out_size[1]}")


if __name__ == "__main__":
    main()
