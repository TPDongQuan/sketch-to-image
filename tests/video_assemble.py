
import argparse
import os
import subprocess

import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "video_proto")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def check_output_path(out_path: str) -> str:
    out_path = resolve(out_path)
    out_path_norm = os.path.normpath(out_path)
    root_norm = os.path.normpath(OUTPUT_ROOT)
    if os.path.commonpath([out_path_norm, root_norm]) != root_norm:
        raise SystemExit(f"--out must be under {OUTPUT_ROOT}, got {out_path_norm}")
    return out_path_norm


def read_clip_frames(path: str):
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    frames = [f for f in reader]
    fps = meta.get("fps", 7.0)
    reader.close()
    return frames, fps


def frame_diff_stats(frames, downscale: int = 64):
    grays = []
    for f in frames:
        im = Image.fromarray(f).convert("L").resize((downscale, downscale), Image.BILINEAR)
        grays.append(np.asarray(im, dtype=np.float32))
    diffs = np.array([float(np.mean(np.abs(grays[i + 1] - grays[i]))) for i in range(len(grays) - 1)])
    max_pair_idx = int(np.argmax(diffs))
    return diffs, float(diffs.mean()), float(diffs.max()), max_pair_idx


def qa_clip(path: str, warn_threshold: float = 30.0, dense_every: int = 4,
            cell: int = 256, cols: int = 4):

    frames, fps = read_clip_frames(path)
    diffs, mean_d, max_d, max_idx = frame_diff_stats(frames)
    status = "WARNING (exceeds threshold)" if max_d > warn_threshold else "OK"
    print(f"QA [{os.path.basename(path)}]: {len(frames)} frames @ {fps}fps -- "
          f"mean diff={mean_d:.2f}, max diff={max_d:.2f} at frame {max_idx}->{max_idx + 1} "
          f"({status}, threshold={warn_threshold})")

    sample_idx = sorted(set(list(range(0, len(frames), dense_every))
                             + [len(frames) - 1, max_idx, max_idx + 1]))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = font_small = ImageFont.load_default()

    n = len(sample_idx)
    rows = (n + cols - 1) // cols
    label_h = 34
    sheet = Image.new("RGB", (cell * cols, label_h + cell * rows), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), f"{os.path.basename(path)} -- every {dense_every}th frame + worst jump "
                       f"(max={max_d:.1f} at {max_idx}->{max_idx + 1})", fill="white", font=font)
    for j, idx in enumerate(sample_idx):
        im = Image.fromarray(frames[idx]).resize((cell, cell))
        x = (j % cols) * cell
        y = label_h + (j // cols) * cell
        sheet.paste(im, (x, y))
        is_worst = idx in (max_idx, max_idx + 1)
        tag = f"f{idx}" + (" *WORST*" if is_worst else "")
        draw.text((x + 6, y + 4), tag, fill=("red" if is_worst else "yellow"), font=font_small)

    sheet_path = check_output_path(
        os.path.join("outputs", "video_proto",
                      f"{os.path.splitext(os.path.basename(path))[0]}_qa_contactsheet.png")
    )
    sheet.save(sheet_path)
    print(f"  saved QA contact sheet -> {sheet_path}")
    return diffs, mean_d, max_d, max_idx, sheet_path


def boomerang(frames):
    """Forward + reverse, dropping the duplicated first/last frame."""
    if len(frames) < 2:
        return frames
    return frames + frames[-2:0:-1]


def stretch_to_min_frames(frames, min_frames: int):
    """Nearest-neighbor temporal upsampling (frame repetition) to reach at
    least `min_frames`. No-op if already long enough."""
    n = len(frames)
    if n >= min_frames:
        return frames
    indices = [round(i * (n - 1) / (min_frames - 1)) for i in range(min_frames)]
    return [frames[i] for i in indices]


def process_clip(frames, loop_mode: str, target_frames: int):
    looped = boomerang(frames) if loop_mode == "boomerang" else frames
    return stretch_to_min_frames(looped, target_frames)


def crossfade(tail, head):
    """Alpha-blend two equal-length frame lists -- causes double-exposure
    ghosting when the two scenes differ a lot (see module docstring)."""
    n = len(tail)
    out = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        blended = (tail[i].astype(np.float32) * (1 - alpha)
                   + head[i].astype(np.float32) * alpha)
        out.append(blended.astype(np.uint8))
    return out


def fade_to_black(tail):
    n = len(tail)
    out = []
    for i in range(n):
        alpha = 1 - (i + 1) / (n + 1)
        out.append((tail[i].astype(np.float32) * alpha).astype(np.uint8))
    return out


def fade_from_black(head):
    n = len(head)
    out = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        out.append((head[i].astype(np.float32) * alpha).astype(np.uint8))
    return out


def assemble(clips, transition: str, n_fade: int):
    if transition == "cut" or n_fade <= 0:
        final = []
        for c in clips:
            final.extend(c)
        return final

    final = list(clips[0][:-n_fade])
    for i in range(1, len(clips)):
        prev_tail = clips[i - 1][-n_fade:]
        curr_head = clips[i][:n_fade]
        if transition == "crossfade":
            final.extend(crossfade(prev_tail, curr_head))
        elif transition == "fadeblack":
            final.extend(fade_to_black(prev_tail))
            final.extend(fade_from_black(curr_head))
        else:
            raise ValueError(f"Unknown transition: {transition}")
        final.extend(clips[i][n_fade:])
    return final


def check_minterpolate_available(ffmpeg_exe: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-h", "filter=minterpolate"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return False
    combined = (result.stdout + result.stderr).lower()
    return "unknown filter" not in combined and "minterpolate" in combined


def interpolate_video(in_path: str, out_path: str, target_fps: int, method: str):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if method == "minterpolate":
        vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    else:  # "dup" -- plain duplication, no motion estimation, always safe
        vf = f"fps={target_fps}"
    cmd = [ffmpeg_exe, "-y", "-i", in_path, "-vf", vf,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-clips", nargs="+", default=None, dest="qa_clips",
                     help="QA MODE: run the frame-diff check + dense contact sheet on each given "
                          "clip (native SVD output or assembled video) and exit, skipping assembly "
                          "entirely. Mutually exclusive with --clips/assembly.")
    ap.add_argument("--qa-threshold", type=float, default=30.0, dest="qa_threshold",
                     help="QA MODE: warn if max frame-to-frame diff exceeds this (default 30.0)")
    ap.add_argument("--qa-dense-every", type=int, default=4, dest="qa_dense_every",
                     help="QA MODE: contact sheet shows every Nth frame, plus the worst jump "
                          "explicitly even if off that grid (default 4)")
    ap.add_argument("--clips", nargs=3, default=None,
                     help="Paths to the 3 scene clips, in order (relative to project root unless absolute)")
    ap.add_argument("--out", default="outputs/video_proto/story_svd.mp4",
                     help="Output mp4 path, MUST be under outputs/video_proto/")
    ap.add_argument("--target-seconds-per-scene", type=float, default=4.0, dest="target_seconds",
                     help="Minimum seconds per scene; stretched via frame repetition only if "
                          "needed (default 4.0)")
    ap.add_argument("--loop-mode", choices=["slow", "boomerang"], default="slow",
                     help="'slow': forward-only, stretched via frame repetition (default, no "
                          "reverse motion). 'boomerang': forward+reverse loop (old default).")
    ap.add_argument("--transition", choices=["cut", "fadeblack", "crossfade"], default="crossfade",
                     help="'cut': hard concatenation. 'fadeblack': fade-out/fade-in through black "
                          "(no ghosting). 'crossfade': alpha-blend both scenes (old default, "
                          "causes double-exposure ghosting). Ignored (treated as 'cut') if "
                          "--crossfade-seconds is 0.")
    ap.add_argument("--crossfade-seconds", type=float, default=0.5, dest="crossfade_seconds",
                     help="Transition duration for fadeblack/crossfade (default 0.5)")
    ap.add_argument("--interpolate-fps", type=int, default=0, dest="interpolate_fps",
                     help="If >0, post-process the assembled (native ~7fps) video up to this fps")
    ap.add_argument("--interp-method", choices=["minterpolate", "dup"], default="minterpolate",
                     help="'minterpolate': motion-compensated interpolation (checked for "
                          "availability first, errors out if not present -- does not silently "
                          "fall back, so the caller can decide). 'dup': plain frame duplication, "
                          "always safe. Only used when --interpolate-fps > 0.")
    args = ap.parse_args()

    if args.qa_clips is not None:
        for p in args.qa_clips:
            p_resolved = resolve(p)
            if not os.path.isfile(p_resolved):
                raise SystemExit(f"Clip not found: {p_resolved}")
            qa_clip(p_resolved, warn_threshold=args.qa_threshold, dense_every=args.qa_dense_every)
        return

    if args.clips is None:
        raise SystemExit("--clips is required (or use --qa-clips for QA mode)")

    clip_paths = [resolve(p) for p in args.clips]
    for p in clip_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Clip not found: {p}")

    out_path = check_output_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    processed_clips = []
    fps_values = []
    for p in clip_paths:
        frames, fps = read_clip_frames(p)
        fps_values.append(fps)
        min_frames = round(args.target_seconds * fps)
        processed = process_clip(frames, args.loop_mode, min_frames)
        processed_clips.append(processed)
        print(f"{os.path.basename(p)}: {len(frames)} source frames @ {fps}fps "
              f"-> {len(processed)} after {args.loop_mode} processing "
              f"({len(processed) / fps:.2f}s)")

    if len(set(fps_values)) != 1:
        raise SystemExit(f"Clips have different fps values {fps_values} -- refusing to mix without resampling.")
    fps = fps_values[0]

    n_fade = round(args.crossfade_seconds * fps)
    final_frames = assemble(processed_clips, args.transition, n_fade)

    native_path = out_path
    if args.interpolate_fps > 0:
        native_path = out_path + ".native7fps.mp4"

    writer = imageio.get_writer(native_path, fps=fps, codec="libx264", quality=8)
    for frame in final_frames:
        writer.append_data(frame)
    writer.close()

    duration = len(final_frames) / fps
    h, w = final_frames[0].shape[:2]
    print(f"\nAssembled (native): {native_path}")
    print(f"Native duration: {duration:.2f}s ({len(final_frames)} frames @ {fps}fps), {w}x{h}")

    if args.interpolate_fps > 0:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        method = args.interp_method
        if method == "minterpolate":
            available = check_minterpolate_available(ffmpeg_exe)
            print(f"minterpolate filter available in this ffmpeg build: {available}")
            if not available:
                raise SystemExit(
                    "--interp-method minterpolate requested but the filter is not available "
                    "in this ffmpeg build. Re-run with --interp-method dup instead."
                )
        print(f"Interpolating to {args.interpolate_fps}fps via '{method}' ...")
        interpolate_video(native_path, out_path, args.interpolate_fps, method)
        os.remove(native_path)

        r = imageio.get_reader(out_path)
        meta = r.get_meta_data()
        n_frames_final = r.count_frames()
        r.close()
        print(f"\nWrote {out_path}")
        print(f"Duration: {meta.get('duration'):.2f}s ({n_frames_final} frames @ {meta.get('fps')}fps)")
        print(f"Resolution: {meta.get('size')}")
    else:
        print(f"\nWrote {out_path}")
        print(f"Duration: {duration:.2f}s ({len(final_frames)} frames @ {fps}fps)")
        print(f"Resolution: {w}x{h}")


if __name__ == "__main__":
    main()
