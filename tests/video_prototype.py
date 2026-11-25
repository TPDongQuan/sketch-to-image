
import argparse
import gc
import os
import signal
import subprocess
import sys
import time

import numpy as np
import imageio.v2 as imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "video_proto")
MIN_FREE_VRAM_GB = 9.0


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def export_frames_to_mp4(frames, out_path: str, fps: int):

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
        writer.append_data(arr)
    writer.close()


class TimeoutError_(Exception):
    pass


class ProjectedOverrunError(Exception):
    pass

STAGE1_DECODE_S_PER_FRAME = 4.8 / 6


def _alarm_handler(signum, frame):
    raise TimeoutError_("Wall-clock timeout reached (--max-seconds)")


# ── GPU safety checks (run BEFORE importing torch/diffusers, so a bad
#    GPU state is caught before any model loading even starts) 

def _nvidia_smi(args):
    return subprocess.run(
        ["nvidia-smi"] + args, capture_output=True, text=True, timeout=10, check=True
    ).stdout.strip()


def check_gpu_safe(min_free_gb: float = MIN_FREE_VRAM_GB):
    try:
        free_mib = int(_nvidia_smi(
            ["--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).splitlines()[0])
    except Exception as e:
        print(f"Could not query nvidia-smi for free VRAM -- aborting as a precaution: {e}")
        sys.exit(1)

    free_gb = free_mib / 1024
    print(f"GPU free VRAM: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        print(f"ABORT: free VRAM ({free_gb:.2f} GB) is below the {min_free_gb} GB safety threshold.")
        sys.exit(1)

    try:
        apps_out = _nvidia_smi(
            ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        )
    except Exception as e:
        print(f"Could not query nvidia-smi for compute processes -- aborting as a precaution: {e}")
        sys.exit(1)

    my_uid = os.getuid()
    foreign_found = False
    for line in apps_out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str = line.split(",")[0].strip()
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            owner_uid = os.stat(f"/proc/{pid}").st_uid
        except FileNotFoundError:
            continue  # process exited between the query and this check
        if owner_uid != my_uid:
            print(f"ABORT: PID {pid} is an existing GPU compute process owned by a "
                  f"different user (uid {owner_uid}).")
            foreign_found = True
        else:
            print(f"NOTE: PID {pid} is an existing GPU compute process owned by YOU -- "
                  f"make sure nothing else of yours is still running before proceeding.")
    if foreign_found:
        sys.exit(1)

    print("GPU safety checks passed.")


def check_output_path(out_path: str) -> str:
    out_path = resolve(out_path)
    out_path_norm = os.path.normpath(out_path)
    root_norm = os.path.normpath(OUTPUT_ROOT)
    if os.path.commonpath([out_path_norm, root_norm]) != root_norm:
        print(f"ABORT: --out must be under {OUTPUT_ROOT}, got {out_path_norm}")
        sys.exit(1)
    return out_path_norm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Source image (relative to project root unless absolute)")
    ap.add_argument("--out", required=True, help="Output mp4 path, MUST be under outputs/video_proto/")
    ap.add_argument("--seed", type=int, default=-1, help="Seed; -1 = random")
    ap.add_argument("--frames", type=int, default=25, help="num_frames (SVD-XT default/trained: 25)")
    ap.add_argument("--steps", type=int, default=25, dest="steps",
                     help="num_inference_steps -- denoising steps, NOT related to --frames "
                          "(pipeline default: 25; this is the dominant cost driver, ~8.6s/step "
                          "measured on this GPU -- reduce this for a cheap smoke test)")
    ap.add_argument("--fps", type=int, default=7, help="fps conditioning (pipeline default: 7)")
    ap.add_argument("--motion-bucket", type=int, default=127, dest="motion_bucket",
                     help="motion_bucket_id -- higher = more motion (pipeline default: 127)")
    ap.add_argument("--noise-aug", type=float, default=0.02, dest="noise_aug",
                     help="noise_aug_strength -- higher = more deviation from source image (default: 0.02)")
    ap.add_argument("--decode-chunk", type=int, default=1, dest="decode_chunk",
                     help="decode_chunk_size -- lower = less VRAM at VAE-decode time, slower (default: 1)")
    ap.add_argument("--max-seconds", type=int, default=300, dest="max_seconds",
                     help="Wall-clock timeout in seconds (best-effort, SIGALRM-based)")
    ap.add_argument("--save-frames", type=int, default=0, dest="save_frames",
                     help="If >0, save this many evenly-spaced frames as PNGs next to --out "
                          "(e.g. for inspecting identity drift)")
    args = ap.parse_args()

    image_path = resolve(args.image)
    if not os.path.isfile(image_path):
        print(f"ABORT: image not found: {image_path}")
        sys.exit(1)
    out_path = check_output_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    check_gpu_safe()

    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    native_size = img.size  # (width, height) -- see module docstring: no resize/pad
    print(f"Input image: {image_path} ({native_size[0]}x{native_size[1]}, used as-is)")

    print("Loading StableVideoDiffusionPipeline (fp16, variant='fp16') ...")
    t_load_start = time.time()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()  # NOTE: do not also call pipe.to("cuda")
    load_elapsed = time.time() - t_load_start
    print(f"Pipeline loaded in {load_elapsed:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed >= 0 else None

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(args.max_seconds)

    t_start = time.time()
    step_timing = {"last_step_end": None, "steps_seen": 0, "step_durations": []}

    def _on_step_end(pipe_obj, step_index, timestep, callback_kwargs):
        now = time.time()
        prev = step_timing["last_step_end"] if step_timing["last_step_end"] is not None else t_start
        step_duration = now - prev
        step_timing["step_durations"].append(step_duration)
        step_timing["last_step_end"] = now
        step_timing["steps_seen"] = step_index + 1
        elapsed = now - t_start
        print(f"  step {step_timing['steps_seen']}/{args.steps}: {step_duration:.2f}s "
              f"(elapsed {elapsed:.1f}s)", flush=True)

        if step_timing["steps_seen"] == 3:
            avg_step = sum(step_timing["step_durations"]) / len(step_timing["step_durations"])
            projected_denoise = avg_step * args.steps
            projected_decode = STAGE1_DECODE_S_PER_FRAME * args.frames
            projected_total = projected_denoise + projected_decode
            print(f"  projection after 3 steps: avg {avg_step:.2f}s/step -> "
                  f"~{projected_denoise:.0f}s denoise + ~{projected_decode:.0f}s decode "
                  f"= ~{projected_total:.0f}s vs --max-seconds {args.max_seconds}s", flush=True)
            if projected_total > args.max_seconds:
                raise ProjectedOverrunError(
                    f"Projected total (~{projected_total:.0f}s) exceeds --max-seconds "
                    f"({args.max_seconds}s), based on the first 3 steps averaging "
                    f"{avg_step:.2f}s/step. Aborting now rather than burning the "
                    f"remaining {args.steps - 3} steps on a run that can't finish in time."
                )
        return callback_kwargs

    try:
        result = pipe(
            img,
            height=native_size[1],
            width=native_size[0],
            num_frames=args.frames,
            num_inference_steps=args.steps,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket,
            noise_aug_strength=args.noise_aug,
            decode_chunk_size=args.decode_chunk,
            generator=generator,
            callback_on_step_end=_on_step_end,
        )
        frames = result.frames[0]
        t_pipe_end = time.time()
        signal.alarm(0)

        export_frames_to_mp4(frames, out_path, args.fps)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        total_elapsed = t_pipe_end - t_start
        if step_timing["last_step_end"] is not None:
            denoise_s = step_timing["last_step_end"] - t_start
            decode_s = t_pipe_end - step_timing["last_step_end"]
        else:
            denoise_s, decode_s = None, None  # callback never fired -- report total only

        print(f"Wrote {out_path}")
        print(f"Load time: {load_elapsed:.1f}s (NOT counted against --max-seconds)")
        if denoise_s is not None:
            print(f"Denoise time ({step_timing['steps_seen']}/{args.steps} steps): {denoise_s:.1f}s")
            print(f"Decode/post-process time: {decode_s:.1f}s")
        print(f"Total pipe() time (counted against --max-seconds): {total_elapsed:.1f}s")
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

        if args.save_frames > 0:
            n = len(frames)
            k = min(args.save_frames, n)
            indices = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
            base, _ = os.path.splitext(out_path)
            for rank, idx in enumerate(indices):
                frame_path = f"{base}_frame{rank}_idx{idx}.png"
                frames[idx].save(frame_path)
                print(f"Saved frame {idx}/{n - 1} -> {frame_path}")

    except (TimeoutError_, ProjectedOverrunError) as e:
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"ABORT: {e} (elapsed {time.time() - t_start:.1f}s since pipe() started, "
              f"{step_timing['steps_seen']}/{args.steps} denoise steps completed, "
              f"peak VRAM so far: {peak_vram_gb:.2f} GB)")
        sys.exit(1)
    finally:
        signal.alarm(0)
        try:
            del pipe
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory freed.")


if __name__ == "__main__":
    main()
