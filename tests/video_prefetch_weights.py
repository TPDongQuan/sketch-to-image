"""
Pre-fetch ONLY the fp16 SVD-XT weights into the shared HF cache.
No GPU, no torch import, no model loading -- pure download via
huggingface_hub.snapshot_download(allow_patterns=...).

Fetches: unet/image_encoder/vae fp16 safetensors + every small
config/scheduler/feature-extractor json + LICENSE.md. Skips the fp32
duplicate safetensors and the standalone svd_xt*.safetensors
checkpoint files (not used by the diffusers folder-based loader).

Run with no timeout wrapper -- this step is network-bound, not
GPU-bound, so a tight deadline doesn't apply here (unlike inference).

Usage:
    conda run --no-capture-output -n video-proto python tests/video_prefetch_weights.py
"""
from huggingface_hub import snapshot_download

REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

ALLOW_PATTERNS = [
    "*.json",            # model_index.json, unet/config.json, vae/config.json,
                          # image_encoder/config.json, scheduler/scheduler_config.json,
                          # feature_extractor/preprocessor_config.json
    "*fp16.safetensors",  # unet/diffusion_pytorch_model.fp16.safetensors (~3.05GB)
                          # image_encoder/model.fp16.safetensors (~1.26GB)
                          # vae/diffusion_pytorch_model.fp16.safetensors (~0.20GB)
    "LICENSE.md",
]

if __name__ == "__main__":
    print(f"Pre-fetching {REPO_ID} (fp16 variant only, ~4.5GB expected) ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=ALLOW_PATTERNS,
    )
    print(f"\nDone. Cached at: {path}")

    import os
    print("\nFiles present after fetch:")
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            if os.path.islink(full):
                full = os.path.realpath(full)  # HF cache uses symlinks to blobs
            size_mb = os.path.getsize(full) / 1e6
            rel = os.path.relpath(os.path.join(root, f), path)
            print(f"  {size_mb:9.1f} MB   {rel}")
