import os
from dotenv import load_dotenv

load_dotenv()

MODEL_ID        = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
CONTROLNET_ID   = os.getenv("CONTROLNET_ID", "lllyasviel/sd-controlnet-scribble")

VAE_ID          = os.getenv("VAE_ID", "stabilityai/sd-vae-ft-mse")
IMG_SIZE        = int(os.getenv("IMG_SIZE", 448))
INFERENCE_STEPS = int(os.getenv("INFERENCE_STEPS", 30))
GUIDANCE_SCALE  = float(os.getenv("GUIDANCE_SCALE", 7.5))
UPLOAD_DIR      = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "outputs")

USE_IP_ADAPTER        = os.getenv("USE_IP_ADAPTER", "true").lower() == "true"
IP_ADAPTER_ID         = os.getenv("IP_ADAPTER_ID", "h94/IP-Adapter")
IP_ADAPTER_WEIGHT_NAME = os.getenv("IP_ADAPTER_WEIGHT_NAME", "ip-adapter_sd15.bin")

IP_ADAPTER_SCALE      = float(os.getenv("IP_ADAPTER_SCALE", 0.1))

STORY_SAME_SEED      = os.getenv("STORY_SAME_SEED", "true").lower() == "true"

STORY_BASE_STRENGTH  = float(os.getenv("STORY_BASE_STRENGTH", 0.90))

# Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)