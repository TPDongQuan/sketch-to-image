import torch
import time

# Fix cuDNN engine error
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

from diffusers import StableDiffusionPipeline

print("Loading SD 1.5 model...")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing(1)

print("Generating ảnh test...")
start = time.time()

with torch.no_grad():
    image = pipe(
        "a beautiful mountain landscape, realistic photo",
        num_inference_steps=20,
        width=512,
        height=512,
        guidance_scale=7.5
    ).images[0]

image.save("test_output.jpg")
elapsed = time.time() - start
print(f"Xong! Mat {elapsed:.1f} giay")
print("Xem anh tai: test_output.jpg")
