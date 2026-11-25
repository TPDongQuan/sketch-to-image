import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

import sys
sys.path.append("backend")

from PIL import Image, ImageDraw
from sd_pipeline import generate
from image_utils import preprocess_sketch, save_image
import time

# Tạo sketch giả đơn giản để test — vẽ hình nhà
print("Tao sketch test...")
img = Image.new("RGB", (512, 512), color="white")
draw = ImageDraw.Draw(img)

# Vẽ hình nhà đơn giản
draw.rectangle([150, 250, 350, 450], outline="black", width=3)  # thân nhà
draw.polygon([(120, 250), (250, 120), (380, 250)], outline="black", width=3)  # mái
draw.rectangle([210, 350, 290, 450], outline="black", width=2)  # cửa

img.save("test_sketch.jpg")
print("Da tao sketch: test_sketch.jpg")

# Preprocess
sketch = preprocess_sketch(img)
sketch.save("test_sketch_processed.jpg")
print("Da preprocess sketch")

# Generate
print("Generating tu sketch... (co the mat 10-30 giay)")
start = time.time()

result = generate(
    sketch_image=sketch,
    prompt="a beautiful house, realistic photo, detailed, high quality",
    negative_prompt="blurry, low quality, cartoon",
    steps=20,
    controlnet_scale=0.8
)

result.save("test_controlnet_output.jpg")
elapsed = time.time() - start
print(f"Xong! Mat {elapsed:.1f} giay")
print("Ket qua: test_controlnet_output.jpg")
