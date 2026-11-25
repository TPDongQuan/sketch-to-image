import cv2
import numpy as np
from PIL import Image
import io
import base64
from config import IMG_SIZE


def preprocess_sketch(image: Image.Image, width: int = None, height: int = None) -> Image.Image:
    w = width  or IMG_SIZE
    h = height or IMG_SIZE

    image = image.resize((w, h), Image.LANCZOS)
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    mean_brightness = np.mean(gray)
    if mean_brightness < 127:
        gray = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def save_image(image: Image.Image, path: str):
    image.save(path, format="JPEG", quality=90)
    print(f"Saved: {path}")