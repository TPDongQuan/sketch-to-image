import torch
import gc
import os
import sys
sys.path.append(os.path.dirname(__file__))

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from config import (
    MODEL_ID, CONTROLNET_ID, VAE_ID,
    IMG_SIZE, INFERENCE_STEPS, GUIDANCE_SCALE,
    IP_ADAPTER_ID, IP_ADAPTER_WEIGHT_NAME, IP_ADAPTER_SCALE,
    USE_IP_ADAPTER
)

CONTROLNET_MODELS = {
    "scribble": "lllyasviel/sd-controlnet-scribble",
    "mlsd":     "lllyasviel/sd-controlnet-mlsd",
    "canny":    "lllyasviel/sd-controlnet-canny",
}

_controlnet_pipes = {}
_img2img_pipe     = None
_t2i_pipe         = None

_ip_adapter_loaded = {
    "img2img": False,
    "t2i": False,
}

_shared_image_encoder = None



def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _get_shared_image_encoder():
    global _shared_image_encoder
    if _shared_image_encoder is not None:
        return _shared_image_encoder
    if not USE_IP_ADAPTER:
        return None
    try:
        print("Loading shared IP-Adapter image encoder (fp16, once)...")
        _shared_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_ID,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )
        print("Shared image encoder ready.")
    except Exception as e:
        print(f"Could not load shared image encoder (IP-Adapter will be skipped): {e}")
        _shared_image_encoder = None
    return _shared_image_encoder


def _load_vae():
    try:
        print(f"Loading VAE ({VAE_ID}, fp16, dedicated instance)...")
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
        print("VAE ready.")
        return vae
    except Exception as e:
        print(f"Could not load VAE ({VAE_ID}); this pipeline will keep "
              f"its own default/bundled VAE instead: {e}")
        return None


# ── Loaders

def load_pipeline(controlnet_type="scribble"):
    global _controlnet_pipes

    if controlnet_type in _controlnet_pipes:
        _activate_controlnet()
        return _controlnet_pipes[controlnet_type]

    free_memory()
    cn_model_id = CONTROLNET_MODELS.get(controlnet_type, CONTROLNET_ID)
    print(f"Loading ControlNet: {cn_model_id}")

    controlnet = ControlNetModel.from_pretrained(
        cn_model_id,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline_vae = _load_vae()  # this pipe's own dedicated instance
    if pipeline_vae is not None:
        pipe.vae = pipeline_vae  # must be set before .to("cuda") below
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()

    _controlnet_pipes[controlnet_type] = pipe
    print(f"Pipeline [{controlnet_type}] ready!")
    _activate_controlnet()
    return pipe


def _try_load_ip_adapter(pipe, pipe_key: str):
    if not USE_IP_ADAPTER:
        return False
    if _ip_adapter_loaded.get(pipe_key):
        return True

    shared_encoder = _get_shared_image_encoder()
    if shared_encoder is not None:
        # Attach BEFORE load_ip_adapter(): diffusers skips re-downloading
        # /re-instantiating the image encoder if pipe.image_encoder is
        # already set to a non-None value.
        pipe.image_encoder = shared_encoder

    try:
        pipe.load_ip_adapter(
            IP_ADAPTER_ID,
            subfolder="models",
            weight_name=IP_ADAPTER_WEIGHT_NAME
        )
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        if pipe.image_encoder is not None:
            pipe.image_encoder.to(pipe.unet.device)
        _ip_adapter_loaded[pipe_key] = True
        print(f"IP-Adapter loaded on [{pipe_key}] pipeline (shared encoder).")
        return True
    except Exception as e:
        print(f"IP-Adapter load failed on [{pipe_key}] (continuing without it): {e}")
        _reset_ip_adapter_state(pipe, pipe_key)
        _ip_adapter_loaded[pipe_key] = False
        return False


def _reset_ip_adapter_state(pipe, pipe_key: str):
    if hasattr(pipe, "unload_ip_adapter"):
        try:
            pipe.unload_ip_adapter()
            return
        except Exception as unload_err:
            print(f"Warning: unload_ip_adapter() failed on [{pipe_key}]: {unload_err}")
    try:
        pipe.unet.set_default_attn_processor()
    except Exception as reset_err:
        print(f"Warning: could not reset attn processors on [{pipe_key}]: {reset_err}")


def _run_pipe(pipe, kwargs: dict, pipe_key: str):
    if "ip_adapter_image" not in kwargs:
        return pipe(**kwargs)

    try:
        return pipe(**kwargs)
    except Exception as e:
        print(f"IP-Adapter call failed on [{pipe_key}] at inference time "
              f"(falling back to no IP-Adapter for the rest of this run): {e}")
        _ip_adapter_loaded[pipe_key] = False
        _reset_ip_adapter_state(pipe, pipe_key)
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("ip_adapter_image", None)
        return pipe(**fallback_kwargs)


def load_img2img_pipeline():
    global _img2img_pipe
    if _img2img_pipe is not None:
        _activate_img2img()
        return _img2img_pipe

    free_memory()
    print("Loading img2img refinement pipeline...")
    _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline_vae = _load_vae()  # this pipe's own dedicated instance
    if pipeline_vae is not None:
        _img2img_pipe.vae = pipeline_vae  # must be set before .to("cuda") below
    _img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(
        _img2img_pipe.scheduler.config
    )
    _img2img_pipe = _img2img_pipe.to("cuda")
    ip_adapter_active = _try_load_ip_adapter(_img2img_pipe, "img2img")
    if not ip_adapter_active:
        _img2img_pipe.enable_attention_slicing(1)
    _img2img_pipe.enable_vae_slicing()
    print("img2img pipeline ready!")
    _activate_img2img()
    return _img2img_pipe


def load_t2i_pipeline():
    """Plain text-to-image pipeline, used for story scene 1 and /generate-text."""
    global _t2i_pipe
    if _t2i_pipe is not None:
        _activate_t2i()
        return _t2i_pipe

    free_memory()
    print("Loading text-to-image pipeline...")
    _t2i_pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline_vae = _load_vae()  # this pipe's own dedicated instance
    if pipeline_vae is not None:
        _t2i_pipe.vae = pipeline_vae  # must be set before .to("cuda") below
    _t2i_pipe.scheduler = UniPCMultistepScheduler.from_config(
        _t2i_pipe.scheduler.config
    )
    _t2i_pipe = _t2i_pipe.to("cuda")
    _t2i_pipe.enable_attention_slicing(1)
    _t2i_pipe.enable_vae_slicing()
    print("Text-to-image pipeline ready!")
    _activate_t2i()
    return _t2i_pipe


def count_prompt_tokens(prompt: str):
    pipe = _t2i_pipe or _img2img_pipe
    if pipe is None or getattr(pipe, "tokenizer", None) is None:
        return None
    return len(pipe.tokenizer(prompt, truncation=False)["input_ids"])

def _move(pipe, device: str):
    if pipe is not None and pipe.unet.device.type != device:
        pipe.to(device)


def _activate_controlnet():
    _move(_t2i_pipe, "cpu")
    _move(_img2img_pipe, "cpu")
    free_memory()
    for key in _controlnet_pipes:
        _move(_controlnet_pipes[key], "cuda")
    free_memory()


def _activate_t2i():
    for key in _controlnet_pipes:
        _move(_controlnet_pipes[key], "cpu")
    _move(_img2img_pipe, "cpu")
    free_memory()
    _move(_t2i_pipe, "cuda")
    free_memory()


def _activate_img2img():
    for key in _controlnet_pipes:
        _move(_controlnet_pipes[key], "cpu")
    _move(_t2i_pipe, "cpu")
    free_memory()
    _move(_img2img_pipe, "cuda")
    free_memory()


# ── Sketch-to-image (existing behavior, unchanged) 

def generate(
    sketch_image: Image.Image,
    prompt: str,
    negative_prompt: str = "(worst quality:2), (low quality:2), blurry, bad anatomy",
    steps: int = None,
    guidance: float = None,
    controlnet_scale: float = 0.9,
    controlnet_type: str = "scribble",
    width: int = None,
    height: int = None,
    seed: int = -1
) -> Image.Image:

    pipe     = load_pipeline(controlnet_type)   # also activates controlnet group
    steps    = steps    or INFERENCE_STEPS
    guidance = guidance or GUIDANCE_SCALE
    width    = width    or IMG_SIZE
    height   = height   or IMG_SIZE

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    free_memory()

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch_image,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=controlnet_scale,
            width=width,
            height=height,
            generator=generator
        )

    image = result.images[0]
    free_memory()
    return image


def refine(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "(worst quality:2), (low quality:2), blurry, bad anatomy",
    strength: float = 0.25,
    steps: int = 20
) -> Image.Image:

    pipe = load_img2img_pipeline()   # activates img2img group, evicts controlnet/t2i
    free_memory()

    with torch.no_grad():
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=9.0
        )
        if _ip_adapter_loaded.get("img2img"):
            kwargs["ip_adapter_image"] = image
        result = _run_pipe(pipe, kwargs, "img2img")

    refined = result.images[0]
    # Next call to generate() will re-activate controlnet automatically.
    return refined


# ── Story mode (moved here from main.py)

def generate_story_scene_1(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int
):
    pipe = load_t2i_pipeline()   # activates t2i group, evicts controlnet/img2img
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

    free_memory()
    with torch.no_grad():
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        result = pipe(**kwargs).images[0]
    free_memory()
    return result


def generate_story_scene_next(
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    reference_image: Image.Image,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int
):
    pipe = load_img2img_pipeline()   # activates img2img group, evicts controlnet/t2i
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

    free_memory()
    with torch.no_grad():
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        if _ip_adapter_loaded.get("img2img") and reference_image is not None:
            kwargs["ip_adapter_image"] = reference_image
        result = _run_pipe(pipe, kwargs, "img2img").images[0]
    free_memory()
    return result