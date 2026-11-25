import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

import os, sys, uuid, re, gc, random, traceback

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOAD_DIR   = os.path.join(ROOT_DIR, "uploads")
OUTPUT_DIR   = os.path.join(ROOT_DIR, "outputs")

sys.path.append(BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from sd_pipeline import (
    generate, refine, load_pipeline,
    generate_story_scene_1, generate_story_scene_next,
    count_prompt_tokens, free_memory
)
from image_utils import preprocess_sketch, pil_to_base64, base64_to_pil, save_image
from config import MODEL_ID, STORY_SAME_SEED, STORY_BASE_STRENGTH

app = FastAPI(title="Sketch to Image AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda":   torch.cuda.is_available(),
        "gpu":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model":  MODEL_ID.split("/")[-1]
    }


# ── Pydantic Models 
class GenerateRequest(BaseModel):
    sketch_base64:    str
    prompt:           str
    negative_prompt:  str   = "(worst quality:2), (low quality:2), blurry, bad anatomy"
    steps:            int   = 35
    guidance_scale:   float = 9.0
    controlnet_scale: float = 0.9
    controlnet_type:  str   = "scribble"
    use_refinement:   bool  = True
    refine_strength:  float = 0.25
    refine_steps:     int   = 20
    width:            int   = 512
    height:           int   = 512
    seed:             int   = -1


class StoryScene(BaseModel):
    prompt:          str
    negative_prompt: Optional[str]   = None   # None = dùng global
    strength:        Optional[float] = None   # None = auto-calculate

    character: Optional[str] = None   
    action:    Optional[str] = None   
    location:  Optional[str] = None   
    mood:      Optional[str] = None   
    style:     Optional[str] = None   


class StoryRequest(BaseModel):
    scenes:              List[StoryScene]
    global_negative:     str   = "(worst quality:2), (low quality:2), blurry, bad anatomy, watermark"
    steps:               int   = 30
    guidance_scale:      float = 10.0
    width:               int   = 512
    height:              int   = 512
    seed:                int   = -1
    base_strength:       float = STORY_BASE_STRENGTH


# ── Story Helper Functions 

POSE_ACTION_WORDS = [
    "standing", "sitting", "walking", "running", "looking up",
    "looking down", "reading", "smiling", "laughing", "holding"
]

QUALITY_TAGS = [
    "realistic photo", "4k", "8k", "sharp focus", "detailed",
    "cinematic", "golden hour", "sunny day", "natural light",
    "studio lighting", "high quality", "masterpiece"
]

# Cùng danh sách tư thế, dùng để phát hiện đổi tư thế giữa 2 cảnh
POSE_CHANGE_WORDS = ["standing", "sitting", "walking", "running", "kneeling", "lying"]


def extract_character_anchor(prompt: str) -> str:
    anchor = prompt

    for tag in QUALITY_TAGS:
        anchor = anchor.replace(", " + tag, "").replace(tag + ", ", "")

    for word in POSE_ACTION_WORDS:
        anchor = re.sub(rf"[^,]*\b{word}\b[^,]*,?", "", anchor, flags=re.IGNORECASE)

    anchor = re.sub(r"\s*,\s*,+", ",", anchor)  # dọn dấu phẩy thừa

    if len(anchor) > 120:
        cut = anchor[:120].rfind(",")
        anchor = anchor[:cut] if cut > 0 else anchor[:120]

    return anchor.strip().strip(",")


def extract_style_anchor(prompt: str) -> str:
    prompt_lower = prompt.lower()
    matched = [tag for tag in QUALITY_TAGS if tag in prompt_lower]
    return ", ".join(matched)


def resolve_style_anchor(scenes: List[StoryScene]) -> str:
    text_anchor = extract_style_anchor(scenes[0].prompt)
    structured_style = (scenes[0].style or "").strip()

    if not structured_style:
        return text_anchor
    if not text_anchor:
        return structured_style
    if structured_style.lower() in text_anchor.lower():
        return text_anchor
    return f"{text_anchor}, {structured_style}"


def resolve_character_anchor(scenes: List[StoryScene]) -> str:
    text_anchor = extract_character_anchor(scenes[0].prompt)
    if text_anchor:
        return text_anchor
    return (scenes[0].character or "").strip()


def _idempotent_suffix(current: str, value: Optional[str], template: str) -> str:
    if not value or not value.strip():
        return ""
    value = value.strip()
    if value.lower() in current.lower():
        return ""
    return template.format(value)


def resolve_location_anchor(scenes: List[StoryScene], idx: int) -> str:
    anchor = ""
    for i in range(idx + 1):
        loc = scenes[i].location
        if loc and loc.strip():
            anchor = loc.strip()
    return anchor


def build_scene_prompt(scenes: List[StoryScene], idx: int) -> str:
    current = scenes[idx].prompt.strip()

    if idx == 0:
        structured_style = (scenes[0].style or "").strip()
        style_suffix = (f", {structured_style}"
                         if structured_style and structured_style.lower() not in current.lower()
                         else "")
        action_suffix = _idempotent_suffix(current, scenes[0].action, ", {}")
        mood_suffix    = _idempotent_suffix(current, scenes[0].mood, ", {} mood")
        return f"{current}{action_suffix}{mood_suffix}{style_suffix}"

    character_anchor = resolve_character_anchor(scenes)
    style_anchor      = resolve_style_anchor(scenes)
    style_suffix       = (f", {style_anchor}"
                           if style_anchor and style_anchor.lower() not in current.lower()
                           else "")

    location_anchor = resolve_location_anchor(scenes, idx)
    location_suffix  = (f", at the {location_anchor}"
                         if location_anchor and location_anchor.lower() not in current.lower()
                         else "")

    action_suffix = _idempotent_suffix(current, scenes[idx].action, ", {}")
    mood_suffix    = _idempotent_suffix(current, scenes[idx].mood, ", {} mood")

    if character_anchor and character_anchor.split(",")[0].lower() in current.lower():
        return f"{current}{action_suffix}{mood_suffix}, consistent style, same setting{style_suffix}{location_suffix}"

    if not character_anchor:
        return f"{current}{action_suffix}{mood_suffix}, consistent style, same setting{style_suffix}{location_suffix}"

    return f"{character_anchor}, {current}{action_suffix}{mood_suffix}, consistent lighting and style{style_suffix}{location_suffix}"


CLIP_WARN_THRESHOLD = 70 


def _trailing_suffixes_for_warning(scenes: List[StoryScene], idx: int) -> List[str]:
    current = scenes[idx].prompt.strip()
    present = []

    if idx == 0:
        if _idempotent_suffix(current, scenes[0].action, "x"):
            present.append("action")
        if _idempotent_suffix(current, scenes[0].mood, "x"):
            present.append("mood")
        structured_style = (scenes[0].style or "").strip()
        if structured_style and structured_style.lower() not in current.lower():
            present.append("style")
        return present

    if _idempotent_suffix(current, scenes[idx].action, "x"):
        present.append("action")
    if _idempotent_suffix(current, scenes[idx].mood, "x"):
        present.append("mood")
    present.append('fixed "consistent style/lighting" text')
    style_anchor = resolve_style_anchor(scenes)
    if style_anchor and style_anchor.lower() not in current.lower():
        present.append("style")
    location_anchor = resolve_location_anchor(scenes, idx)
    if location_anchor and location_anchor.lower() not in current.lower():
        present.append("location")
    return present


def _structured_field_changed(curr: Optional[str], prev: Optional[str]) -> Optional[bool]:
    if not curr or not prev:
        return None
    return curr.strip().lower() != prev.strip().lower()


def compare_scenes(scene: StoryScene, prev_scene: Optional[StoryScene]) -> dict:
    if prev_scene is None:
        return {"character_changed": None, "pose_changed": None,
                "location_changed": None, "mood_changed": None,
                "style_changed": None}
    return {
        "character_changed": _structured_field_changed(scene.character, prev_scene.character),
        "pose_changed":      _structured_field_changed(scene.action,    prev_scene.action),
        "location_changed":  _structured_field_changed(scene.location,  prev_scene.location),
        "mood_changed":      _structured_field_changed(scene.mood,      prev_scene.mood),
        "style_changed":     _structured_field_changed(scene.style,     prev_scene.style),
    }


def calculate_strength(
    base: float,
    scene_prompt: str,
    prev_prompt: str,
    scene_idx: int,
    total_scenes: int,
    scene: Optional[StoryScene] = None,
    prev_scene: Optional[StoryScene] = None,
) -> float:
    curr = scene_prompt.lower()
    prev = prev_prompt.lower() if prev_prompt else ""

    structured = compare_scenes(scene, prev_scene) if scene is not None else {}

    # ── Signal thay đổi MẠNH (cần strength cao) ──
    NEW_SUBJECT_PATTERNS = ["and a ", "and an ", "another ", "appears", "enters",
                            "second ", "new ", "joins", "arrives", "comes"]
    has_new_subject = any(p in curr for p in NEW_SUBJECT_PATTERNS)

    if structured.get("location_changed") is not None:
        location_changed = structured["location_changed"]
    else:
        LOCATION_WORDS = ["inside", "outside", "indoors", "outdoors", "forest",
                          "beach", "city", "room", "street", "field", "mountain",
                          "ocean", "lake", "park", "house", "building"]
        loc_curr = [w for w in LOCATION_WORDS if w in curr]
        loc_prev = [w for w in LOCATION_WORDS if w in prev]
        location_changed = bool(loc_curr) and bool(loc_prev) and loc_curr != loc_prev

    TIME_NOW   = ["night", "dark", "evening", "dusk", "midnight"]
    TIME_PREV  = ["day", "sunny", "bright", "morning", "noon", "afternoon", "dawn"]
    TIME_NOW2  = ["day", "sunny", "bright", "morning", "noon"]
    TIME_PREV2 = ["night", "dark", "evening", "dusk", "midnight"]
    time_changed = (any(t in curr for t in TIME_NOW)  and any(t in prev for t in TIME_PREV)) or \
                   (any(t in curr for t in TIME_NOW2) and any(t in prev for t in TIME_PREV2))

    def pose_changed(c, p):
        if structured.get("pose_changed") is not None:
            return structured["pose_changed"]
        curr_pose = [w for w in POSE_CHANGE_WORDS if w in c]
        prev_pose = [w for w in POSE_CHANGE_WORDS if w in p]
        return bool(curr_pose) and bool(prev_pose) and curr_pose != prev_pose

    # ── Signal thay đổi NHẸ (cần strength thấp) ──
    if structured.get("mood_changed") is not None:
        only_emotion = structured["mood_changed"] and \
                       not has_new_subject and not location_changed and \
                       not pose_changed(curr, prev)
    else:
        EMOTION_WORDS = ["smiling", "crying", "laughing", "angry", "happy",
                         "sad", "surprised", "scared", "excited", "calm"]
        only_emotion = any(w in curr for w in EMOTION_WORDS) and \
                       not has_new_subject and not location_changed and \
                       not pose_changed(curr, prev)

    curr_words = set(curr.split())
    prev_words = set(prev.split())
    common_ratio = len(curr_words & prev_words) / max(len(curr_words | prev_words), 1)
    minor_change = common_ratio > 0.5

    # ── Tính strength (thứ tự ưu tiên: subject > location > time > pose > emotion > minor)
    strength = base

    if has_new_subject:
        strength = min(base + 0.20, 0.85)
    elif location_changed:
        strength = min(base + 0.15, 0.82)
    elif time_changed:
        strength = min(base + 0.12, 0.80)
    elif pose_changed(curr, prev):
        strength = min(base + 0.18, 0.85)
    elif only_emotion:
        strength = max(base - 0.10, 0.25)
    elif minor_change:
        strength = max(base - 0.08, 0.28)

    if scene_idx == total_scenes - 1 and not has_new_subject:
        strength = max(strength - 0.05, 0.25)

    return round(strength, 2)


def resolve_negative(scene_neg: Optional[str], global_neg: str) -> str:
    """Dùng negative riêng nếu user nhập, không thì dùng global."""
    if scene_neg and scene_neg.strip():
        base_quality = "(worst quality:2), (low quality:2), blurry"
        return f"{base_quality}, {scene_neg.strip()}"
    return global_neg

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt không được để trống")
    try:
        sketch_pil       = base64_to_pil(req.sketch_base64)
        sketch_processed = preprocess_sketch(sketch_pil, req.width, req.height)
        sketch_id        = str(uuid.uuid4())[:8]
        save_image(sketch_processed, os.path.join(UPLOAD_DIR, f"{sketch_id}_sketch.jpg"))

        result = generate(
            sketch_image=sketch_processed,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance=req.guidance_scale,
            controlnet_scale=req.controlnet_scale,
            controlnet_type=req.controlnet_type,
            width=req.width, height=req.height,
            seed=req.seed
        )
        if req.use_refinement:
            result = refine(image=result, prompt=req.prompt,
                            negative_prompt=req.negative_prompt,
                            strength=req.refine_strength, steps=req.refine_steps)

        save_image(result, os.path.join(OUTPUT_DIR, f"{sketch_id}_output.jpg"))
        return {"success": True, "image_base64": pil_to_base64(result), "image_id": sketch_id}
    except Exception as e:
        raise HTTPException(500, f"Lỗi generate: {str(e)}")


@app.post("/generate-text")
async def generate_text_to_image(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt không được để trống")
    try:
        result = generate_story_scene_1(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width, height=req.height,
            seed=req.seed
        )
        image_id = str(uuid.uuid4())[:8]
        save_image(result, os.path.join(OUTPUT_DIR, f"{image_id}_t2i.jpg"))
        return {"success": True, "image_base64": pil_to_base64(result), "image_id": image_id}
    except Exception as e:
        raise HTTPException(500, f"Lỗi generate text: {str(e)}")


@app.post("/generate-story")
async def generate_story(req: StoryRequest):
    if not req.scenes:
        raise HTTPException(400, "Cần ít nhất 1 scene")
    if len(req.scenes) > 10:
        raise HTTPException(400, "Tối đa 10 scenes")

    results_b64  = []
    prev_image   = None
    first_image  = None  
                           
    total        = len(req.scenes)

    base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)

    for idx, scene in enumerate(req.scenes):
        try:
            final_prompt = build_scene_prompt(req.scenes, idx)
            final_neg    = resolve_negative(scene.negative_prompt, req.global_negative)
            token_count = count_prompt_tokens(final_prompt)
            if token_count is not None and token_count > CLIP_WARN_THRESHOLD:
                drop_order = ", then ".join(reversed(_trailing_suffixes_for_warning(req.scenes, idx))) \
                             or "(none — the scene's own prompt text itself is long)"
                print(f"WARNING: scene {idx+1}/{total} prompt is {token_count} tokens "
                      f"(CLIP limit ~77) — over the {CLIP_WARN_THRESHOLD}-token warning "
                      f"threshold. On overflow, truncation drops from the end first: {drop_order}.")
            if scene.strength is not None:
                strength = scene.strength
            else:
                strength = req.base_strength

            scene_seed = base_seed if STORY_SAME_SEED else base_seed + idx

            print(f"Scene {idx+1}/{total} | strength={strength:.2f}")
            print(f"  base_seed={base_seed} scene_seed={scene_seed}")
            print(f"  full prompt: {final_prompt}")

            if idx == 0:
                result = generate_story_scene_1(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    width=req.width, height=req.height,
                    seed=scene_seed
                )
                first_image = result
            else:
                result = generate_story_scene_next(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    input_image=prev_image,
                    reference_image=first_image,   # identity anchor (IP-Adapter)
                    strength=strength,
                    steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    seed=scene_seed
                )

            save_image(result, os.path.join(OUTPUT_DIR,
                       f"story_{str(uuid.uuid4())[:6]}_s{idx+1}.jpg"))
            results_b64.append(pil_to_base64(result))
            prev_image  = result

        except Exception as e:
            free_memory()
            traceback.print_exc()
            print(f"Scene {idx+1}/{total} failed: {e}")
            return {
                "success":         False,
                "scenes":          results_b64,
                "total":           len(results_b64),
                "requested_total": total,
                "failed_scene":    idx + 1,
                "seed_used":       base_seed,
                "error":           f"Lỗi ở cảnh {idx + 1}/{total}: {e}",
            }

    return {"success": True, "scenes": results_b64, "total": len(results_b64), "seed_used": base_seed}


@app.on_event("startup")
async def startup():
    print(f"Preloading pipeline — Model: {MODEL_ID}")
    load_pipeline()
    print("Server ready!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)