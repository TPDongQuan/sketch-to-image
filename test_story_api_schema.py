"""
Focused tests for Phase 2B.1: structured scene fields exposed through the
actual /generate-story request schema (backend/main.py: StoryRequest,
StoryScene). Unlike test_story_structured_fields.py (which constructs
StoryScene objects directly in Python), these tests build raw dicts —
exactly what arrives over HTTP as JSON — and parse them via
StoryRequest(**payload), the same pydantic validation FastAPI runs on a
real request body. This is what actually proves the API accepts these
fields, not just that the underlying Python objects support them.

Run:
    python test_story_api_schema.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import StoryRequest, build_scene_prompt, calculate_strength, compare_scenes


# ── 1/2: API accepts structured fields; prompt-only stays backward compatible ──
class TestApiSchemaAcceptsStructuredFields(unittest.TestCase):

    def test_prompt_only_payload_still_parses_exactly_as_before(self):
        payload = {
            "scenes": [
                {"prompt": "a young woman standing in a room, cinematic, realistic photo"},
                {"prompt": "the same woman running through the forest"},
            ]
        }
        req = StoryRequest(**payload)
        self.assertEqual(len(req.scenes), 2)
        for s in req.scenes:
            self.assertIsNone(s.character)
            self.assertIsNone(s.action)
            self.assertIsNone(s.location)
            self.assertIsNone(s.mood)
            self.assertIsNone(s.style)

    def test_full_structured_payload_parses_and_fields_are_accessible(self):
        payload = {
            "scenes": [
                {
                    "prompt": "a young woman with short red hair, wearing a blue jacket, "
                              "standing in a room, cinematic, realistic photo",
                    "character": "young woman, short red hair, blue jacket",
                    "action": "standing", "location": "room", "mood": "calm",
                    "style": "realistic photo, cinematic",
                },
                {
                    "prompt": "the same woman running through the forest",
                    "character": "young woman, short red hair, blue jacket",
                    "action": "running", "location": "forest", "mood": "energetic",
                    "style": "realistic photo, cinematic",
                },
            ]
        }
        req = StoryRequest(**payload)
        self.assertEqual(req.scenes[1].action, "running")
        self.assertEqual(req.scenes[1].location, "forest")
        self.assertEqual(req.scenes[1].mood, "energetic")

    # ── 4: partial structured fields ────────────────────────────────
    def test_partial_structured_payload_no_crash_correct_none_handling(self):
        payload = {
            "scenes": [
                {"prompt": "a man in a green coat standing in a park",
                 "character": "man in a green coat", "action": "standing", "style": "cinematic"},
                {"prompt": "the man sitting on a bench",
                 "character": "man in a green coat", "action": "sitting", "style": "cinematic"},
                # location and mood intentionally omitted on both scenes
            ]
        }
        req = StoryRequest(**payload)
        s0, s1 = req.scenes
        self.assertIsNone(s0.location)
        self.assertIsNone(s0.mood)

        cmp = compare_scenes(s1, s0)
        self.assertFalse(cmp["character_changed"])   # supplied, same -> False
        self.assertTrue(cmp["pose_changed"])          # supplied, differs -> True
        self.assertIsNone(cmp["location_changed"])    # omitted -> None (fallback signal)
        self.assertIsNone(cmp["mood_changed"])        # omitted -> None (fallback signal)

        # Must not crash when fed into calculate_strength with omitted fields.
        strength = calculate_strength(
            base=0.55, scene_prompt=s1.prompt, prev_prompt=s0.prompt,
            scene_idx=1, total_scenes=2, scene=s1, prev_scene=s0,
        )
        self.assertIsInstance(strength, float)


# ── 5/6/7: structured action / location / mood changes detected via the real schema ──
class TestStructuredChangeDetectionViaRealSchema(unittest.TestCase):

    def _build(self, scene0_extra, scene1_extra):
        payload = {"scenes": [
            {"prompt": "scene one text", **scene0_extra},
            {"prompt": "scene two text", **scene1_extra},
        ]}
        return StoryRequest(**payload).scenes

    def test_action_change_detected(self):
        s0, s1 = self._build({"action": "standing"}, {"action": "dancing"})
        self.assertTrue(compare_scenes(s1, s0)["pose_changed"])

    def test_location_change_detected(self):
        s0, s1 = self._build({"location": "room"}, {"location": "forest"})
        self.assertTrue(compare_scenes(s1, s0)["location_changed"])

    def test_mood_change_detected(self):
        s0, s1 = self._build({"mood": "calm"}, {"mood": "energetic"})
        self.assertTrue(compare_scenes(s1, s0)["mood_changed"])


# ── 8: Phase 2A style anchor still works with the real schema ──────
class TestPhase2AStillWorksThroughRealSchema(unittest.TestCase):

    def test_style_anchor_survives_through_structured_payload(self):
        payload = {"scenes": [
            {"prompt": "a young woman with short red hair, wearing a blue jacket, "
                       "realistic photo, cinematic"},
            {"prompt": "the same woman running through the forest"},
        ]}
        scenes = StoryRequest(**payload).scenes
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertIn("realistic photo", final_prompt.lower())
        self.assertIn("cinematic", final_prompt.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
