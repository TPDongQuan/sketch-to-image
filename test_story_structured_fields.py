"""
Focused tests for Story Mode Phase 2B: structured scene fields
(backend/main.py: StoryScene.character/action/location/mood/style,
compare_scenes(), and calculate_strength()'s structured wiring).

Pure string/list/dict logic, no torch/GPU dependency (same caveat as
test_story_anchor.py: importing backend/main.py still pulls in the full
FastAPI/sd_pipeline import chain, a pre-existing structural coupling not
addressed by this change).

Run:
    python test_story_structured_fields.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import StoryScene, compare_scenes, calculate_strength


class TestStructuredSceneComparison(unittest.TestCase):

    # ── Case 1 — action change (with location change too) ──────────
    def test_case1_action_and_location_change_character_and_style_unchanged(self):
        scene1 = StoryScene(
            prompt="a young woman with short red hair, wearing a blue jacket, "
                   "standing in a room, cinematic, realistic photo",
            character="young woman, short red hair, blue jacket",
            action="standing", location="room", mood="calm",
            style="realistic photo, cinematic",
        )
        scene2 = StoryScene(
            prompt="the same woman running through the forest",
            character="young woman, short red hair, blue jacket",
            action="running", location="forest", mood="energetic",
            style="realistic photo, cinematic",
        )
        cmp = compare_scenes(scene2, scene1)
        self.assertTrue(cmp["pose_changed"])
        self.assertTrue(cmp["location_changed"])
        self.assertFalse(cmp["character_changed"])
        self.assertFalse(cmp["style_changed"])
        self.assertTrue(cmp["mood_changed"])

        strength = calculate_strength(
            base=0.55, scene_prompt=scene2.prompt, prev_prompt=scene1.prompt,
            scene_idx=1, total_scenes=2, scene=scene2, prev_scene=scene1,
        )
        # location_changed outranks pose_changed in priority: min(0.55+0.15,0.82)
        # = 0.70, then the pre-existing "last scene" rule (-0.05) applies.
        self.assertAlmostEqual(strength, 0.65, places=2)

    # ── Case 2 — action word NOT in the old keyword lists ───────────
    def test_case2_action_word_absent_from_old_keyword_lists(self):
        # "dancing"/"cooking" are absent from BOTH POSE_ACTION_WORDS and
        # POSE_CHANGE_WORDS in main.py — this is the exact brittleness
        # the task describes.
        prose_prev = "a young woman with short red hair dancing in a room"
        prose_curr = "the woman cooking in the room"

        # Free-text-only path (no structured fields at all, matching
        # calling code that never adopted Phase 2B): the keyword scan has
        # NO word from any list to react to (no new-subject pattern, same
        # "room" so no location change, no listed pose word, no emotion
        # word, and low prose word-overlap so minor_change doesn't fire
        # either) — it falls all the way through to the unmodified base
        # strength. This is the pre-existing, still-present limitation
        # for callers who don't use structured fields.
        strength_prose_only = calculate_strength(
            base=0.55, scene_prompt=prose_curr, prev_prompt=prose_prev,
            scene_idx=1, total_scenes=5,
        )
        self.assertEqual(strength_prose_only, 0.55)

        # Structured-field path: same underlying actions, provided
        # explicitly — detected correctly regardless of vocabulary.
        scene1 = StoryScene(prompt=prose_prev, action="dancing")
        scene2 = StoryScene(prompt=prose_curr, action="cooking")
        cmp = compare_scenes(scene2, scene1)
        self.assertTrue(cmp["pose_changed"])

        strength_structured = calculate_strength(
            base=0.55, scene_prompt=scene2.prompt, prev_prompt=scene1.prompt,
            scene_idx=1, total_scenes=5, scene=scene2, prev_scene=scene1,
        )
        # min(0.55+0.18, 0.85) = 0.73; scene_idx(1) != total_scenes-1(4),
        # so the "last scene" adjustment does not apply here.
        self.assertAlmostEqual(strength_structured, 0.73, places=2)
        self.assertGreater(strength_structured, strength_prose_only)

    # ── Case 3 — location change WITHOUT action change ──────────────
    def test_case3_location_change_only(self):
        scene1 = StoryScene(prompt="a woman standing in a room", action="standing", location="room")
        scene2 = StoryScene(prompt="the woman standing in a forest", action="standing", location="forest")

        cmp = compare_scenes(scene2, scene1)
        self.assertTrue(cmp["location_changed"])
        self.assertFalse(cmp["pose_changed"])

        strength = calculate_strength(
            base=0.55, scene_prompt=scene2.prompt, prev_prompt=scene1.prompt,
            scene_idx=1, total_scenes=5, scene=scene2, prev_scene=scene1,
        )
        self.assertAlmostEqual(strength, 0.70, places=2)  # min(0.55+0.15, 0.82)

    # ── Case 4 — same scene characteristics, no false major-change ──
    def test_case4_identical_characteristics_not_flagged_as_major_change(self):
        common_kwargs = dict(character="a man in a green coat", action="standing",
                              location="park", mood="calm", style="cinematic")
        scene1 = StoryScene(prompt="a man in a green coat standing in a park, cinematic", **common_kwargs)
        scene2 = StoryScene(prompt="a man in a green coat standing in a park, cinematic", **common_kwargs)

        cmp = compare_scenes(scene2, scene1)
        self.assertFalse(cmp["character_changed"])
        self.assertFalse(cmp["pose_changed"])
        self.assertFalse(cmp["location_changed"])
        self.assertFalse(cmp["mood_changed"])
        self.assertFalse(cmp["style_changed"])

        strength = calculate_strength(
            base=0.55, scene_prompt=scene2.prompt, prev_prompt=scene1.prompt,
            scene_idx=1, total_scenes=5, scene=scene2, prev_scene=scene1,
        )
        # No major-change branch (subject/location/time/pose) should
        # fire — strength must stay at or below base, never boosted.
        self.assertLessEqual(strength, 0.55)

    # ── Backward compatibility ───────────────────────────────────────
    def test_no_structured_fields_matches_pre_phase2b_behavior(self):
        # Omitting scene/prev_scene entirely (the exact old call
        # signature) must behave identically to before this change —
        # pure keyword-based detection, unaffected by this phase.
        strength_old_signature = calculate_strength(
            base=0.55, scene_prompt="the woman sitting on a bench",
            prev_prompt="a woman standing in a park", scene_idx=1, total_scenes=5,
        )
        # StoryScene with no structured fields set (all None) must give
        # the identical result via compare_scenes()'s all-None fallback.
        scene1 = StoryScene(prompt="a woman standing in a park")
        scene2 = StoryScene(prompt="the woman sitting on a bench")
        cmp = compare_scenes(scene2, scene1)
        self.assertIsNone(cmp["pose_changed"])
        self.assertIsNone(cmp["location_changed"])

        strength_with_empty_structured = calculate_strength(
            base=0.55, scene_prompt=scene2.prompt, prev_prompt=scene1.prompt,
            scene_idx=1, total_scenes=5, scene=scene2, prev_scene=scene1,
        )
        self.assertEqual(strength_old_signature, strength_with_empty_structured)


if __name__ == "__main__":
    unittest.main(verbosity=2)
