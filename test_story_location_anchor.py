"""
Run:
    python test_story_location_anchor.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import (
    StoryScene, build_scene_prompt, resolve_location_anchor,
    extract_character_anchor, extract_style_anchor,
)


class TestLocationAnchor(unittest.TestCase):

    def _scenes(self, *locations_and_prompts):
        """locations_and_prompts: list of (prompt, location_or_None) tuples."""
        return [StoryScene(prompt=p, location=loc) for p, loc in locations_and_prompts]

    #  1. Explicit location establishes anchor 
    def test_explicit_location_establishes_anchor(self):
        scenes = self._scenes(
            ("a woman standing in a room", "room"),
            ("the woman sitting", "room"),
        )
        self.assertEqual(resolve_location_anchor(scenes, 0), "room")
        self.assertEqual(resolve_location_anchor(scenes, 1), "room")

    #  2. Missing location inherits previous anchor 
    def test_missing_location_inherits_previous_anchor(self):
        scenes = self._scenes(
            ("a woman standing in a room", "room"),
            ("the woman smiling", None),   # no location on this scene
        )
        self.assertEqual(resolve_location_anchor(scenes, 1), "room")

    #  3. Explicit location change updates the anchor 
    def test_explicit_location_change_updates_anchor(self):
        scenes = self._scenes(
            ("a woman standing in a room", "room"),
            ("the woman on a beach", "beach"),
        )
        self.assertEqual(resolve_location_anchor(scenes, 1), "beach")

    #  4. Missing location AFTER a change inherits the NEW location 
    def test_missing_location_after_change_inherits_new_location(self):
        scenes = self._scenes(
            ("a woman standing in a room", "room"),
            ("the woman on a beach", "beach"),
            ("the woman talking with a friend", None),  # no location -> inherit "beach", not "room"
        )
        self.assertEqual(resolve_location_anchor(scenes, 2), "beach")

    #  Full regression scenario matching Phase 4H's actual story 
    def test_full_five_scene_story_matches_phase4h_expectation(self):
        scenes = self._scenes(
            ("a young woman with short red hair, wearing a blue jacket, standing in a "
             "cinematic forest scene, realistic photo, cinematic", "forest"),
            ("the same woman running through the forest", "forest"),
            ("the same woman walking along a beach at sunset", "beach"),
            ("the same woman sitting beside a red vintage car on the beach", "beach"),
            ("the same woman talking with a young man beside the red vintage car", None),
        )
        self.assertEqual(resolve_location_anchor(scenes, 0), "forest")
        self.assertEqual(resolve_location_anchor(scenes, 1), "forest")
        self.assertEqual(resolve_location_anchor(scenes, 2), "beach")
        self.assertEqual(resolve_location_anchor(scenes, 3), "beach")
        self.assertEqual(resolve_location_anchor(scenes, 4), "beach")  # inherited, scene 5 itself has no location

        final_prompt_5 = build_scene_prompt(scenes, 4)
        self.assertIn("beach", final_prompt_5.lower())
        print("Scene 5 final_prompt:", final_prompt_5)

    #  7. No duplicate injection when the scene's own text already
    #      contains the location 
    def test_no_duplicate_when_prompt_already_contains_location(self):
        scenes = self._scenes(
            ("a woman standing in a room", "room"),
            ("the woman walking along a beach at sunset", "beach"),  # prompt already says "beach"
        )
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(final_prompt.lower().count("beach"), 1)  # not duplicated

    # ── 5. Character/style anchors remain unchanged by this feature ───
    def test_character_and_style_anchors_unaffected(self):
        scene1_prompt = ("a young woman with short red hair, wearing a blue jacket, "
                          "standing in a room, cinematic, realistic photo")
        self.assertEqual(
            extract_character_anchor(scene1_prompt),
            "a young woman with short red hair, wearing a blue jacket",
        )
        self.assertIn("cinematic", extract_style_anchor(scene1_prompt))
        self.assertIn("realistic photo", extract_style_anchor(scene1_prompt))

        scenes = [
            StoryScene(prompt=scene1_prompt, character="young woman with short red hair, wearing a blue jacket",
                       action="standing", location="room", mood="calm", style="realistic photo, cinematic"),
            StoryScene(prompt="the same woman running", location="room"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        # Character anchor and style anchor still present exactly as before Phase 4I.
        self.assertIn("short red hair", final_prompt.lower())
        self.assertIn("blue jacket", final_prompt.lower())
        self.assertIn("cinematic", final_prompt.lower())
        self.assertIn("realistic photo", final_prompt.lower())

    # ── 6. Existing prompt-only (no structured fields at all) behavior
    #      remains fully backward compatible ──────────────────────────
    def test_prompt_only_story_unaffected_by_location_anchor(self):
        scenes = [
            StoryScene(prompt="a young woman standing in a room, cinematic, realistic photo"),
            StoryScene(prompt="the same woman walking along a beach at sunset"),
        ]
        # No structured location anywhere -> anchor must be empty, no injection.
        self.assertEqual(resolve_location_anchor(scenes, 0), "")
        self.assertEqual(resolve_location_anchor(scenes, 1), "")

        final_prompt = build_scene_prompt(scenes, 1)
        # Behavior identical to pre-Phase-4I: only character+style anchor
        # suffix logic applies, no "at the ..." location clause appended
        # beyond what the scene's own text already says.
        self.assertNotIn(", at the", final_prompt.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
