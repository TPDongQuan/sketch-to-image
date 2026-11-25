"""
Run:
    python test_story_anchor.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import (
    extract_character_anchor, extract_style_anchor, build_scene_prompt,
    calculate_strength, StoryScene,
)

# Scene 1 establishes: character = "a young woman with short red hair,
# wearing a blue jacket", pose = "standing", location = "a room",
# style = "cinematic, realistic photo".
SCENE_1_PROMPT = (
    "a young woman with short red hair, wearing a blue jacket, "
    "standing in a room, cinematic, realistic photo"
)


class TestCharacterStyleAnchorSeparation(unittest.TestCase):

    def test_style_anchor_extracts_only_style_tags_present_in_prompt(self):
        style = extract_style_anchor(SCENE_1_PROMPT)
        self.assertIn("cinematic", style)
        self.assertIn("realistic photo", style)
        # A tag NOT present in the prompt must not appear.
        self.assertNotIn("golden hour", style)

    def test_character_anchor_excludes_style_and_pose_words(self):
        anchor = extract_character_anchor(SCENE_1_PROMPT)
        self.assertNotIn("cinematic", anchor.lower())
        self.assertNotIn("realistic photo", anchor.lower())
        self.assertNotIn("standing", anchor.lower())
        # Character-identifying content must survive.
        self.assertIn("red hair", anchor.lower())
        self.assertIn("blue jacket", anchor.lower())

    # ── Test 1: same character, changed pose AND location ──────────
    def test_scene_with_pose_and_location_change_keeps_style(self):
        scenes = [
            StoryScene(prompt=SCENE_1_PROMPT),
            StoryScene(prompt="the woman running through the forest"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)

        # Style must be present even though scene 2's own text omits it.
        self.assertIn("cinematic", final_prompt.lower())
        self.assertIn("realistic photo", final_prompt.lower())
        # Character info must still be present (not silently dropped).
        self.assertIn("red hair", final_prompt.lower())
        # The scene's own new action/location text must still be there.
        self.assertIn("forest", final_prompt.lower())

        strength = calculate_strength(
            base=0.55, scene_prompt=scenes[1].prompt, prev_prompt=scenes[0].prompt,
            scene_idx=1, total_scenes=2,
        )
        self.assertAlmostEqual(strength, 0.65, places=2)

    # ── Test 2: same character, no meaningful scene change ──────────
    def test_scene_with_no_change_keeps_style_not_duplicated(self):
        scenes = [
            StoryScene(prompt=SCENE_1_PROMPT),
            # Minor variation only (emotion), same location, no pose word.
            StoryScene(prompt="the woman smiling in the room"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)

        self.assertIn("cinematic", final_prompt.lower())
        self.assertIn("realistic photo", final_prompt.lower())
        self.assertEqual(final_prompt.lower().count("cinematic"), 1)
        self.assertEqual(final_prompt.lower().count("realistic photo"), 1)

        strength = calculate_strength(
            base=0.55, scene_prompt=scenes[1].prompt, prev_prompt=scenes[0].prompt,
            scene_idx=1, total_scenes=2,
        )
        self.assertAlmostEqual(strength, 0.40, places=2)

    def test_scene_1_prompt_is_unchanged(self):
        scenes = [StoryScene(prompt=SCENE_1_PROMPT)]
        self.assertEqual(build_scene_prompt(scenes, 0), SCENE_1_PROMPT)

    def test_no_style_tags_in_scene_1_yields_empty_style_anchor(self):
        scenes = [
            StoryScene(prompt="a man wearing a green coat, standing in a park"),
            StoryScene(prompt="the man sitting on a bench"),
        ]
        style = extract_style_anchor(scenes[0].prompt)
        self.assertEqual(style, "")
        final_prompt = build_scene_prompt(scenes, 1)
        # No trailing/dangling comma artifact from an empty style_suffix.
        self.assertFalse(final_prompt.rstrip().endswith(","))
        self.assertNotIn(",,", final_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
