"""
Focused tests for wiring StoryScene.character/action/mood into
build_scene_prompt() (previously dead fields — see the Story Mode static
review). Mirrors the idempotent-injection pattern resolve_location_anchor()
already established: only append a field's text if the scene's own
free-text prompt doesn't already contain it.

Pure string logic, no torch/GPU dependency (same caveat as the other
test_story_*.py files: importing backend/main.py still pulls in the full
FastAPI/sd_pipeline import chain).

Run:
    python test_story_action_mood_character.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import StoryScene, build_scene_prompt, resolve_character_anchor


class TestActionMoodCharacterWiring(unittest.TestCase):

    # ── action ───────────────────────────────────────────────────────
    def test_action_field_only_gets_injected(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman in the forest", action="crouching low"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertIn("crouching low", final_prompt.lower())

    def test_action_text_only_not_duplicated(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman is crouching low in the forest"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(final_prompt.lower().count("crouching low"), 1)

    def test_action_both_field_and_text_no_duplication(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman is crouching low in the forest", action="crouching low"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(final_prompt.lower().count("crouching low"), 1)

    def test_action_neither_matches_prior_behavior(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman in the forest"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(
            final_prompt,
            "a young woman with short red hair, the same woman in the forest, "
            "consistent lighting and style",
        )

    # ── mood ─────────────────────────────────────────────────────────
    def test_mood_field_only_gets_injected_in_short_form(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman in the forest", mood="calm"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertIn("calm mood", final_prompt.lower())

    def test_mood_text_only_not_duplicated(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman looks calm in the forest"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertNotIn("calm mood", final_prompt.lower())
        self.assertEqual(final_prompt.lower().count("calm"), 1)

    def test_mood_both_field_and_text_no_duplication(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman looks calm in the forest", mood="calm"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(final_prompt.lower().count("calm"), 1)

    def test_mood_neither_matches_prior_behavior(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest"),
            StoryScene(prompt="the same woman in the forest"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertNotIn("mood", final_prompt.lower())

    # ── character ────────────────────────────────────────────────────
    def test_character_field_only_used_when_free_text_has_no_character_desc(self):
        # Scene 1's own prompt has nothing left to anchor on once pose
        # words are stripped (it's purely an action/location sentence) —
        # extract_character_anchor() would return "" here.
        scenes = [
            StoryScene(prompt="standing calmly in a forest", character="a young woman with short red hair"),
            StoryScene(prompt="walking through the city"),
        ]
        self.assertEqual(resolve_character_anchor(scenes), "a young woman with short red hair")
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertIn("short red hair", final_prompt.lower())

    def test_character_text_only_matches_prior_behavior(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing calmly in a forest"),
            StoryScene(prompt="walking through the city"),
        ]
        self.assertEqual(
            resolve_character_anchor(scenes),
            "a young woman with short red hair",
        )

    def test_character_both_prefers_text_anchor_no_duplication(self):
        # Free text already yields a non-empty anchor -> structured field
        # is NOT additionally appended (character_anchor sits once, at
        # the very front of final_prompt).
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing calmly in a forest",
                       character="a young woman with short red hair"),
            StoryScene(prompt="walking through the city"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(final_prompt.lower().count("short red hair"), 1)

    def test_character_neither_matches_prior_behavior(self):
        scenes = [
            StoryScene(prompt="standing calmly in a forest"),
            StoryScene(prompt="walking through the city"),
        ]
        self.assertEqual(resolve_character_anchor(scenes), "")
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertEqual(
            final_prompt,
            "walking through the city, consistent style, same setting",
        )

    # ── ordering: new terms must not push style/location further toward
    #    truncation — they must appear BEFORE style_suffix/location_suffix
    def test_action_mood_appear_before_style_and_location_suffixes(self):
        scenes = [
            StoryScene(prompt="a young woman with short red hair, standing in a forest, cinematic"),
            StoryScene(prompt="the same woman in the city", action="crouching", mood="tense",
                       location="rooftop"),
        ]
        final_prompt = build_scene_prompt(scenes, 1)
        self.assertLess(final_prompt.lower().index("crouching"), final_prompt.lower().index("rooftop"))
        self.assertLess(final_prompt.lower().index("tense mood"), final_prompt.lower().index("rooftop"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
