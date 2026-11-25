"""
Focused tests for Phase 2C: the alpha-blend multi-character compositor
(maybe_composite()) has been removed from backend/main.py, not merely
made unreachable.

Run:
    python test_story_compositor_removed.py
"""
import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

import main as story_main


class TestCompositorGenuinelyRemoved(unittest.TestCase):

    def test_maybe_composite_function_no_longer_exists(self):
        self.assertFalse(hasattr(story_main, "maybe_composite"))

    def test_multi_patterns_list_no_longer_exists(self):
        self.assertFalse(hasattr(story_main, "MULTI_PATTERNS"))

    def test_unrelated_helpers_still_present(self):
        # Sanity check that removal was scoped correctly — everything
        # else from prior phases must still be there.
        for name in ("extract_character_anchor", "extract_style_anchor",
                     "build_scene_prompt", "calculate_strength",
                     "compare_scenes", "resolve_negative", "StoryScene",
                     "StoryRequest"):
            self.assertTrue(hasattr(story_main, name), f"{name} missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
