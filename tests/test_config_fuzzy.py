import math
import unittest

from emosense.config import EMOTION_ORDER, VAD_COORDINATES
from emosense.fuzzy import initial_state_from_attributes, label_brightness, label_colorfulness, stability_reward


class ConfigFuzzyTests(unittest.TestCase):
    def test_emotion_order_matches_paper_table(self):
        self.assertEqual(
            EMOTION_ORDER,
            (
                "amusement",
                "awe",
                "contentment",
                "excitement",
                "anger",
                "disgust",
                "fear",
                "sadness",
            ),
        )
        self.assertEqual(VAD_COORDINATES["anger"], (0.0, 1.0, 1.0))
        self.assertEqual(VAD_COORDINATES["sadness"], (0.0, 0.0, 0.0))

    def test_fuzzy_labels(self):
        self.assertEqual(label_brightness(0.1, (0.33, 0.66)), "low brightness")
        self.assertEqual(label_brightness(0.9, (0.33, 0.66)), "high brightness")
        self.assertEqual(label_colorfulness(0.02, (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)), "monochromatic")
        self.assertEqual(label_colorfulness(0.95, (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)), "vibrant")

    def test_stability_reward_matches_eq_13(self):
        states = [
            (0.0, 0.0),
            (3.0, 4.0),
            (6.0, 8.0),
        ]
        # Two deltas, each with ||delta||_2 = 5, so Eq. 13 returns exp(-5).
        self.assertTrue(math.isclose(stability_reward(states), math.exp(-5.0), rel_tol=1e-9))

    def test_initial_fuzzy_state_uses_reference_attributes(self):
        low_state = initial_state_from_attributes(0.2, 0.1)
        high_state = initial_state_from_attributes(0.8, 0.9)

        self.assertEqual(len(low_state), 7)
        self.assertEqual(len(high_state), 7)
        self.assertNotEqual(low_state, high_state)
        self.assertEqual(tuple(sorted(low_state[:2])), low_state[:2])
        self.assertEqual(tuple(sorted(low_state[2:])), low_state[2:])
        self.assertEqual(tuple(sorted(high_state[:2])), high_state[:2])
        self.assertEqual(tuple(sorted(high_state[2:])), high_state[2:])


if __name__ == "__main__":
    unittest.main()
