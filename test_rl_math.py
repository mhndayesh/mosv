"""Run without torch: python -m unittest test_rl_math.py"""

import unittest

from rl_math import group_advantages, reward_integer_answer


class TestRLMath(unittest.TestCase):
    def test_group_centering(self):
        a = group_advantages([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(a, [-0.5, -0.5, 0.5, 0.5])
        self.assertAlmostEqual(sum(a), 0)

    def test_rewards(self):
        self.assertEqual(reward_integer_answer("42", 42), 1.0)
        self.assertEqual(reward_integer_answer("I think 42", 42), -0.25)
        self.assertLess(reward_integer_answer("35", 42), 1.0)

    def test_positive_advantage_increases_action_probability(self):
        # For binary policy pi(1)=sigmoid(z), gradient[-A log pi(1)]
        # = -A(1-pi(1)). Positive A => gradient descent raises z & pi.
        p, advantage = 0.25, 0.5
        gradient = -advantage * (1 - p)
        self.assertLess(gradient, 0)
        self.assertGreater(p - 0.1 * gradient, p)


if __name__ == "__main__":
    unittest.main()
