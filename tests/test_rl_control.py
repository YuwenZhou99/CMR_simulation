"""Run with: python -m unittest discover -s tests -p test_rl_control.py"""
import unittest
import numpy as np

from ur_simulation.rl_control.environment import ManipulationEnv, TaskConfig, INITIAL_Q


class RLEnvironmentTests(unittest.TestCase):
    def test_velocity_servo_does_not_secretly_follow_reference(self):
        env = ManipulationEnv(TaskConfig())
        try:
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs.shape, (69,))
            for _ in range(20):
                obs, _, done, truncated, info = env.step(np.zeros(6))
                self.assertFalse(done or truncated)
            np.testing.assert_allclose(env.robot.joints()[0], INITIAL_Q, atol=1e-6)
            self.assertGreater(np.linalg.norm(info["tau"]), 0.)
            env.step(np.array([.2,0,0,0,0,0]))
            self.assertGreater(env.robot.joints()[1][0], 0.)
        finally:
            env.close()

    def test_clients_and_resets_are_independent(self):
        first = ManipulationEnv(TaskConfig(task="trajectory", randomize=True))
        second = None
        try:
            second = ManipulationEnv(TaskConfig(task="push"))
            a, _ = first.reset(seed=42)
            second.reset(seed=42)
            second_before = second.robot.joints()[0].copy()
            first.step(np.ones(6)*.1)
            np.testing.assert_allclose(second.robot.joints()[0], second_before)
            again, _ = first.reset(seed=42)
            np.testing.assert_allclose(a, again)
            self.assertEqual(second.action_space.shape, (6,))
            self.assertIsNotNone(second.tip_id)
            with self.assertRaises(ValueError):
                first.step(np.full(6, np.nan))
        finally:
            first.close()
            if second is not None:
                second.close()

    def test_random_scenes_and_idle_reward(self):
        env = ManipulationEnv(TaskConfig(task="push", random_positions=True, reward_version=2))
        try:
            _, first = env.reset(seed=123)
            refs = env.references.copy()
            _, second = env.reset(seed=124)
            self.assertFalse(np.allclose(first["box_start"], second["box_start"]))
            self.assertFalse(np.allclose(first["goal"], second["goal"]))
            _, repeated = env.reset(seed=123)
            np.testing.assert_allclose(first["goal"], repeated["goal"])
            np.testing.assert_allclose(refs, env.references)
            np.testing.assert_allclose(env._box_state()[0][:2], first["box_start"])
            actual_goal = env.pb.getBasePositionAndOrientation(env.goal_id)[0]
            np.testing.assert_allclose(actual_goal[:2], first["goal"])
            rewards = [env.step(np.zeros(6))[1]]
            self.assertLess(sum(rewards), 0.)
        finally:
            env.close()

    def test_horizon_is_truncation_not_a_success(self):
        env = ManipulationEnv(TaskConfig(task="push"))
        try:
            env.reset(seed=0)
            env.index = env.max_steps-1
            _, _, terminated, truncated, info = env.step(np.zeros(6))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["is_success"])
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
