"""Analytical-coordinate and power consistency checks for task-space PD."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from ur_simulation.pybullet import PyBullet

spec = importlib.util.spec_from_file_location('push_task_test', Path(__file__).resolve().parents[1]/'scripts/classic_control_env.py')
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)

class TaskPDTests(unittest.TestCase):
    def test_analytical_jacobian_and_downward_chart(self):
        def headless(**kw):
            return PyBullet(**kw, render_mode='rgb_array', renderer='Tiny')
        with patch('ur_simulation.classic_control.robots.ur7e.PyBullet', headless):
            robot = push.UR7e(block_gripper=True, neutral_joints=push.INIT_Q)
        try:
            for q in (push.INIT_Q, push.INIT_Q + [.15, -.1, .12, .1, -.2, .05]):
                robot.set_joint_angles(q)
                _, ja = push.task_state(robot, q)
                numerical = np.zeros_like(ja)
                for j in range(6):
                    delta = np.eye(6)[j]*.001
                    robot.set_joint_angles(q+delta)
                    xp, _ = push.task_state(robot, q+delta)
                    robot.set_joint_angles(q-delta)
                    xm, _ = push.task_state(robot, q-delta)
                    numerical[:, j] = (xp-xm)/.002
                np.testing.assert_allclose(ja, numerical, atol=2e-4)
            np.testing.assert_allclose(push.orientation_coordinates(push.DOWNWARD), 0, atol=1e-12)
            np.testing.assert_allclose(push.orientation_coordinates(-np.array(push.DOWNWARD)), 0, atol=1e-12)
        finally:
            robot.sim.close()

    def test_virtual_work_and_damping(self):
        rng = np.random.default_rng(8)
        ja = rng.normal(size=(6, 6))
        dq = rng.normal(size=6)
        gravity = rng.normal(size=6)
        x = rng.normal(size=6)
        kd = np.arange(1., 7.)
        tau = push.task_pd_torque(x, ja@dq, x, ja, gravity, np.ones(6), kd)
        self.assertAlmostEqual(float(dq@(tau-gravity)), -float((ja@dq)@(kd*(ja@dq))))
        np.testing.assert_allclose(push.task_pd_torque(x, np.zeros(6), x, ja, gravity, np.ones(6), kd), gravity)

if __name__ == '__main__':
    unittest.main()
