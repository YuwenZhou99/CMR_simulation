"""Velocity-action RL with model-based inverse-dynamics velocity servo.
IK supplies task references, not policy actions.
"""
from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from ur_simulation.pybullet import PyBullet

ROOT = Path(__file__).resolve().parents[2]
INITIAL_Q = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])


@dataclass
class TaskConfig:
    task: str = "trajectory"
    mode: str = "path"
    target: tuple = (0.4, 0.3, 0.5)
    box_start: tuple = (0.48, -0.16)
    goal: tuple = (0.48, 0.10)
    goal_radius: float = 0.10
    push_height: float = 0.05
    policy_hz: int = 50
    max_velocity: float = 1.0
    velocity_gain: float = 30.0
    stage: str = "base"
    randomize: bool = False
    random_positions: bool = False
    reward_version: int = 1
    push_stage: str = "push"  # Optional reach-only diagnostic/training task

    def validate(self):
        if self.reward_version not in (1, 2) or self.push_stage not in ("reach", "push"):
            raise ValueError("Invalid reward version or push stage")
        if not 0 < self.max_velocity <= 3 or not 0 < self.velocity_gain <= 100:
            raise ValueError("Invalid velocity bound or servo gain")
        if self.task not in ("trajectory", "push") or self.mode not in ("point", "path"):
            raise ValueError("Use task trajectory/push and mode point/path")
        if self.stage not in ("base", "limits", "smooth", "safe"):
            raise ValueError("Unknown fine-tuning stage")
        if self.policy_hz <= 0 or 1000 % self.policy_hz:
            raise ValueError("policy_hz must divide the 1000 Hz physics rate")
        if len(self.target) != 3 or len(self.box_start) != 2 or len(self.goal) != 2:
            raise ValueError("target must be XYZ; box_start and goal must be XY")
        if not np.all(np.isfinite([*self.target, *self.box_start, *self.goal,
                                   self.goal_radius, self.push_height])):
            raise ValueError("Task parameters must be finite")
        if not 0.01 <= self.goal_radius <= 0.3 or self.push_height <= 0.018:
            raise ValueError("Invalid goal radius or pusher height")


def _load_script(name, pb):
    # A private module copy lets existing IK functions use this client's API.
    spec = importlib.util.spec_from_file_location(f"_rl_{name}_{pb._client}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.p = pb
    return module


class Robot:
    """Minimal client-bound adapter for the existing planning/scene helpers."""
    arm_indices = np.array([2, 3, 4, 5, 6, 7])
    arm_joint_forces = np.array([150., 150., 150., 28., 28., 28.])
    ee_link = 9

    def __init__(self, sim):
        self.sim = sim
        self.pb = sim.physics_client
        self.robot_id = self.pb.loadURDF(
            str(ROOT / "ur_simulation/assets/urdf/ur7e.urdf"), useFixedBase=True)
        self.set_joint_angles(INITIAL_Q)
        self.pb.setJointMotorControlArray(self.robot_id, self.arm_indices,
                                         p.VELOCITY_CONTROL, forces=np.zeros(6))
        for j in self.arm_indices:
            self.pb.changeDynamics(self.robot_id, int(j), linearDamping=0,
                                   angularDamping=0, jointDamping=0)
        self.robot_state = SimpleNamespace(
            get_joint_angles=lambda: self.joints()[0],
            get_joint_velocities=lambda: self.joints()[1],
            get_end_effector_position=lambda: self.pose()[0],
            get_end_effector_orientation=lambda: self.pose()[1])

    def joints(self):
        states = self.pb.getJointStates(self.robot_id, self.arm_indices)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def pose(self):
        s = self.pb.getLinkState(self.robot_id, self.ee_link, computeLinkVelocity=1,
                                 computeForwardKinematics=1)
        return tuple(np.array(s[i]) for i in (4, 5, 6, 7))

    def set_joint_angles(self, angles):
        for j, q in zip(self.arm_indices, angles):
            self.pb.resetJointState(self.robot_id, int(j), float(q), targetVelocity=0)


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or TaskConfig()
        self.config.validate()
        self.box_start = np.asarray(self.config.box_start, dtype=float)
        self.goal = np.asarray(self.config.goal, dtype=float)
        self.render_mode = render_mode
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        self.substeps = 1000 // self.config.policy_hz
        self.dt = self.substeps / 1000
        self.sim = PyBullet(render_mode="human" if render_mode else "rgb_array",
                            renderer="Tiny", n_substeps=1, step_frequency=1000)
        self.pb = self.sim.physics_client
        try:
            self.robot = Robot(self.sim)
            self.sim.create_plane(0)
            self.plane = self.sim.bodies_idx["plane"]
            self.classic = _load_script("classic_control", self.pb)
            self.push = _load_script("classic_control_env", self.pb)
            self.box_id = self.tip_id = None
            self._prepare_reference()
            if self.config.task == "push":
                self._create_box()
            self.lower, self.upper = np.array([
                self.pb.getJointInfo(self.robot.robot_id, int(j))[8:10]
                for j in self.robot.arm_indices]).T
            self.saved_state = self.pb.saveState()
            self.action_space = spaces.Box(-1., 1., (6,), dtype=np.float32)
            # Same schema permits continuing a checkpoint on a modified task.
            self.observation_space = spaces.Box(-np.inf, np.inf, (69,), dtype=np.float32)
        except Exception:
            self.sim.close()
            raise

    def _prepare_reference(self):
        if self.config.task == "trajectory":
            plan = self.classic.build_plan(self.robot, self.config.mode, self.config.target)
            _, qs, knots, vs, accs, self.path_start, self.path_duration = plan
            sample = lambda t: self.classic.trajectory(t, knots, qs, vs, accs)
        else:
            points, durations = self.push.build_push_path(
                np.array(self.box_start), np.array(self.goal), self.config.push_height)
            qs = np.vstack([INITIAL_Q, self.push.ik_waypoints(self.robot, points)])
            knots = np.r_[0., np.cumsum(durations)]
            sample = lambda t: self.push.sample_quintic(t, knots, qs)
            self.path_start, self.path_duration = np.nan, np.nan
        self.max_steps = int(np.ceil((knots[-1]+2) / self.dt))
        times = np.arange(self.max_steps + 1) * self.dt
        self.references = np.array([sample(t) for t in times])
        self.reference_xyz = []
        self.reference_orn = []
        try:
            for q in self.references[:, 0]:
                self.robot.set_joint_angles(q)
                xyz, orn, _, _ = self.robot.pose()
                self.reference_xyz.append(xyz)
                self.reference_orn.append(orn)
        finally:
            self.robot.set_joint_angles(INITIAL_Q)
        self.reference_xyz = np.array(self.reference_xyz)

    def _create_box(self):
        half = self.push.BOX_SIZE / 2
        collision = self.pb.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
        visual = self.pb.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3,
                                           rgbaColor=[0.9, 0.35, 0.1, 1])
        self.box_id = self.pb.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=collision,
                                             baseVisualShapeIndex=visual,
                                             basePosition=[*self.box_start, half])
        self.pb.changeDynamics(self.box_id, -1, lateralFriction=0.6,
                               rollingFriction=0.002, spinningFriction=0.002)
        goal = self.pb.createVisualShape(p.GEOM_CYLINDER, radius=self.config.goal_radius,
                                         length=0.002, rgbaColor=[0.1, 0.85, 0.25, 0.45])
        self.goal_id = self.pb.createMultiBody(baseMass=0, baseVisualShapeIndex=goal,
                                basePosition=[*self.goal, 0.002])
        self.tip_id = self.push.attach_pusher(self.robot)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pb.restoreState(self.saved_state)
        self.sim.elapsed_time = 0
        self.robot.set_joint_angles(INITIAL_Q)
        if self.config.task == "push" and self.config.random_positions:
            # Conservative world-XY patch; reject unreachable entire push paths.
            for attempt in range(30):
                self.box_start = self.np_random.uniform([.42, -.20], [.53, -.10])
                self.goal = self.np_random.uniform([.42, .06], [.53, .18])
                if np.linalg.norm(self.goal-self.box_start) <= self.config.goal_radius+.04:
                    continue
                try:
                    self._prepare_reference()
                    if np.any(self.references[:, 0] < self.lower) or np.any(self.references[:, 0] > self.upper):
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("Could not sample a reachable push scene in 30 attempts")
            self.pb.resetBasePositionAndOrientation(self.box_id, [*self.box_start, .03], [0, 0, 0, 1])
            self.pb.resetBaseVelocity(self.box_id, [0, 0, 0], [0, 0, 0])
            self.pb.resetBasePositionAndOrientation(self.goal_id, [*self.goal, .002], [0, 0, 0, 1])
        if self.config.randomize:
            # Task perturbation, not a claim of sim-to-real domain coverage.
            if self.box_id is not None:
                self.pb.changeDynamics(self.box_id, -1,
                    mass=float(self.np_random.uniform(0.09, 0.15)),
                    lateralFriction=float(self.np_random.uniform(0.4, 0.8)))
            else:
                self.robot.set_joint_angles(INITIAL_Q + self.np_random.uniform(-0.02, 0.02, 6))
        self.index = 0
        self.previous_action = np.zeros(6)
        self.contact_force = 0.
        self.contact_ever = False
        self.contact_steps = self.clipped_substeps = self.physics_steps = 0
        self.pusher_contact_steps = self.robot_contact_steps = 0
        self.error_sum = 0.
        self.hold_steps = 0
        self.finished = False
        self.previous_box_error = float(np.linalg.norm(np.array(self.goal)-self.box_start))
        self.previous_reach = self._reach_distance()
        return self._observation(), {"task": self.config.task,
                                    "box_start": self.box_start.copy(), "goal": self.goal.copy()}

    def _reach_distance(self):
        box = self._box_state()[0]
        direction = self.goal-box[:2]
        unit = direction/max(np.linalg.norm(direction), 1e-8)
        rear = np.r_[box[:2]-unit*(self.push.BOX_SIZE/2+self.push.PUSHER_RADIUS), self.config.push_height]
        return float(np.linalg.norm(self.robot.pose()[0]-rear))

    def _box_state(self):
        if self.box_id is None:
            return np.zeros(3), np.array([0., 0., 0., 1.]), np.zeros(3), np.zeros(3)
        pos, orn = self.pb.getBasePositionAndOrientation(self.box_id)
        vel, omega = self.pb.getBaseVelocity(self.box_id)
        return tuple(np.asarray(x) for x in (pos, orn, vel, omega))

    def _observation(self):
        q, dq = self.robot.joints()
        xyz, orn, vel, omega = self.robot.pose()
        box, boxorn, boxvel, boxomega = self._box_state()
        goal = np.r_[self.goal, 0.03] if self.box_id is not None else np.array(self.config.target)
        qd, dqd, _ = self.references[self.index]
        obs = np.r_[q, dq, xyz, orn, vel, omega, qd-q, dqd,
                    self.reference_xyz[self.index]-xyz, self.index/self.max_steps,
                    self.previous_action, box, boxorn, boxvel, boxomega,
                    goal-box if self.box_id is not None else goal-xyz,
                    box-xyz if self.box_id is not None else np.zeros(3), self.contact_force,
                    [self.box_id is None, self.box_id is not None]]
        return obs.astype(np.float32)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset() after termination/truncation")
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain six finite values")
        action = np.clip(action, -1, 1)
        old_dq = self.robot.joints()[1]
        bad_contact = False
        count = 0
        self.contact_force = 0.
        for k in range(self.substeps):
            alpha = (k + 1) / self.substeps
            dq_command = ((1-alpha)*self.previous_action + alpha*action)*self.config.max_velocity
            q, dq = self.robot.joints()
            mass = np.asarray(self.pb.calculateMassMatrix(self.robot.robot_id, q.tolist()))
            nonlinear = np.asarray(self.pb.calculateInverseDynamics(
                self.robot.robot_id, q.tolist(), dq.tolist(), [0.]*6))
            tau_command = mass @ (self.config.velocity_gain*(dq_command-dq)) + nonlinear
            tau = np.clip(tau_command, -self.robot.arm_joint_forces, self.robot.arm_joint_forces)
            self.clipped_substeps += int(np.any(np.abs(tau-tau_command) > 1e-8))
            self.pb.setJointMotorControlArray(self.robot.robot_id, self.robot.arm_indices,
                                              p.TORQUE_CONTROL, forces=tau)
            self.sim.step(realtime=self.render_mode == "human")
            self.physics_steps += 1
            if self.tip_id is not None:
                n, force, _ = self.push.contact_measurements(self.pb, self.tip_id, self.box_id, .001)
                # In the original geometry, the flange can touch before the ball.
                # Record both physical contacts; never infer contact from proximity.
                arm_contacts = [c for c in self.pb.getContactPoints(self.robot.robot_id, self.box_id)
                                if c[3] >= 0 and c[9] > 1e-6]
                self.pusher_contact_steps += int(n > 0)
                self.robot_contact_steps += int(bool(arm_contacts))
                count += int(n > 0 or bool(arm_contacts))
                force += sum(float(c[9]) for c in arm_contacts)
                self.contact_force = max(self.contact_force, force)
                bad_contact |= any(c[9] > 1e-6 for c in self.pb.getContactPoints(self.tip_id, self.plane))
            bad_contact |= any(c[3] >= 0 and c[9] > 1e-6
                               for c in self.pb.getContactPoints(self.robot.robot_id, self.plane))
        self.index += 1
        action_change = float(np.mean((action-self.previous_action)**2))
        self.previous_action = action.copy()
        self.contact_steps += count
        self.contact_ever |= count > 0
        q, dq = self.robot.joints()
        xyz, orn, vel, _ = self.robot.pose()
        box, _, boxvel, _ = self._box_state()
        qd, dqd, _ = self.references[self.index]
        q_error = float(np.linalg.norm(q-qd))
        self.error_sum += float(np.mean((q-qd)**2))
        joint_rmse = float(np.sqrt(self.error_sum/self.index))
        near_limit = np.minimum(q-self.lower, self.upper-q) < .15*(self.upper-self.lower)
        limit_failure = bool(np.any(q < self.lower) or np.any(q > self.upper))
        failure = bool(bad_contact or limit_failure or np.max(np.abs(dq)) > 8)
        failure_reason = "ground_contact" if bad_contact else "joint_limit" if limit_failure else "overspeed" if failure else ""
        position_error = float(np.linalg.norm(xyz-self.reference_xyz[self.index]))
        orientation_error = float(2*np.arccos(np.clip(abs(np.dot(orn, self.reference_orn[self.index])), 0, 1)))
        box_error = float(np.linalg.norm(box[:2]-self.goal)) if self.box_id is not None else 0.
        if self.box_id is None:
            reward = (1-np.tanh(q_error/.25)) + .3*(1-np.tanh(np.linalg.norm(dq-dqd)/.5))
            reward += .3*(1-np.tanh(position_error/.05))
            success = bool(self.index == self.max_steps and joint_rmse < .04
                           and position_error < .02 and orientation_error < np.deg2rad(5))
        else:
            direction = np.array(self.goal)-box[:2]
            unit = direction/max(np.linalg.norm(direction), 1e-8)
            rear = np.r_[box[:2]-unit*(self.push.BOX_SIZE/2+self.push.PUSHER_RADIUS), self.config.push_height]
            reach = float(np.linalg.norm(xyz-rear))
            reward = .6*(1-np.tanh(reach/.1)) + .8*(1-np.tanh(box_error/.2))
            reward += 20*(self.previous_box_error-box_error)
            reward += .2*float(count > 0)
            if reach < .08:
                reward += .3*(1-np.tanh(np.linalg.norm(vel)/.2))
            if self.config.reward_version == 2:
                # Progress pays only for improvement; idle scenes earn no distance bonus.
                # Reference guidance makes approach exploration feasible without overriding actions.
                down_error = 2*np.arccos(np.clip(abs(np.dot(orn, self.push.DOWNWARD)), 0, 1))
                reward = 10*(self.previous_reach-reach) + 40*(self.previous_box_error-box_error)
                reward -= self.dt*(.5 + 2*min(reach, 1.) + .5*min(q_error, 3.))
                reward -= self.dt*.3*down_error*float(reach < .15)
                reward -= .02*action_change + .002*float(np.mean(action**2))
                self.previous_reach = reach
            self.previous_box_error = box_error
            achieved = (box_error <= self.config.goal_radius and self.contact_ever
                        and np.linalg.norm(box[:2]-self.box_start) > .03
                        and np.linalg.norm(boxvel) < .05)
            self.hold_steps = self.hold_steps + 1 if achieved else 0
            success = self.hold_steps >= max(1, round(.2/self.dt))
            if self.config.reward_version == 2 and self.config.push_stage == "reach":
                success = bool(reach < .025 and down_error < np.deg2rad(10)
                               and np.linalg.norm(vel) < .08)
            if box[2] < 0 or np.linalg.norm(box[:2]) > 1.5:
                failure, failure_reason = True, "box_outside_workspace"
        # Sequential fine-tuning objectives inspired by paper eqs. (5)-(7).
        if self.config.stage in ("limits", "smooth", "safe"):
            reward -= .1*np.sum(near_limit)
        if self.config.stage in ("smooth", "safe"):
            reward -= .1*np.tanh(np.mean(np.abs(dq-old_dq))/self.dt)
        if self.config.stage == "safe" and bad_contact:
            reward -= 5
        success = bool(success and not failure)
        reward += (20 if self.config.reward_version == 2 else 5)*success
        reward -= (100 if self.config.reward_version == 2 else 5)*failure
        terminated = bool(success or failure)
        truncated = bool(self.index >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        info = dict(is_success=success, failure_reason=failure_reason,
                    time=self.index*self.dt, joint_rmse=joint_rmse,
                    position_error=position_error, orientation_error=orientation_error,
                    box_error=box_error, contact_force=self.contact_force,
                    contact_steps=self.contact_steps, clipped_substeps=self.clipped_substeps,
                    pusher_contact_steps=self.pusher_contact_steps,
                    robot_contact_steps=self.robot_contact_steps,
                    q=q, dq=dq, q_des=qd.copy(), dq_des=dqd.copy(),
                    ee_xyz=xyz, reference_xyz=self.reference_xyz[self.index].copy(),
                    box_xyz=box, box_start=self.box_start.copy(), goal=self.goal.copy(),
                    action=action.copy(), tau=tau.copy())
        return self._observation(), float(reward), terminated, truncated, info

    def diagnostic_action(self):
        """Explicit classical baseline for tests only; NEVER used by RL step()."""
        q, _ = self.robot.joints()
        qd, dqd, _ = self.references[self.index]
        return np.clip((dqd+4*(qd-q))/self.config.max_velocity, -1, 1).astype(np.float32)

    def close(self):
        if hasattr(self, "sim"):
            self.sim.close()
