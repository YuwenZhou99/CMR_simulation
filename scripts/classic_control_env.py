import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p

from ur_simulation.classic_control.robots.ur7e import UR7e


INIT_Q = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
DOWNWARD = p.getQuaternionFromEuler([0, np.pi / 2, 0])
BOX_SIZE = 0.06
PUSHER_RADIUS = 0.018
GOAL_RADIUS = 0.10
PUSH_HEIGHT = 0.05
TASK_KP = np.array([400., 400., 400., 8., 8., 8.])
TASK_KD = np.array([30., 30., 30., .1, .1, .1])


def attach_pusher(robot):
    """Collision-enabled ball fixed to the EE frame; IK target is ball centre."""
    pb = robot.sim.physics_client
    ee_pos = robot.robot_state.get_end_effector_position().tolist()
    ee_orn = robot.robot_state.get_end_effector_orientation().tolist()
    shape = pb.createCollisionShape(p.GEOM_SPHERE, radius=PUSHER_RADIUS)
    visual = pb.createVisualShape(p.GEOM_SPHERE, radius=PUSHER_RADIUS,
                                  rgbaColor=[0.15, 0.45, 0.95, 1])
    tip_id = pb.createMultiBody(baseMass=0.03,
                                baseCollisionShapeIndex=shape,
                                baseVisualShapeIndex=visual,
                                basePosition=ee_pos, baseOrientation=ee_orn)
    pb.createConstraint(robot.robot_id, robot.ee_link, tip_id, -1,
                        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    # Ignore pusher / robot self-collisions, but allow ball / box contact.
    for link in range(-1, pb.getNumJoints(robot.robot_id)):
        pb.setCollisionFilterPair(robot.robot_id, tip_id, link, -1, 0)
    return tip_id


def ik_waypoints(robot, points):
    """Solve IK using each prior solution as the next seed, then reset scene."""
    pb = robot.sim.physics_client
    initial = robot.robot_state.get_joint_angles().copy()
    out = []
    try:
        for point in points:
            angles = np.asarray(pb.calculateInverseKinematics(
                robot.robot_id, robot.ee_link, point.tolist(), DOWNWARD,
                maxNumIterations=2000, residualThreshold=1e-8), dtype=float)
            angles = angles[:len(initial)]
            robot.set_joint_angles(angles)
            actual = robot.robot_state.get_end_effector_position()
            actual_orientation = robot.robot_state.get_end_effector_orientation()
            if (np.linalg.norm(actual - point) > 0.008 or
                abs(np.dot(actual_orientation, DOWNWARD)) < np.cos(np.deg2rad(3) / 2)):
                raise ValueError(f"IK cannot accurately reach {np.round(point, 3)}")
            out.append(angles.copy())
    finally:
        robot.set_joint_angles(initial)
    return np.asarray(out)


def build_push_path(start, goal, push_height):
    """Approach behind box, touch its rear face, push, then retreat upward."""
    direction = goal - start
    length = np.linalg.norm(direction)
    if length < 0.03:
        raise ValueError("Goal must be at least 3 cm from the box start")
    unit = direction / length
    # Desired sphere-centre location when touching the rear face of the box.
    touch = start - unit * (BOX_SIZE / 2 + PUSHER_RADIUS)
    back = touch - 0.075 * unit
    finish = goal - unit * (BOX_SIZE / 2 + PUSHER_RADIUS)
    xy = [back, back, touch, finish, finish]
    heights = [max(push_height + 0.12, 0.17), push_height,
               push_height, push_height, max(push_height + 0.12, 0.17)]
    points = np.array([[v[0], v[1], h] for v, h in zip(xy, heights)])
    # Initial -> pre-approach -> behind box -> contact -> push -> retreat
    durations = np.array([3.0, 2.0, 2.0, max(4.0, length / 0.025), 2.0])
    return points, durations


def sample_quintic(t, knots, qs):
    """Point-to-point joint path; stop (dq=ddq=0) at each knot."""
    if t >= knots[-1]:
        return qs[-1].copy(), np.zeros(qs.shape[1]), np.zeros(qs.shape[1])
    i = max(0, np.searchsorted(knots, t, side="right") - 1)
    dt = knots[i + 1] - knots[i]
    s = np.clip((t - knots[i]) / dt, 0., 1.)
    delta = qs[i + 1] - qs[i]
    q = qs[i] + (10*s**3 - 15*s**4 + 6*s**5) * delta
    dq = (30*s**2 - 60*s**3 + 30*s**4) * delta / dt
    ddq = (60*s - 180*s**2 + 120*s**3) * delta / dt**2
    return q, dq, ddq


def skew(v):
    """Matrix such that skew(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def orientation_coordinates(quaternion):
    """Rotation-vector chart about DOWNWARD, expressed in world axes.

    Unlike Euler pitch, this chart is nonsingular at our downward tool pose.
    It is a local chart: rotations at pi from DOWNWARD are excluded.
    """
    q = np.asarray(quaternion, float)
    d = np.asarray(DOWNWARD, float)
    # Relative rotation R(q) R(DOWNWARD)^T, quaternion [xyz, w].
    xyz = -q[3]*d[:3] + d[3]*q[:3] - np.cross(q[:3], d[:3])
    w = q[3]*d[3] + np.dot(q[:3], d[:3])
    if w < 0:
        xyz, w = -xyz, -w
    length = np.linalg.norm(xyz)
    angle = 2*np.arctan2(length, w)
    if angle >= np.pi-1e-3:
        raise ValueError("Task orientation reached the rotation-vector chart boundary")
    return 2*xyz if length < 1e-9 else xyz*(angle/length)


def task_state(robot, q):
    """Return x=[position, rotation coordinates] and its analytical Jacobian.

    Bullet's geometric angular Jacobian maps dq to world angular velocity;
    the inverse SO(3) left Jacobian maps that velocity to coordinate rates.
    Both position and Jacobian refer to the flange link-frame origin.
    """
    pb = robot.sim.physics_client
    link = pb.getLinkState(robot.robot_id, robot.ee_link, computeForwardKinematics=1)
    phi = orientation_coordinates(link[5])
    jv, jw = pb.calculateJacobian(robot.robot_id, robot.ee_link, [0., 0., 0.],
                                  q.tolist(), [0.]*len(q), [0.]*len(q))
    theta = np.linalg.norm(phi)
    hat = skew(phi)
    coefficient = (1/12 + theta**2/720 if theta < 1e-4 else
                   (1 - .5*theta/np.tan(.5*theta))/theta**2)
    left_inverse = np.eye(3) - .5*hat + coefficient*(hat @ hat)
    jacobian = np.vstack([jv, left_inverse @ np.asarray(jw)])
    return np.r_[link[4], phi], jacobian


def task_pd_torque(x, dx, xd, jacobian, gravity, kp, kd):
    """Slide law: u = JA.T @ (KP*(xd-x) - KD*dx) + g(q).

    No desired-velocity feedforward, inertia or Coriolis cancellation is added.
    xd can be scheduled along a path; finite tracking lag is then expected.
    """
    return jacobian.T @ (kp*(xd-x) - kd*dx) + gravity


def contact_measurements(pb, tip_id, box_id, dt):
    """Measure pusher/box contact after a simulation step.

    PyBullet getContactPoints returns normalForce in N (not an impulse).
    The sum is total *normal* force across contact points; it does not
    include tangential friction. normal impulse approximated as F * dt.
    """
    contacts = pb.getContactPoints(bodyA=tip_id, bodyB=box_id)
    forces = [max(0.0, float(pt[9])) for pt in contacts]
    return len(contacts), sum(forces), max(forces, default=0.0)


def plot_results(data, output_path):
    """Save joint, torque, contact and box trajectory figures alongside the NPZ."""
    stem = output_path.with_suffix("")
    t = data["t"]
    contact = data["contact_count"] > 0
    outputs = []

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, data["q_des"][:, j], "--", lw=1.3, label="Desired")
        ax.plot(t, data["q"][:, j], lw=1.2, label="Actual")
        ax.set_ylabel(f"Joint {j+1} [rad]")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Joint motion (IK guide is not the task-PD feedback target)"
                 if str(data.get("control_space", "joint")) == "task" else "Joint position tracking")
    fig.tight_layout()
    outputs.append(stem.with_name(stem.name + "_joints.png"))
    fig.savefig(outputs[-1], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, data["contact_normal_force"], label="Sphere / box normal force [N]")
    ax.plot(t, data["contact_peak_force"], label="Largest sphere contact force [N]", alpha=0.75)
    if "arm_contact_normal_force" in data:
        ax.plot(t, data["arm_contact_normal_force"], label="Arm-link / box normal force [N]", alpha=.75)
    # Indicate contact time with a translucent strip at the bottom.
    if np.any(contact):
        ymax = max(float(np.max(data["contact_normal_force"])), 1.0)
        ax.fill_between(t, 0, ymax, where=contact, step="post",
                        alpha=0.12, label="Contact active")
    ax.set(xlabel="Time [s]", ylabel="Normal force [N]",
           title="Robot / box contact forces")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    outputs.append(stem.with_name(stem.name + "_contacts.png"))
    fig.savefig(outputs[-1], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, data["tau"][:, j], lw=1.2)
        ax.set_ylabel(f"Joint {j+1} torque [N·m]")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Applied joint torques (clipped commands)")
    fig.tight_layout()
    outputs.append(stem.with_name(stem.name + "_torques.png"))
    fig.savefig(outputs[-1], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    box_xy = data["box_xy"]
    ax.plot(box_xy[:, 0], box_xy[:, 1], label="Actual box path")
    ax.scatter(*data["box_start"], marker="s", label="Start")
    ax.scatter(*data["goal"], marker="x", s=100, label="Goal")
    ax.add_patch(plt.Circle(data["goal"], float(data["goal_radius"]),
                            fill=False, linestyle="--", label="Goal region"))
    ax.set(xlabel="World X [m]", ylabel="World Y [m]", title="Box movement in world XY")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outputs.append(stem.with_name(stem.name + "_box_path.png"))
    fig.savefig(outputs[-1], dpi=160)
    plt.close(fig)

    if "x" in data:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(t, data["x_des"][:, j], "--", label="Desired")
            ax.plot(t, data["x"][:, j], label="Actual")
            ax.set_ylabel(["x [m]", "y [m]", "z [m]", "phi_x [rad]",
                           "phi_y [rad]", "phi_z [rad]"][j])
            ax.grid(alpha=.3)
        axes[0, 0].legend()
        axes[2, 0].set_xlabel("Time [s]")
        axes[2, 1].set_xlabel("Time [s]")
        fig.tight_layout()
        outputs.append(stem.with_name(stem.name + "_task.png"))
        fig.savefig(outputs[-1], dpi=160)
        plt.close(fig)

    import webbrowser
    for f in outputs:
        webbrowser.open(f.resolve().as_uri())
    return outputs


def run(args):
    start = np.asarray(args.box_start, float)
    goal = np.asarray(args.goal, float)
    if not np.all(np.isfinite(np.r_[start, goal])) or not 0.01 <= GOAL_RADIUS <= 0.3:
        raise ValueError("Invalid start, goal, or goal radius (use 0.01 to 0.3 m)")
    if PUSH_HEIGHT <= PUSHER_RADIUS:
        raise ValueError("Pusher height must be larger than the ball radius")

    robot = None
    recorder = None
    log = {k: [] for k in ("t", "box_xy", "ee_xyz", "q", "q_des", "dq", "dq_des",
                            "tau", "contact_count", "contact_normal_force", "contact_peak_force",
                            "x", "x_des", "dx", "arm_contact_count", "arm_contact_normal_force",
                            "tau_command")}
    try:
        robot = UR7e(block_gripper=True, neutral_joints=INIT_Q)
        robot.sim.create_plane(0)
        points, durations = build_push_path(start, goal, PUSH_HEIGHT)
        with robot.sim.no_rendering():
            joint_waypoints = ik_waypoints(robot, points)
        q_start = robot.robot_state.get_joint_angles().copy()
        qs = np.vstack([q_start, joint_waypoints])
        knots = np.r_[0.0, np.cumsum(durations)]
        total_time = knots[-1] + 2.0
        nsteps = int(np.ceil(total_time / robot.sim.dt))
        # Preserve the existing waypoint approach/IK branch.
        # FK turns the planned joint path into x_d; feedback below uses x, not q_d-q.
        task_references = []
        with robot.sim.no_rendering():
            try:
                for k in range(nsteps + 1):
                    reference_q, _, _ = sample_quintic(k*robot.sim.dt, knots, qs)
                    robot.set_joint_angles(reference_q)
                    xyz = robot.robot_state.get_end_effector_position()
                    orientation = robot.robot_state.get_end_effector_orientation()
                    task_references.append(np.r_[xyz, orientation_coordinates(orientation)])
            finally:
                robot.set_joint_angles(q_start)
        task_references = np.asarray(task_references)
        # Create the block AFTER IK planning so teleported joints don't hit the block.
        pb = robot.sim.physics_client
        half = BOX_SIZE / 2
        collision = pb.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
        visual = pb.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3,
                                      rgbaColor=[0.9, 0.35, 0.1, 1])
        box_id = pb.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=[*start, half])
        pb.changeDynamics(box_id, -1, lateralFriction=0.6,
                          rollingFriction=0.002, spinningFriction=0.002)
        target_visual = pb.createVisualShape(p.GEOM_CYLINDER,
                                             radius=GOAL_RADIUS, length=0.002,
                                             rgbaColor=[0.1, 0.85, 0.25, 0.45])
        pb.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual,
                           basePosition=[*goal, 0.002])
        tip_id = attach_pusher(robot)

        task_kp, task_kd = TASK_KP, TASK_KD
        if getattr(args, "record", None) is not None:
            from ur_simulation.gui_recording import GUIRecorder
            video_path = args.record or str(Path(args.output).with_suffix(".mp4"))
            recorder = GUIRecorder(robot.sim, video_path)
        for step in range(nsteps):
            t = step * robot.sim.dt
            q = robot.robot_state.get_joint_angles().copy()
            dq = robot.robot_state.get_joint_velocities().copy()
            dynamics = robot.robot_model.get_dynamics()
            x, jacobian = task_state(robot, q)
            xd = task_references[step]
            tau = task_pd_torque(x, jacobian @ dq, xd, jacobian,
                                 dynamics.gravity_vector, task_kp, task_kd)
            tau_command = tau.copy()
            # Match the physical effort limit recorded by the UR simulation.
            tau = np.clip(tau, -robot.arm_joint_forces, robot.arm_joint_forces)
            robot.control_torques(tau)
            robot.sim.step()
            if recorder is not None:
                recorder.capture(step * robot.sim.dt)
            # Sample all observed quantities *after* stepping, when contact forces
            # reflect this physics step. Reference is sampled at the same time.
            t_obs = (step + 1) * robot.sim.dt
            q_obs = robot.robot_state.get_joint_angles().copy()
            dq_obs = robot.robot_state.get_joint_velocities().copy()
            q_des_obs, dq_des_obs, _ = sample_quintic(t_obs, knots, qs)
            box_xyz, _ = pb.getBasePositionAndOrientation(box_id)
            ee_xyz = robot.robot_state.get_end_effector_position().copy()
            count, normal_force, peak_force = contact_measurements(
                pb, tip_id, box_id, robot.sim.dt)
            x_obs, ja_obs = task_state(robot, q_obs)
            xd_obs = task_references[step + 1]
            arm_contacts = [c for c in pb.getContactPoints(robot.robot_id, box_id)
                            if c[3] >= 0 and c[9] > 1e-6]
            log["x"].append(x_obs)
            log["x_des"].append(xd_obs)
            log["dx"].append(ja_obs @ dq_obs)
            log["arm_contact_count"].append(len(arm_contacts))
            log["arm_contact_normal_force"].append(sum(c[9] for c in arm_contacts))
            log["tau_command"].append(tau_command)
            log["t"].append(t_obs)
            log["box_xy"].append(box_xyz[:2])
            log["ee_xyz"].append(ee_xyz)
            log["q"].append(q_obs)
            log["q_des"].append(q_des_obs)
            log["dq"].append(dq_obs)
            log["dq_des"].append(dq_des_obs)
            log["tau"].append(tau)
            log["contact_count"].append(count)
            log["contact_normal_force"].append(normal_force)
            log["contact_peak_force"].append(peak_force)
        box_xyz, _ = pb.getBasePositionAndOrientation(box_id)
        err = np.linalg.norm(np.asarray(box_xyz[:2]) - goal)
        print("Controller: gravity_pd")
        print(f"Box XY: {np.round(box_xyz[:2], 4)} m")
        print(f"Goal XY: {np.round(goal, 4)} m")
        print(f"Distance to goal: {err:.4f} m")
        print("SUCCESS" if err <= GOAL_RADIUS else "NOT REACHED")
        print(f"Arm-link contact samples: {np.count_nonzero(log['arm_contact_count'])}")
        print(f"Final task position error: {np.linalg.norm(log['x_des'][-1][:3]-log['x'][-1][:3]):.4f} m")
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            if robot is not None:
                robot.sim.close()
    if log["t"]:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: np.asarray(value) for key, value in log.items()}
        data.update(box_start=start, goal=goal, goal_radius=GOAL_RADIUS,
                    controller="gravity_pd", dt=robot.sim.dt,
                    control_space="task",
                    task_kp=task_kp, task_kd=task_kd,
                    orientation_chart="log(R R_downward^T)")
        data["joint_error"] = data["q_des"] - data["q"]
        data["contact_normal_impulse"] = data["contact_normal_force"] * data["dt"]
        np.savez_compressed(path, **data)
        active = np.flatnonzero(data["contact_count"] > 0)
        print(f"Contact steps: {len(active)} / {len(data['t'])}")
        if active.size:
            print(f"First contact: {data['t'][active[0]]:.3f} s")
            print(f"Last contact:  {data['t'][active[-1]]:.3f} s")
            print(f"Peak total normal contact force: {data['contact_normal_force'].max():.2f} N")
            print(f"Accumulated normal impulse (approx.): {data['contact_normal_impulse'].sum():.3f} N·s")
        else:
            print("No pusher–box contact registered.")
        print(f"Simulation data: {path}")
        for fig_path in plot_results(data, path):
            print(f"Figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="UR7e PD-based box pushing")
    parser.add_argument("--box-start", nargs=2, type=float, default=[0.48, -0.16],
                        metavar=("X", "Y"), help="Initial box centre XY, metres")
    parser.add_argument("--goal", nargs=2, type=float, default=[0.48, 0.10],
                        metavar=("X", "Y"), help="Target-region centre XY, metres")
    parser.add_argument("--output", default="results/box_push_task_pd.npz")
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="MP4",
                        help="Record simulation view to MP4 (default: same stem as --output)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
