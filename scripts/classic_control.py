import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot


def plan_joint_waypoints(robot, positions):
    # tool0 +Z is flange +X: rotate flange +X toward world -Z.
    orientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
    initial_angles = robot.robot_state.get_joint_angles().copy()
    waypoints = []
    try:
        for position in positions:
            # Seed each IK solve with the previous pose to keep the same branch.
            angles = np.array(p.calculateInverseKinematics(
                robot.robot_id, robot.ee_link, position, orientation,
                maxNumIterations=2000, residualThreshold=1e-8,
            ))
            robot.set_joint_angles(angles)
            actual = robot.robot_state.get_end_effector_position()
            actual_orientation = robot.robot_state.get_end_effector_orientation()
            alignment = abs(np.dot(actual_orientation, orientation))
            limits = [p.getJointInfo(robot.robot_id, int(j))[8:10]
                      for j in robot.arm_indices]
            if (np.linalg.norm(actual - position) > 0.001
                    or alignment < np.cos(np.deg2rad(1) / 2)
                    or any(q < lo or q > hi for q, (lo, hi) in zip(angles, limits))):
                raise ValueError(f"Cannot reach downward-facing waypoint: {position}")
            waypoints.append(angles)
    finally:
        robot.set_joint_angles(initial_angles)
    return np.array(waypoints)


def path_position(s):
    # Predefined curve in the vertical world Y-Z plane, in metres.
    centre = np.array([0.45, 0.15, 0.48])
    radius = 0.12  # Circle radius in metres.
    theta = 2 * np.pi * (10*s**3 - 15*s**4 + 6*s**5)
    return centre + radius * np.array([0, np.sin(theta), np.cos(theta)])


def trajectory(t, knot_times, waypoints, velocities, accelerations):
    # Quintic Hermite interpolation matches q, dq, ddq at each knot.
    # Interior path samples do not impose stops.
    if t >= knot_times[-1]:
        return waypoints[-1].copy(), np.zeros(6), np.zeros(6)
    i = max(0, np.searchsorted(knot_times, t, side="right") - 1)
    h = knot_times[i + 1] - knot_times[i]
    s = (t - knot_times[i]) / h
    q0, q1 = waypoints[i:i+2]
    v0, v1 = velocities[i:i+2] * h
    a0, a1 = accelerations[i:i+2] * h**2
    delta = q1 - q0
    c0, c1, c2 = q0, v0, a0 / 2
    c3 = 10*delta - 6*v0 - 4*v1 - 1.5*a0 + 0.5*a1
    c4 = -15*delta + 8*v0 + 7*v1 + 1.5*a0 - a1
    c5 = 6*delta - 3*v0 - 3*v1 - 0.5*a0 + 0.5*a1
    q_des = c0 + c1*s + c2*s**2 + c3*s**3 + c4*s**4 + c5*s**5
    dq_des = (c1 + 2*c2*s + 3*c3*s**2 + 4*c4*s**3 + 5*c5*s**4) / h
    ddq_des = (2*c2 + 6*c3*s + 12*c4*s**2 + 20*c5*s**3) / h**2
    return q_des, dq_des, ddq_des


def inverse_dynamics(q, dq, q_des, dq_des, ddq_des, dynamics, kp, kd):
    M = dynamics.mass_matrix
    C = dynamics.coriolis_vector
    g = dynamics.gravity_vector
    # Calculate the expected acceleration
    y = ddq_des + kp * (q_des - q) + kd * (dq_des - dq)
    # Convert desired acceleration into joint torques
    tau = (M @ y) + C + g
    return tau


def build_plan(robot, mode, target):
    desired_position = np.asarray(target, dtype=float)
    if mode == "point":
        # User-editable absolute Cartesian waypoints (world frame, metres).
        positions = np.array([
            [0.5, 0.4, 0.65],
            [0.3, -0.2, 0.2],
            desired_position,
        ])
    else:
        approach_duration, path_duration, target_duration = 3.0, 12.0, 3.0
        path_times = np.linspace(0.0, path_duration, 301)
        path_positions = np.array([path_position(u / path_duration)
                                    for u in path_times])
        positions = np.vstack([path_positions, desired_position])

    if not np.all(np.isfinite(positions)) or np.any(positions[:, 2] <= 0):
        raise ValueError("Waypoints must be finite world coordinates with z > 0 m.")

    initial_angles = robot.robot_state.get_joint_angles().copy()
    initial_position = robot.robot_state.get_end_effector_position().copy()
    with robot.sim.no_rendering():
        planned = plan_joint_waypoints(robot, positions)

    if mode == "point":
        waypoints = np.vstack([initial_angles, planned])
        knot_times = np.arange(len(waypoints), dtype=float) * 3.0
        # A stop at each waypoint: zero joint velocity and acceleration.
        velocities = np.zeros_like(waypoints)
        accelerations = np.zeros_like(waypoints)
        path_start, path_duration = np.nan, np.nan
    else:
        if np.max(np.abs(np.diff(planned[:-1], axis=0))) > 0.2:
            raise ValueError("IK changed branch along the path; change its centre or radius.")
        if np.max(np.abs(planned[-2] - planned[0])) > 0.01:
            raise ValueError("IK did not close the path on the same joint branch.")
        planned[-2] = planned[0]
        waypoints = np.vstack([initial_angles, planned])
        knot_times = np.r_[0.0, approach_duration + path_times,
                           approach_duration + path_duration + target_duration]
        path_velocities = np.gradient(planned[:-1], path_times, axis=0, edge_order=2)
        path_accelerations = np.gradient(path_velocities, path_times, axis=0, edge_order=2)
        path_velocities[[0, -1]] = 0
        path_accelerations[[0, -1]] = 0
        velocities = np.vstack([np.zeros(6), path_velocities, np.zeros(6)])
        accelerations = np.vstack([np.zeros(6), path_accelerations, np.zeros(6)])
        path_start = approach_duration

    return (np.vstack([initial_position, positions]), waypoints,
            knot_times, velocities, accelerations, path_start, path_duration)


def run(mode, target, record=None):
    desired_position = np.asarray(target, dtype=float)
    robot = create_scene()
    try:
        (positions, waypoints, knot_times, velocities, accelerations,
         path_start, path_duration) = build_plan(robot, mode, target)
    except Exception:
        robot.sim.close()
        raise

    kp = np.array([10, 10, 10, 10, 10, 10])
    kd = np.array([50, 50, 50, 50, 50, 50])
    duration = knot_times[-1] + 2.0
    times, joint_angles, joint_velocities = [], [], []
    desired_angles, desired_velocities, desired_accelerations = [], [], []
    commanded_torques, applied_torques = [], []
    end_effector_positions, reference_positions = [], []
    previous_trace_position = None
    step = 0
    recorder = None

    try:
        if record is not None:
            from ur_simulation.gui_recording import GUIRecorder
            video = record or str(Path(__file__).resolve().parents[1] / "results" / f"inverse_dynamics_{mode}.mp4")
            recorder = GUIRecorder(robot.sim, video)
        while True:
            t = step * robot.sim.dt
            if t >= duration:
                break
            q = robot.robot_state.get_joint_angles().copy()
            dq = robot.robot_state.get_joint_velocities().copy()
            dynamics = robot.robot_model.get_dynamics()
            q_des, dq_des, ddq_des = trajectory(
                t, knot_times, waypoints, velocities, accelerations)
            tau = inverse_dynamics(q, dq, q_des, dq_des, ddq_des,
                                   dynamics, kp, kd)
            actual_position = robot.robot_state.get_end_effector_position().copy()
            end_effector_positions.append(actual_position)

            drawing_path = (mode == "path" and
                             path_start <= t <= path_start + path_duration)
            reference_positions.append(
                path_position((t - path_start) / path_duration)
                if drawing_path else np.full(3, np.nan))
            # Keep the original 3D debug trail in the PyBullet simulation.
            # No separate path figure is generated.
            if drawing_path and step % 20 == 0:
                if previous_trace_position is not None:
                    robot.sim.physics_client.addUserDebugLine(
                        previous_trace_position, actual_position,
                        [1, 0.65, 0], lineWidth=3)
                previous_trace_position = actual_position.copy()

            times.append(t)
            joint_angles.append(q)
            joint_velocities.append(dq)
            desired_angles.append(q_des.copy())
            desired_velocities.append(dq_des.copy())
            desired_accelerations.append(ddq_des.copy())
            commanded_torques.append(tau.copy())
            applied_torques.append(np.clip(tau, -robot.arm_joint_forces,
                                           robot.arm_joint_forces))
            robot.control_torques(tau)
            robot.sim.step()
            if recorder is not None:
                recorder.capture(t)
            step += 1
    finally:
        try:
            if recorder is not None:
                recorder.close()
            actual_position = robot.robot_state.get_end_effector_position()
            print("Mode:", mode)
            print("Desired position:", desired_position)
            print("Actual position:", actual_position)
            print("Position error [m]:", np.linalg.norm(desired_position - actual_position))
            print("Joint error [rad]:", waypoints[-1] - robot.robot_state.get_joint_angles())
        finally:
            robot.sim.close()

    joint_angles = np.array(joint_angles)
    desired_angles = np.array(desired_angles)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(times, desired_angles[:, i], color="#66c2a5", linewidth=1,
                label=f"q_{i} (desired)")
        ax.plot(times, joint_angles[:, i], color="coral", linewidth=1,
                label=f"q_{i} (actual)")
        ax.legend()
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"q_{i} (rad)")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("0.6")
        ax.tick_params(colors="0.5", length=0)

    fig.tight_layout()
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    fig.savefig(results / f"inverse_dynamics_{mode}.png", dpi=150)
    np.savez_compressed(
        results / f"inverse_dynamics_{mode}.npz",
        time=np.array(times), q=joint_angles, q_des=desired_angles,
        dq=np.array(joint_velocities),
        dq_des=np.array(desired_velocities),
        ddq_des=np.array(desired_accelerations),
        error=desired_angles - joint_angles,
        tau_command=np.array(commanded_torques),
        tau_applied=np.array(applied_torques),
        waypoint_positions=positions, waypoint_angles=waypoints,
        knot_times=knot_times,
        end_effector_positions=np.array(end_effector_positions),
        reference_positions=np.array(reference_positions),
        path_start=path_start, path_duration=path_duration,
    )
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR7e inverse-dynamics trajectory control")
    parser.add_argument("--mode", choices=["point", "path"], default="path",
                        help="Point-to-point waypoints or continuous circular trajectory")
    parser.add_argument("--target", nargs=3, type=float, default=[0.4, 0.3, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="Final end-effector position in world coordinates (m)")
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="MP4",
                        help="Save simulation video; default results/inverse_dynamics_MODE.mp4")
    args = parser.parse_args()
    run(args.mode, args.target, args.record)
