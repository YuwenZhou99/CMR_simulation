import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ur_simulation.rl_control.environment import ManipulationEnv, TaskConfig



class RecordingEnv(gym.Wrapper):
    """Capture after physics, before DummyVecEnv automatically resets a done episode."""
    recorder = None

    def step(self, action):
        result = self.env.step(action)
        if self.recorder is not None:
            self.recorder.capture(max(0., result[4]["time"] - self.env.unwrapped.dt))
        return result


def episode_video_path(option, folder, episode, episodes):
    if not option:
        return folder/f"episode_{episode:03d}.mp4"
    path = Path(option)
    if path.suffix.lower() != ".mp4":
        raise ValueError("--record path must end in .mp4")
    return path if episodes == 1 else path.with_name(f"{path.stem}_{episode:03d}.mp4")


def make_env(config, gui=False, monitor=None, recording=False):
    def factory():
        raw = ManipulationEnv(config, "human" if gui else None)
        if recording:
            raw = RecordingEnv(raw)
        return Monitor(raw, filename=str(monitor) if monitor else None,
                       info_keywords=("is_success", "box_error", "contact_steps", "failure_reason", "joint_rmse", "position_error"))
    return DummyVecEnv([factory])


def save_model(model, env, folder, config, algorithm, seed):
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model")
    env.save(folder / "normalize.pkl")
    sources = ["scripts/classic_control.py", "scripts/classic_control_env.py",
               "scripts/rl_control.py", "ur_simulation/rl_control/environment.py"]
    record = dict(config=asdict(config), algorithm=algorithm, seed=seed,
                  optimization=dict(gamma=model.gamma, learning_rate=float(model.lr_schedule(1.)),
                                    target_kl=getattr(model, "target_kl", None)),
                  timesteps=int(model.num_timesteps),
                  status="Training checkpoint; performance requires separate evaluation.",
                  source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (folder / "config.json").write_text(json.dumps(record, indent=2), encoding="utf8")


def load_record(folder):
    record = json.loads((Path(folder)/"config.json").read_text(encoding="utf8"))
    if record["config"].get("action_mode") is not None or record["config"].get("reward_version", 1) not in (1, 2):
        raise ValueError("This is a torque/PD experimental checkpoint, not a velocity policy. Use the original path_ppo, point_ppo or push_ppo checkpoint, or train afresh.")
    if record["algorithm"] != "ppo":
        raise ValueError("This script supports PPO checkpoints only")
    return record


def train(args):
    # Original push_ppo settings; continuation changes only scene positions.
    if args.resume:
        prior = Path(args.resume)
        record = load_record(prior)
        if record["algorithm"] != "ppo":
            raise ValueError("Continuation requires a PPO checkpoint")
        config = TaskConfig(**record["config"])
        if config.reward_version != 2:
            raise ValueError("Continuation requires the original v2 reward")
        config.random_positions = args.random_positions
    else:
        config = TaskConfig(task=args.task, mode=args.mode, target=tuple(args.target),
                            random_positions=args.random_positions,
                            reward_version=2)
    if config.random_positions and config.task != "push":
        raise ValueError("Random scene positions apply only to pushing")
    folder = Path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Use a new output directory to preserve checkpoints and logs")
    folder.mkdir(parents=True, exist_ok=True)
    base = make_env(config, monitor=folder/"monitor.csv")
    env = evaluation_env = None
    try:
        if args.resume:
            env = VecNormalize.load(prior/"normalize.pkl", base)
            env.training, env.norm_reward = True, True
            model = PPO.load(prior/"model.zip", env=env, device="cpu")
            model.set_random_seed(args.seed)
        else:
            env = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
            model = PPO("MlpPolicy", env=env, learning_rate=3e-4, gamma=.99,
                        n_steps=2048, batch_size=512, n_epochs=10,
                        gae_lambda=.95, clip_range=.2, ent_coef=1e-4,
                        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                                           net_arch=dict(pi=[64,64], vf=[128,128])),
                        seed=args.seed, device="cpu", verbose=1)
        evaluation_env = VecNormalize(make_env(config), training=False, norm_reward=False)
        evaluation_env.seed(args.seed + 10000)

        class SaveBest(BaseCallback):
            def _on_step(self):
                save_model(self.model, env, folder/"best", config, "ppo", args.seed)
                return True

        evaluation = EvalCallback(evaluation_env, eval_freq=10000,
                                  n_eval_episodes=5, deterministic=True,
                                  log_path=str(folder/"validation"),
                                  callback_on_new_best=SaveBest())
        (folder/"training_settings.json").write_text(json.dumps(dict(
            resume=str(prior.resolve()) if args.resume else None,
            starting_timesteps=int(model.num_timesteps), additional_steps=args.steps,
            validation_episodes=5, selection="mean validation reward"), indent=2))
        try:
            model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume),
                        callback=evaluation)
        except KeyboardInterrupt:
            print("Interrupted: saving current checkpoint.")
        save_model(model, env, folder, config, "ppo", args.seed)
        print(f"Checkpoint saved: {folder}")
    finally:
        (env or base).close()
        if evaluation_env is not None:
            evaluation_env.close()


def plot_push_details(data, filename):
    """PD-style plots using the contact quantity actually recorded by RL."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    path = np.vstack([data["box_start"], data["box_xyz"][:, :2]])
    ax.plot(path[:, 0], path[:, 1], label="Actual box path")
    ax.scatter(*data["box_start"], marker="s", label="Start")
    ax.scatter(*data["goal"], marker="x", s=100, label="Goal")
    if "goal_radius" in data:
        ax.add_patch(plt.Circle(data["goal"], float(data["goal_radius"]),
                               fill=False, linestyle="--", label="Goal region"))
    ax.set(xlabel="World X [m]", ylabel="World Y [m]", title="Box movement in world XY")
    ax.axis("equal")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename.with_name(filename.stem+"_box_path.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    t, force = data["time"], data["contact_force"]
    ax.plot(t, force, label="Robot / box normal force (peak per policy interval)")
    if np.any(force > 0):
        ax.fill_between(t, 0, max(float(np.max(force)), 1.), where=force > 0,
                        step="pre", alpha=.12, label="Interval containing contact")
    ax.set(xlabel="Time [s]", ylabel="Normal force [N]",
           title="Robot / box contact forces (sphere + arm links)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename.with_name(filename.stem+"_contacts.png"), dpi=160)
    plt.close(fig)


def plot_rollout(data, filename, task):
    import matplotlib.pyplot as plt
    if task == "push":
        plot_push_details(data, filename)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(data["time"], data["q_des"][:, j], "--", label="Reference")
        ax.plot(data["time"], data["q"][:, j], label="Actual")
        ax.set(xlabel="Time (s)", ylabel=f"Joint {j+1} (rad)")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename.with_suffix(".png"), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if task == "push":
        axes[0].plot(data["box_xyz"][:, 0], data["box_xyz"][:, 1], label="Box")
        axes[0].scatter(*data["goal"], marker="x", label="Goal")
        axes[0].set(xlabel="World x (m)", ylabel="World y (m)")
        axes[1].plot(data["time"], data["contact_force"])
        axes[1].set(xlabel="Time (s)", ylabel="Peak normal contact force per policy interval (N)")
    else:
        axes[0].plot(data["reference_xyz"][:, 1], data["reference_xyz"][:, 2], "--", label="Reference")
        axes[0].plot(data["ee_xyz"][:, 1], data["ee_xyz"][:, 2], label="Actual")
        axes[0].set(xlabel="World y (m)", ylabel="World z (m)")
        axes[1].plot(data["time"], data["position_error"])
        axes[1].set(xlabel="Time (s)", ylabel="Position tracking error (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(filename.with_name(filename.stem+"_task.png"), dpi=150)
    plt.close(fig)


def evaluate(args):
    prior = Path(args.checkpoint)
    record = load_record(prior)
    config = TaskConfig(**record["config"])
    scene_mode = getattr(args, "scenes", "checkpoint")
    if scene_mode != "checkpoint":
        if config.task != "push":
            raise ValueError("Scene overrides apply only to pushing")
        config.random_positions = scene_mode == "random"
        config.randomize = False  # Isolate position generalization.
    folder = Path(args.output or prior/"evaluation")
    folder.mkdir(parents=True, exist_ok=True)
    recording = getattr(args, "record", None) is not None
    if recording:
        for i in range(args.episodes):
            video = episode_video_path(args.record, folder, i, args.episodes)
            if video.exists():
                raise FileExistsError(f"Video exists; select a new --record or --output: {video}")
    base = make_env(config, args.gui or recording, recording=recording)
    capture_env = base.envs[0].env if recording else None
    env = base
    try:
        env = VecNormalize.load(prior/"normalize.pkl", base)
        env.training, env.norm_reward = False, False
        model = PPO.load(prior/"model.zip", env=env, device="cpu")
        summaries = []
        keys = ("time", "q", "dq", "q_des", "dq_des", "ee_xyz", "reference_xyz",
                "box_xyz", "action", "tau", "contact_force", "position_error", "orientation_error")
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            observation = env.reset()
            if recording:
                from ur_simulation.gui_recording import GUIRecorder
                capture_env.recorder = GUIRecorder(
                    capture_env.unwrapped.sim,
                    episode_video_path(args.record, folder, episode, args.episodes),
                    fps=min(25, config.policy_hz))
            log = {key: [] for key in keys}
            total_reward = 0.
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, dones, infos = env.step(action)
                info = infos[0]
                total_reward += float(rewards[0])
                for key in keys:
                    log[key].append(info[key])
                if dones[0]:
                    break
            if recording:
                recorder, capture_env.recorder = capture_env.recorder, None
                recorder.close()
            data = {key: np.asarray(value) for key, value in log.items()}
            data.update(goal=np.asarray(info["goal"]), box_start=np.asarray(info["box_start"]), target=np.asarray(config.target),
                        policy_type=record["algorithm"],
                        task=config.task, goal_radius=config.goal_radius, policy_hz=config.policy_hz, action_mode="velocity_inverse_dynamics")
            filename = folder/f"episode_{episode:03d}.npz"
            np.savez_compressed(filename, **data)
            plot_rollout(data, filename, config.task)
            summary = {key: info[key] for key in ("is_success", "failure_reason", "time",
                       "joint_rmse", "position_error", "box_error", "contact_steps", "clipped_substeps",
                       "pusher_contact_steps", "robot_contact_steps")}
            summary.update(episode=episode, scene_seed=args.seed+episode, reward=total_reward,
                           peak_contact_force=float(np.max(data["contact_force"])),
                           goal=info["goal"].tolist(), box_start=info["box_start"].tolist())
            summaries.append(summary)
            print(json.dumps(summary))
        result = dict(policy=record["algorithm"],
                      training_steps=record["timesteps"], seed=args.seed,
                      config=asdict(config), episodes=summaries,
                      success_rate=float(np.mean([s["is_success"] for s in summaries])))
        (folder/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    finally:
        try:
            if capture_env is not None and capture_env.recorder is not None:
                capture_env.recorder.close()
                capture_env.recorder = None
        finally:
            env.close()
    # Open saved figures only after the simulation has closed.
    import webbrowser
    for episode in range(args.episodes):
        for figure in sorted(folder.glob(f"episode_{episode:03d}*.png")):
            webbrowser.open(figure.resolve().as_uri())
    return result


def add_task_options(parser):
    parser.add_argument("--task", choices=["trajectory", "push"], default="trajectory")
    parser.add_argument("--mode", choices=["point", "path"], default="path")
    parser.add_argument("--target", nargs=3, type=float, default=[.4, .3, .5])
    parser.add_argument("--random-positions", action="store_true", help="Resample reachable box and goal XY each episode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    add_task_options(training)
    training.add_argument("--steps", type=int, default=1000000)
    training.add_argument("--resume", help="Checkpoint directory for continued training/fine-tuning")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--scenes", choices=["checkpoint", "fixed", "random"], default="checkpoint")
    evaluation.add_argument("--episodes", type=int, default=1)
    evaluation.add_argument("--gui", action="store_true", help="Show simulation during evaluation")
    evaluation.add_argument("--record", nargs="?", const="", default=None, metavar="MP4",
                            help="Save video; automatically enables the GUI")
    for sub in (training, evaluation):
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--output", required=sub is training)
    args = parser.parse_args()
    if min(getattr(args, "steps", 1), getattr(args, "episodes", 1)) <= 0:
        parser.error("Steps and episodes must be positive")
    torch.set_num_threads(1)
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
