# Example usage of run_train.py
#
# To execute the training pipeline defined in this script, you need to provide
# a json config file that specifies the training parameters and model configuration.
#
# Usage:
#     python run_train.py --config /path/to/your/config.json
#

import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent_architecture import AgentParams, PPOAgent, RandomAgent, save_agents
from env_internals import BoardsImplementation
from env_wrapper import BoardsWrapper
from misc_utils import smooth_list
from training_loop import save_stats, train_agents


DEFAULT_OUTPUT_ROOT = Path("experiments")


@dataclass
class RunArtifacts:
    run_dir: Path
    stats_dir: Path
    plots_dir: Path
    models_dir: Path
    configs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MARL agents from a JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where experiment outputs should be stored.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name. Defaults to '<config_stem>_<timestamp>'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Optional override for config device (e.g. cpu/cuda/cuda:0). If none, inferred by PyTorch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for torch/env/agent seeds. Applied on top-level config or each curriculum stage.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_run_name(config_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config_path.stem}_{timestamp}"


def ensure_required(config: dict[str, Any], keys: list[str], *, context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing required keys in {context}: {missing}")


def choose_device(config: dict[str, Any], cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    requested = config.get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_artifact_dirs(output_root: Path, run_name: str) -> RunArtifacts:
    run_dir = output_root / run_name
    stats_dir = run_dir / "stats"
    plots_dir = run_dir / "plots"
    models_dir = run_dir / "models"
    configs_dir = run_dir / "configs"
    for directory in (run_dir, stats_dir, plots_dir, models_dir, configs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        stats_dir=stats_dir,
        plots_dir=plots_dir,
        models_dir=models_dir,
        configs_dir=configs_dir,
    )


def build_env(game_cfg: dict[str, Any], device: str) -> BoardsWrapper:
    ensure_required(
        game_cfg,
        [
            "size",
            "n_landmarks",
            "n_clues",
            "n_questions",
            "max_moves",
            "history_len",
            "instant_reward_multiplier",
            "end_reward_multiplier",
        ],
        context="game config",
    )

    env_internals = BoardsImplementation(
        game_cfg["size"],
        game_cfg["n_landmarks"],
        game_cfg["n_clues"],
        game_cfg["n_questions"],
        seed=game_cfg.get("env_seed"),
        receiver_goal_visibility_mode=game_cfg.get("receiver_goal_visibility_mode", "none"),
        receiver_goal_visibility_ratio=game_cfg.get("receiver_goal_visibility_ratio", 0.0),
        disable_sender=game_cfg.get("disable_sender", False),
    )
    return BoardsWrapper(
        env_internals,
        game_cfg["max_moves"],
        game_cfg["history_len"],
        game_cfg["instant_reward_multiplier"],
        game_cfg["end_reward_multiplier"],
        device,
        shaping_multiplier=float(game_cfg.get("shaping_multiplier", 0.0)),
    )


def build_agent_params(training_cfg: dict[str, Any], seed: int | None) -> AgentParams:
    ensure_required(
        training_cfg,
        ["gamma", "alpha", "gae_lambda", "policy_clip", "batch_size", "n_epochs"],
        context="training config",
    )
    return AgentParams(
        training_cfg["gamma"],
        training_cfg["alpha"],
        training_cfg["gae_lambda"],
        training_cfg["policy_clip"],
        training_cfg["batch_size"],
        training_cfg["n_epochs"],
        seed,
    )


def infer_hidden_size(game_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> int:
    if "hidden_size" in agent_cfg:
        return int(agent_cfg["hidden_size"])
    return int(game_cfg["size"]) ** 2 * 32


def build_agent(
    *,
    role: str,
    env: BoardsWrapper,
    game_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    device: str,
):
    kind = agent_cfg.get("kind", "ppo").lower()
    if kind == "random":
        permitted_actions = list(range(env.sender_n_actions))
        return RandomAgent(permitted_actions)

    if kind != "ppo":
        raise ValueError(f"Unsupported agent kind for '{role}': {kind}")

    n_actions = env.sender_n_actions if role == "sender" else env.receiver_n_actions
    n_channels = env.sender_n_channels if role == "sender" else env.receiver_n_channels
    seed = agent_cfg.get("seed")
    params = build_agent_params(training_cfg, seed)
    hidden_size = infer_hidden_size(game_cfg, agent_cfg)
    return PPOAgent(
        game_cfg["size"],
        game_cfg["history_len"],
        n_actions,
        hidden_size,
        device,
        params,
        n_channels_per_frame=n_channels,
    )


def is_reusable_agent(existing_agent: Any, role: str, env: BoardsWrapper) -> bool:
    if existing_agent is None:
        return False
    if isinstance(existing_agent, RandomAgent):
        return True

    n_actions = env.sender_n_actions if role == "sender" else env.receiver_n_actions
    for attr_name in ("n_actions", "action_dim"):
        if hasattr(existing_agent, attr_name):
            return int(getattr(existing_agent, attr_name)) == int(n_actions)

    # If we cannot inspect action space, keep the agent and let runtime fail loudly
    # only if the underlying implementation is incompatible.
    return True


def maybe_rebuild_agents(
    *,
    env: BoardsWrapper,
    game_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    agents_cfg: dict[str, Any],
    existing_sender: Any,
    existing_receiver: Any,
    device: str,
    allow_stage_warm_start: bool,
) -> tuple[Any, Any]:
    sender_cfg = agents_cfg.get("sender", {"kind": "ppo"})
    receiver_cfg = agents_cfg.get("receiver", {"kind": "ppo"})

    if allow_stage_warm_start and is_reusable_agent(existing_sender, "sender", env):
        sender_agent = existing_sender
    else:
        sender_agent = build_agent(
            role="sender",
            env=env,
            game_cfg=game_cfg,
            training_cfg=training_cfg,
            agent_cfg=sender_cfg,
            device=device,
        )

    if allow_stage_warm_start and is_reusable_agent(existing_receiver, "receiver", env):
        receiver_agent = existing_receiver
    else:
        receiver_agent = build_agent(
            role="receiver",
            env=env,
            game_cfg=game_cfg,
            training_cfg=training_cfg,
            agent_cfg=receiver_cfg,
            device=device,
        )

    return sender_agent, receiver_agent


def merge_stats(accumulator: dict[str, list[Any]], new_stats: dict[str, list[Any]]) -> dict[str, list[Any]]:
    for key, values in new_stats.items():
        accumulator.setdefault(key, [])
        accumulator[key].extend(values)
    return accumulator


def resolve_output_root(raw_config: dict[str, Any], args: argparse.Namespace) -> Path:
    if "output_dir" in raw_config and raw_config["output_dir"]:
        return Path(raw_config["output_dir"])
    return Path(args.output_root)


def plot_series(values: list[float], title: str, output_path: Path, smoothing_window: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        return

    smoothed = smooth_list(values, smoothing_window)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_plots(stats: dict[str, list[Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not stats or not stats.get("performances_dist"):
        return

    smoothing_window = max(len(stats["performances_dist"]) // 20, 1)
    plot_series(stats["performances_dist"], "Overall performance", output_dir / "overall_performance.png", smoothing_window)
    plot_series(stats["final_rewards_dist"], "Final rewards", output_dir / "final_rewards.png", smoothing_window)
    plot_series(stats["episode_lengths_dist"], "Episode lengths", output_dir / "episode_lengths.png", smoothing_window)

    episode_lengths = stats["episode_lengths_dist"]
    if episode_lengths:
        receiver_avg = [
            total / max(length, 1)
            for total, length in zip(stats["receiver_instant_rewards_dist"], episode_lengths)
        ]
        sender_avg = [
            total / max(length, 1)
            for total, length in zip(stats["sender_instant_rewards_dist"], episode_lengths)
        ]
        useless_sender = [
            total / max(length, 1)
            for total, length in zip(stats["useless_actions_sender_dist"], episode_lengths)
        ]
        useless_receiver = [
            total / max(length, 1)
            for total, length in zip(stats["useless_actions_receiver_dist"], episode_lengths)
        ]
        plot_series(receiver_avg, "Avg receiver instant rewards per episode", output_dir / "avg_receiver_instant_rewards.png", smoothing_window)
        plot_series(sender_avg, "Avg sender instant rewards per episode", output_dir / "avg_sender_instant_rewards.png", smoothing_window)
        plot_series(useless_sender, "Avg useless sender actions per episode", output_dir / "avg_useless_sender_actions.png", smoothing_window)
        plot_series(useless_receiver, "Avg useless receiver actions per episode", output_dir / "avg_useless_receiver_actions.png", smoothing_window)


def save_stage_outputs(
    *,
    artifacts: RunArtifacts,
    stage_name: str,
    stage_config: dict[str, Any],
    stage_stats: dict[str, list[Any]],
    sender_agent: Any,
    receiver_agent: Any,
    make_plots: bool,
) -> None:
    safe_stage_name = stage_name.replace(" ", "_")
    save_json(stage_config, artifacts.configs_dir / f"{safe_stage_name}.json")
    save_stats(stage_stats, str(artifacts.stats_dir / f"{safe_stage_name}_stats.pkl"))
    save_agents(sender_agent, receiver_agent, str(artifacts.models_dir / f"{safe_stage_name}_agents.pkl"))
    if make_plots:
        stage_plot_dir = artifacts.plots_dir / safe_stage_name
        stage_plot_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(stage_stats, stage_plot_dir)


def normalize_single_run_config(config: dict[str, Any]) -> dict[str, Any]:
    ensure_required(config, ["game", "training"], context="single-run config")
    merged_game = deepcopy(config["game"])
    if "seeds" in config and isinstance(config["seeds"], dict):
        merged_game["env_seed"] = config["seeds"].get("env_seed", merged_game.get("env_seed"))

    agents_cfg = deepcopy(config.get("agents", {}))
    seeds_cfg = deepcopy(config.get("seeds", {}))
    agents_cfg.setdefault("sender", {})
    agents_cfg.setdefault("receiver", {})
    if seeds_cfg:
        agents_cfg["sender"].setdefault("seed", seeds_cfg.get("sender_seed"))
        agents_cfg["receiver"].setdefault("seed", seeds_cfg.get("receiver_seed"))

    return {
        "name": config.get("name", "train"),
        "device": config.get("device", "auto"),
        "game": merged_game,
        "training": deepcopy(config["training"]),
        "agents": agents_cfg,
        "curriculum": [
            {
                "name": config.get("name", "train"),
                "game": merged_game,
                "training": deepcopy(config["training"]),
                "agents": agents_cfg,
                "warm_start_from_previous_stage": False,
            }
        ],
    }


def normalize_curriculum_config(config: dict[str, Any]) -> dict[str, Any]:
    ensure_required(config, ["base", "curriculum"], context="curriculum config")
    ensure_required(config["base"], ["game", "training"], context="curriculum base config")

    base = deepcopy(config["base"])
    base_game = deepcopy(base["game"])
    if "seeds" in base and isinstance(base["seeds"], dict):
        base_game["env_seed"] = base["seeds"].get("env_seed", base_game.get("env_seed"))

    base_agents = deepcopy(base.get("agents", {}))
    base_seeds = deepcopy(base.get("seeds", {}))
    base_agents.setdefault("sender", {})
    base_agents.setdefault("receiver", {})
    if base_seeds:
        base_agents["sender"].setdefault("seed", base_seeds.get("sender_seed"))
        base_agents["receiver"].setdefault("seed", base_seeds.get("receiver_seed"))

    normalized_stages: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(config["curriculum"]):
        stage_name = raw_stage.get("name", f"stage_{index + 1}")
        stage_game = deep_update(base_game, raw_stage.get("game", {}))
        stage_training = deep_update(base["training"], raw_stage.get("training", {}))
        stage_agents = deep_update(base_agents, raw_stage.get("agents", {}))
        if "seeds" in raw_stage:
            seeds_cfg = raw_stage["seeds"]
            stage_game["env_seed"] = seeds_cfg.get("env_seed", stage_game.get("env_seed"))
            stage_agents["sender"]["seed"] = seeds_cfg.get("sender_seed", stage_agents["sender"].get("seed"))
            stage_agents["receiver"]["seed"] = seeds_cfg.get("receiver_seed", stage_agents["receiver"].get("seed"))

        normalized_stages.append(
            {
                "name": stage_name,
                "game": stage_game,
                "training": stage_training,
                "agents": stage_agents,
                "warm_start_from_previous_stage": raw_stage.get("warm_start_from_previous_stage", index > 0),
            }
        )

    return {
        "name": config.get("name", "curriculum_train"),
        "device": config.get("device", base.get("device", "auto")),
        "base": base,
        "curriculum": normalized_stages,
    }


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if "curriculum" in config and "base" in config:
        return normalize_curriculum_config(config)
    return normalize_single_run_config(config)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = deepcopy(config)
    if args.device is not None:
        updated["device"] = args.device

    if args.seed is None:
        return updated

    for stage in updated["curriculum"]:
        stage["game"]["env_seed"] = args.seed
        stage.setdefault("agents", {}).setdefault("sender", {})["seed"] = args.seed + 1
        stage.setdefault("agents", {}).setdefault("receiver", {})["seed"] = args.seed + 2
    return updated


def print_stage_header(stage_idx: int, num_stages: int, stage_name: str, stage_cfg: dict[str, Any], device: str) -> None:
    print("=" * 80)
    print(f"Stage {stage_idx}/{num_stages}: {stage_name}")
    print(f"Device: {device}")
    print(
        "Game:",
        {
            k: stage_cfg["game"][k]
            for k in (
                "size",
                "n_landmarks",
                "n_clues",
                "n_questions",
                "max_moves",
                "history_len",
            )
            if k in stage_cfg["game"]
        },
    )
    print(
        "Training:",
        {
            k: stage_cfg["training"][k]
            for k in (
                "training_epochs",
                "n_episodes",
                "batch_size",
                "n_epochs",
                "alpha",
            )
            if k in stage_cfg["training"]
        },
    )


def run_stage(
    *,
    stage_cfg: dict[str, Any],
    sender_agent: Any,
    receiver_agent: Any,
    device: str,
) -> tuple[dict[str, list[Any]], Any, Any]:
    env = build_env(stage_cfg["game"], device)
    sender_agent, receiver_agent = maybe_rebuild_agents(
        env=env,
        game_cfg=stage_cfg["game"],
        training_cfg=stage_cfg["training"],
        agents_cfg=stage_cfg.get("agents", {}),
        existing_sender=sender_agent,
        existing_receiver=receiver_agent,
        device=device,
        allow_stage_warm_start=stage_cfg.get("warm_start_from_previous_stage", False),
    )

    stage_stats: dict[str, list[Any]] = {}
    training_epochs = int(stage_cfg["training"].get("training_epochs", 1))
    n_episodes = int(stage_cfg["training"]["n_episodes"])
    for epoch in range(training_epochs):
        print(f"  Training epoch {epoch + 1}/{training_epochs}...")
        learn_interval = int(stage_cfg["training"].get("learn_interval", 32))
        epoch_stats = train_agents(env, sender_agent, receiver_agent, n_episodes, learn_interval)
        merge_stats(stage_stats, epoch_stats)

    return stage_stats, sender_agent, receiver_agent


def main():
    args = parse_args()
    raw_config = load_json(args.config)
    normalized = normalize_config(raw_config)
    normalized = apply_cli_overrides(normalized, args)

    device = choose_device(normalized, args.device)
    config_path = Path(args.config)
    run_name = make_run_name(config_path, args.run_name)
    output_root = resolve_output_root(raw_config, args)
    artifacts = create_artifact_dirs(output_root, run_name)

    set_global_seeds(args.seed)
    save_json(raw_config, artifacts.configs_dir / "input_config.json")
    save_json(normalized, artifacts.configs_dir / "resolved_config.json")

    sender_agent = None
    receiver_agent = None
    all_stats: dict[str, list[Any]] = {}
    stage_summaries: list[dict[str, Any]] = []

    total_stages = len(normalized["curriculum"])
    for idx, stage_cfg in enumerate(normalized["curriculum"], start=1):
        print_stage_header(idx, total_stages, stage_cfg["name"], stage_cfg, device)
        stage_stats, sender_agent, receiver_agent = run_stage(
            stage_cfg=stage_cfg,
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            device=device,
        )
        merge_stats(all_stats, stage_stats)
        save_stage_outputs(
            artifacts=artifacts,
            stage_name=stage_cfg["name"],
            stage_config=stage_cfg,
            stage_stats=stage_stats,
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            make_plots=not args.no_plots,
        )

        stage_summary = {
            "stage": stage_cfg["name"],
            "episodes": len(stage_stats.get("performances_dist", [])),
            "final_mean_performance_last_50": float(np.mean(stage_stats.get("performances_dist", [])[-50:]))
            if stage_stats.get("performances_dist")
            else None,
            "final_mean_reward_last_50": float(np.mean(stage_stats.get("final_rewards_dist", [])[-50:]))
            if stage_stats.get("final_rewards_dist")
            else None,
        }
        stage_summaries.append(stage_summary)
        print(f"Finished stage '{stage_cfg['name']}'. Summary: {stage_summary}")

    save_stats(all_stats, str(artifacts.stats_dir / "all_stats.pkl"))
    if sender_agent is not None and receiver_agent is not None:
        save_agents(sender_agent, receiver_agent, str(artifacts.models_dir / "final_agents.pkl"))
    if not args.no_plots:
        generate_plots(all_stats, artifacts.plots_dir / "combined")
    save_json({"run_name": run_name, "stages": stage_summaries}, artifacts.run_dir / "summary.json")

    print("=" * 80)
    print(f"Training finished. Outputs saved to: {artifacts.run_dir}")

if __name__ == "__main__":
    main()
