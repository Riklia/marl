import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent_architecture import load_agents, PPOAgent, RandomAgent
from env_wrapper import BoardsWrapper
from misc_utils import create_animation
from run_train import build_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a saved MARL run by playing and rendering games.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the run outputs (configs/, models/, ...).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Optional stage name to evaluate. If omitted, uses the last stage or single-run config.",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=1,
        help="Number of episodes to play for validation.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the most interesting episodes after all games finish.",
    )
    parser.add_argument(
        "--n-render",
        type=int,
        default=1,
        help="How many episodes to render (picks the most interesting ones). Only used when --render is set.",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=0,
        help="Minimum number of moves for an episode to be considered interesting enough to render.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for saved GIF animations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override seed for episode randomness.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_agent_checkpoint(models_dir: Path, stage_name: str | None = None) -> Path:
    if stage_name:
        explicit = models_dir / f"{stage_name}_agents.pkl"
        if explicit.exists():
            return explicit

    final_path = models_dir / "final_agents.pkl"
    if final_path.exists():
        return final_path

    candidates = sorted(models_dir.glob("*agents.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No agent checkpoint files found in {models_dir}")
    return candidates[-1]


def select_stage_config(config: dict[str, Any], stage_name: str | None = None) -> dict[str, Any]:
    if "curriculum" in config:
        stages = config["curriculum"]
        if stage_name is None:
            return stages[-1]
        for stage in stages:
            if stage.get("name") == stage_name:
                return stage
        raise KeyError(f"Stage '{stage_name}' not found in curriculum")
    return config


def build_validation_env(run_dir: Path, stage_name: str | None = None, device: str = "cpu") -> tuple[BoardsWrapper, Any, Any, dict[str, Any]]:
    configs_dir = run_dir / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found under {run_dir}")

    resolved_config_path = configs_dir / "resolved_config.json"
    input_config_path = configs_dir / "input_config.json"
    if resolved_config_path.exists():
        config = load_json(resolved_config_path)
    elif input_config_path.exists():
        config = load_json(input_config_path)
    else:
        raise FileNotFoundError("Neither resolved_config.json nor input_config.json exist in configs/")

    stage_cfg = select_stage_config(config, stage_name)
    env = build_env(stage_cfg["game"], device)

    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found under {run_dir}")

    agent_ckpt = find_best_agent_checkpoint(models_dir, stage_name)
    sender_agent, receiver_agent = load_agents(str(agent_ckpt))

    if hasattr(sender_agent, "to"):
        sender_agent = sender_agent
    if hasattr(receiver_agent, "to"):
        receiver_agent = receiver_agent

    return env, sender_agent, receiver_agent, stage_cfg


def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_episode(env: BoardsWrapper, sender_agent: Any, receiver_agent: Any) -> tuple[float, float, int, list]:
    env.reset()
    done = False
    while not done:
        sender_state = env.sender_observe()
        sender_action, _, _ = sender_agent.choose_action(sender_state)
        _, done = env.sender_act(sender_action)
        if done:
            break

        receiver_state = env.receiver_observe()
        receiver_action, _, _ = receiver_agent.choose_action(receiver_state)
        _, done = env.receiver_act(receiver_action)

    final_reward = env.get_final_reward()
    final_performance = env.get_final_performance()
    num_moves = env.num_moves
    frames = list(env.animation_frames)

    return final_reward, final_performance, num_moves, frames


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    env, sender_agent, receiver_agent, stage_cfg = build_validation_env(
        args.run_dir,
        stage_name=args.stage,
        device="cpu",
    )

    print(f"Loaded run directory: {args.run_dir}")
    print(f"Evaluating stage: {stage_cfg.get('name', 'single_run')}")
    print(f"Game config: size={stage_cfg['game']['size']}, landmarks={stage_cfg['game']['n_landmarks']}, clues={stage_cfg['game']['n_clues']}, questions={stage_cfg['game']['n_questions']}")

    episodes = []
    for episode in range(args.n_games):
        reward, performance, num_moves, frames = run_episode(env, sender_agent, receiver_agent)
        print(f"Episode {episode}: reward={reward:.4f}, performance={performance:.4f}, moves={num_moves}")
        episodes.append((reward, performance, num_moves, frames))

    mean_reward = sum(r for r, *_ in episodes) / len(episodes)
    mean_perf = sum(p for _, p, *_ in episodes) / len(episodes)
    print("\nValidation summary:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Mean reward: {mean_reward:.4f}")
    print(f"  Mean performance: {mean_perf:.4f}")

    if args.render:
        candidates = [
            (perf, num_moves, frames)
            for _, perf, num_moves, frames in episodes
            if num_moves >= args.min_moves
        ]
        # Sort by performance desc, then by moves desc (more moves = more interesting communication)
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        to_render = candidates[:args.n_render]
        if not to_render:
            print(f"No episodes matched --min-moves={args.min_moves}; nothing rendered.")
        for i, (perf, num_moves, frames) in enumerate(to_render):
            freeze = args.fps * 2
            padded = [frames[0]] * freeze + frames + [frames[-1]] * freeze
            title = f"Performance: {perf:.2f}  Moves: {num_moves}"
            print(f"Rendering animation {i + 1}/{len(to_render)} (perf={perf:.2f}, moves={num_moves})")
            create_animation(padded, title=title, fps=args.fps)


if __name__ == "__main__":
    main()
