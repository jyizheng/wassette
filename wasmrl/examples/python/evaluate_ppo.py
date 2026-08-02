#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evaluate a trained WasmRL PPO model against a random-policy baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from stable_baselines3 import PPO

from wasmrl_sb3 import WasmRLVecEnv, device_summary, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", type=Path)
    parser.add_argument("--env-config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--min-reward-improvement", type=float, default=0.20)
    return parser.parse_args()


def load_training_metadata(model_path: Path) -> Dict[str, Any]:
    metadata_path = model_path.parent / "training.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Training metadata not found at {}; pass a model produced by ppo_training.py".format(
                metadata_path
            )
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def rollout(
    env: WasmRLVecEnv,
    episodes: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    observations = env.reset()
    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []
    vector_steps = 0
    max_vector_steps = max(10_000, episodes * 1_000)

    while len(rewards) < episodes and vector_steps < max_vector_steps:
        actions = action_fn(observations)
        observations, _, dones, infos = env.step(actions)
        vector_steps += 1
        for done, info in zip(dones, infos):
            if not done or "episode" not in info:
                continue
            rewards.append(float(info["episode"]["r"]))
            lengths.append(int(info["episode"]["l"]))
            successes.append(bool(info["wasmrl.terminated"]))
            if len(rewards) >= episodes:
                break

    if len(rewards) < episodes:
        raise RuntimeError(
            "Only completed {} of {} evaluation episodes".format(len(rewards), episodes)
        )
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def make_env(
    component: Path, env_config: str, num_envs: int, seed: int
) -> WasmRLVecEnv:
    env = WasmRLVecEnv(
        component_path=component,
        num_envs=num_envs,
        config_json=env_config,
        seed=seed,
    )
    env.seed(seed)
    return env


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    metadata = load_training_metadata(model_path)
    component = (
        args.component.expanduser().resolve()
        if args.component is not None
        else Path(metadata["component"])
    )
    env_config = args.env_config or json.dumps(metadata["env_config"], separators=(",", ":"))
    num_envs = args.num_envs or int(metadata["num_envs"])
    if args.episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    device = resolve_device(args.device)
    model = PPO.load(str(model_path), device=device)

    trained_env = make_env(component, env_config, num_envs, args.seed)
    try:
        trained = rollout(
            trained_env,
            args.episodes,
            lambda obs: model.predict(obs, deterministic=True)[0],
        )
    finally:
        trained_env.close()

    random_generator = np.random.default_rng(args.seed)
    random_env = make_env(component, env_config, num_envs, args.seed)
    try:
        action_count = int(random_env.action_space.n)

        def random_policy(observations: np.ndarray) -> np.ndarray:
            return random_generator.integers(
                0, action_count, size=observations.shape[0], dtype=np.int32
            )

        random_baseline = rollout(random_env, args.episodes, random_policy)
    finally:
        random_env.close()

    reward_improvement = trained["mean_reward"] - random_baseline["mean_reward"]
    result = {
        "model": str(model_path),
        "component": str(component),
        "device": device,
        "accelerator": device_summary(device),
        "trained": trained,
        "random_baseline": random_baseline,
        "mean_reward_improvement": reward_improvement,
        "thresholds": {
            "min_success_rate": args.min_success_rate,
            "min_reward_improvement": args.min_reward_improvement,
        },
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_path.parent / "evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("trained: success={:.1%}, reward={:.4f}, length={:.2f}".format(
        trained["success_rate"], trained["mean_reward"], trained["mean_length"]
    ))
    print("random:  success={:.1%}, reward={:.4f}, length={:.2f}".format(
        random_baseline["success_rate"],
        random_baseline["mean_reward"],
        random_baseline["mean_length"],
    ))
    print("mean reward improvement: {:.4f}".format(reward_improvement))
    print("Evaluation: {}".format(output_path))

    failures: List[str] = []
    if trained["success_rate"] < args.min_success_rate:
        failures.append(
            "success rate {:.1%} < {:.1%}".format(
                trained["success_rate"], args.min_success_rate
            )
        )
    if reward_improvement < args.min_reward_improvement:
        failures.append(
            "reward improvement {:.4f} < {:.4f}".format(
                reward_improvement, args.min_reward_improvement
            )
        )
    if failures:
        print("Evaluation failed: {}".format("; ".join(failures)))
        return 2
    print("Evaluation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
