#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Train a Stable-Baselines3 PPO policy against a WasmRL component."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from wasmrl_sb3 import WasmRLVecEnv, device_summary, resolve_device


WASMRL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPONENT = WASMRL_ROOT / "target/wasm32-wasip2/release/counter_env.wasm"
DEFAULT_ENV_CONFIG = '{"initial_value":0,"target":5,"max_steps":20}'


class EpisodeLogger(BaseCallback):
    """Collect completed episode statistics from the WasmRL adapter."""

    def __init__(self, log_every: int = 100) -> None:
        super().__init__(verbose=0)
        self.log_every = log_every
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_rewards.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))
            if self.log_every and len(self.episode_rewards) % self.log_every == 0:
                print(
                    "episodes={} mean_reward={:.4f} mean_length={:.2f}".format(
                        len(self.episode_rewards),
                        np.mean(self.episode_rewards[-self.log_every :]),
                        np.mean(self.episode_lengths[-self.log_every :]),
                    ),
                    flush=True,
                )
        return True


def compatible_batch_size(num_envs: int, n_steps: int, requested: int) -> int:
    """Choose the largest requested-or-smaller batch that divides the rollout."""
    rollout_size = num_envs * n_steps
    upper = min(requested, rollout_size)
    for candidate in range(upper, 0, -1):
        if rollout_size % candidate == 0:
            return candidate
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo-counter"))
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-memory-mb", type=int, default=64)
    parser.add_argument("--fuel-per-step", type=int, default=1_000_000)
    parser.add_argument("--timeout-step-ms", type=int, default=100)
    parser.add_argument("--timeout-reset-ms", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs < 1 or args.n_steps < 2 or args.total_timesteps < 1:
        raise ValueError("num-envs, total-timesteps, and n-steps must be positive")
    try:
        parsed = json.loads(args.env_config)
    except json.JSONDecodeError as error:
        raise ValueError("--env-config must be valid JSON: {}".format(error)) from error
    if not isinstance(parsed, dict):
        raise ValueError("--env-config must contain a JSON object")


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed, using_cuda=device.startswith("cuda"))

    component = args.component.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = compatible_batch_size(args.num_envs, args.n_steps, args.batch_size)
    if batch_size != args.batch_size:
        print(
            "Adjusted batch size from {} to {} so it divides rollout size {}.".format(
                args.batch_size, batch_size, args.num_envs * args.n_steps
            )
        )

    accelerator = device_summary(device)
    print("WasmRL PPO training")
    print("  component: {}".format(component))
    print("  device: {}".format(device))
    if "cuda_device_name" in accelerator:
        print("  gpu: {} ({} GiB)".format(
            accelerator["cuda_device_name"], accelerator["cuda_memory_gb"]
        ))
    print("  vec envs: {} (current runtime dispatch is serial)".format(args.num_envs))
    print("  timesteps: {}".format(args.total_timesteps))

    env = WasmRLVecEnv(
        component_path=component,
        num_envs=args.num_envs,
        config_json=args.env_config,
        max_memory_mb=args.max_memory_mb,
        fuel_per_step=args.fuel_per_step,
        timeout_step_ms=args.timeout_step_ms,
        timeout_reset_ms=args.timeout_reset_ms,
        seed=args.seed,
    )
    env.seed(args.seed)
    callback = EpisodeLogger()
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        seed=args.seed,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=args.progress,
        )
        elapsed = time.perf_counter() - started
        model_path = output_dir / "model"
        model.save(str(model_path))
    finally:
        env.close()

    metadata: Dict[str, Any] = {
        "component": str(component),
        "env_config": json.loads(args.env_config),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_timesteps": args.total_timesteps,
        "trained_timesteps": model.num_timesteps,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "environment_steps_per_second": model.num_timesteps / elapsed,
        "completed_episodes": len(callback.episode_rewards),
        "mean_training_reward_last_100": (
            float(np.mean(callback.episode_rewards[-100:]))
            if callback.episode_rewards
            else None
        ),
        "device": device,
        "accelerator": accelerator,
        "stable_baselines3_version": stable_baselines3.__version__,
    }
    (output_dir / "training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Training complete in {:.2f}s ({:.1f} env steps/s)".format(
        elapsed, metadata["environment_steps_per_second"]
    ))
    print("Model: {}".format(output_dir / "model.zip"))
    print("Metadata: {}".format(output_dir / "training.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
