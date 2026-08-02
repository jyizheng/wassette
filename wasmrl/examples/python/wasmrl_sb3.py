#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Stable-Baselines3 integration helpers for WasmRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import wasmrl_py as wasmrl
from stable_baselines3.common.vec_env import VecEnv


IndexSpec = Optional[Union[int, Iterable[int]]]


def resolve_device(requested: str) -> str:
    """Resolve an SB3 device and fail early when requested hardware is absent."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch wheel and expose the GPU to this process."
        )
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable in this PyTorch build.")
    return requested


def device_summary(device: str) -> Dict[str, Any]:
    """Return serializable accelerator diagnostics for logs and metadata."""
    summary: Dict[str, Any] = {
        "requested_device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        requested_index = torch.device(device).index
        index = requested_index if requested_index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "cuda_device": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_memory_gb": round(properties.total_memory / (1024**3), 2),
            }
        )
    return summary


class WasmRLVecEnv(VecEnv):
    """Adapt WasmRL's vector API to the Stable-Baselines3 VecEnv contract."""

    def __init__(
        self,
        component_path: Union[str, Path],
        num_envs: int = 8,
        config_json: str = "{}",
        max_memory_mb: int = 64,
        fuel_per_step: int = 1_000_000,
        timeout_step_ms: int = 100,
        timeout_reset_ms: int = 500,
        seed: int = 0,
    ) -> None:
        component = Path(component_path).expanduser().resolve()
        if not component.is_file():
            raise FileNotFoundError(
                "Wasm environment component not found at {}. Build it with "
                "`cargo build -p counter-env --target wasm32-wasip2 --release`.".format(
                    component
                )
            )
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")

        self.render_mode = None
        self._closed = False
        self._actions: Optional[np.ndarray] = None
        self.component_path = str(component)
        self.config_json = config_json

        config = wasmrl.EnvConfig(
            config_json=config_json,
            num_envs=num_envs,
            max_memory_mb=max_memory_mb,
            fuel_per_step=fuel_per_step,
            timeout_step_ms=timeout_step_ms,
            timeout_reset_ms=timeout_reset_ms,
            auto_reset=True,
            seed=seed,
        )
        self.wasmrl_env = wasmrl.WasmVecEnv(str(component), config)
        observation_space = self._observation_space(
            self.wasmrl_env.single_observation_space
        )
        action_space = self._action_space(self.wasmrl_env.single_action_space)
        super().__init__(num_envs, observation_space, action_space)

    @staticmethod
    def _observation_space(space: Any) -> gym.Space:
        shape = tuple(space.shape)
        low = np.asarray(space.low, dtype=np.float32).reshape(shape)
        high = np.asarray(space.high, dtype=np.float32).reshape(shape)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def _action_space(space: Any) -> gym.Space:
        if hasattr(space, "n"):
            return gym.spaces.Discrete(int(space.n), start=int(space.start))
        shape = tuple(space.shape)
        low = np.asarray(space.low, dtype=np.float32).reshape(shape)
        high = np.asarray(space.high, dtype=np.float32).reshape(shape)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self) -> np.ndarray:
        reset_seed = next((seed for seed in self._seeds if seed is not None), None)
        observations, _ = self.wasmrl_env.reset(seed=reset_seed)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return np.asarray(observations, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        if self._actions is not None:
            raise RuntimeError("step_async called while another step is pending")
        if isinstance(self.action_space, gym.spaces.Discrete):
            self._actions = np.asarray(actions, dtype=np.int32).reshape(self.num_envs)
        else:
            self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called without step_async")

        actions = self._actions
        self._actions = None
        observations, rewards, terminated, truncated, batch_info = self.wasmrl_env.step(
            actions
        )
        observations = np.asarray(observations, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        dones = np.logical_or(terminated, truncated)

        final_observations = batch_info.get(
            "final_observation", [None] * self.num_envs
        )
        final_rewards = batch_info.get(
            "final_episode_rewards", [None] * self.num_envs
        )
        final_lengths = batch_info.get(
            "final_episode_lengths", [None] * self.num_envs
        )

        infos: List[Dict[str, Any]] = []
        for index in range(self.num_envs):
            info: Dict[str, Any] = {
                "TimeLimit.truncated": bool(truncated[index] and not terminated[index]),
                "wasmrl.terminated": bool(terminated[index]),
                "wasmrl.truncated": bool(truncated[index]),
            }
            if dones[index]:
                terminal_observation = final_observations[index]
                if terminal_observation is not None:
                    info["terminal_observation"] = np.asarray(
                        terminal_observation, dtype=np.float32
                    )
                if final_rewards[index] is not None and final_lengths[index] is not None:
                    info["episode"] = {
                        "r": float(final_rewards[index]),
                        "l": int(final_lengths[index]),
                    }
            infos.append(info)

        return observations, rewards, dones, infos

    def close(self) -> None:
        if not self._closed:
            self.wasmrl_env.close()
            self._closed = True

    def render(self, mode: str = "human") -> None:
        del mode
        return None

    def get_attr(self, attr_name: str, indices: IndexSpec = None) -> List[Any]:
        selected = self._get_indices(indices)
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
        elif hasattr(self.wasmrl_env, attr_name):
            value = getattr(self.wasmrl_env, attr_name)
        else:
            raise AttributeError("WasmRLVecEnv has no attribute {!r}".format(attr_name))
        return [value for _ in selected]

    def set_attr(self, attr_name: str, value: Any, indices: IndexSpec = None) -> None:
        selected = self._get_indices(indices)
        if len(selected) != self.num_envs:
            raise NotImplementedError("WasmRL attributes can only be set for all environments")
        if not hasattr(self, attr_name):
            raise AttributeError("WasmRLVecEnv has no mutable attribute {!r}".format(attr_name))
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: IndexSpec = None,
        **method_kwargs: Any,
    ) -> List[Any]:
        del method_kwargs
        selected = self._get_indices(indices)
        if method_name == "snapshot":
            snapshots = self.wasmrl_env.snapshot_all()
            return [snapshots[index] for index in selected]
        if method_name == "restore":
            if len(selected) != self.num_envs or len(method_args) != 1:
                raise ValueError("restore requires one complete snapshot list")
            self.wasmrl_env.restore_all(method_args[0])
            return [None for _ in selected]
        raise AttributeError("WasmRL environment has no method {!r}".format(method_name))

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: IndexSpec = None
    ) -> List[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]


# Preserve the original example's public class name.
WasmRLVecEnvWrapper = WasmRLVecEnv
