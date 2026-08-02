#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Python contract tests for the WasmRL Stable-Baselines3 adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


WASMRL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WASMRL_ROOT / "examples/python"))

from wasmrl_sb3 import WasmRLVecEnv  # noqa: E402


class WasmRLVecEnvTest(unittest.TestCase):
    """Exercise a real component through the Python and SB3 layers."""

    def test_counter_episode_and_auto_reset(self) -> None:
        component = WASMRL_ROOT / "target/wasm32-wasip2/release/counter_env.wasm"
        self.assertTrue(
            component.is_file(),
            "build counter-env for wasm32-wasip2 before running this test",
        )
        env = WasmRLVecEnv(
            component,
            num_envs=4,
            config_json='{"initial_value":0,"target":5,"max_steps":20}',
            seed=42,
        )
        try:
            env.seed(42)
            observations = env.reset()
            self.assertEqual(observations.shape, (4, 1))
            self.assertEqual(observations.dtype, np.float32)

            # SB3 emits int64 actions for Discrete spaces; the adapter must lower
            # them to the i32 tensor expected by CounterEnv.
            for _ in range(5):
                observations, rewards, dones, infos = env.step(
                    np.ones(4, dtype=np.int64)
                )

            self.assertTrue(dones.all())
            self.assertTrue(np.allclose(rewards, 1.0))
            self.assertTrue(np.allclose(observations, 0.0))
            for info in infos:
                self.assertTrue(info["wasmrl.terminated"])
                self.assertFalse(info["wasmrl.truncated"])
                self.assertEqual(info["episode"]["l"], 5)
                self.assertAlmostEqual(info["episode"]["r"], 0.96)
                self.assertTrue(np.allclose(info["terminal_observation"], [5.0]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
