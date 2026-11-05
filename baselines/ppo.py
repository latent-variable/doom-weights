"""The trained yardstick: PPO on the same inputs and the same network shape.

    .venv/bin/python -m baselines.ppo --scenario basic --mode objects --steps 2000000

The policy network is exactly harness.net's shape (D -> 128 -> 128 -> A, ReLU), so its
weights are exported to the same .npz and scored by the same harness.evaluate, played
greedily (argmax), the way a written network is played. Checkpoints along the way are
exported too, so a written network can be placed on the training curve: "as good as
PPO after N steps". Uses no budget: it runs on this Mac's CPU.
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from harness import env as E

REPO = Path(__file__).resolve().parent.parent


class DoomGym(gym.Env):
    def __init__(self, scenario, mode, seed):
        self.scenario, self.mode, self.seed_ = scenario, mode, seed
        self.game = None
        d, a = E.input_size(scenario), len(E.SCENARIOS[scenario]["actions"])
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (d,), np.float32)
        self.action_space = gym.spaces.Discrete(a)

    def reset(self, seed=None, options=None):
        if self.game is None:
            self.game = E.Doom(self.scenario, self.mode, self.seed_)
        return self.game.reset(), {}

    def step(self, action):
        x, r, done = self.game.step(int(action))
        if done:
            x = np.zeros(self.observation_space.shape, np.float32)
        return x, float(r), done, False, {}


def export(model, path):
    pol = model.policy
    lin = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)] + [pol.action_net]
    assert len(lin) == 3
    w = {}
    for i, l in enumerate(lin, 1):
        w[f"W{i}"] = l.weight.detach().cpu().numpy().T.astype(np.float32)
        w[f"b{i}"] = l.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, **w)


class Checkpoints(BaseCallback):
    def __init__(self, folder, every):
        super().__init__()
        self.folder, self.every, self.next = folder, every, every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            export(self.model, self.folder / f"steps{self.num_timesteps:09d}.npz")
            self.next += self.every
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(E.SCENARIOS))
    ap.add_argument("--mode", required=True, choices=sorted(E.MODES))
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--checkpoints", type=int, default=8)
    a = ap.parse_args()
    out = REPO / "results" / "ppo" / f"{a.scenario}-{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = VecMonitor(SubprocVecEnv([(lambda i=i: DoomGym(a.scenario, a.mode, 1000 + i)) for i in range(a.envs)]))
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, n_epochs=4, learning_rate=2.5e-4,
                ent_coef=0.01, gamma=0.99, seed=0, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.ReLU))
    t = time.time()
    model.learn(a.steps, callback=Checkpoints(out, a.steps // a.checkpoints))
    export(model, out / "final.npz")
    (out / "train.json").write_text(json.dumps({"steps": a.steps, "seconds": round(time.time() - t),
                                                "envs": a.envs}, indent=1))
    venv.close()
    print(f"{a.scenario} {a.mode}: {a.steps} steps in {time.time() - t:.0f} s -> {out}")


if __name__ == "__main__":
    main()
