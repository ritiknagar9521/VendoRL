"""Train the PPO agent with several random seeds.
Usage: python train.py                       (Default : 3 seeds, 400 iterations)
       python train.py --seeds 0 1 2 3 4 --iters 600
"""
import argparse, os, random
import numpy as np
import torch
from vending_env import VendingEnv
from ppo import PPOAgent

REWARD_SCALE = 0.01   # rewards are in hundreds of Rs; scaling keeps network targets small and stable


def collect(env, agent, n_episodes):
    """Play n_episodes with the current policy and store what the PPO update needs."""
    b = {k: [] for k in ["states", "actions", "log_probs", "rewards", "values", "dones"]}
    profits = []
    for _ in range(n_episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            a, lp, v = agent.act(s)
            s2, r, done, _ = env.step(a)
            for k, x in zip(b, [s, a, lp, r * REWARD_SCALE, v, float(done)]):
                b[k].append(x)
            s, total = s2, total + r
        profits.append(total)                       # true profit in Rs, used for the learning curve
    b["states"] = np.array(b["states"], dtype=np.float32)
    b["actions"] = np.array(b["actions"], dtype=np.int64)
    b["log_probs"] = np.array(b["log_probs"], dtype=np.float32)
    return b, float(np.mean(profits))


def train(seed, iters, batch_episodes):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # makes the run reproducible
    env = VendingEnv(seed=seed)
    agent = PPOAgent(env.state_dim, env.action_dim)
    curve = []
    for it in range(iters):
        agent.entropy_coef = 0.02 * (1 - it / iters) + 0.002          # explore a lot early, less later
        batch, mean_profit = collect(env, agent, batch_episodes)
        agent.update(batch)
        curve.append(mean_profit)
        if (it + 1) % 50 == 0:
            print(f"[seed {seed}] iter {it + 1}/{iters}  avg profit Rs {np.mean(curve[-10:]):.1f}")
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    agent.save(f"models/ppo_seed{seed}.pt")
    np.save(f"results/ppo_seed{seed}_curve.npy", np.array(curve))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--batch_episodes", type=int, default=32)      # 32 episodes ~ 380 steps per update
    a = p.parse_args()
    for seed in a.seeds:
        train(seed, a.iters, a.batch_episodes)
    print("Done. Now run: python evaluate.py")
