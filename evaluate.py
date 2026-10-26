"""Compare trained PPO agents with baselines on the SAME 500 unseen test episodes.
Prints a table and saves results/dashboard.png.   Usage: python evaluate.py
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vending_env import VendingEnv
from ppo import PPOAgent

N_EVAL, SEED0 = 500, 10000          # test episodes use seeds 10000-10499 (never used in training)
DAYS = ["Low demand day", "Normal day", "High demand day"]

def rollout(env, policy, n=N_EVAL):
    """Run a policy for n episodes; record profit, unsold units, day type and prices per hour."""
    out = {"profit": [], "leftover": [], "day": [],
           "sp": np.full((n, env.horizon), np.nan), "dp": np.full((n, env.horizon), np.nan)}
    for i in range(n):
        s, done, total, t = env.reset(seed=SEED0 + i), False, 0.0, 0
        while not done:
            a = policy(env, s)
            out["sp"][i, t], out["dp"][i, t] = env.actions[a]
            s, r, done, info = env.step(a)
            total, t = total + r, t + 1
        out["profit"].append(total); out["leftover"].append(info["leftover"]); out["day"].append(info["day_type"])
    for k in ["profit", "leftover", "day"]:
        out[k] = np.array(out[k])
    return out

def main():
    env, rng = VendingEnv(), np.random.default_rng(0)

    # Baselines (NOT reinforcement learning, used only for comparison)
    base = {"Random": rollout(env, lambda e, s: int(rng.integers(9)))}
    fixed = {a: rollout(env, lambda e, s, a=a: a) for a in range(9)}
    best_a = max(fixed, key=lambda a: fixed[a]["profit"].mean())
    best_fixed = fixed[best_a]
    base[f"Best fixed {env.actions[best_a]}"] = best_fixed
    base["Time rule"] = rollout(env, lambda e, s: 0 if e.competitor_open() else 4)   # cheap while competitor open

    # Trained PPO agents (one per seed, greedy actions)
    runs = []
    for path in sorted(glob.glob("models/ppo_seed*.pt")):
        ag = PPOAgent(env.state_dim, env.action_dim); ag.load(path)
        runs.append(rollout(env, lambda e, s, ag=ag: ag.act(s, deterministic=True)[0]))
    assert runs, "No trained models found. Run python train.py first."

    # Results table
    rows = [(n, r["profit"].mean(), r["profit"].std(), r["leftover"].mean()) for n, r in base.items()]
    m = [r["profit"].mean() for r in runs]
    rows.append((f"PPO ({len(runs)} seeds)", np.mean(m), np.std(m), np.mean([r["leftover"].mean() for r in runs])))
    print(f"\n{'Policy':<28}{'Profit Rs':>12}{'+/-':>10}{'Unsold units':>15}")
    for n, mu, sd, lo in rows:
        print(f"{n:<28}{mu:>12.1f}{sd:>10.1f}{lo:>15.2f}")
    print("(Baselines: +/- = std over episodes. PPO: +/- = std across training seeds.)")
    for d in range(3):
        print(f"{DAYS[d]:<18} best fixed Rs {best_fixed['profit'][best_fixed['day']==d].mean():.0f}"
              f" | PPO Rs {np.mean([r['profit'][r['day']==d].mean() for r in runs]):.0f}")
    os.makedirs("results", exist_ok=True)
    with open("results/table.csv", "w") as f:
        f.write("policy,profit,std,unsold\n")
        for n, mu, sd, lo in rows:
            f.write(f'"{n}",{mu:.2f},{sd:.2f},{lo:.2f}\n')

    # Dashboard
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VendoRL: PPO Dynamic Pricing Dashboard (INR)", fontsize=16, fontweight="bold")

    curves = np.array([np.load(f) for f in sorted(glob.glob("results/ppo_seed*_curve.npy"))])
    sm = np.array([np.convolve(c, np.ones(10) / 10, mode="valid") for c in curves])
    mu, sd = sm.mean(0), sm.std(0)
    ax[0, 0].plot(mu, color="tab:purple", label="PPO (mean of seeds)")
    ax[0, 0].fill_between(range(len(mu)), mu - sd, mu + sd, color="tab:purple", alpha=0.2)
    ax[0, 0].axhline(best_fixed["profit"].mean(), color="red", ls="--", label="Best fixed price")
    ax[0, 0].set(title="1. Learning curve (mean +/- std over seeds)", xlabel="Update iteration", ylabel="Profit per episode (Rs)")
    ax[0, 0].legend()

    names = [r[0] for r in rows]; cols = ["gray"] * len(base) + ["tab:purple"]
    ax[0, 1].bar(range(len(rows)), [r[1] for r in rows], yerr=[r[2] for r in rows], color=cols, capsize=4)
    ax[0, 2].bar(range(len(rows)), [r[3] for r in rows], color=cols)
    for a_, t in [(ax[0, 1], "2. Mean profit per episode (Rs)"), (ax[0, 2], "3. Mean unsold units at shift end")]:
        a_.set_title(t); a_.set_xticks(range(len(rows))); a_.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    best = max(runs, key=lambda r: r["profit"].mean())      # best seed, for the behaviour plots
    for key, a_, title in [("sp", ax[1, 0], "4. Snack price by hour"), ("dp", ax[1, 1], "5. Drink price by hour")]:
        for d, c in zip(range(3), ["tab:blue", "tab:gray", "tab:red"]):
            a_.plot(np.nanmean(best[key][best["day"] == d], axis=0), marker="o", color=c, label=DAYS[d])
        a_.axvspan(3.5, 7.5, color="orange", alpha=0.15, label="Competitor open")
        a_.set(title=f"{title} (PPO, by hidden day type)", xlabel="Hour of shift (0 = 08:00)", ylabel="Avg price (Rs)")
        a_.legend(fontsize=8)

    w = 0.35
    ax[1, 2].bar(np.arange(3) - w / 2, [best_fixed["profit"][best_fixed["day"] == d].mean() for d in range(3)], w, color="red", label="Best fixed")
    ax[1, 2].bar(np.arange(3) + w / 2, [best["profit"][best["day"] == d].mean() for d in range(3)], w, color="tab:purple", label="PPO")
    ax[1, 2].set_xticks(range(3)); ax[1, 2].set_xticklabels(DAYS); ax[1, 2].legend()
    ax[1, 2].set(title="6. Profit by hidden day type", ylabel="Mean profit (Rs)")

    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("results/dashboard.png", dpi=200)
    print("Saved results/dashboard.png and results/table.csv")


if __name__ == "__main__":
    main()
