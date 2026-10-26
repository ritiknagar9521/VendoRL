"""PPO (Proximal Policy Optimization) agent.
PPO is an Actor-Critic policy-gradient method:
  Actor  = policy network pi(a|s)  (chooses prices)
  Critic = value network V(s)      (predicts future profit, used to judge actions)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(inp, out):
    """Small neural network: 2 hidden layers of 64 units with Tanh."""
    return nn.Sequential(nn.Linear(inp, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, entropy_coef=0.02, epochs=8, minibatch=128):
        self.gamma, self.lam = gamma, lam          # discount factor, GAE smoothing
        self.clip_eps = clip_eps                   # PPO clip range
        self.entropy_coef = entropy_coef           # exploration bonus
        self.epochs, self.minibatch = epochs, minibatch
        self.actor = mlp(state_dim, action_dim)    # outputs one score (logit) per price combination
        self.critic = mlp(state_dim, 1)            # outputs V(s)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, state, deterministic=False):
        """Training: sample an action from pi. Evaluation: take the most likely action."""
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = Categorical(logits=self.actor(s))
            value = self.critic(s).item()
        action = dist.probs.argmax() if deterministic else dist.sample()
        return action.item(), dist.log_prob(action).item(), value

    def compute_gae(self, rewards, values, dones):
        """Generalised Advantage Estimation: how much better was an action than the critic expected.
        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma*lam*A_{t+1}"""
        adv, running = np.zeros(len(rewards), dtype=np.float32), 0.0
        for t in reversed(range(len(rewards))):
            next_v = 0.0 if dones[t] else values[t + 1]       # no future after the shift ends
            delta = rewards[t] + self.gamma * next_v - values[t]
            running = delta + self.gamma * self.lam * (1.0 - dones[t]) * running
            adv[t] = running
        return adv

    def update(self, b):
        """Several epochs of mini-batch updates on one batch of experience."""
        S = torch.as_tensor(b["states"]); A = torch.as_tensor(b["actions"])
        old_lp = torch.as_tensor(b["log_probs"])
        adv_np = self.compute_gae(b["rewards"], b["values"], b["dones"])
        target = torch.as_tensor(adv_np + np.asarray(b["values"], dtype=np.float32))   # critic target
        adv = torch.as_tensor(adv_np)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalise advantages

        n = len(S)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                dist = Categorical(logits=self.actor(S[idx]))
                ratio = torch.exp(dist.log_prob(A[idx]) - old_lp[idx])           # pi_new / pi_old
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                # PPO clipped objective (take the pessimistic one) minus entropy bonus
                actor_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean() \
                             - self.entropy_coef * dist.entropy().mean()
                critic_loss = nn.functional.mse_loss(self.critic(S[idx]).squeeze(1), target[idx])
                self.opt_actor.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); self.opt_actor.step()
                self.opt_critic.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5); self.opt_critic.step()

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
