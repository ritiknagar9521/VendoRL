

# VendoRL: Adaptive Vending Machine Pricing with Reinforcement Learning (PPO)

A simulated vending machine adjusts prices based on remaining inventory, time, and uncertain customer demand to maximize profit without excessive unsold stock.

A simulated vending machine sells a snack and a drink over a 12-hour shift. 
A PPO agent (actor-critic policy-gradient RL) chooses the prices every hour to maximise profit without leaving too much stock unsold.
Customer demand is uncertain: the day type (low / normal / high demand) is hidden and must be inferredfrom recent sales.

## Installation (Windows / Linux / macOS; CPU is enough, the networks are tiny)
    python -m venv venv
    venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
    pip install -r requirements.txt

## Execution
    python train.py                  # trains PPO with seeds 0,1,2 -> models/ppo_seed*.pt, results/*_curve.npy
    python evaluate.py               # table + results/dashboard.png on 500 unseen test episodes
Options: `python train.py --seeds 0 1 2 3 4 --iters 600`

## Files
| File | Purpose |
|---|---|
| `vending_env.py` | MDP: 8-d state, 9 discrete actions, profit reward, hidden demand level 
| `ppo.py` | PPO agent 
| `train.py` 
| `evaluate.py` | Baselines 

## Dependencies
Python 3.9+, numpy, matplotlib, torch 

## Reproduction
Training seeds 0, 1, 2. Test episodes use seeds 10000-10499, never seen in training. Baselines are only
comparison points; the pricing policy itself is learned by RL (PPO).
