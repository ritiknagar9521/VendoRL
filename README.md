# VendoRL: Adaptive Vending Machine Pricing Using PPO

VendoRL is a reinforcement learning mini-project that explores how a vending machine can automatically adjust the prices of snacks and drinks to improve its daily profit. Instead of following a fixed pricing strategy, the machine learns from its interactions with a simulated environment and adapts its decisions according to stock availability, customer demand, time of day, and competition.

The project uses **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm, to learn pricing decisions under uncertain customer demand.

**Team Members**
- Shiv JEE Yadav — B23EE1095
- Ritik Nagar — B23EE1086

---

## 1. Problem Statement
Setting the right price is an important challenge for a vending machine owner. Low prices may attract more customers but reduce the profit earned per product. High prices may increase the profit per sale but can also reduce demand and leave products unsold.

The challenge becomes more difficult because customer demand is not the same every day. Some days have low demand, some are normal, and others have high demand. A nearby cafeteria can also affect customer purchases during its operating hours.

The goal of VendoRL is to train an agent that learns when to increase or decrease prices to earn better overall profit while accounting for unsold inventory.

## 2. Project Objectives
- Build a simulated vending machine environment.
- Model changing customer demand and random purchasing behaviour.
- Allow the agent to choose prices for snacks and drinks.
- Train a PPO agent to improve its pricing decisions through experience.
- Compare the learned policy with simple pricing strategies.
- Study the relationship between profit, inventory, time of day, and competition.

## 3. How the Environment Works
The environment represents a vending machine operating for a 12-hour shift, from 08:00 to 20:00 in the simulation's hourly time model.

| Parameter | Description |
|---|---|
| Products | One snack and one drink |
| Initial inventory | 30 units of each product |
| Shift duration | 12 hours |
| Snack prices | Rs 20, Rs 40, Rs 70 |
| Drink prices | Rs 30, Rs 50, Rs 90 |
| Snack purchase cost | Rs 15 per unit |
| Drink purchase cost | Rs 20 per unit |
| Holding cost | Rs 1 per remaining unit per hour |

Customer demand is generated randomly using a Poisson distribution. Expected demand depends on the selected price, the hidden demand level of the day, and the time of day.

A lunch rush increases demand between 12:00 and 14:00. A competing cafeteria operates between 12:00 and 16:00, making customers more sensitive to prices during this period.

The environment has three hidden demand levels:
- **Low demand:** 30% probability
- **Normal demand:** 40% probability
- **High demand:** 30% probability

The agent is not directly told the demand level. It must use recent sales and other observations to make its decisions.

## 4. Reinforcement Learning Formulation
VendoRL models the pricing problem as a Markov Decision Process (MDP).

### State Space
The agent receives an observation containing eight values:
1. Remaining snack inventory.
2. Remaining drink inventory.
3. Current time in the shift.
4. Whether the competitor is open.
5. Snack sales in the previous hour.
6. Drink sales in the previous hour.
7. Previous snack price level.
8. Previous drink price level.

These values are scaled to approximately the range [0, 1] before being passed to the neural networks. The hidden demand category is not included in the observation, forcing the agent to infer customer demand realistically.

### Action Space
The agent has nine possible actions. Each action represents one combination of snack and drink prices (e.g., low snack price with a high drink price). This allows it to adjust the two products independently.

### Reward Function
The reward is based on the profit earned during each hour.
- Money earned from sales is calculated using the difference between selling price and purchase cost.
- A holding cost is charged for the remaining inventory.
- At the end of the shift, the purchase cost of unsold products is deducted.

This design encourages the agent to consider both profit margins and inventory rather than simply maximizing the number of products sold.

## 5. Description of PPO - Policy learning method
Proximal Policy Optimization (PPO) is a policy-gradient reinforcement learning algorithm. It learns a policy that selects actions based on the current observation and improves that policy using experience collected from the environment.

The implementation uses an Actor-Critic architecture:
- **Actor network:** Produces scores for the nine possible pricing actions.
- **Critic network:** Estimates the expected future reward from the current state.

Both networks use two hidden layers with 64 units and Tanh activation functions. PPO uses a clipped objective to limit large policy changes, helping make learning more stable. The default training script runs 400 iterations for each of three random seeds.

## 6. Baseline Strategies
To understand whether learning provides an advantage, PPO is compared with three simpler strategies.

| Strategy | Description |
|---|---|
| Random pricing | Selects a pricing action randomly |
| Best fixed price | Always uses Rs 40 for snacks and Rs 50 for drinks |
| Time-based rule | Uses cheaper prices while the competitor is open and normal prices otherwise |
| PPO | Learns pricing decisions from interaction with the environment |

## 7. Results and Observations
The supplied dashboard summarizes training progress, average profit, unsold inventory, hourly prices, and performance under different demand conditions.

### Main observations
- **Learning progress:** The PPO learning curve rises during early training and later fluctuates around a higher profit level.
- **Overall profit:** PPO earns approximately Rs 1,280 per shift, compared with roughly Rs 1,100 for the best fixed-price strategy.
- **Normal-demand days:** PPO shows a modest profit improvement over fixed pricing.
- **High-demand days:** PPO shows a larger profit advantage by adjusting prices to demand.
- **Low-demand days:** PPO earns less than the fixed-price strategy in the displayed results.
- **Unsold inventory:** PPO leaves more products unsold on average, showing that higher profit does not necessarily mean lower leftover stock.

## 8. Installation, Execution & Reproduction Instructions
Follow these steps to setup the environment and reproduce the trained RL agent.

**Dependencies:**
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

**Installation Steps:**
```bash
# Clone the repository
git clone [https://github.com/ritiknagar9521/VendoRL.git](https://github.com/ritiknagar9521/VendoRL.git)
cd VendoRL

# Create and activate a virtual environment (Optional but recommended)
python -m venv virtenv
source virtenv/bin/activate  # On Windows use: virtenv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib
