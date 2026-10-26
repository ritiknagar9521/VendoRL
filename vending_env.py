"""VendoRL environment: a vending machine selling a snack and a drink for one 12-hour shift.

MDP summary (explained in the report):
  State  (8 numbers, all scaled to ~[0,1]):
      snack stock, drink stock, time of day, competitor open,
      last hour's snack sales, last hour's drink sales, last snack price tier, last drink price tier
  Action (9 discrete): every (snack price tier, drink price tier) pair
  Reward: profit this hour = margin on sales - holding cost, and at shift end - cost of unsold stock
  Hidden variable: today's demand level (low / normal / high). The agent never sees it directly and
  must infer it from recent sales. This is the "uncertain customer demand" of the project title.
"""
import numpy as np


class VendingEnv:
    SNACK_PRICES = [20.0, 40.0, 70.0]   # Low / Normal / High  (Rs)
    DRINK_PRICES = [30.0, 50.0, 90.0]
    SNACK_COST = 15.0                   # what the owner paid per snack (Rs)
    DRINK_COST = 20.0
    HOLDING_COST = 1.0                  # Rs per unsold unit per hour (storage / cooling)
    DAY_MULT = [0.6, 1.0, 1.5]          # hidden demand multiplier: low, normal, high day
    DAY_PROBS = [0.3, 0.4, 0.3]

    def __init__(self, seed=None, max_stock=30, horizon=12):
        self.max_stock = max_stock
        self.horizon = horizon          # 12 hours: t=0 is 08:00, t=11 is 19:00
        self.actions = [(s, d) for s in self.SNACK_PRICES for d in self.DRINK_PRICES]
        self.state_dim = 8
        self.action_dim = len(self.actions)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- helpers -------------------------------------------------------
    def competitor_open(self, t=None):
        t = self.t if t is None else t
        return 4 <= t < 8               # cafeteria open 12:00-16:00

    def _state(self):
        return np.array([
            self.snack_stock / self.max_stock,
            self.drink_stock / self.max_stock,
            self.t / self.horizon,
            float(self.competitor_open()),
            min(self.last_snack_sales / 10.0, 1.0),
            min(self.last_drink_sales / 10.0, 1.0),
            self.last_snack_tier / 2.0,
            self.last_drink_tier / 2.0,
        ], dtype=np.float32)

   
    def reset(self, seed=None):
        if seed is not None:            # same seed -> same day type and same random customers
            self.rng = np.random.default_rng(seed)
        self.day_type = int(self.rng.choice(3, p=self.DAY_PROBS))   # HIDDEN from the agent
        self.snack_stock = float(self.max_stock)
        self.drink_stock = float(self.max_stock)
        self.t = 0
        self.last_snack_sales = 0.0
        self.last_drink_sales = 0.0
        self.last_snack_tier = 1        # start as "normal"
        self.last_drink_tier = 1
        return self._state()

    def step(self, action):
        sp, dp = self.actions[action]
        comp = self.competitor_open()
        rush = 1.8 if self.t in (4, 5) else 1.0          # lunch rush 12:00-14:00

        # Price sensitivity is higher while the competitor is open
        a_snack = 0.05 if comp else 0.025
        a_drink = 0.04 if comp else 0.020

        # Expected customers this hour: base demand * hidden day level * rush * exp(-alpha*(price-ref))
        lam_snack = 2.5 * self.DAY_MULT[self.day_type] * rush * np.exp(-a_snack * (sp - 40.0))
        lam_drink = 3.0 * self.DAY_MULT[self.day_type] * rush * np.exp(-a_drink * (dp - 50.0))

        # Actual purchases are random (Poisson) and limited by stock
        snack_sales = min(self.snack_stock, float(self.rng.poisson(lam_snack)))
        drink_sales = min(self.drink_stock, float(self.rng.poisson(lam_drink)))
        self.snack_stock -= snack_sales
        self.drink_stock -= drink_sales

        # Reward = profit this hour
        reward = (sp - self.SNACK_COST) * snack_sales + (dp - self.DRINK_COST) * drink_sales
        reward -= self.HOLDING_COST * (self.snack_stock + self.drink_stock)

        self.t += 1
        self.last_snack_sales, self.last_drink_sales = snack_sales, drink_sales
        self.last_snack_tier, self.last_drink_tier = divmod(action, 3)   # tier indices 0..2

        done = self.t >= self.horizon or (self.snack_stock == 0 and self.drink_stock == 0)
        if done:   # unsold stock was bought but never sold: lose its purchase cost
            reward -= self.SNACK_COST * self.snack_stock + self.DRINK_COST * self.drink_stock

        info = {"snack_sales": snack_sales, "drink_sales": drink_sales, "day_type": self.day_type,
                "leftover": self.snack_stock + self.drink_stock}
        return self._state(), float(reward), done, info
