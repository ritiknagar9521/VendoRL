import numpy as np

class ComplexVendingEnv:
    def __init__(self):
        self.max_inventory = 10
        self.max_time = 10
        
        # Prices for Snacks and Drinks
        self.price_tiers = [1, 3, 5]
        
        # Base demand rate (lambda) for each price tier: High price = low demand
        # E.g., At $1, average 2.5 customers/hr. At $5, average 0.5 customers/hr.
        self.base_demand = {1: 2.5, 3: 1.2, 5: 0.5}
        
        # Build the 9 possible action combinations
        self.actions = []
        for snack_price in self.price_tiers:
            for drink_price in self.price_tiers:
                self.actions.append((snack_price, drink_price))
                
        self.reset()

    def reset(self):
        self.snack_inv = self.max_inventory
        self.drink_inv = self.max_inventory
        self.time = 0
        return self._get_state()

    def _get_state(self):
        return (self.snack_inv, self.drink_inv, self.time)

    def step(self, action_idx):
        snack_price, drink_price = self.actions[action_idx]
        
        # 1. Check for Rush Hour (Lunchtime spike at Hour 5)
        traffic_multiplier = 2.0 if self.time == 5 else 1.0
        
        # 2. Calculate Poisson lambda (average expected sales)
        snack_lambda = self.base_demand[snack_price] * traffic_multiplier
        drink_lambda = self.base_demand[drink_price] * traffic_multiplier
        
        # 3. Simulate actual sales using Poisson distribution
        # You can't sell more than you have in inventory!
        snack_sales = min(self.snack_inv, np.random.poisson(snack_lambda))
        drink_sales = min(self.drink_inv, np.random.poisson(drink_lambda))
        
        # Update inventory
        self.snack_inv -= snack_sales
        self.drink_inv -= drink_sales
        
        # 4. Calculate Financials
        revenue = (snack_sales * snack_price) + (drink_sales * drink_price)
        holding_cost = 0.1 * (self.snack_inv + self.drink_inv)
        
        self.time += 1
        done = (self.time >= self.max_time or (self.snack_inv == 0 and self.drink_inv == 0))
        
        reward = revenue - holding_cost
        
        # Terminal Penalty
        if done:
            reward -= 2.0 * (self.snack_inv + self.drink_inv)
            
        return self._get_state(), reward, done

# --- TEST SCRIPT ---
if __name__ == "__main__":
    env = ComplexVendingEnv()
    state = env.reset()
    print("Starting Day -> State:", state)
    
    done = False
    while not done:
        # Blindly choose Action 4 (Snack $3, Drink $3)
        action = 4 
        prices = env.actions[action]
        next_state, reward, done = env.step(action)
        print(f"Hour {next_state[2]} | Prices: S${prices[0]} D${prices[1]} | Inv left: (S:{next_state[0]}, D:{next_state[1]}) | Profit: ${reward:.2f}")