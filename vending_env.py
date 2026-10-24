import random

class VendingEnvironment:
    def __init__(self):
        # Maximum inventory the machine can hold
        self.max_inventory = 3
        
        # Time blocks: 0 (Morning), 1 (Afternoon), 2 (Evening)
        self.max_time_steps = 3
        
        # Action space: 0 ($1), 1 ($2), 2 ($3)
        self.prices = [1, 2, 3]
        
        self.reset()

    def reset(self):
        """Starts a new day (episode). Restocks and sets time to Morning (0)."""
        self.inventory = self.max_inventory
        self.time_step = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state tuple: (Inventory, Time Block)"""
        return (self.inventory, self.time_step)

    def step(self, action):
        """
        Executes the chosen price action, simulates stochastic customer demand,
        calculates reward, and advances time.
        """
        current_price = self.prices[action]
        reward = 0
        
        # 1. Simulate Stochastic Customer Demand
        if self.inventory > 0:
            # Base probability drops as price increases
            if action == 0:     # $1
                buy_chance = 0.8
            elif action == 1:   # $2
                buy_chance = 0.5
            else:               # $3
                buy_chance = 0.2
            
            # Time of day modifies the probability
            if self.time_step == 0:      # Morning (Busy)
                buy_chance += 0.1
            elif self.time_step == 2:    # Evening (Slow)
                buy_chance -= 0.1
                
            # Roll the dice: Did a customer buy?
            if random.random() < buy_chance:
                self.inventory -= 1
                reward = current_price  # Positive reward for sale
                
        # 2. Advance Time
        self.time_step += 1
        done = False
        
        # 3. Check for End of Day & Apply Penalty
        if self.time_step >= self.max_time_steps:
            done = True
            # Penalty: -1 for every unsold item
            reward -= (self.inventory * 1) 
            
        return self.get_state(), reward, done

# --- Quick Test to verify the environment works ---
if __name__ == "__main__":
    env = VendingEnvironment()
    state = env.reset()
    print(f"Start of Day State: {state}")
    
    # Test setting price to $2 (Action 1)
    next_state, reward, done = env.step(action=1)
    
    print(f"Action Taken: Normal Price ($2)")
    print(f"Reward Received: ${reward}")
    print(f"Next State: {next_state} | Is day over? {done}")