import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from vending_env import VendingEnv
import matplotlib.pyplot as plt

# 1. Define the Neural Network Architecture
class VendingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VendingDQN, self).__init__()
        # 3 inputs (Snack Inv, Drink Inv, Time) -> 64 hidden nodes
        self.fc1 = nn.Linear(input_dim, 64)
        # 64 hidden nodes -> 64 hidden nodes
        self.fc2 = nn.Linear(64, 64)
        # 64 hidden nodes -> 9 outputs (Q-values for each action)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # ReLU activation function adds non-linearity
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. Define the Agent with Experience Replay
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = 3
        self.action_size = 9 # 3 snack prices * 3 drink prices
        
        # Hyperparameters
        self.gamma = 0.95        # Discount rate
        self.epsilon = 1.0       # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Experience Replay Memory (stores last 2,000 steps)
        self.memory = deque(maxlen=2000)
        
        # The Neural Network (Brain)
        self.model = VendingDQN(self.state_size, self.action_size)
        # Adam Optimizer is standard for DQN
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Mean Squared Error Loss
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Saves the memory to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy using the neural network."""
        # Explore
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        # Exploit: Ask the neural network for the best action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): # Don't track gradients during prediction
            q_values = self.model(state_tensor)
        return np.argmax(q_values.numpy())

    def replay(self):
        """Trains the neural network using random memories from the buffer."""
        # Wait until we have enough memories to form a batch
        if len(self.memory) < self.batch_size:
            return
            
        # Randomly sample a batch of memories
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = float(reward)
            if not done:
                # Bellman Equation: target = Reward + Gamma * Max(Future Q)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * self.model(next_state_tensor).detach().max().item()
                
            # Get the current predictions from the network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            
            # Update the specific action's Q-value with our new target
            target_f[0][action] = target
            
            # Perform gradient descent to train the network
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- TRAINING SCRIPT ---
if __name__ == "__main__":
    env = VendingEnv()
    agent = DQNAgent(env)
    
    # DQN takes longer to train than tables. We will do 1,000 episodes.
    # Depending on your laptop, this might take 1-3 minutes.
    episodes = 1000
    rewards_history = []
    
    print("Starting DQN Training... This will take a moment.")
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Save to memory
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train the network
            agent.replay()
            
        rewards_history.append(total_reward)
        if e % 50 == 0:
            print(f"Episode {e}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    print("Training complete!")
    
    # Save the advanced graph
    smoothed = [np.mean(rewards_history[i:i+50]) for i in range(0, len(rewards_history), 50)]
    plt.plot(smoothed)
    plt.title("DQN Agent Profit Over Time (Complex Environment)")
    plt.xlabel("Episodes (Fifties)")
    plt.ylabel("Average Profit ($)")
    plt.savefig("dqn_training_results.png")
    print("Graph saved as 'dqn_training_results.png'")