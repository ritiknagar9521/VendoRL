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

