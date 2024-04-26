import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Randomly select an action, ensure it's a tensor
            return torch.tensor([random.randrange(self.action_size)], device=device)
        else:
            # Compute the action using the policy net
            with torch.no_grad():
                state = state.unsqueeze(0)  # Add batch dimension if not already added
            q_values = self.policy_net(state)
            # Return the action as a single-element tensor
            return q_values.max(1)[1].view(1)  # Flatten to [1] instead of [1,1] or something similar

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()
    
        # Prepare the states
        states = np.stack(mini_batch[0], axis=0)
        states = np.float32(states[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).to(device)  # Convert to tensor and move to device
    
        # Prepare the actions
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).to(device)
    
        # Prepare the rewards
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).to(device)
    
        # Prepare the next states
        next_states = np.stack(mini_batch[3], axis=0)
        next_states = np.float32(next_states[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).to(device)  # Convert to tensor and move to device
    
        # Prepare the done masks
        dones = mini_batch[4]  # Assuming this is where dones are stored in your batch
        mask = torch.tensor(list(map(int, dones == False)), dtype=torch.uint8).to(device)
    
        # Compute Q(s_t, a), the Q-value of the current state
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # Compute Q function of next state
        next_q_values = self.policy_net(next_states).max(1)[0]
    
        # Compute the expected Q values
        expected_q_values = (next_q_values * mask * self.discount_factor) + rewards
    
        # Compute the loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.detach())
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
        # Decrease epsilon
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
