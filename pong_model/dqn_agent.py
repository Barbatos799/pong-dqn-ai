import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 64
        self.update_target_steps = 200
        self.step_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="pong_dqn.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="pong_dqn.pth"):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = 0.0

#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import pygame
# import os
# from collections import deque
#
# class DQN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(5, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3)
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
# class Agent:
#     def __init__(self):
#         self.model = DQN()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.memory = deque(maxlen=5000)
#         self.gamma = 0.99
#         self.epsilon = 1.0
#         self.epsilon_min = 0.05
#         self.epsilon_decay = 0.995
#
#     def act(self, state):
#         if random.random() < self.epsilon:
#             return random.randint(0, 2)
#
#         state = torch.FloatTensor(state)
#         with torch.no_grad():
#             return torch.argmax(self.model(state)).item()
#
#     def remember(self, s, a, r, s2, d):
#         self.memory.append((s, a, r, s2, d))
#
#     def train(self, batch_size=32):
#         if len(self.memory) < batch_size:
#             return
#
#         batch = random.sample(self.memory, batch_size)
#
#         states = torch.from_numpy(np.array([b[0] for b in batch])).float()
#         actions = torch.LongTensor([b[1] for b in batch])
#         rewards = torch.FloatTensor([b[2] for b in batch])
#         next_states = torch.from_numpy(np.array([b[3] for b in batch])).float()
#         dones = torch.FloatTensor([b[4] for b in batch])
#
#         q_values = self.model(states)
#         q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()
#
#         with torch.no_grad():
#             q_next = self.model(next_states).max(1)[0]
#             q_target = rewards + self.gamma * q_next * (1 - dones)
#
#         loss = nn.MSELoss()(q_action, q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#
#     def save(self, path="pong_dqn.pth"):
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         full_path = os.path.join(base_dir, path)
#         torch.save(self.model.state_dict(), full_path)
#         print(f"✅ Model saved at: {full_path}")
#
#     def load(self, path="pong_dqn.pth"):
#         self.model.load_state_dict(torch.load(path))
#         self.epsilon = 0.05




