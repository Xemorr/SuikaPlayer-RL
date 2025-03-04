import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

import main
from c_adamw import CAdamW

reward_history = deque(maxlen=5000)

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=4, stride=2),
            nn.GroupNorm(1, 24),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=1),
            nn.GroupNorm(1, 32),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        print(conv_out_size)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Agent:
    def __init__(self, state_shape, n_actions):
        self.n_actions = n_actions
        self.net = RainbowDQN(state_shape, n_actions).to(device)
        self.target_net = RainbowDQN(state_shape, n_actions).to(device)
        self.optimizer = CAdamW(self.net.parameters(), lr=0.0001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.net(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device).unsqueeze(1)
        dones = torch.BoolTensor(dones).to(device)

        # Q-values for the selected actions from the main network
        q_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Double DQN: select the action using the main network (self.net)
        best_actions = self.net(next_states).argmax(1)  # Select actions with the main network

        # Use the target network (self.target_net) to get the Q-values for those selected actions
        next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(-1)).squeeze(-1)

        # Set Q-value to 0 for terminal states
        next_q_values[dones] = 0.0

        target = rewards + self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # Gamma growth
        self.gamma = min(self.gamma * 1.0005, 0.999)

    def update_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

if __name__ == '__main__':
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = (1, 100, 100)  # Example input shape (1 color channels, 100x100 pixels)
    n_actions = 8  # Adjust this to the number of possible actions in the game
    update_interval = 1000
    agent = Agent(state_shape, n_actions)

    # Main training loop
    previous_score = 0
    previous_game = -1
    gym = main.Gym()
    step = 0
    while True:
        for i in trange(600):
            step += 1
            # Update target network every fixed interval
            if step % update_interval == 0:
                agent.update_target_network()

            state = gym.take_screenshot()
            action = agent.select_action(state)

            xCoord = 45 * (action - 4)
            gym.click_on_canvas(xCoord, 20)
            time.sleep(0.6)

            next_state = gym.take_screenshot()
            done = gym.game_over()  # Implement a check to determine if the game is over

            new_score = gym.extract_score()
            if new_score == previous_game:
                new_score = 0
            reward = -300 if done else max(new_score - previous_score, -300)
            reward_history.append(reward)
            if random.randint(1, 10) == 5:
                print(agent.epsilon)
                print(sum(reward_history) / len(reward_history))
            previous_score = new_score
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            if done:
                previous_game = new_score
                gym.click_next()
        torch.save(agent.net.state_dict(), "agent.pth")
