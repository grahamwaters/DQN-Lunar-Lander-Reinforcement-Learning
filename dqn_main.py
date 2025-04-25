import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ============================ CONFIG ============================
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
NUM_EPISODES = 1000
RENDER_EVERY = 50
# ================================================================

# =========================== SETUP ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logdir = f"runs/DQN_LunarLander_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(logdir)

# Gym env creation helper
def render_env(ep):
    return gym.make("LunarLander-v3", render_mode="human") if ep % RENDER_EVERY == 0 or ep == NUM_EPISODES else gym.make("LunarLander-v3", render_mode=None)

env = render_env(0)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# =========================== DQN Model ==========================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

policy_net = DQN(obs_dim, n_actions).to(device)
target_net = DQN(obs_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# ========================= INIT BUFFER ==========================
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.append((obs, action, reward, next_obs, done))
    obs = next_obs if not done else env.reset()[0]

# ========================== TRAIN LOOP ==========================
epsilon = EPSILON_START
step_count = 0

for episode in range(1, NUM_EPISODES + 1):
    env.close()
    env = render_env(episode)
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(policy_net(obs_tensor)).item()

        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.append((obs, action, reward, next_obs, done))
        obs = next_obs
        total_reward += reward
        step_count += 1

        # Train step
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            act_tensor = torch.tensor(act_batch, dtype=torch.int64, device=device).unsqueeze(1)
            rew_tensor = torch.tensor(rew_batch, dtype=torch.float32, device=device).unsqueeze(1)
            next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)

            # Q targets
            with torch.no_grad():
                max_next_q = target_net(next_obs_tensor).max(1, keepdim=True)[0]
                target_q = rew_tensor + GAMMA * max_next_q * (1 - done_tensor)

            # Q estimates
            current_q = policy_net(obs_tensor).gather(1, act_tensor)
            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss", loss.item(), global_step=step_count)

        # Sync target network
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # Logging
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    writer.add_scalar("Episode Reward", total_reward, global_step=episode)
    print(f"Ep {episode:4d} | Reward: {total_reward:7.2f} | Epsilon: {epsilon:.3f}")

# ========================== CLEANUP ==========================
env.close()
writer.close()