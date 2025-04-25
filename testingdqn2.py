import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import uuid
import imageio
import cv2
import pickle
import glob
import subprocess

# =========================== CONFIG ===========================
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
RENDER_EVERY = 25
VIDEO_DIR = "videos"
MODEL_PATH = "dqn_model.pkl"

os.makedirs(VIDEO_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(f"runs/DQN_LunarLander_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# ========================= ENV SETUP ===========================
def make_env(record=False):
    return gym.make("LunarLander-v3", render_mode="rgb_array" if record else None)

env = make_env()
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# ========================= Q-NETWORK ===========================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

policy_net = DQN(obs_dim, n_actions).to(device)
target_net = DQN(obs_dim, n_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# ===================== MODEL RESUME ============================
start_episode = 1
epsilon = EPSILON_START
step_count = 0

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        state = pickle.load(f)
        policy_net.load_state_dict(state["policy_state"])
        target_net.load_state_dict(state["target_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epsilon = state["epsilon"]
        step_count = state["step"]
        start_episode = state["episode"] + 1
    print(f"[✓] Loaded model from {MODEL_PATH}, resuming from episode {start_episode}")
else:
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    print("[i] No saved model found, starting fresh.")

replay_buffer = deque(maxlen=BUFFER_SIZE)

# ===================== INITIAL BUFFER ==========================
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.append((obs, action, reward, next_obs, done))
    obs = next_obs if not done else env.reset()[0]

# ====================== VIDEO UTILS ============================
def annotate_frame(frame, score, epoch, landed=False):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    cv2.putText(img, f"Score: {score:.1f}", (10, 30), font, 0.8, color, 2)
    cv2.putText(img, f"Epoch: {epoch}", (10, 60), font, 0.8, color, 2)
    if landed:
        cv2.putText(img, "LANDED!", (200, 200), font, 1.5, (0, 255, 0), 4)
        cv2.circle(img, (300, 150), 30, (0, 255, 0), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_gif(frames, episode):
    uid = uuid.uuid4().hex[:6]
    filename = os.path.join(VIDEO_DIR, f"episode_{episode}_{uid}.gif")
    imageio.mimsave(filename, frames, fps=30)
    print(f"[✓] Saved video: {filename}")

# ====================== TRAINING LOOP ==========================
for episode in range(start_episode, NUM_EPISODES + 1):
    record_video = episode % RENDER_EVERY == 0 or episode == NUM_EPISODES
    env.close()
    env = make_env(record=record_video)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    frames = []

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(policy_net(obs_tensor)).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.append((obs, action, reward, next_obs, done))
        obs = next_obs
        total_reward += reward
        step_count += 1

        if record_video:
            raw_frame = env.render()
            annotated = annotate_frame(raw_frame, total_reward, episode)
            frames.append(annotated)

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            act_tensor = torch.tensor(act_batch, dtype=torch.int64, device=device).unsqueeze(1)
            rew_tensor = torch.tensor(rew_batch, dtype=torch.float32, device=device).unsqueeze(1)
            next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                max_next_q = target_net(next_obs_tensor).max(1, keepdim=True)[0]
                target_q = rew_tensor + GAMMA * max_next_q * (1 - done_tensor)

            current_q = policy_net(obs_tensor).gather(1, act_tensor)
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss", loss.item(), global_step=step_count)

        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    writer.add_scalar("Episode Reward", total_reward, global_step=episode)
    print(f"Ep {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    if record_video:
        landed = total_reward > 200
        if landed:
            for _ in range(15):
                frames.append(annotate_frame(frames[-1], total_reward, episode, landed=True))
        save_gif(frames, episode)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "policy_state": policy_net.state_dict(),
            "target_state": target_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epsilon": epsilon,
            "step": step_count,
            "episode": episode
        }, f)

# ====================== FINAL VIDEO COMBINE ==========================
gif_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.gif")))
mp4_clips = []

for gif in gif_files:
    mp4_path = os.path.splitext(gif)[0] + ".mp4"
    try:
        # Convert each GIF to MP4
        subprocess.run([
            "ffmpeg", "-y", "-i", gif, "-vf", "scale=600:-2", "-pix_fmt", "yuv420p", mp4_path
        ], check=True)
        mp4_clips.append(mp4_path)
    except subprocess.CalledProcessError:
        print(f"[!] Failed to convert {gif} to MP4.")

# Create concat list
if mp4_clips:
    concat_file = os.path.join(VIDEO_DIR, "concat_list.txt")
    with open(concat_file, "w") as f:
        for mp4 in mp4_clips:
            f.write(f"file '{os.path.abspath(mp4)}'\n")

    combined_output = os.path.join(VIDEO_DIR, "combined_output.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", combined_output
        ], check=True)
        print(f"[✓] Combined MP4 created at {combined_output}")
    except subprocess.CalledProcessError:
        print("[!] Failed to combine MP4 files.")
else:
    print("[i] No MP4 files to combine.")

env.close()
writer.close()
