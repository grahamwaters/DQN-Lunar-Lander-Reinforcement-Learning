import gymnasium as gym
import torch
import torch.nn as nn
import pickle

MODEL_PATH = "dqn_model.pkl"

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

# Load environment
env = gym.make("LunarLander-v3", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Load trained model
policy_net = DQN(obs_dim, n_actions)

with open(MODEL_PATH, "rb") as f:
    state = pickle.load(f)
    policy_net.load_state_dict(state["policy_state"])

policy_net.eval()

# Run the agent
obs, _ = env.reset()
done = False

while not done:
    env.render()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = torch.argmax(policy_net(obs_tensor)).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print("Episode finished.")
env.close()
