import gymnasium as gym
import numpy as np
import random

# Parameters
state_space_bins = 20
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000
render_every = 100  # Render every N episodes

# Initialize environment
env = gym.make("LunarLander-v3", render_mode="human")

# Setup Q-table
action_space_size = env.action_space.n
obs_dim = env.observation_space.shape[0]
q_table = np.zeros((state_space_bins,) * obs_dim + (action_space_size,))

# Discretization helpers
obs_low = np.clip(env.observation_space.low, -1e10, 1e10)  # sanitize infs
obs_high = np.clip(env.observation_space.high, -1e10, 1e10)

def discretize(obs):
    ratios = (obs - obs_low) / (obs_high - obs_low)
    ratios = np.clip(ratios, 0, 1)
    discrete = (ratios * (state_space_bins - 1)).astype(int)
    return tuple(discrete)

def select_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + discount_factor * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += learning_rate * td_error

# Training loop
for episode in range(1, num_episodes + 1):
    observation, _ = env.reset()
    state = discretize(observation)
    total_reward = 0
    done = False

    while not done:
        if episode % render_every == 0:
            env.render()

        action = select_action(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_observation)

        update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        done = terminated or truncated

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

env.close()