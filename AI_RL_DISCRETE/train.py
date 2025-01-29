"""
train.py

Coordinates training by:
  - Creating the environment (environment.py)
  - Instantiating NeuralSAQAgent (model.py in the file you are looking for)
  - Running episodes, collecting transitions, and calling agent.train_step(...)
"""

from AI_RL_DISCRETE.environment import MinecraftParkourEnv
from AI_RL_DISCRETE.DQN.agent import NeuralDQNAgent

def convert_observation_to_state(obs):
    """
    A helper to possibly reduce or parse the environment's observation.
    In a basic scenario, you might just return obs directly if it is already
    a 1D float array that matches agent.obs_dim.
    """

    return obs

def train_dqn(num_episodes=100, max_steps=200, name = "Bob"):
    env = MinecraftParkourEnv(
        server_host="localhost",
        server_port=3000,
        name=name,
        view_xz_radius=3,
        view_y_up=3,
        view_y_down=3,
        sleep_after_action=0.2
    )

    # Wait a little for the bot to connect/spawn
    import time
    time.sleep(5.0)

    sample_obs = env.reset()
    obs_dim = sample_obs.shape[0]
    action_size = env.action_space.n # 18


    # Create the agent with replay buffer + batch size
    agent = NeuralDQNAgent(
        obs_dim=obs_dim,
        action_size=action_size,
        buffer_capacity=5000,  # e.g. 5k transitions
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999
    )

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)

            # The agent will store the transition + call train_on_batch if enough data
            agent.train_step(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            if done:
                break

        print(f"Episode {episode+1}/{num_episodes} | Steps: {step+1} "
              f"| Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    print("Training complete!")



