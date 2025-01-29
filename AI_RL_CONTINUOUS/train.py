import time
from AI_RL_CONTINUOUS.environment import MinecraftParkourEnv
from AI_RL_CONTINUOUS.SAC.agent import SACAgent
from AI_RL_CONTINUOUS.PPO.agent import PPOAgent, Transition



def train_sac(num_episodes=100, max_steps=200, eval_mode=False):
    env = MinecraftParkourEnv(
        server_host="localhost",
        server_port=3000,
        view_xz_radius=5,
        view_y_up=7,
        view_y_down=3,
        sleep_after_action=0.2
    )

    time.sleep(5.0)

    sample_obs = env.reset()
    obs_dim = sample_obs.shape[0]
    action_dim = 6  # [yaw, pitch, forward, left, right, jump]

    try:
        agent = SACAgent.load_from_checkpoint("storage/sac_agent.pth")
    except:
        agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_capacity=50000,
        batch_size=64
    )

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            # Continuous action from your SAC agent
            raw_action = agent.select_action(obs, eval_mode=eval_mode)
            # raw_action is in [-1,1]^6 (assuming you're using tanh)

            # Scale the first two dims to [-0.3, 0.3]
            delta_yaw = 0.3 * raw_action[0]
            delta_pitch = 0.3 * raw_action[1]

            # The last four we just pass as is, or interpret them directly in [-1,1].
            # So the final action array:
            action = [
                delta_yaw,
                delta_pitch,
                raw_action[2],  # forward?
                raw_action[3],  # left?
                raw_action[4],  # right?
                raw_action[5],  # jump?
            ]

            next_obs, reward, done, info = env.step(action)

            # Note: store 'raw_action' in the buffer if your policy heads produce that in [-1,1].
            # Or store the scaled one, but be consistent.
            agent.store_transition(obs, raw_action, reward, next_obs, done)

            agent.update()

            obs = next_obs
            total_reward += reward

            if done:
                break

        print(f"Episode {episode+1}/{num_episodes} | Steps: {step+1} | Reward: {total_reward:.2f}")

    agent.save("storage/sac_agent.pth")
    env.close()
    print("Training complete!")

def train_ppo(num_episodes=100, max_steps=200, eval_mode=False):
    # 1) Create your environment
    env = MinecraftParkourEnv(
        server_host="localhost",
        server_port=3000,
        view_xz_radius=3,
        view_y_up=3,
        view_y_down=3,
        sleep_after_action=0.2
    )

    time.sleep(5.0)  # wait for bot to connect
    sample_obs = env.reset()
    obs_dim = sample_obs.shape[0]
    action_dim = 6  # [yaw, pitch, forward, left, right, jump]

    try:
        agent = PPOAgent.load_from_checkpoint("storage/ppo_agent.pth")

    except:
        print("this is the first time")
        # 2) Create the PPO agent
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            eps_clip=0.2,
            K_epochs=10,
            batch_size=64,
            hidden_dim=128
        )


    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # 1) Select action using current policy
            action, log_prob, value = agent.select_action(state, eval_mode=eval_mode)

            # The first two dimensions in [-1,1], scale them to [-0.3, 0.3]
            delta_yaw = 0.3 * action[0]
            delta_pitch = 0.3 * action[1]

            # The other four are used as-is in [-1,1]
            # forward, left, right, jump
            final_action = [
                delta_yaw,
                delta_pitch,
                action[2],
                action[3],
                action[4],
                action[5]
            ]

            # 2) Step environment
            next_state, reward, done, info = env.step(final_action)

            total_reward += reward

            # Store transition in the buffer
            # We don't know "next_value" yet, but we know "value" for the current state
            transition = Transition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                next_state=next_state,
                done=done,
                value=value
            )
            agent.store_transition(transition)

            # Move on
            state = next_state

            if done:
                # Finish the episode's data, compute advantage
                agent.finish_trajectory()
                # Perform PPO update
                agent.update()
                break

        print(f"Episode {episode+1}/{num_episodes} | Steps: {step+1} | Reward: {total_reward:.2f}")

    agent.save("storage/ppo_agent.pth")
    env.close()
    print("PPO training complete!")