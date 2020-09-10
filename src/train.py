from collections import deque
import numpy as np
import torch
from tqdm import tqdm


def train_ddpg(agent, env, n_episodes=2000, max_t=700):
    initial_score = 5
    scores_deque = deque(maxlen=100)
    scores = []
    brain_name = env.brain_names[0]
    scores_window = deque(maxlen=100)  # last 100 scores
    # eps = eps_start  # initialize epsilon
    episode_loop = tqdm(range(1, n_episodes + 1), desc="Episode 0 | Avg Score: None", leave=False)
    max_score = -np.Inf
    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action, memory=None)[brain_name]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        episode_loop.set_description_str('Episode {} | Avg Score: {:.2f}'.format(episode_loop.n, np.mean(scores_window)))
        if np.mean(scores_window) > initial_score:
            episode_loop.set_description_str('Achieved in {:d} episode | Avg Score: {:.2f}'.format(episode_loop.n - 100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            initial_score = np.mean(scores_window)
    return scores
