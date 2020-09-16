from collections import deque
import numpy as np
import torch
from tqdm import tqdm


def train_ddpg(agent, env, num_agents=20, n_episodes=2000, max_t=200):
    initial_score = 5
    episode_scores = []
    brain_name = env.brain_names[0]
    scores_window = deque(maxlen=100)  # last 100 scores
    # eps = eps_start  # initialize epsilon
    episode_loop = tqdm(range(1, n_episodes + 1), desc="Episode 0 | Avg Score: None", leave=False)
    #max_score = -np.Inf
    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(num_agents)
        #for t in range(max_t):
        done = False
        step = 0
        while not done:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            agent_scores += rewards
            if np.any(dones):
                done = True

        episode_scores.append(np.mean(agent_scores))
        average_score = np.mean(episode_scores[i_episode - min(i_episode, scores_average_window):i_episode + 1])
        #scores.append(score)
        print(episode_loop.n)
        episode_loop.set_description_str('Episode {} | Avg Score: {:.2f}'.format(episode_loop.n, np.mean(scores_window)))
        if agent_scores.mean() > initial_score:
            episode_loop.set_description_str('Achieved in {:d} episode | Avg Score: {:.2f}'.format(episode_loop.n - 100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            initial_score = np.mean(scores_window)
    return scores
