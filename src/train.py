from collections import deque
import numpy as np
import torch
from tqdm import tqdm


def train_ddpg(agent, env, num_agents=20, n_episodes=120, max_t=200):
    previous_score = 1
    episode_scores = []  # list containing scores from each episode
    average_scores = []
    brain_name = env.brain_names[0]
    # eps = eps_start  # initialize epsilon
    episode_loop = tqdm(range(1, n_episodes + 1), desc="Episode 0 | Avg Score: None", leave=False)

    for _ in episode_loop:
        # reset the unity environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # reset the agent for the new episode
        agent.reset()
        agent_scores = np.zeros(num_agents)

        scores_average_window = 100

        while True:
            # determine actions for the unity agents from current sate
            actions = agent.act(states)

            # send the actions to the unity agents in the environment and receive resultant environment information
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations  # get the next states for each unity agent in the environment
            rewards = env_info.rewards  # get the rewards for each unity agent in the environment
            dones = env_info.local_done  # see if episode has finished for each unity agent in the environment

            # Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
            agent.step(states, actions, rewards, next_states, dones)

            # set new states to current states for determining next actions
            states = next_states

            # Update episode score for each unity agent
            agent_scores += rewards

            # If any unity agent indicates that the episode is done,
            # then exit episode loop, to begin new episode
            if np.any(dones):
                break

        #print("Came out of the game loop")
        episode_scores.append(np.mean(agent_scores))
        average_score = np.mean(episode_scores[episode_loop.n - min(episode_loop.n, scores_average_window):episode_loop.n + 1])
        average_scores.append(average_score)
        #print(episode_loop.n)
        episode_loop.set_description_str('Episode {} | Avg Score: {:.2f}'.format(episode_loop.n, average_score))
        if agent_scores.mean() > previous_score:
            previous_score = agent_scores.mean()
            episode_loop.set_description_str('Achieved in {} episode | Avg Score: {:.2f}'.format(episode_loop.n, average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            #torch.save(agent.actor_local, 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            #torch.save(agent.critic_local, 'checkpoint_critic.pth')
    return episode_scores, average_scores
