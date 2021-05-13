import Environment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from nav_dqn_agent_pixels import Agent
import torch
import os
from model_pixels import QNetworkFull

def dqn_train(env, pixels, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, train_m=True):  # original 2000 episodes
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        if pixels:
            state = env.reset()
        else:
            env_info = env.reset(train_mode=train_m)[brain_name]
            state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            if pixels:
                next_state, reward, done, _ = env.step(action)
            else:    
                env_info = env.step(action)[brain_name]
                done = env_info.local_done[0]
                reward = env_info.rewards[0]
                next_state = env_info.vector_observations[0]
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def dqn_evaluate(env, agent, state_size, action_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_model = QNetworkFull(state_size, action_size, 0).to(device)
    trained_model.load_state_dict(torch.load(root + "/checkpoint.pth"))
    trained_model.eval()

    agent.qnetwork_local = trained_model

    for i in range(10):

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:

            action = agent.act(state, 0.1)
            env_info = env.step(action)[brain_name]
            done = env_info.local_done[0]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        print("Evaluating Agent. Episode " + str(i) + " Score = " + str(score))