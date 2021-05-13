import Environment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from nav_dqn_agent_pixels import Agent
import torch
import os
from model_pixels import QNetworkFull
from dqn_functions import dqn_train, dqn_evaluate

PIXELS = True

if not PIXELS:

    # please do not modify the line below
    root = os.path.dirname(__file__)
    print(root)
    path = root + "/banana_standard/Banana.x86_64"
    print(path)
    env = UnityEnvironment(file_name=path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)

    # initialize the Nav Deep Q network agent
    agent = Agent(state_size=state_size,
                action_size=action_size, seed=0, pixels=PIXELS)

else:

    # please do not modify the line below
    root = os.path.dirname(__file__)
    path = root + "/VisualBanana_Linux/Banana.x86_64"
    env = Environment.CollectBanana(path)

    # initialize the Nav Deep Q network agent
    agent = Agent(state_size=env.state_size,
                action_size=env.action_size, seed=0, pixels=PIXELS)

train = True
evaluate = False

if train:

    scores = dqn_train(env, PIXELS, agent, n_episodes=2000, max_t=1000, eps_start=1.0,
                 eps_end=0.01, eps_decay=0.995, train_m=True)

if evaluate:

    dqn_evaluate(env, agent, state_size, action_size)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
