### Udacity Deep Reinforcment Learning Nanodegree 
# Project: Navigation with full Conv Net, from pixels to actions

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Files and Folders
* navigation.py: the main routine for the basic-banana project. It is the high-level calls to construct the environment, agent, train and play. 
* dqn_functions.py: the deep reinforcement learning algorithm routine.
* nav_dqn_agent_pixels.py: the class definitions of the DQN agent and replay buffer.
* model_pixels.py: the deep neural network models (DQN) are defined in this file.
* Environment.py: a wrapper for the UnityEnvironment. The wrapper makes the 
environment interface similar to the OpenAI gym environment, so that 
the DQN routines are more general.
 
#### Environment Wrapper Class
The ``CollectBanana`` Class is a wrapper for the ``UnityEnvironment`` class, which 
includes the following main methods:
* step
* reset
* get_state
* close

The ``name`` parameter in the 
constructor allows the selection of ``state`` format returned:

* The state contains four frames by calling ``get_state()``. 
To fit the PyTorch format, the original frame format is transposed from NHWC (Batch, Height, Width, Channels) to NCHW 
 by numpy transpose function as follows:
``frame = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))`` 
The current frame, together with previous frame ``last_frame`` and the second previous frame, ``last2_frame``
are then assemblied into variable ``CollectBanana.state`` variable.

#### Training: DQN and Agent

The DQN used in this implementation is the simple DQN with two networks: one is local, and one is target.

In nav_dqn_agent_pixels.py, the network is created with these lines:
```python
    self.qnetwork_local = QNetworkFull(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetworkFull(state_size, action_size, seed).to(device)
```

For both basic- and visual-banana project, the training parameters are as follows:

In the dqn_functions.py: function dqn_train() method:
```python
n_episodes=2000, max_t=1000, eps_start=1.0,
eps_end=0.01, eps_decay=0.995,
score_window_size=100, target_score=13.0
```

In the nav_dqn_agent_pixels.py file: the Agent class has parameters to learn():
```python
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

#### Neural Network Models

The neural networks are defined in the model.py file. It contains two classes: 
* VisualQNetwork
The visualQNetwork employs three 3D convolutional layers and 2 fully connected layers. It is defined as 
follows:
```python
    n_filter_1 = 128
    n_filter_2 = 256
    n_filter_3 = 256
    # NHWC (Batch, Height, Width, Channels)

    self.conv_layer_1 = nn.Conv3d(in_channels=3, out_channels=n_filter_1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    self.maxpool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    self.batch_norm_1 = nn.BatchNorm3d(n_filter_1)

    self.conv_layer_2 = nn.Conv3d(in_channels=n_filter_1, out_channels=n_filter_2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    self.batch_norm_2 = nn.BatchNorm3d(n_filter_2)

    self.conv_layer_3 = nn.Conv3d(in_channels=n_filter_2, out_channels=n_filter_3, kernel_size=(4, 2, 2), stride=(1, 2, 2))
    self.batch_norm_3 = nn.BatchNorm3d(n_filter_3)

    conv_net_output_size = self._get_conv_out_size(state_size)

    self.fully_connected_1 = nn.Linear(conv_net_output_size, 1024)     
    # finally, create action_size output channel
    self.fully_connected_2 = nn.Linear(1024, action_size)
```
The network is visualized as shown below:
![alt text](./VisualDQN.png)

#### Results

##### Visual Banana
```angular2html
Episode 100	Average Score: 0.255
Episode 200	Average Score: 2.70
Episode 300	Average Score: 5.45
Episode 400	Average Score: 8.95
Episode 500	Average Score: 10.62
Episode 600	Average Score: 11.36
Episode 700	Average Score: 12.29
Episode 832	Average Score: 13.01
Environment solved in 832 episodes!	Average Score: 13.01
```

#### Ideas learnt
From the challenge project, I observed the following:
* The use of 3D convolutional layers improves performance in the training process.
* Max Pooling 3D helps diminish the number of weights to be learnt.

## Project Instruction from Udacity
### Getting Started

1. Install UNITY and ML-Agent following this instruction: 
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

To install Unity on Ubuntu, see this post:
https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/page-2

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
