from unityagents import UnityEnvironment
import numpy as np

class CollectBanana():

    def __init__(self, path):

        env = UnityEnvironment(file_name=path)

        self.base = env
        # get the default brain
        self.brain_name = self.base.brain_names[0]
        self.brain = self.base.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.train_mode = True
        self.last_frame = None
        self.last2_frame = None
        self.last3_frame = None
        self.reset()
        self.state_size = self.state.shape


    def get_state(self):
        
        # state size is 1,84,84,3 Width = 84 px Height = 84 px Nchannels = 3
        # Rearrange from NHWC (Batch, Height, Width, Channels) to NCHW 
        frame = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))[:,:,:,:] #cut the image partially
        frame_size = frame.shape  # 1,3,84,84
        #print(frame_size)
        # NCDHW
        nframes = 4
        self.state = np.zeros((1, frame_size[1], nframes, frame_size[2], frame_size[3]))
        #print(self.state.shape)
        self.state[0, :, 0, :, :] = frame
        if not(self.last_frame is None):
            self.state[0, :, 1, :, :] = self.last_frame
        if not(self.last2_frame is None):
            self.state[0, :, 2, :, :] = self.last2_frame
        if not (self.last3_frame is None):
            self.state[0, :, 3, :, :] = self.last3_frame
        self.last3_frame = self.last2_frame
        self.last2_frame = self.last_frame
        self.last_frame = frame


    def reset(self):

        self.env_info = self.base.reset(train_mode=self.train_mode)[self.brain_name]
        self.get_state()
        return self.state

    def render(self):
        pass

    def step(self, action):
        self.env_info = self.base.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards[0]  # get the reward
        done = self.env_info.local_done[0]  # see if episode has finished
        return self.state, reward, done, None #info is none

    def close(self):
        self.base.close()