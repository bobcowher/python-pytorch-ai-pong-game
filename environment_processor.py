import gym
import torch
from PIL import Image
import numpy as np


class PreprocessEnv(gym.Wrapper):
    
    def __init__(self, env, device):
        gym.Wrapper.__init__(self, env)
        self.device = device
    
    def reset(self):
        obs, info = self.env.reset()
        return self.process_observation(observation=obs)
    
    def step(self, action):
        # print(action)
        action = action.item()
        next_state, reward, done, truncated, info = self.env.step(action)
        next_state = self.process_observation(next_state)
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)

        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        return next_state, reward, done, info

    def process_observation(self, observation):

        PRINT_SHAPE = False

        IMG_SHAPE = (84, 84)
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0

        img = img.to(self.device)

        return img