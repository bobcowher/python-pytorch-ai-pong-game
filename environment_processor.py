import gym
import torch
from PIL import Image
import numpy as np


class PreprocessEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
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
        return next_state, reward, done, info

    def process_observation(self, observation):

        PRINT_SHAPE = False

        IMG_SHAPE = (84, 84)
        print(observation.shape) if PRINT_SHAPE else None
        img = Image.fromarray(observation)
        print("Break 1", img.size) if PRINT_SHAPE else None
        img = img.resize(IMG_SHAPE)
        print("Break 2", img.size) if PRINT_SHAPE else None
        img = img.convert("L")
        print("Break 3", img.size) if PRINT_SHAPE else None
        img = np.array(img)
        print("Break 4", img.shape) if PRINT_SHAPE else None
        img = torch.from_numpy(img)
        print("Break 5", img.shape) if PRINT_SHAPE else None
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        print("Break 6", img.shape) if PRINT_SHAPE else None
        img = img / 255.0
        print("Break 7", img.shape) if PRINT_SHAPE else None
        
        return img