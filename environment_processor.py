import gym
import torch
from PIL import Image
import numpy as np


class PreprocessEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        obs = self.env.reset()
        return self.process_observation(observation=obs)
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        next_state = self.process_observation(next_state)
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done, info

    def process_observation(self, observation):
        IMG_SHAPE = (84, 84)
        print(observation)
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        return img

    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        #         processed_batch = batch.astype('float32')/255.0

        return processed_batch