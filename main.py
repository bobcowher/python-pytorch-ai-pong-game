import torch

from PIL import Image
import numpy as np
import gym
import random
import time
from model import *
import copy
from torchsummary import summary
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from environment_processor import PreprocessEnv
from memory import ReplayMemory
from utils import plot_stats

eps = 1.0
# eps = 0

epochs = 100

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12
BATCH_SIZE = 128
GAMMA = 0.99

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train(model, target_model, batch_size, env, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    memory = ReplayMemory()

    stats = {'MSELoss': [], 'Returns': []}

    for epoch in tqdm(range(1, epochs + 1)):

        state = env.reset()
        done = False
        ep_return = 0

        while not done:

            action = policy(state=state, model=model, epsilon=0.2)
            # print(action)
            next_state, reward, done, info = env.step(action)

            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = model(state_b).gather(1, action_b)
                
                next_qsa_b = target_model(next_state_b)
                next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                
                target_b = reward_b + ~done_b * GAMMA * next_qsa_b
                
                loss = F.mse_loss(qsa_b, target_b)
                
                model.zero_grad()
                
                loss.backward()
                
                optimizer.step()

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)
        
        if epoch % 10 == 0:
            target_model.load_state_dict(model.state_dict())
    
    return stats


def policy(state, model, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(6, (1, 1))
    else:
        av = model(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)


model = build_the_model()

target_model = copy.deepcopy(model).eval()

env = gym.make("ALE/Pong-v5", render_mode='human')

env = PreprocessEnv(env)

print(summary(model=model, input_size=(1, 84, 84), device='cpu'))

state = env.reset()

print(f"State shape is: {state.shape}")

action = model(state)


print(f"Demo action is {action}")

stats = train(model=model,
      target_model=target_model,
      batch_size=32,
      env=env,
      epochs=1000)

plot_stats(stats)