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

epochs = 100

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12
BATCH_SIZE = 128
GAMMA = 0.99

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train(model, target_model, batch_size, epochs, epsilon, epsilon_min, gamma, validate=False):

    model.to(device)
    target_model.to(device)

    if validate:
        env = gym.make("ALE/Pong-v5", render_mode='human')
    else:
        env = gym.make("ALE/Pong-v5")

    env = PreprocessEnv(env, device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    memory = ReplayMemory()

    stats = {'MSELoss': [], 'Returns': []}

    for epoch in tqdm(range(1, epochs + 1)):

        state = env.reset()
        state = state.to(device)
        done = False
        ep_return = 0

        while not done:

            action = policy(state=state, model=model, epsilon=epsilon)

            action = action.to(device)

            next_state, reward, done, info = env.step(action)

            reward = reward + 0.001

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
        print("")
        print(f"Epoch {epoch} out of {epochs} completed")
        print(f"Epoch return: {ep_return}")
        print(f"Epsilon is {epsilon}")

        # Reduce epsilon
        if epsilon > epsilon_min:
            epsilon = epsilon * gamma
        
        if epoch % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        if epoch % 100 == 0:
            save_the_model(model, 'models/latest.pt')
            save_the_model(model, f"models/model_iter_{epoch}.pt")
    
    return stats

def test(model):

    model = model.to(device)

    env = gym.make("ALE/Pong-v5", render_mode='human')

    env = PreprocessEnv(env, device=device)

    state = env.reset()

    state = state.to(device)

    done = False

    while not done:

        action = policy(state=state, model=model, epsilon=0.)

        # action = model(state)
        # print(action)

        print(action)

        action = action.to(device)

        next_state, reward, done, info = env.step(action)

        state = next_state



def policy(state, model, epsilon):
    if torch.rand(1) < epsilon:
        return torch.randint(6, (1, 1))
    else:
        av = model(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)


model = build_the_model(weights_filename='models/latest.pt')

target_model = copy.deepcopy(model).eval()

TRAIN = True

if TRAIN:

    stats = train(model=model,
          target_model=target_model,
          batch_size=64,
          epochs=5000,
          epsilon=1,
          epsilon_min=0.1,
          gamma=0.99)

    plot_stats(stats)

train(model=model,
          target_model=target_model,
          batch_size=64,
          epochs=1,
          epsilon=0,
          epsilon_min=0.1,
          gamma=0.995,
          validate=True)