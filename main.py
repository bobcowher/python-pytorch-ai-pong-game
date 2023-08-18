import torch

from PIL import Image
import numpy as np
import gym
import random
import time
from model import *

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque

from environment_processor import PreprocessEnv

eps = 1.0
# eps = 0

epochs = 100

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12
BATCH_SIZE = 128
GAMMA = 0.99

input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train(model, env, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    state = env.reset()

    for epoch in range(epochs):

        action = choose_action(state=state, model=model, eps_decay=0.1, training=True)
        print(action)
        next_state, reward, done, info = env.step(action)

        time.sleep(0.1)

        # Append to replay memory and then trim replay memory if it's too long
        # next_state = torch.from_numpy(next_state)

        state = next_state





def choose_action(state, model, eps_decay=0, training=True):
    """
    Choose an action from either a random action space or the model passed in, using epsilon greedy.
    Decay the eps value by a factor of eps_decay every 1000 moves.
    :param state:
    :param model:
    :param eps_decay:
    :param training:
    :return:
    """

    global eps

    eps_threshold = (random.randint(1,100) / 100)

    print(f"Eps threshold is...{eps_threshold}")
    print(f"Eps is {eps}")

    # If a random number between 0 and 1 is greater than our epsilon greedy threshold, play a random move.
    # Otherwise, consult the model for a move.
    if eps_threshold > eps and training is True:
        output = model.forward(state)[0]
        action = torch.argmax(output)
    else:
        print("Grabbing random action")
        action = env.action_space.sample()

    if eps_decay > 0:
        eps = eps - (eps * (eps_decay/1000))

    return action





model = build_the_model(input_shape=input_shape)

env = gym.make("ALE/Pong-v5", render_mode='human')

env = PreprocessEnv(env)

train(model, env, 1000)