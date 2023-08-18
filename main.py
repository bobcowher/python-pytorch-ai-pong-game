import torch

from image_processor import ImageProcessor
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


eps = 1.0
# eps = 0

epochs = 100

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12
BATCH_SIZE = 128
GAMMA = 0.99

input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
def optimize_model(optimizer, replay_memory, model):




    if len(replay_memory) < BATCH_SIZE:
        return
    transitions = replay_memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(torch.tensor(batch.action))
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()

def train(model, env, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    replay_memory = ReplayMemory(model.replay_memory_size)

    state, info = env.reset()

    for epoch in range(epochs):
        state = processor.process_observation(state)

        state = processor.process_state_batch(state)

        action = choose_action(state=state, model=model, eps_decay=0.1, training=True)
        print(action)
        next_state, reward, done, truncated, info = env.step(action)

        time.sleep(0.1)

        # Append to replay memory and then trim replay memory if it's too long
        # next_state = torch.from_numpy(next_state)

        replay_memory.push(state, action, next_state, reward)

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        state = next_state

        optimize_model(optimizer=optimizer, replay_memory=replay_memory, model=model)








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


processor = ImageProcessor()





model = build_the_model(input_shape=input_shape)

env = gym.make("ALE/Pong-v5", render_mode='human')

train(model, env, 1000)