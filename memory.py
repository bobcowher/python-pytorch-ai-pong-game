import torch
import random


class ReplayMemory:
    
    def __init__(self, capacity=500000, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def insert(self, transition):
        transition = [item.cpu() for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)