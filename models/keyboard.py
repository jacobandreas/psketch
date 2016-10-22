from misc import util
import net
import trpo

from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

N_UPDATE = 2000
N_BATCH = 2000

N_HIDDEN = 128
N_EMBED = 64

DISCOUNT = 0.9

class KeyboardModel(object):
    def __init__(self, config):
        self.world = None

    def prepare(self, world, trainer):
        assert self.world is None
        self.world = world

    def init(self, states, tasks):
        self.n_tasks = len(tasks)

    def save(self):
        pass

    def load(self):
        pass

    def experience(self, episode):
        pass

    def act(self, states):
        print states[0].pp()
        k = raw_input("action: ")
        action = int(k)
        return [action] * len(states), [False] * len(states)

    def get_state(self):
        return [None] * self.n_tasks

    def train(self, action=None, update_actor=True, update_critic=True):
        pass
