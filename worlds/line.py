from cookbook import Cookbook
import numpy as np

SIZE = 5

class LineWorld(object):
    def __init__(self, config):
        self.n_features = SIZE + 1
        self.n_actions = 2
        self.cookbook = Cookbook(config.recipes)

    def sample_scenario_with_goal(self, goal):
        return self.sample_scenario()

    def sample_scenario(self):
        return LineScenario(0)

class LineScenario(object):
    def __init__(self, init_pos):
        self.init_pos = init_pos
        self.terminal = LineState(0, reached_right=True)

    def init(self):
        return LineState(self.init_pos, reached_right=False)

class LineState(object):
    def __init__(self, pos, reached_right):
        self.pos = pos
        self.reached_right = reached_right

    def features(self):
        feats = np.zeros(SIZE + 1)
        feats[self.pos] = 1
        feats[SIZE] = self.reached_right
        return feats

    def step(self, action):
        npos = self.pos
        if action == 0:
            npos -= 1
        elif action == 1:
            npos += 1
        else:
            assert False

        npos = min(max(npos, 0), SIZE - 1)
        reached_right = self.reached_right or npos == SIZE - 1
        return 0, LineState(npos, reached_right)
