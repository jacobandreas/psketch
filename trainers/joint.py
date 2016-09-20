from misc import util
from misc.experience import Transition
from worlds.cookbook import Cookbook

from collections import defaultdict
import numpy as np
import yaml

N_ITERS = 1000000
N_UPDATE = 100
MAX_TIMESTEPS = 20

class CurriculumTrainer(object):
    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.action_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)

        self.tasks_by_action = defaultdict(list)
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            goal = (self.action_index.index(goal[0]), self.cookbook.index[goal[1]])
            steps = [util.parse_fexp(s) for s in hint]
            steps = [(self.action_index.index(a), self.cookbook.index[b])
                    for a, b in steps]
            for action, arg in steps:
                self.tasks_by_action[action].append((goal, steps))

    def do_rollout(self, model, world, max_size):
        transitions = []
        goal = None
        while goal is None:
            g = np.random.choice(self.hints.keys())
            if len(self.hints[g]) <= max_size:
                goal = g
        goal_action, goal_arg = util.parse_fexp(goal)
        steps = [util.parse_fexp(s) for s in self.hints[goal]]
        steps = [(self.action_index.index(a), self.cookbook.index[b])
                for a, b in steps]

        scenario = world.sample_scenario_with_goal(goal_arg)
        state_before = scenario.init()
        if goal == "fakemake[plank]":
            state_before.inventory[self.cookbook.index["wood"]] += 4
        ### if goal == "left[iron]":
        ###     state_before.pos = len(state_before.features()) - 1
        ###     state_before.reached_right = True

        model.init(state_before, steps)

        total_reward = 0
        hit = 0
        for t in range(MAX_TIMESTEPS):
            model_state_before = model.get_state()
            action, terminate = model.act(state_before)
            model_state_after = model.get_state()
            if terminate:
                win = state_before.inventory[self.cookbook.index[goal_arg]] > 0
                ### if goal == "right[iron]":
                ###     win = state_before.reached_right
                ### elif goal == "left[iron]":
                ###     win = state_before.pos == 0
                ### elif goal == "both[iron]":
                ###     win = state_before.pos == 0 and state_before.reached_right

                reward = 1 if win else 0
                state_after = state_before.as_terminal()
            elif action >= world.n_actions:
                state_after = state_before
                reward = 0
            else:
                reward, state_after = state_before.step(action)

            reward = max(min(reward, 1), -1)
            transitions.append(Transition(state_before, model_state_before,
                action, state_after, model_state_after, reward))
            total_reward += reward
            if terminate:
                break

            state_before = state_after

        #if hit == 2: print "both", total_reward

        return transitions, total_reward, steps

    def train(self, model, world):
        model.prepare(world)
        total_err = 0.
        total_reward = 0.
        by_steps = defaultdict(lambda: 0.)
        max_len = 2

        i_rollout = 0
        i_update = 0

        while i_update < N_ITERS:
            transitions, reward, steps = self.do_rollout(model, world, max_len)
            model.experience(transitions)
            #by_len[len(steps)] += reward
            by_steps[tuple(steps)] += reward
            total_reward += reward
            i_rollout += 1
            err = model.train()
            if err is None:
                continue
            i_update += 1
            total_err += err

            if (i_update + 1) % N_UPDATE == 0:
                #world.visualize(transitions)
                print "[max len]", max_len
                print "[transitions]", [t.a for t in transitions]
                print "[hint]", steps
                print "[reward]", reward
                #print transitions[-1].s2.inventory
                #print [t.s1.pos for t in transitions]
                #print [t.s2.pos for t in transitions]
                print dict(by_steps)
                #print "%5.3f %5.3f" % \
                #        (total_err / N_UPDATE, total_reward / i_rollout)
                print total_err / N_UPDATE
                print total_reward / i_rollout
                print

                if total_reward / i_rollout > 0.8:
                    max_len += 1

                total_err = 0.
                total_reward = 0.
                i_rollout = 0
                by_len = {1: 0, 2: 0}
                by_steps = defaultdict(lambda: 0.)
                model.roll()
