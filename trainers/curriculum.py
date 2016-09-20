from misc import util
from misc.experience import Transition
from worlds.cookbook import Cookbook

from collections import defaultdict
import itertools
import numpy as np
import yaml

N_ITERS = 1000000
N_UPDATE = 100
MAX_TIMESTEPS = 20

class CurriculumTrainer(object):
    #@profile
    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.action_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)

        self.tasks_by_action = defaultdict(list)
        self.tasks = []
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            #goal = (self.action_index.index(goal[0]), self.cookbook.index[goal[1]])
            steps = [util.parse_fexp(s) for s in hint]
            steps = [(self.action_index.index(a), self.cookbook.index[b])
                    for a, b in steps]
            self.tasks.append((goal, steps))
            for action, arg in steps:
                self.tasks_by_action[action].append((goal, steps))

    #@profile
    def do_rollout(self, model, world, goal, steps):
        goal_name, goal_arg = goal
        scenario = world.sample_scenario_with_goal(goal_arg)
        state_before = scenario.init()
        model.init(state_before, steps)
        transitions = []

        total_reward = 0
        timer = MAX_TIMESTEPS
        while timer > 0:
            model_state_before = model.get_state()
            action, terminate = model.act(state_before)
            model_state_after = model.get_state()
            if terminate:
                win = state_before.inventory[self.cookbook.index[goal_arg]] > 0
                reward = 1 if win else 0
                state_after = state_before.as_terminal()
            elif action >= world.n_actions:
                state_after = state_before
                timer = MAX_TIMESTEPS
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
            timer -= 1

        return transitions, total_reward, steps

    #@profile
    def train(self, model, world):
        model.prepare(world)
        #model.load()
        #actions = itertools.cycle(self.tasks_by_action.keys())
        actions = self.tasks_by_action.keys()
        #actions = [2]
        max_steps = 1

        while True:
            print "MAX_STEPS", max_steps
            min_reward = np.inf
            for action in actions:
            ### for _ in range(1):
                possible_tasks = self.tasks_by_action[action]
                #possible_tasks = self.tasks
                possible_tasks = [t for t in possible_tasks if len(t[1]) <= max_steps]
                if len(possible_tasks) == 0:
                    continue

                total_reward = 0.
                total_err = 0.
                n_rollouts = 0.
                for j in range(N_UPDATE):
                    err = None
                    while err is None:
                        goal, steps = possible_tasks[np.random.choice(len(possible_tasks))]
                        transitions, reward, steps = self.do_rollout(model, world, goal, steps)
                        total_reward += reward
                        n_rollouts += 1
                        model.experience(transitions)
                        #err = model.train(action, update_critic=False)
                        err = model.train(action)
                        #err = model.train()
                    total_err += err

                #world.visualize(transitions)
                print
                print [t.a for t in transitions]
                print total_reward / n_rollouts
                print err / N_UPDATE
                print
                min_reward = min(min_reward, total_reward / n_rollouts)

            print "MIN REWARD", min_reward
            print
            if min_reward > 0.95:
                max_steps += 1
                model.save()
                #exit()
