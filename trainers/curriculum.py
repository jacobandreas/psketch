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

        self.random = np.random.RandomState(0)

    #@profile
    def do_rollout(self, model, world, possible_tasks):

        n_batch = 100

        states_before = []
        steps = []
        goal_args = []
        for _ in range(n_batch):
            goal, st = possible_tasks[self.random.choice(len(possible_tasks))]
            goal_name, goal_arg = goal
            scenario = world.sample_scenario_with_goal(goal_arg)
            states_before.append(scenario.init())
            steps.append(st)
            goal_args.append(goal_arg)
        #for state in states_before:
        #    print state.grid.sum(axis=2).sum(axis=1)

        model.init(states_before, steps)
        transitions = [[] for _ in range(n_batch)]

        total_reward = 0.
        #timer = np.asarray([MAX_TIMESTEPS for _ in range(n_batch)])
        timer = MAX_TIMESTEPS
        done = [False for _ in range(n_batch)]

        #while (timer > 0).all() and not all(done):
        while not all(done) and timer > 0:
            mstates_before = model.get_state()
            action, terminate = model.act(states_before)
            #print action
            mstates_after = model.get_state()
            states_after = [None for _ in range(n_batch)]
            for i in range(n_batch):
                if action[i] is None:
                    assert done[i]
                elif terminate[i]:
                    win = states_before[i].inventory[self.cookbook.index[goal_args[i]]] > 0
                    reward = 1 if win else 0
                    states_after[i] = states_before[i].as_terminal()
                elif action[i] >= world.n_actions:
                    states_after[i] = states_before[i]
                    #timer[i] = MAX_TIMESTEPS
                    reward = 0
                else:
                    reward, states_after[i] = states_before[i].step(action[i])

                #reward = max(min(reward, 1), -1)
                if not done[i]:
                    #transitions.append(Transition(state_before, model_state_before,
                    #    action, state_after, model_state_after, reward))
                    transitions[i].append(Transition(states_before[i], mstates_before[i],
                            action[i], states_after[i], mstates_after[i],
                            reward))
                    total_reward += reward

                #if terminate[i] or timer[i] <= 0:
                if terminate[i]:
                    done[i] = True

            states_before = states_after
            timer -= 1

        #print [t.a for t in transitions[0]]
        #world.visualize(transitions[0])
        #assert not any(len(t) == 0 for t in transitions)
        return transitions, total_reward / n_batch, steps

    #@profile
    def train(self, model, world):
        model.prepare(world)
        #model.load()
        #actions = itertools.cycle(self.tasks_by_action.keys())
        actions = self.tasks_by_action.keys()
        #actions = [2]
        max_steps = 1

        while True:
        #for _ in range(1):
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
                count = 0
                by_task = defaultdict(lambda: 0)
                for j in range(N_UPDATE):
                    #counter = defaultdict(lambda: 0)
                    err = None
                    while err is None:
                    #for _ in range(2):
                        #goal, steps = possible_tasks[self.random.choice(len(possible_tasks))]
                        transitions, reward, steps = self.do_rollout(model, world, possible_tasks) #, goal, steps)
                        for t in transitions:
                            for tt in t:
                                if tt.r > 0:
                                    by_task[tt.m1.task] += 1
                        total_reward += reward
                        n_rollouts += 1
                        for t in transitions:
                            #print [tt.a for tt in t]
                            model.experience(t)
                        #print
                        #err = model.train(action, update_critic=False)
                        err = model.train(action)
                        count += 1
                        #err = model.train()
                    total_err += err
                    #print ">>", counter

                #world.visualize(transitions)
                print
                print action
                print [t.a for t in transitions[-1]]
                print total_reward / n_rollouts
                print err / N_UPDATE
                print dict(by_task)
                print count
                print
                min_reward = min(min_reward, total_reward / n_rollouts)

            print "MIN REWARD", min_reward
            print
            if min_reward > 0.95:
                max_steps += 1
                model.save()
                #exit()
