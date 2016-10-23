from misc import util
from misc.experience import Transition
from worlds.cookbook import Cookbook

from collections import defaultdict, namedtuple
import itertools
import logging
import numpy as np
import yaml

N_ITERS = 1000000
N_UPDATE = 500
N_BATCH = 100
IMPROVEMENT_THRESHOLD = 0.8

Task = namedtuple("Task", ["goal", "steps"])

class CurriculumTrainer(object):
    def __init__(self, config):
        # load configs
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)

        # initialize randomness
        self.random = np.random.RandomState(0)

        # organize task and subtask indices
        self.tasks_by_subtask = defaultdict(list)
        self.tasks = []
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]), self.cookbook.index[goal[1]])
            if config.model.use_args:
                steps = [util.parse_fexp(s) for s in hint]
                steps = [(self.subtask_index.index(a), self.cookbook.index[b])
                        for a, b in steps]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask, _ in steps:
                    self.tasks_by_subtask[subtask].append(task)
            else:
                steps = [self.subtask_index.index(a) for a in hint]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask in steps:
                    self.tasks_by_subtask[subtask].append(task)
            self.tasks.append(task)
            self.task_index.index(task)

    def do_rollout(self, model, world, possible_tasks, task_probs):
        states_before = []
        tasks = []
        goal_names = []
        goal_args = []

        # choose tasks and initialize model
        for _ in range(N_BATCH):
            task = possible_tasks[self.random.choice(
                len(possible_tasks), p=task_probs)]
            goal, _ = task
            goal_name, goal_arg = goal
            scenario = world.sample_scenario_with_goal(goal_arg)
            states_before.append(scenario.init())
            tasks.append(task)
            goal_names.append(goal_name)
            goal_args.append(goal_arg)
        model.init(states_before, tasks)
        transitions = [[] for _ in range(N_BATCH)]

        # initialize timer
        total_reward = 0.
        timer = self.config.trainer.max_timesteps
        done = [False for _ in range(N_BATCH)]

        # act!
        while not all(done) and timer > 0:
            mstates_before = model.get_state()
            action, terminate = model.act(states_before)
            mstates_after = model.get_state()
            states_after = [None for _ in range(N_BATCH)]
            for i in range(N_BATCH):
                if action[i] is None:
                    assert done[i]
                elif terminate[i]:
                    win = states_before[i].satisfies(goal_names[i], goal_args[i])
                    reward = 1 if win else 0
                    states_after[i] = None
                elif action[i] >= world.n_actions:
                    states_after[i] = states_before[i]
                    reward = 0
                else:
                    reward, states_after[i] = states_before[i].step(action[i])

                if not done[i]:
                    transitions[i].append(Transition(
                            states_before[i], mstates_before[i], action[i], 
                            states_after[i], mstates_after[i], reward))
                    total_reward += reward

                if terminate[i]:
                    done[i] = True

            states_before = states_after
            timer -= 1

        return transitions, total_reward / N_BATCH

    def train(self, model, world):
        model.prepare(world, self)
        #model.load()
        subtasks = self.tasks_by_subtask.keys()
        max_steps = 1

        task_probs = []
        while True:
            logging.info("[max steps] %d", max_steps)
            min_reward = np.inf

            # TODO refactor
            for _ in range(1):
                # make sure there's something of this length
                possible_tasks = self.tasks
                possible_tasks = [t for t in possible_tasks if len(t.steps) <= max_steps]
                if len(possible_tasks) == 0:
                    continue

                # re-initialize task probs if necessary
                if len(task_probs) != len(possible_tasks):
                    task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

                total_reward = 0.
                total_err = 0.
                count = 0.
                task_rewards = defaultdict(lambda: 0)
                task_counts = defaultdict(lambda: 0)
                for j in range(N_UPDATE):
                    err = None
                    # get enough samples for one training step
                    while err is None:
                        transitions, reward = self.do_rollout(model, world, 
                                possible_tasks, task_probs)
                        for t in transitions:
                            tr = sum(tt.r for tt in t)
                            task_rewards[t[0].m1.task] += tr
                            task_counts[t[0].m1.task] += 1
                        total_reward += reward
                        count += 1
                        for t in transitions:
                            model.experience(t)
                        err = model.train()
                    total_err += err

                # recompute task probs
                task_probs = np.zeros(len(possible_tasks))
                for i, task in enumerate(possible_tasks):
                    i_task = self.task_index[task]
                    task_probs[i] = 1. * task_rewards[i_task] / task_counts[i_task]
                task_probs = 1 - task_probs
                task_probs += 0.01
                task_probs /= task_probs.sum()

                # log
                logging.info("[tasks] %s", [p[0] for p in  possible_tasks])
                logging.info("[probs] %s", task_probs)
                logging.info("")
                logging.info("[rollout0] %s", [t.a for t in transitions[0]])
                logging.info("[rollout1] %s", [t.a for t in transitions[1]])
                logging.info("[rollout2] %s", [t.a for t in transitions[2]])
                logging.info("[reward] %s", total_reward / count)
                logging.info("[error] %s", err / N_UPDATE)
                score_dict = {self.task_index.get(k): 1. * task_rewards[k] / task_counts[k] for k in task_rewards}
                logging.info("[scores] %s", score_dict)
                logging.info("")
                min_reward = min(min_reward, min(score_dict.values()))

            logging.info("[min reward] %s", min_reward)
            logging.info("")
            if min_reward > self.config.trainer.improvement_threshold:
                max_steps += 1
                model.save()
