from misc import util
import net

from collections import namedtuple, defaultdict
import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework.ops import IndexedSlicesValue

N_UPDATE = 2000
N_BATCH = 2000

N_HIDDEN = 128
N_EMBED = 64

DISCOUNT = 0.9

ActorModule = namedtuple("ActorModule", ["t_probs", "t_chosen_prob", "params",
        "t_decrement_op"])
CriticModule = namedtuple("CriticModule", ["t_value", "params"])
Trainer = namedtuple("Trainer", ["t_loss", "t_grad", "t_train_op"])
InputBundle = namedtuple("InputBundle", ["t_arg", "t_step", "t_feats", 
        "t_action_mask", "t_reward"])

ModelState = namedtuple("ModelState", ["action", "arg", "remaining", "task", "step"])

def increment_sparse_or_dense(into, increment):
    assert isinstance(into, np.ndarray)
    if isinstance(increment, IndexedSlicesValue):
        for i in range(increment.values.shape[0]):
            into[increment.indices[i], :] += increment.values[i, :]
    else:
        into += increment

class ModularACModel(object):
    def __init__(self, config):
        self.experiences = []
        self.world = None
        tf.set_random_seed(0)
        self.next_actor_seed = 0
        self.config = config

    def prepare(self, world, trainer):
        assert self.world is None
        self.world = world
        self.trainer = trainer

        self.n_tasks = len(trainer.task_index)
        self.n_modules = len(trainer.subtask_index)
        self.max_task_steps = max(len(t.steps) for t in trainer.task_index.contents.keys())
        if self.config.model.featurize_plan:
            self.n_features = world.n_features + self.n_modules * self.max_task_steps
        else:
            self.n_features = world.n_features

        self.n_actions = world.n_actions + 1
        self.t_n_steps = tf.Variable(1., name="n_steps")
        self.t_inc_steps = self.t_n_steps.assign(self.t_n_steps + 1)
        # TODO configurable optimizer
        self.optimizer = tf.train.RMSPropOptimizer(0.0003)

        def build_actor(index, t_input, t_action_mask, extra_params=[]):
            with tf.variable_scope("actor_%s" % index):
                t_action_score, v_action = net.mlp(t_input, (N_HIDDEN, self.n_actions))

                # TODO this is pretty gross
                v_bias = v_action[-1]
                assert "b1" in v_bias.name
                t_decrement_op = v_bias[-1].assign(v_bias[-1] - 3)

                t_action_logprobs = tf.nn.log_softmax(t_action_score)
                t_chosen_prob = tf.reduce_sum(t_action_mask * t_action_logprobs, 
                        reduction_indices=(1,))

            return ActorModule(t_action_logprobs, t_chosen_prob, 
                    v_action+extra_params, t_decrement_op)

        def build_critic(index, t_input, t_reward, extra_params=[]):
            with tf.variable_scope("critic_%s" % index):
                if self.config.model.baseline in ("task", "common"):
                    t_value = tf.get_variable("b", shape=(),
                            initializer=tf.constant_initializer(0.0))
                    v_value = [t_value]
                elif self.config.model.baseline == "state":
                    t_value, v_value = net.mlp(t_input, (1,))
                    t_value = tf.squeeze(t_value)
                else:
                    raise NotImplementedError(
                            "Baseline %s is not implemented" % self.config.model.baseline)
            return CriticModule(t_value, v_value + extra_params)

        def build_actor_trainer(actor, critic, t_reward):
            t_advantage = t_reward - critic.t_value
            # TODO configurable entropy regularizer
            actor_loss = -tf.reduce_sum(actor.t_chosen_prob * t_advantage) + \
                    0.001 * tf.reduce_sum(tf.exp(actor.t_probs) * actor.t_probs)
            actor_grad = tf.gradients(actor_loss, actor.params)
            actor_trainer = Trainer(actor_loss, actor_grad, 
                    self.optimizer.minimize(actor_loss, var_list=actor.params))
            return actor_trainer

        def build_critic_trainer(t_reward, critic):
            t_advantage = t_reward - critic.t_value
            critic_loss = tf.reduce_sum(tf.square(t_advantage))
            critic_grad = tf.gradients(critic_loss, critic.params)
            critic_trainer = Trainer(critic_loss, critic_grad,
                    self.optimizer.minimize(critic_loss, var_list=critic.params))
            return critic_trainer

        # placeholders
        t_arg = tf.placeholder(tf.int32, shape=(None,))
        t_step = tf.placeholder(tf.float32, shape=(None, 1))
        t_feats = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))

        if self.config.model.use_args:
            t_embed, v_embed = net.embed(t_arg, len(trainer.cookbook.index),
                    N_EMBED)
            xp = v_embed
            t_input = tf.concat(1, (t_embed, t_feats))
        else:
            t_input = t_feats
            xp = []

        actors = {}
        actor_trainers = {}
        critics = {}
        critic_trainers = {}

        if self.config.model.featurize_plan:
            actor = build_actor(0, t_input, t_action_mask, extra_params=xp)
            for i_module in range(self.n_modules):
                actors[i_module] = actor
        else:
            for i_module in range(self.n_modules):
                actor = build_actor(i_module, t_input, t_action_mask, extra_params=xp)
                actors[i_module] = actor

        if self.config.model.baseline == "common":
            common_critic = build_critic(0, t_input, t_reward, extra_params=xp)
        for i_task in range(self.n_tasks):
            if self.config.model.baseline == "common":
                critic = common_critic
            else:
                critic = build_critic(i_task, t_input, t_reward, extra_params=xp)
            for i_module in range(self.n_modules):
                critics[i_task, i_module] = critic

        for i_module in range(self.n_modules):
            for i_task in range(self.n_tasks):
                critic = critics[i_task, i_module]
                critic_trainer = build_critic_trainer(t_reward, critic)
                critic_trainers[i_task, i_module] = critic_trainer

                actor = actors[i_module]
                actor_trainer = build_actor_trainer(actor, critic, t_reward)
                actor_trainers[i_task, i_module] = actor_trainer

        self.t_gradient_placeholders = {}
        self.t_update_gradient_op = None

        params = []
        for module in actors.values() + critics.values():
            params += module.params
        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.session.run([actor.t_decrement_op for actor in actors.values()])

        self.actors = actors
        self.critics = critics
        self.actor_trainers = actor_trainers
        self.critic_trainers = critic_trainers
        self.inputs = InputBundle(t_arg, t_step, t_feats, t_action_mask, t_reward)

    def init(self, states, tasks):
        n_act_batch = len(states)
        self.subtasks = []
        self.args = []
        self.i_task = []
        for i in range(n_act_batch):
            if self.config.model.use_args:
                subtasks, args = zip(*tasks[i].steps)
            else:
                subtasks = tasks[i].steps
                args = [None] * len(subtasks)
            self.subtasks.append(tuple(subtasks))
            self.args.append(tuple(args))
            self.i_task.append(self.trainer.task_index[tasks[i]])
        self.i_subtask = [0 for _ in range(n_act_batch)]
        self.i_step = np.zeros((n_act_batch, 1))
        self.randoms = []
        for _ in range(n_act_batch):
            self.randoms.append(np.random.RandomState(self.next_actor_seed))
            self.next_actor_seed += 1

    def save(self):
        self.saver.save(self.session, 
                os.path.join(self.config.experiment_dir, "modular_ac.chk"))

    def load(self):
        path = os.path.join(self.config.experiment_dir, "modular_ac.chk")
        logging.info("loaded %s", path)
        self.saver.restore(self.session, path)

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            running_reward = running_reward * DISCOUNT + transition.r
            n_transition = transition._replace(r=running_reward)
            if n_transition.a < self.n_actions:
                self.experiences.append(n_transition)

    def featurize(self, state, mstate):
        if self.config.model.featurize_plan:
            task_features = np.zeros((self.max_task_steps, self.n_modules))
            for i, m in enumerate(self.trainer.task_index.get(mstate.task).steps):
                task_features[i, m] = 1
            return np.concatenate((state.features(), task_features.ravel()))
        else:
            return state.features()

    def act(self, states):
        mstates = self.get_state()
        self.i_step += 1
        by_mod = defaultdict(list)
        n_act_batch = len(self.i_subtask)

        for i in range(n_act_batch):
            by_mod[self.i_task[i], self.i_subtask[i]].append(i)

        action = [None] * n_act_batch
        terminate = [None] * n_act_batch

        for k, indices in by_mod.items():
            i_task, i_subtask = k
            assert len(set(self.subtasks[i] for i in indices)) == 1
            if i_subtask >= len(self.subtasks[indices[0]]):
                continue
            actor = self.actors[self.subtasks[indices[0]][i_subtask]]
            feed_dict = {
                self.inputs.t_feats: [self.featurize(states[i], mstates[i]) for i in indices],
            }
            if self.config.model.use_args:
                feed_dict[self.inputs.t_arg] = [mstates[i].arg for i in indices]

            logprobs = self.session.run([actor.t_probs], feed_dict=feed_dict)[0]
            probs = np.exp(logprobs)
            for pr, i in zip(probs, indices):

                if self.i_step[i] >= self.config.model.max_subtask_timesteps:
                    a = self.n_actions
                else:
                    a = self.randoms[i].choice(self.n_actions, p=pr)

                if a >= self.world.n_actions:
                    self.i_subtask[i] += 1
                    self.i_step[i] = 0.
                t = self.i_subtask[i] >= len(self.subtasks[indices[0]])
                action[i] = a
                terminate[i] = t

        return action, terminate

    def get_state(self):
        out = []
        for i in range(len(self.i_subtask)):
            if self.i_subtask[i] >= len(self.subtasks[i]):
                out.append(ModelState(None, None, None, None, [0.]))
            else:
                out.append(ModelState(
                        self.subtasks[i][self.i_subtask[i]], 
                        self.args[i][self.i_subtask[i]], 
                        len(self.args) - self.i_subtask[i],
                        self.i_task[i],
                        self.i_step[i].copy()))
        return out

    def train(self, action=None, update_actor=True, update_critic=True):
        if action is None:
            experiences = self.experiences
        else:
            experiences = [e for e in self.experiences if e.m1.action == action]
        if len(experiences) < N_UPDATE:
            return None
        batch = experiences[:N_UPDATE]

        by_mod = defaultdict(list)
        for e in batch:
            by_mod[e.m1.task, e.m1.action].append(e)

        grads = {}
        params = {}
        for module in self.actors.values() + self.critics.values():
            for param in module.params:
                if param.name not in grads:
                    grads[param.name] = np.zeros(param.get_shape(), np.float32)
                    params[param.name] = param
        touched = set()

        total_actor_err = 0
        total_critic_err = 0
        for i_task, i_mod1 in by_mod:
            actor = self.actors[i_mod1]
            critic = self.critics[i_task, i_mod1]
            actor_trainer = self.actor_trainers[i_task, i_mod1]
            critic_trainer = self.critic_trainers[i_task, i_mod1]

            all_exps = by_mod[i_task, i_mod1]
            for i_batch in range(int(np.ceil(1. * len(all_exps) / N_BATCH))):
                exps = all_exps[i_batch * N_BATCH : (i_batch + 1) * N_BATCH]
                s1, m1, a, s2, m2, r = zip(*exps)
                feats1 = [self.featurize(s, m) for s, m in zip(s1, m1)]
                args1 = [m.arg for m in m1]
                steps1 = [m.step for m in m1]
                a_mask = np.zeros((len(exps), self.n_actions))
                for i_datum, aa in enumerate(a):
                    a_mask[i_datum, aa] = 1

                feed_dict = {
                    self.inputs.t_feats: feats1,
                    self.inputs.t_action_mask: a_mask,
                    self.inputs.t_reward: r
                }
                if self.config.model.use_args:
                    feed_dict[self.inputs.t_arg] = args1

                actor_grad, actor_err = self.session.run([actor_trainer.t_grad, actor_trainer.t_loss],
                        feed_dict=feed_dict)
                critic_grad, critic_err = self.session.run([critic_trainer.t_grad, critic_trainer.t_loss], 
                        feed_dict=feed_dict)

                total_actor_err += actor_err
                total_critic_err += critic_err

                if update_actor:
                    for param, grad in zip(actor.params, actor_grad):
                        increment_sparse_or_dense(grads[param.name], grad)
                        touched.add(param.name)
                if update_critic:
                    for param, grad in zip(critic.params, critic_grad):
                        increment_sparse_or_dense(grads[param.name], grad)
                        touched.add(param.name)

        global_norm = 0
        for k in params:
            grads[k] /= N_UPDATE
            global_norm += (grads[k] ** 2).sum()
        rescale = min(1., 1. / global_norm)

        # TODO precompute this part of the graph
        updates = []
        feed_dict = {}
        for k in params:
            param = params[k]
            grad = grads[k]
            grad *= rescale
            if k not in self.t_gradient_placeholders:
                self.t_gradient_placeholders[k] = tf.placeholder(tf.float32, grad.shape)
            feed_dict[self.t_gradient_placeholders[k]] = grad
            updates.append((self.t_gradient_placeholders[k], param))

        #    if "embed" in k:
        #        for i, row in enumerate(grads[k]):
        #            print self.trainer.cookbook.index.get(i)
        #            print row

        #exit()

        if self.t_update_gradient_op is None:
            self.t_update_gradient_op = self.optimizer.apply_gradients(updates)
        self.session.run(self.t_update_gradient_op, feed_dict=feed_dict)

        self.experiences = []
        self.session.run(self.t_inc_steps)

        return np.asarray([total_actor_err, total_critic_err]) / N_UPDATE
