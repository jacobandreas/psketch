from misc import util
import net
import trpo

from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

N_BATCH = 1000

N_HIDDEN = 256
N_EMBED = 64

N_MODULES = 5
N_TASKS = 5

DISCOUNT = 0.9
EPS = 0.1
TEMP = 1000
MAX_EXPERIENCES = 10000

ActorModule = namedtuple("ActorModule", ["t_probs", "t_chosen_prob", "params"])
CriticModule = namedtuple("CriticModule", ["t_value", "t_value_target",
        "t_advantage", "params", "params_target", "t_assign_ops"])
Trainer = namedtuple("Trainer", ["t_loss", "t_grad", "t_train_op"])
InputBundle = namedtuple("InputBundle", ["t_arg", "t_arg_next", "t_feats", "t_feats_next",
        "t_action_mask", "t_reward"])

def increment_sparse_or_dense(into, increment):
    assert isinstance(into, np.ndarray)
    if isinstance(increment, np.ndarray):
        into += increment
    elif isinstance(increment, tf.python.framework.ops.IndexedSlicesValue):
        for i in range(increment.values.shape[0]):
            into[increment.indices[i], :] += increment.values[i, :]
    else:
        assert False

class ModularACModel(object):
    def __init__(self, config):
        self.experiences = []
        self.world = None
        self.task_index = util.Index()

    def prepare(self, world):
        assert self.world is None
        self.world = world
        self.n_actions = world.n_actions + 1
        self.t_n_steps = tf.Variable(1., name="n_steps")
        self.t_inc_steps = self.t_n_steps.assign(self.t_n_steps + 1)
        #optimizer = tf.train.RMSPropOptimizer(0.00001)
        self.optimizer = tf.train.AdamOptimizer()

        def build_actor(index, t_input, t_action_mask, extra_params=()):
            with tf.variable_scope("actor_%s" % index):
                t_action_score, v_action = net.mlp(t_input, (N_HIDDEN, self.n_actions))
                #t_w_action_score = t_action_score / (1000 / self.t_n_steps + 1)
                #t_w_action_score = t_action_score * 0
                t_w_action_score = t_action_score
                t_action_logprobs = tf.nn.log_softmax(t_w_action_score)
                t_chosen_prob = tf.reduce_sum(
                        t_action_mask * t_action_logprobs, reduction_indices=(1,))
            return ActorModule(t_action_logprobs, t_chosen_prob, v_action + extra_params)

        def build_critic(index, t_input, t_input_next, t_reward, extra_params=(),
                extra_params_target=()):
            with tf.variable_scope("critic_%s_target" % index):
                t_value_target, v_value_target = net.mlp(t_input_next, (N_HIDDEN, 1))
                t_value_target = tf.squeeze(t_value_target)
            with tf.variable_scope("critic_%s" % index):
                t_value, v_value = net.mlp(t_input, (N_HIDDEN, 1))
                t_value = tf.squeeze(t_value)
                #t_advantage = t_reward + DISCOUNT * t_value_target - t_value
                t_advantage = None
            t_assign_ops = [v_target.assign(v) for v, v_target 
                    in zip(v_value, v_value_target)]
            return CriticModule(t_value, t_value_target, t_advantage, v_value + extra_params,
                    v_value_target + extra_params_target, t_assign_ops)

        def build_actor_trainer(actor, t_reward, critic, critic_target):
            #if critic_target is None:
            #    t_advantage = t_reward - critic.t_value
            #else:
            #    t_advantage = t_reward + DISCOUNT * critic_target.t_value - critic.t_value
            t_advantage = t_reward - critic.t_value
            actor_loss = -tf.reduce_sum(actor.t_chosen_prob * t_advantage)
            #actor_loss = tf.reduce_sum(0 * actor.t_chosen_prob + 1)
            actor_grad = tf.gradients(actor_loss, actor.params)
            actor_trainer = Trainer(actor_loss, actor_grad, self.optimizer.minimize(actor_loss, 
                    var_list=actor.params))
            return actor_trainer

        def build_critic_trainer(t_reward, critic, critic_target):
            #if critic_target is None:
            #    t_advantage = t_reward - critic.t_value
            #else:
            #    t_advantage = t_reward + DISCOUNT * critic_target.t_value - critic.t_value
            t_advantage = t_reward - critic.t_value
            critic_loss = tf.reduce_sum(tf.square(t_advantage))
            #critic_loss = tf.reduce_sum(0 * t_advantage + 1)
            critic_grad = tf.gradients(critic_loss, critic.params)
            critic_trainer = Trainer(critic_loss, critic_grad, self.optimizer.minimize(critic_loss, 
                    var_list=critic.params))
            return critic_trainer

        # placeholders
        t_arg = tf.placeholder(tf.int32, shape=(None,))
        t_arg_next = tf.placeholder(tf.int32, shape=(None,))
        t_feats = tf.placeholder(tf.float32, shape=(None, world.n_features))
        t_feats_next = tf.placeholder(tf.float32, shape=(None, world.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))

        # input representation
        with tf.variable_scope("embed"):
            t_embed_arg, v_embed_arg = net.embed(t_arg, self.world.cookbook.n_kinds, N_EMBED)
        with tf.variable_scope("embed_next"):
            t_embed_arg_next, v_embed_arg_next = net.embed(t_arg, self.world.cookbook.n_kinds, N_EMBED)
        t_assign_embed_ops = [v_target.assign(v) for v, v_target in zip(v_embed_arg, v_embed_arg_next)]
        t_input = tf.concat(1, (t_embed_arg, t_feats))
        t_input_next = tf.concat(1, (t_embed_arg_next, t_feats_next))

        # modules
        ### actors = {}
        ### scratch_actors = {}
        ### for i_module in range(N_MODULES):
        ###     actor = build_actor(i_module, t_input, t_action_mask)
        ###     scratch_actor = build_actor("%d_scratch" % i_module, t_input, t_action_mask)
        ###     actors[i_module] = actor
        ###     scratch_actors[i_module] = scratch_actor

        actors = {}
        actor_trainers = {}
        critics = {}
        critic_trainers = {}

        for i_module in range(N_MODULES):
            actor = build_actor(i_module, t_input, t_action_mask, extra_params=v_embed_arg)
            actors[i_module] = actor
            for i_task in range(N_TASKS):
                critic = build_critic("%d_%d" % (i_task, i_module), t_input, t_input_next, t_reward,
                        extra_params=v_embed_arg, extra_params_target=v_embed_arg_next)
                critics[i_task, i_module] = critic

        for i_module in range(N_MODULES):
            for i_module_target in list(range(N_MODULES)) + [None]:
                for i_task in range(N_TASKS):
                    critic = critics[i_task, i_module]
                    critic_target = None if i_module_target is None else critics[i_task, i_module_target]
                    critic_trainer = build_critic_trainer(t_reward, critic, critic_target)
                    critic_trainers[i_task, i_module, i_module_target] = critic_trainer

                    actor = actors[i_module]
                    actor_trainer = build_actor_trainer(actor, t_reward, critic, critic_target)
                    actor_trainers[i_task, i_module, i_module_target] = actor_trainer

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.t_assign_embed_ops = t_assign_embed_ops
        self.actors = actors
        self.critics = critics
        self.actor_trainers = actor_trainers
        self.critic_trainers = critic_trainers
        self.inputs = InputBundle(t_arg, t_arg_next, t_feats, t_feats_next, t_action_mask, t_reward)
        ### self.optimizer = trpo.TrustRegionOptimizer(actors, scratch_actors, self.inputs, self.session)

    def init(self, state, hint):
        self.actions, self.args = zip(*hint)
        self.i_task = self.task_index.index(tuple(hint))
        self.i_subtask = 0

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            running_reward = running_reward * DISCOUNT + transition.r
            self.experiences.append(transition._replace(r = running_reward))

    #def experience(self, episode):
    #    self.experiences += episode

    def act(self, state):
        actor = self.actors[self.actions[self.i_subtask]]
        logprobs = self.session.run([actor.t_probs],
                feed_dict={
                    self.inputs.t_arg: [self.args[self.i_subtask]],
                    self.inputs.t_feats: [state.features()]
                })[0][0, :]
        probs = np.exp(logprobs)
        action = np.random.choice(self.n_actions, p=probs)
        if action >= self.world.n_actions:
            self.i_subtask += 1
        terminate = self.i_subtask >= len(self.actions)
        return action, terminate

    def get_state(self):
        if self.i_subtask >= len(self.actions):
            return (None, None, None, None)
        return (self.actions[self.i_subtask], 
                self.args[self.i_subtask], 
                len(self.args) - self.i_subtask,
                self.i_task)

    ### def train(self):
    ###     if len(self.experiences) < N_BATCH:
    ###         return None
    ###     batch = self.experiences[:N_BATCH]
    ###     self.optimizer.update(batch)
    ###     self.experiences = []
    ###     return 0

    def train(self):
        if len(self.experiences) < N_BATCH:
            return None

        #batch_indices = [np.random.choice(len(self.experiences)) for _ in range(N_BATCH)]
        #batch = [self.experiences[i] for i in batch_indices]
        #self.experiences = self.experiences[-MAX_EXPERIENCES:]
        batch = self.experiences[:N_BATCH]

        by_mod = defaultdict(list)
        for e in batch:
            by_mod[e.m1[3], e.m1[0], e.m2[0]].append(e)
            #by_mod[0, e.m1[0], e.m1[0]].append(e)

        #for k in by_mod:
        #    print k, len(by_mod[k])

        grads = {}
        params = {}
        for module in self.actors.values() + self.critics.values():
            for param in module.params:
                if param.name not in grads:
                    grads[param.name] = np.zeros(param.get_shape(), np.float32)
                    params[param.name] = param

        total_actor_err = 0
        total_critic_err = 0
        for i_task, i_mod1, i_mod2 in by_mod:
            actor = self.actors[i_mod1]
            critic = self.critics[i_task, i_mod1]
            actor_trainer = self.actor_trainers[i_task, i_mod1, i_mod2]
            critic_trainer = self.critic_trainers[i_task, i_mod1, i_mod2]

            exps = by_mod[i_task, i_mod1, i_mod2]
            s1, m1, a, s2, m2, r = zip(*exps)
            feats1 = [s.features() for s in s1]
            args1 = [m[1] for m in m1]
            feats2 = [s.features() for s in s2]
            args2 = [m[1] or 0 for m in m2]
            a_mask = np.zeros((len(exps), self.n_actions))
            for i_datum, aa in enumerate(a):
                a_mask[i_datum, aa] = 1

            feed_dict = {
                self.inputs.t_arg: args1,
                self.inputs.t_arg_next: args2,
                self.inputs.t_feats: feats1,
                self.inputs.t_feats_next: feats2,
                self.inputs.t_action_mask: a_mask,
                self.inputs.t_reward: r
            }

            actor_grad, actor_err = self.session.run([actor_trainer.t_grad, actor_trainer.t_loss],
                    feed_dict=feed_dict)
            critic_grad, critic_err = self.session.run([critic_trainer.t_grad, critic_trainer.t_loss],
                    feed_dict=feed_dict)

            total_actor_err += actor_err
            total_critic_err += critic_err

            for param, grad in zip(actor.params, actor_grad):
                increment_sparse_or_dense(grads[param.name], grad)
            for param, grad in zip(critic.params, critic_grad):
                increment_sparse_or_dense(grads[param.name], grad)

        updates = []
        for k in params:
            param = params[k]
            grad = grads[k]
            grad /= N_BATCH
            #if (grad ** 2).sum() > 0:
                #print k, (grad ** 2).sum()
            updates.append((tf.constant(grad), param))
        update_op = self.optimizer.apply_gradients(updates)
        self.session.run(update_op)

        self.experiences = []
        self.session.run(self.t_inc_steps)

        return np.asarray([total_actor_err, total_critic_err]) / N_BATCH

    def roll(self):
        pass
        ### for critic in self.critics.values():
        ###     self.session.run(critic.t_assign_ops)
        ### self.session.run(self.t_assign_embed_ops)
