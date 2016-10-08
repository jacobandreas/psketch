from misc import util
import net
import trpo

from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

N_UPDATE = 10000
N_BATCH = 5000
#N_UPDATE = 1000
#N_BATCH = 1000

#N_UPDATE = 100
#N_BATCH = 100

N_HIDDEN = 128
N_EMBED = 64

#N_MODULES = 6
#N_TASKS = 10
#N_MODULES = 6
N_MODULES = 8
N_TASKS = 9

DISCOUNT = 0.9
EPS = 0.1
TEMP = 1000
MAX_EXPERIENCES = 10000

ActorModule = namedtuple("ActorModule", ["t_probs", "t_chosen_prob", "params"])
CriticModule = namedtuple("CriticModule", ["t_value", "t_value_target",
        "t_advantage", "params", "params_target", "t_assign_ops"])
Trainer = namedtuple("Trainer", ["t_loss", "t_grad", "t_train_op", "foo"])
InputBundle = namedtuple("InputBundle", ["t_arg", "t_arg_next", "t_step", "t_step_next", 
        "t_feats", "t_feats_next", "t_action_mask", "t_reward"])

ModelState = namedtuple("ModelState", ["action", "arg", "remaining", "task", "step"])

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
        tf.set_random_seed(0)
        self.next_actor_seed = 0

    def prepare(self, world):
        assert self.world is None
        self.world = world
        #self.n_actions = world.n_actions + 1
        self.n_actions = world.n_actions
        self.t_n_steps = tf.Variable(1., name="n_steps")
        self.t_inc_steps = self.t_n_steps.assign(self.t_n_steps + 1)
        self.optimizer = tf.train.RMSPropOptimizer(0.001)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        def build_actor(index, t_input, t_action_mask, extra_params=[]):
            with tf.variable_scope("actor_%s" % index):
                t_action_score, v_action = net.mlp(t_input, (N_HIDDEN, self.n_actions))
                #t_action_score, v_action = net.mlp(t_input, (self.n_actions,))
                #t_action_score, v_action = net.mlp(t_input, (self.n_actions,))
                #t_w_action_score = t_action_score / (1000 / self.t_n_steps + 1)
                #t_w_action_score = t_action_score * 0.1
                t_w_action_score = t_action_score
                t_action_logprobs = tf.nn.log_softmax(t_w_action_score)
                t_chosen_prob = tf.reduce_sum(
                        t_action_mask * t_action_logprobs, reduction_indices=(1,))
            return ActorModule(t_action_logprobs, t_chosen_prob, v_action + extra_params)

        def build_critic(index, t_input, t_input_next, t_reward, extra_params=[],
                extra_params_target=[]):
            with tf.variable_scope("critic_%s_target" % index):
                #t_value_target, v_value_target = net.mlp(t_input_next, (N_HIDDEN, 1))
                t_value_target, v_value_target = net.mlp(t_input_next, (1,))
                t_value_target = tf.squeeze(t_value_target)
            with tf.variable_scope("critic_%s" % index):
                #t_value, v_value = net.mlp(t_input, (N_HIDDEN, 1))

                t_value, v_value = net.mlp(t_input, (1,))
                t_value = tf.squeeze(t_value)

                #t_advantage = t_reward + DISCOUNT * t_value_target - t_value
                t_advantage = None
            t_assign_ops = [v_target.assign(v) for v, v_target 
                    in zip(v_value, v_value_target)]
            return CriticModule(t_value, t_value_target, t_advantage, v_value + extra_params,
                    v_value_target + extra_params_target, t_assign_ops)

        ### def build_actor_trainer(actor, t_reward, critic, critic_target):
        def build_actor_trainer(actor, t_reward, critic):
            #if critic_target is None:
            #    t_advantage = t_reward - critic.t_value
            #else:
            #    t_advantage = t_reward + DISCOUNT * critic_target.t_value - critic.t_value
            t_advantage = t_reward - critic.t_value
            actor_loss = -tf.reduce_sum(actor.t_chosen_prob * t_advantage) + \
                    0.001 * tf.reduce_sum(tf.exp(actor.t_probs) * actor.t_probs)
            #actor_loss = tf.reduce_sum(0 * actor.t_chosen_prob + 1)
            actor_grad = tf.gradients(actor_loss, actor.params)
            actor_trainer = Trainer(actor_loss, actor_grad, self.optimizer.minimize(actor_loss, 
                    var_list=actor.params), None)
            return actor_trainer

        ### def build_critic_trainer(t_reward, critic, critic_target):
        ###     #if critic_target is None:
        ###     #    t_advantage = t_reward - critic.t_value
        ###     #else:
        ###     #    t_advantage = t_reward + DISCOUNT * critic_target.t_value - critic.t_value
        ###     t_advantage = t_reward - critic.t_value
        ###     critic_loss = tf.reduce_sum(tf.square(t_advantage))
        ###     #critic_loss = tf.reduce_sum(0 * t_advantage + 1)
        ###     critic_grad = tf.gradients(critic_loss, critic.params)
        ###     critic_trainer = Trainer(critic_loss, critic_grad, self.optimizer.minimize(critic_loss, 
        ###             var_list=critic.params), None)
        ###     return critic_trainer

        def build_critic_trainer(t_reward, critic):
            t_advantage = t_reward - critic.t_value
            critic_loss = tf.reduce_sum(tf.square(t_advantage))
            critic_grad = tf.gradients(critic_loss, critic.params)
            critic_trainer = Trainer(critic_loss, critic_grad,
                    self.optimizer.minimize(critic_loss,
                        var_list=critic.params), None)
            return critic_trainer

        # placeholders
        t_arg = tf.placeholder(tf.int32, shape=(None,))
        t_arg_next = tf.placeholder(tf.int32, shape=(None,))
        t_step = tf.placeholder(tf.float32, shape=(None, 1))
        t_step_next = tf.placeholder(tf.float32, shape=(None, 1))
        t_feats = tf.placeholder(tf.float32, shape=(None, world.n_features))
        t_feats_next = tf.placeholder(tf.float32, shape=(None, world.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))

        # input representation
        #with tf.variable_scope("embed"):
        #    t_embed_arg, v_embed_arg = net.embed(t_arg, self.world.cookbook.n_kinds, N_EMBED)
        #with tf.variable_scope("embed_next"):
        #    t_embed_arg_next, v_embed_arg_next = net.embed(t_arg, self.world.cookbook.n_kinds, N_EMBED)
        #t_assign_embed_ops = [v_target.assign(v) for v, v_target in zip(v_embed_arg, v_embed_arg_next)]
        #t_input = tf.concat(1, (t_embed_arg, t_feats, t_step))
        #t_input_next = tf.concat(1, (t_embed_arg_next, t_feats_next, t_step_next))
        t_input = t_feats
        t_input_next = t_feats_next
        extra_params = extra_params_target = []

        #with tf.variable_scope("input"):
        #    t_input, v_input = net.mlp(t_feats, [N_HIDDEN])
        #    t_input = tf.nn.relu(t_input)
        #with tf.variable_scope("input_next"):
        #    t_input_next, v_input_next = net.mlp(t_feats_next, [N_HIDDEN])
        #    t_input_next = tf.nn.relu(t_input_next)
        #extra_params = v_input
        #extra_params_target = v_input_next

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
            #actor = build_actor(i_module, t_input, t_action_mask, extra_params=v_embed_arg)
            actor = build_actor(i_module, t_input, t_action_mask, extra_params=extra_params)
            actors[i_module] = actor
            #for i_task in range(N_TASKS):
            #    critic = build_critic("%d_%d" % (i_task, i_module), t_input, t_input_next, t_reward,
            #            extra_params=v_embed_arg, extra_params_target=v_embed_arg_next)
            #    critics[i_task, i_module] = critic
        for i_task in range(N_TASKS):
            critic = build_critic("%d" % i_task, t_input, t_input_next,
                    t_reward, extra_params=extra_params, extra_params_target=extra_params_target)
                    #extra_params=v_embed_arg, extra_params_target=v_embed_arg_next)
            for i_module in range(N_MODULES):
                critics[i_task, i_module] = critic

        ### for i_module in range(N_MODULES):
        ###     for i_module_target in list(range(N_MODULES)) + [None]:
        ###     #for i_module_target in [None]:
        ###         for i_task in range(N_TASKS):
        ###             critic = critics[i_task, i_module]
        ###             critic_target = None if i_module_target is None else critics[i_task, i_module_target]
        ###             critic_trainer = build_critic_trainer(t_reward, critic, critic_target)
        ###             critic_trainers[i_task, i_module, i_module_target] = critic_trainer

        ###             actor = actors[i_module]
        ###             actor_trainer = build_actor_trainer(actor, t_reward, critic, critic_target)
        ###             actor_trainers[i_task, i_module, i_module_target] = actor_trainer
        for i_module in range(N_MODULES):
            for i_task in range(N_TASKS):
                critic = critics[i_task, i_module]
                critic_trainer = build_critic_trainer(t_reward, critic)
                critic_trainers[i_task, i_module] = critic_trainer

                actor = actors[i_module]
                actor_trainer = build_actor_trainer(actor, t_reward, critic)
                actor_trainers[i_task, i_module] = actor_trainer

        self.t_gradient_placeholders = {}
        self.t_update_gradient_op = None

        params = []
        for module in actors.values() + critics.values():
            params += module.params
        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        #self.t_assign_embed_ops = t_assign_embed_ops
        self.actors = actors
        self.critics = critics
        self.actor_trainers = actor_trainers
        self.critic_trainers = critic_trainers
        self.inputs = InputBundle(t_arg, t_arg_next, t_step, t_step_next, 
                t_feats, t_feats_next, t_action_mask, t_reward)
        ### self.optimizer = trpo.TrustRegionOptimizer(actors, scratch_actors, self.inputs, self.session)

    def init(self, states, hints):
        n_act_batch = len(states)
        #self.actions, self.args = zip(*hint)
        self.actions = []
        self.args = []
        self.i_task = []
        for i in range(n_act_batch):
            actions, args = zip(*hints[i])
            self.actions.append(actions)
            self.args.append(args)
            self.i_task.append(self.task_index.index(tuple(hints[i])))
        self.i_subtask = [0 for _ in range(n_act_batch)]
        self.i_step = np.zeros((n_act_batch, 1))
        self.randoms = []
        for _ in range(n_act_batch):
            self.randoms.append(np.random.RandomState(self.next_actor_seed))
            self.next_actor_seed += 1

    def save(self):
        self.saver.save(self.session, "modular_ac.chk")

    def load(self):
        self.saver.restore(self.session, "modular_ac.chk")
        #for var in tf.get_collection(tf.GraphKeys.VARIABLES):
        #    print var.name
        #exit()

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            running_reward = running_reward * DISCOUNT + transition.r
            #if transition.a < self.world.n_actions:
            #    running_reward = running_reward * DISCOUNT + transition.r
            #elif transition.a < self.n_actions:
            #    running_reward = running_reward * DISCOUNT - 10
            n_transition = transition._replace(r=running_reward)
            if n_transition.a < self.n_actions:
                self.experiences.append(n_transition)

    #def experience(self, episode):
    #    self.experiences += episode

    #@profile
    def act(self, states):
        self.i_step += 0.1
        by_mod = defaultdict(list)
        n_act_batch = len(self.i_subtask)

        for i in range(n_act_batch):
            by_mod[self.i_task[i], self.i_subtask[i]].append(i)

        action = [None] * n_act_batch
        terminate = [None] * n_act_batch

        for k, indices in by_mod.items():
            i_task, i_subtask = k
            assert len(set(self.actions[i] for i in indices)) == 1
            if i_subtask >= len(self.actions[indices[0]]):
                continue
            actor = self.actors[self.actions[indices[0]][i_subtask]]
            feed_dict = {
                #self.inputs.t_arg: [self.args[i][i_subtask] for i in indices],
                self.inputs.t_feats: [states[i].features() for i in indices],
                #self.inputs.t_step: [self.i_step[i] for i in indices]
            }
            logprobs = self.session.run([actor.t_probs], feed_dict=feed_dict)[0]
            probs = np.exp(logprobs)
            for pr, i in zip(probs, indices):

                #a = self.randoms[i].choice(self.n_actions, p=pr)
                if self.i_step[i] >= 1.4999:
                    a = self.world.n_actions
                else:
                    a = self.randoms[i].choice(self.n_actions, p=pr)

                if a == self.world.n_actions:
                    self.i_subtask[i] += 1
                    self.i_step[i] = 0.
                t = self.i_subtask[i] >= len(self.actions[indices[0]])
                action[i] = a
                terminate[i] = t

        #assert None not in action
        #assert None not in terminate
        return action, terminate

        #actor = self.actors[self.actions[self.i_subtask]]
        #feed_dict = {
        #    self.inputs.t_arg: [self.args[self.i_subtask]],
        #    self.inputs.t_feats: [state.features()],
        #    self.inputs.t_step: self.i_step
        #}
        #logprobs = self.session.run([actor.t_probs], feed_dict=feed_dict)[0][0, :]
        #probs = np.exp(logprobs)
        #action = np.random.choice(self.n_actions, p=probs)
        #if action >= self.world.n_actions:
        #    self.i_subtask += 1
        #    self.i_step = np.asarray([[0.]])
        #terminate = self.i_subtask >= len(self.actions)
        #return action, terminate

    def get_state(self):
        out = []
        for i in range(len(self.i_subtask)):
            if self.i_subtask[i] >= len(self.actions[i]):
                out.append(ModelState(None, None, None, None, [0.]))
            else:
                out.append(ModelState(
                        self.actions[i][self.i_subtask[i]], 
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
        #batch_indices = [np.random.choice(len(self.experiences)) for _ in range(N_BATCH)]
        #batch = [self.experiences[i] for i in batch_indices]
        #self.experiences = self.experiences[-MAX_EXPERIENCES:]
        batch = experiences[:N_UPDATE]

        #print hash(tuple(t.a for t in batch))
        #print hash(tuple(t.m1.arg for t in batch))
        #print hash(tuple(t.m2.arg for t in batch))
        ##print [t.m1.task for t in batch]
        #print hash(tuple(tuple(t.m1.step) for t in batch))
        #print hash(tuple(tuple(t.m2.step) for t in batch))
        #print hash(tuple(t.r for t in batch))
        #exit()

        #print len(batch)
        #_, _, a, _, _, r = zip(*batch)
        #print np.mean(r)
        #print np.mean(np.asarray(r) ** 2)
        #counter = defaultdict(lambda: 0)
        #for aa in a:
        #    counter[aa] += 1
        #print counter
        #print

        ### !!!
        #self.experiences = []
        #return 0

        by_mod = defaultdict(list)
        for e in batch:
            ### by_mod[e.m1.task, e.m1.action, e.m2.action].append(e)
            by_mod[e.m1.task, e.m1.action].append(e)

            #by_mod[0, e.m1[0], e.m1[0]].append(e)

        #print {k: len(v) for k, v in by_mod.items()}

        #for k in by_mod:
        #    print k, len(by_mod[k])

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
        ### for i_task, i_mod1, i_mod2 in by_mod:
        for i_task, i_mod1 in by_mod:
            actor = self.actors[i_mod1]
            critic = self.critics[i_task, i_mod1]
            actor_trainer = self.actor_trainers[i_task, i_mod1]
            critic_trainer = self.critic_trainers[i_task, i_mod1]

            all_exps = by_mod[i_task, i_mod1]
            for i_batch in range(int(np.ceil(1. * len(all_exps) / N_BATCH))):
                exps = all_exps[i_batch * N_BATCH : (i_batch + 1) * N_BATCH]
                s1, m1, a, s2, m2, r = zip(*exps)
                feats1 = [s.features() for s in s1]
                args1 = [m.arg for m in m1]
                #steps1 = [[m.step / 10.] for m in m1]
                steps1 = [m.step for m in m1]
                feats2 = [s.features() for s in s2]
                args2 = [m.arg or 0 for m in m2]
                #steps2 = [[m.step / 10.] for m in m2]
                steps2 = [m.step for m in m2]
                a_mask = np.zeros((len(exps), self.n_actions))
                for i_datum, aa in enumerate(a):
                    a_mask[i_datum, aa] = 1

                feed_dict = {
                    #self.inputs.t_arg: args1,
                    #self.inputs.t_arg_next: args2,
                    #self.inputs.t_step: steps1,
                    #self.inputs.t_step_next: steps2,
                    self.inputs.t_feats: feats1,
                    self.inputs.t_feats_next: feats2,
                    self.inputs.t_action_mask: a_mask,
                    self.inputs.t_reward: r
                }

                actor_grad, actor_err = self.session.run([actor_trainer.t_grad, actor_trainer.t_loss],
                        feed_dict=feed_dict)
                critic_grad, critic_err = self.session.run([critic_trainer.t_grad, critic_trainer.t_loss], 
                        feed_dict=feed_dict)

                #print args1
                #print steps1
                #print args2
                #print steps2
                #print hash(tuple(a))
                #print hash(tuple(r))

                #print i_batch, i_task, i_mod1, i_mod2
                #print actor_err
                #print critic_err
                #print len(exps)
                #print len(all_exps)
                #exit()

                #print actor_err
                #print critic_err
                #exit()

                #print i_task, \
                #        i_mod1, \
                #        sum([ag.sum() for ag in actor_grad]) / len(exps)
                #        #actor_err / len(exps), \
                #        #critic_err / len(exps), \
                #        #sum(r) / len(exps)

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

        #print total_actor_err
        #print total_critic_err
        #exit()

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
            #if (grad ** 2).sum() > 0:
                #print k, (grad ** 2).sum()
            if k not in self.t_gradient_placeholders:
                self.t_gradient_placeholders[k] = tf.placeholder(tf.float32, grad.shape)
            feed_dict[self.t_gradient_placeholders[k]] = grad
            updates.append((self.t_gradient_placeholders[k], param))
        if self.t_update_gradient_op is None:
            self.t_update_gradient_op = self.optimizer.apply_gradients(updates)
        self.session.run(self.t_update_gradient_op, feed_dict=feed_dict)

        self.experiences = []
        self.session.run(self.t_inc_steps)

        return np.asarray([total_actor_err, total_critic_err]) / N_UPDATE

    def roll(self):
        pass
        ### for critic in self.critics.values():
        ###     self.session.run(critic.t_assign_ops)
        ### self.session.run(self.t_assign_embed_ops)
