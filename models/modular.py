from misc import util
import net

from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

N_BATCH = 100

N_HIDDEN = 256
N_EMBED = 64

N_MODULES = 3

DISCOUNT = 0.9
EPS = 0.1
TEMP = 1000
MAX_EXPERIENCES = 50000

Module = namedtuple("Module", ["t_arg", "t_arg_n", "t_feats", "t_feats_n", 
        "t_scores", "t_scores_n", "v"])

class ModularModel(object):
    def __init__(self, config):
        self.experiences = []
        self.good_experiences = defaultdict(list)
        self.world = None

    def prepare(self, world):
        assert self.world is None
        self.world = world
        self.n_actions = world.n_actions + 1

        def predictor(scope):
            with tf.variable_scope(scope):
                t_arg = tf.placeholder(tf.int32, shape=[None])
                t_embed, v_embed = net.embed(t_arg, self.world.cookbook.n_kinds, N_EMBED)
                t_feats = tf.placeholder(tf.float32, shape=[None, world.n_features + 1])
                t_comb = tf.concat(1, (t_embed, t_feats))
                t_scores, v_weights = net.mlp(t_comb, [N_HIDDEN, self.n_actions])
                #t_scores, v_weights = net.mlp(t_comb, [self.n_actions])
            return t_arg, t_feats, t_scores, v_embed + v_weights

        t_rewards = tf.placeholder(tf.float32, shape=[None])
        t_actions = tf.placeholder(tf.float32, shape=[None, self.n_actions])
        t_assign_ops = []
        modules = {}
        for i_module in range(N_MODULES):
            t_arg, t_feats, t_scores, v = predictor("now_%d" % i_module)
            t_arg_n, t_feats_n, t_scores_n, v_n = predictor("next_%d" % i_module)
            varz = v + v_n
            t_assign_ops += [w_n.assign(w) for w, w_n in zip(v, v_n)]
            modules[i_module] = Module(t_arg, t_arg_n, t_feats, t_feats_n,
                    t_scores, t_scores_n, v)


        self.session = tf.Session()

        self.t_rewards = t_rewards
        self.t_actions = t_actions
        self.t_assign_ops = t_assign_ops
        self.modules = modules
        self.n_steps = [0 for _ in range(N_MODULES)]


        def trainer(t_rewards, t_actions, mod, mod_n, opt):
            t_score_chosen = tf.reduce_sum(mod.t_scores * t_actions, reduction_indices=(1,))
            t_score_n_best = tf.reduce_max(mod_n.t_scores_n, reduction_indices=(1,))
            t_td = t_rewards + DISCOUNT * t_score_n_best - t_score_chosen
            t_err = tf.reduce_mean(tf.minimum(tf.square(t_td), 1))
            t_train_op = opt.minimize(t_err, var_list=mod.v)
            return (t_train_op, t_err)

        self.cached_train_ops = {}
        for mod1 in range(N_MODULES):
            for mod2 in range(N_MODULES):
                opt = tf.train.AdamOptimizer()
                self.cached_train_ops[mod1, mod2] = trainer(self.t_rewards,
                        self.t_actions, self.modules[mod1], self.modules[mod2],
                        opt)
        self.session.run(tf.initialize_all_variables())

    def t_train_ops(self, mod1, mod2):
        return self.cached_train_ops[mod1, mod2]

    def init(self, state, hint):
        self.actions, self.args = zip(*hint)
        self.hint_index = 0

    def experience(self, transitions):
        touched = set()
        if sum(t.r for t in transitions) > 0:
            for t in transitions:
                hint = (t.m1[:2], t.m2[:2])
                self.good_experiences[hint].append(t)
                touched.add(hint)
        for t in touched:
            self.good_experiences[t] = self.good_experiences[t][-MAX_EXPERIENCES:]
        self.experiences += transitions
        self.experiences = self.experiences[-MAX_EXPERIENCES:]

    def act(self, state):
        eps = max(1. - self.n_steps[self.actions[self.hint_index]] / 50000., 0.1)
        module = self.modules[self.actions[self.hint_index]]
        if np.random.random() < eps:
            action = np.random.randint(self.n_actions)
        else:
            preds = self.session.run([module.t_scores],
                    feed_dict={
                        module.t_arg: [self.args[self.hint_index]],
                        module.t_feats: [np.concatenate((state.features(),
                            [self.hint_index]))]
                    })[0][0, :]
            action = np.argmax(preds)

        if action >= self.world.n_actions:
            self.hint_index += 1
        terminate = self.hint_index >= len(self.actions)
        return action, terminate

    def get_state(self):
        if self.hint_index >= len(self.actions):
            return (0, 0, 0)
        return (self.actions[self.hint_index], 
                self.args[self.hint_index], 
                len(self.args) - self.hint_index)

    def train(self):
        if len(self.experiences) < N_BATCH:
            return 0

        batch_indices = [np.random.randint(len(self.experiences)) 
                for _ in range(N_BATCH)]
        batch = [self.experiences[i] for i in batch_indices]

        by_mods = defaultdict(list)
        for e in batch:
            by_mods[e.m1[0], e.m2[0]].append(e)

        total_err = 0

        for i_mod0, i_mod1 in by_mods:
            mod0 = self.modules[i_mod0]
            mod1 = self.modules[i_mod1]
            exps = by_mods[i_mod0, i_mod1]
            s1, m1, a, s2, m2, r = zip(*exps)
            feats1 = [s.features() for s in s1]
            feats2 = [s.features() for s in s2]
            args1 = [m[1] for m in m1]
            args2 = [m[1] for m in m2]
            states1 = [[m[2]] for m in m1]
            states2 = [[m[2]] for m in m2]
            a_mask = np.zeros((len(exps), self.n_actions))
            for i, act in enumerate(a):
                a_mask[i, act] = 1

            feed_dict = {
                mod0.t_feats: np.concatenate((feats1, states1), axis=1),
                mod1.t_feats_n: np.concatenate((feats2, states2), axis=1),
                mod0.t_arg: args1,
                mod1.t_arg_n: args2,
                self.t_actions: a_mask,
                self.t_rewards: r
            }

            _, err = self.session.run(self.t_train_ops(i_mod0, i_mod1),
                    feed_dict=feed_dict)
            total_err += err

            self.n_steps[i_mod0] += 1

        return total_err

    def roll(self):
        self.session.run(self.t_assign_ops)
