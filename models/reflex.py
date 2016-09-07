import net

import numpy as np
import tensorflow as tf

N_BATCH = 100

N_HIDDEN = 256
N_EMBED = 64

DISCOUNT = 0.9
EPS = 0.1
TEMP = 1000
MAX_EXPERIENCES = 50000

class ReflexModel(object):
    def __init__(self, config):
        self.experiences = []
        self.world = None

    def prepare(self, world):
        assert self.world is None
        self.world = world
        self.n_actions = world.n_actions + 1

        def predictor(scope):
            with tf.variable_scope(scope):
                t_plan = tf.placeholder(tf.int32, shape=[None, 2])
                t_embed_plan, v_emb = net.embed(t_plan, self.world.cookbook.n_kinds, N_EMBED, multi=True)
                t_features = tf.placeholder(tf.float32, shape=[None, world.n_features])
                t_comb = tf.concat(1, (t_embed_plan, t_features))
                t_scores, v_weights = net.mlp(t_comb, [N_HIDDEN, self.n_actions])
            return t_features, t_plan, t_scores, v_weights + v_emb

        t_features, t_plan, t_scores, v_weights = predictor("now")
        t_features_next, t_plan_next, t_scores_next, v_weights_next = predictor("next")
        t_rewards = tf.placeholder(tf.float32, shape=[None])
        t_actions = tf.placeholder(tf.float32, shape=[None, self.n_actions])
        t_chosen_scores = tf.reduce_sum(t_scores * t_actions, reduction_indices=(1,))
        t_max_scores_next = tf.reduce_max(t_scores_next, reduction_indices=(1,))
        t_td = t_rewards + DISCOUNT * t_max_scores_next - t_chosen_scores
        t_err = tf.reduce_mean(tf.minimum(tf.square(t_td), 1))
        opt = tf.train.AdamOptimizer()
        t_train_op = opt.minimize(t_err, var_list=v_weights)
        t_assign_ops = [wn.assign(w) for (w, wn) in zip(v_weights, v_weights_next)]

        self.session = tf.Session()

        self.session.run(tf.initialize_all_variables())

        self.t_features = t_features
        self.t_plan = t_plan
        self.t_actions = t_actions
        self.t_features_next = t_features_next
        self.t_plan_next = t_plan_next
        self.t_rewards = t_rewards
        self.t_scores = t_scores
        self.t_err = t_err
        self.t_train_op = t_train_op
        self.t_assign_ops = t_assign_ops

        self.step_count = 0

    def init(self, state, hint):
        self.hint = hint
        self.hint_index = 0

    def experience(self, transitions):
        self.experiences += transitions
        self.experiences = self.experiences[-MAX_EXPERIENCES:]

    def act(self, state):
        #if np.random.random() < EPS * np.exp(-self.step_count / TEMP):
        eps = max(1. - self.step_count / 100000., 0)
        #eps = EPS
        #if self.step_count > 10000 == 0:
        #    eps /= 2
        if np.random.random() < eps:
            action = np.random.randint(self.n_actions)
        else:
            preds = self.session.run([self.t_scores], 
                    feed_dict={
                        self.t_features: [state.features()],
                        self.t_plan: [self.get_state()]
                    })[0][0, :]
            action = np.argmax(preds)

        if action >= self.world.n_actions:
            self.hint_index += 1
        terminate = self.hint_index >= len(self.hint)

        return action, terminate

    def get_state(self):
        if self.hint_index >= len(self.hint):
            return (0, 0)
        return self.hint[self.hint_index]

    def train(self):
        if len(self.experiences) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.experiences))
                for _ in range(N_BATCH)]
        batch_exp = [self.experiences[i] for i in batch_indices]
        s1, m1, a, s2, m2, r = zip(*batch_exp)
        feats1 = [s.features() for s in s1]
        feats2 = [s.features() for s in s2]
        a_mask = np.zeros((len(batch_indices), self.n_actions))
        for i, act in enumerate(a):
            a_mask[i, act] = 1

        feed_dict = {
            self.t_features: feats1,
            self.t_plan: m1,
            self.t_actions: a_mask,
            self.t_features_next: feats2,
            self.t_plan_next: m2,
            self.t_rewards: r
        }

        _, err = self.session.run([self.t_train_op, self.t_err], feed_dict=feed_dict)

        self.step_count += 1

        return err

    def roll(self):
        self.session.run(self.t_assign_ops)
