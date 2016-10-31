import numpy as np
import tensorflow as tf

import net

DISCOUNT = 0.9
BATCH_SIZE = 1000

CHOICES = [4, 7, 2, 9, 10, 3, 6, 5, 0]
SOLUTION = [4, 5, 2, 6, 0]

class ReflexMetaModel(object):
    def __init__(self, world, subtask_index):
        self.world = world
        self.subtask_index = subtask_index
        self.random = np.random.RandomState(0)
        self.experiences = []
        self.good_experiences = []
        self.iter = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.t_feats = tf.placeholder(tf.float32, shape=(None,
                world.n_features))
            #self.t_feats = tf.placeholder(tf.float32, shape=(None,
            #    1))
            self.t_scores, self.v = net.mlp(self.t_feats, [len(CHOICES)])
            self.t_reward = tf.placeholder(tf.float32, shape=(None,))
            with tf.variable_scope("target"):
                self.t_scores_target, self.v_target = net.mlp(self.t_feats,
                        [len(CHOICES)])
            self.t_assign_ops = [vv_t.assign(vv) for vv, vv_t in zip(self.v, self.v_target)]
            self.t_action_mask = tf.placeholder(tf.float32, shape=(None,
                    len(CHOICES)))
            #self.t_pred = tf.reduce_sum(self.t_action_mask * self.t_scores,
            #        reduction_indices=(1,))
            #self.t_err = tf.reduce_mean(tf.square(self.t_pred - self.t_reward))
            self.t_score_now = tf.reduce_sum(self.t_action_mask * self.t_scores,
                    reduction_indices=(1,))
            self.t_score_next = tf.reduce_max(self.t_scores_target,
                    reduction_indices=(1,))
            self.t_err = tf.reduce_mean(tf.square(self.t_reward + 0.9 *
                    self.t_score_next - self.t_score_now))
            self.opt = tf.train.RMSPropOptimizer(0.001)
            self.t_train_op = self.opt.minimize(self.t_err, var_list=self.v)
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())

    def act(self, state, init=False):
        #if init:
        #    self.counters = np.zeros(len(state), dtype=int)
        ##print self.counters
        #return [
        #    (None if self.counters[i] >= len(SOLUTION) else SOLUTION[self.counters[i]], 
        #     None) 
        #    for i in range(len(state))]

        choices = CHOICES
        features = []
        for s in state:
            if s is None:
                features.append(np.zeros(self.world.n_features))
                #features.append([0])
            else:
                features.append(s.features())
                #features.append([0])
        feed_dict = {
            self.t_feats: features
        }
        scores = self.session.run([self.t_scores], feed_dict=feed_dict)[0]
        actions = []
        args = []
        for i, s in enumerate(state):
            sc = scores[i, :]
            ch = CHOICES
            if init:
                sc = sc[:-1]
                ch = ch[:-1]

            if s is None:
                action = None
            elif np.random.random() < 0.5:
                action = np.random.choice(ch)
            else:
                action = CHOICES[np.argmax(sc)]
            actions.append(action)
            args.append(None)

        return zip(actions, args)

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            #running_reward = running_reward * DISCOUNT + transition.r
            #n_transition = transition._replace(r=running_reward)
            n_transition = transition
            if n_transition.a >= self.world.n_actions:
                self.experiences.append(n_transition)
                if n_transition.r > 0:
                    self.good_experiences.append(n_transition)

    def train(self):
        self.iter += 1
        total_err = 0
        if len(self.experiences) < BATCH_SIZE:
            return None
        for i in range(100):
            #self.experiences = self.experiences[-BATCH_SIZE:]
            self.good_experiences = self.good_experiences[-BATCH_SIZE/10:]
            self.experiences = self.good_experiences + \
                    self.experiences[-(BATCH_SIZE-len(self.good_experiences)):]

            action_mask = np.zeros((BATCH_SIZE, len(CHOICES)))
            for i, e in enumerate(self.experiences):
                action_mask[i, CHOICES.index(e.m2.action)] = 1

            feed_dict = {
                self.t_feats: [e.s1.features() for e in self.experiences],
                #self.t_feats: [[0] for e in self.experiences],
                self.t_action_mask: action_mask,
                self.t_reward: [e.r for e in self.experiences]
            }

            err, _ = self.session.run([self.t_err, self.t_train_op], feed_dict=feed_dict)
            total_err += err

        self.session.run(self.t_assign_ops)

        print len(self.good_experiences)
        return total_err
