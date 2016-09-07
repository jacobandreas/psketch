import net

import numpy as np
import tensorflow as tf

N_BATCH = 20

N_HIDDEN = 256

DISCOUNT = 0.9
EPS = 0.1
MAX_REPLAY_LEN = 5
MAX_EXPERIENCES = 10000

class AttentiveModel(object):
    def __init__(self, config):
        self.experiences = []

    def prepare(self, world):
        self.world = world
        self.initializer = tf.nn.rnn_cell.LSTMStateTuple(
                np.zeros((1, N_HIDDEN), dtype=np.float32), 
                np.zeros((1, N_HIDDEN), dtype=np.float32))
        self.batch_initializer = tf.nn.rnn_cell.LSTMStateTuple(
                np.zeros((N_BATCH, N_HIDDEN), dtype=np.float32), 
                np.zeros((N_BATCH, N_HIDDEN), dtype=np.float32))

        def predictor(scope):
            with tf.variable_scope(scope) as vs:
                t_episode_features = tf.placeholder(tf.float32,
                            shape=(N_BATCH, MAX_REPLAY_LEN, world.n_features))
                t_episode_lengths = tf.placeholder(tf.int32, shape=(N_BATCH,))
                t_state_features = tf.placeholder(tf.float32,
                            shape=(1, world.n_features))
                t_rnn_state = (
                        tf.placeholder(tf.float32, shape=(1, N_HIDDEN)),
                        tf.placeholder(tf.float32, shape=(1, N_HIDDEN)))
                t_episode_state = tf.nn.rnn_cell.LSTMStateTuple(
                        tf.placeholder(tf.float32, shape=(N_BATCH, N_HIDDEN)),
                        tf.placeholder(tf.float32, shape=(N_BATCH, N_HIDDEN)))

                cell = tf.nn.rnn_cell.LSTMCell(N_HIDDEN, state_is_tuple=True)
                proj_cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, world.n_actions)

                # TODO input projection
                t_episode_scores, _ = tf.nn.dynamic_rnn(proj_cell,
                        t_episode_features, t_episode_lengths,
                        initial_state=t_episode_state,
                        dtype=tf.float32, scope=vs)
                vs.reuse_variables()
                t_scores, t_next_rnn_state = \
                        proj_cell(t_state_features, t_rnn_state)

                v_weights = tf.get_collection(tf.GraphKeys.VARIABLES,
                        scope=vs.name)


                #feats_rs = tf.reshape(t_episode_features, (N_BATCH * MAX_REPLAY_LEN, world.n_features))
                #t_episode_scores_rs, v1 = net.mlp(feats_rs, [N_HIDDEN, world.n_actions])
                #t_episode_scores = tf.reshape(t_episode_scores_rs, (N_BATCH,
                #    MAX_REPLAY_LEN, world.n_actions))
                #vs.reuse_variables()
                #t_scores, v2 = net.mlp(t_state_features, [N_HIDDEN, world.n_actions])
                #t_next_rnn_state = cell.zero_state(1, tf.float32)
                #assert v1 == v2
                #v_weights = v1

            return t_episode_features, t_episode_state, t_episode_lengths, t_episode_scores, \
                t_state_features, t_rnn_state, t_scores, t_next_rnn_state, \
                v_weights

        #t_features, t_scores, v_weights = predictor("now")
        #t_features_next, t_scores_next, v_weights_next = predictor("next")
        t_ep_features, t_ep_state, t_ep_lengths, t_ep_scores, t_features, t_rnn_state, \
                t_scores, t_next_rnn_state, v_weights = predictor("now")
        t_ep_features_next, t_ep_state_next, t_ep_lengths_next, t_ep_scores_next, _, _, _, _, \
                v_weights_next = predictor("next")

        t_rewards = tf.placeholder(tf.float32, shape=(N_BATCH, MAX_REPLAY_LEN))
        t_actions = tf.placeholder(tf.float32, shape=(N_BATCH, MAX_REPLAY_LEN, world.n_actions))
        t_chosen_scores = tf.reduce_sum(t_ep_scores * t_actions, reduction_indices=(2,))
        t_max_scores_next = tf.reduce_max(t_ep_scores_next, reduction_indices=(2,))
        t_td = t_rewards + DISCOUNT * t_max_scores_next - t_chosen_scores
        t_err = tf.reduce_mean(tf.square(t_td))
        opt = tf.train.AdamOptimizer()
        t_train_op = opt.minimize(t_err, var_list=v_weights)
        t_assign_ops = [wn.assign(w) for (w, wn) in zip(v_weights, v_weights_next)]

        self.session = tf.Session()

        self.session.run(tf.initialize_all_variables())

        self.t_ep_features = t_ep_features
        self.t_ep_lengths = t_ep_lengths
        self.t_features = t_features
        self.t_ep_state = t_ep_state
        self.t_rnn_state = t_rnn_state
        self.t_scores = t_scores
        self.t_next_rnn_state = t_next_rnn_state
        self.t_ep_features_next = t_ep_features_next
        self.t_ep_state_next = t_ep_state_next
        self.t_ep_lengths_next = t_ep_lengths_next

        self.t_actions = t_actions
        self.t_rewards = t_rewards
        self.t_err = t_err
        self.t_train_op = t_train_op
        self.t_assign_ops = t_assign_ops

    def init(self, state, hint):
        self.rnn_state = self.initializer

    def experience(self, episode):
        self.experiences.append(episode)
        self.experiences = self.experiences[-MAX_EXPERIENCES:]

    def act(self, state):
        if np.random.random() < EPS:
            return np.random.randint(self.world.n_actions)
        preds, self.rnn_state = self.session.run(
                [self.t_scores, self.t_next_rnn_state],
                feed_dict={
                    self.t_features: [state.features()],
                    self.t_rnn_state: self.rnn_state
                })
        preds = preds[0, :]
        return np.argmax(preds)

    def get_state(self):
        return (self.rnn_state[0][0, :], self.rnn_state[1][0, :])

    def train(self):
        if len(self.experiences) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.experiences))
                for _ in range(N_BATCH)]
        batch_episodes = [self.experiences[i] for i in batch_indices]
        batch_offsets = [np.random.randint(len(e)) for e in batch_episodes]
        sliced_episodes = [e[o:o+MAX_REPLAY_LEN] 
                for e, o in zip(batch_episodes, batch_offsets)]
        s1 = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_features))
        m1a = np.zeros((N_BATCH, N_HIDDEN))
        m1b = np.zeros((N_BATCH, N_HIDDEN))
        a = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_actions))
        s2 = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_features))
        m2a = np.zeros((N_BATCH, N_HIDDEN))
        m2b = np.zeros((N_BATCH, N_HIDDEN))
        r = np.zeros((N_BATCH, MAX_REPLAY_LEN))
        l = np.zeros(N_BATCH)

        for i_episode, episode in enumerate(sliced_episodes):
            s1[i_episode, :len(episode), :] = [t.s1.features() for t in episode]
            m1a[i_episode, :] = episode[0].m1[0]
            m1b[i_episode, :] = episode[0].m1[1]
            actions = [t.a for t in episode]
            a[i_episode, :len(episode), actions] = 1
            s2[i_episode, :len(episode), :] = [t.s2.features() for t in episode]
            m2a[i_episode, :] = episode[0].m2[0]
            m2b[i_episode, :] = episode[0].m2[1]
            r[i_episode, :len(episode)] = [t.r for t in episode]
            l[i_episode] = len(episode)

        feed_dict = {
            self.t_ep_features: s1,
            self.t_ep_state: self.batch_initializer, #tf.nn.rnn_cell.LSTMStateTuple(m1a, m1b),
            self.t_actions: a,
            self.t_ep_features_next: s2,
            self.t_ep_state_next: self.batch_initializer, #tf.nn.rnn_cell.LSTMStateTuple(m2a, m2b),
            self.t_rewards: r,
            self.t_ep_lengths: l,
            self.t_ep_lengths_next: l
        }

        _, err = self.session.run([self.t_train_op, self.t_err], feed_dict=feed_dict)

        return err

    def roll(self):
        self.session.run(self.t_assign_ops)
