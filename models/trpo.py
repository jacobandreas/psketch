from collections import defaultdict
import numpy as np
import tensorflow as tf

DELTA = 0.01

class TrustRegionOptimizer(object):
    def __init__(self, actors, scratch_actors, inputs, session):
        self.actors = actors
        self.scratch_actors = scratch_actors
        self.inputs = inputs
        self.session = session
        assert actors.keys() == scratch_actors.keys()

        self.n_actions = actors.values()[0].t_probs.get_shape()[1]

        self.loss_grads = {}
        self.kls = {}
        for k in self.actors:
            actor = self.actors[k]
            scratch_actor = self.scratch_actors[k]
            t_loss = -tf.reduce_sum(tf.exp(scratch_actor.t_chosen_prob - actor.t_chosen_prob) *
                    self.inputs.t_reward)
            #t_kl = tf.reduce_sum(actor.t_probs * tf.log(actor.t_probs - scratch_actor.t_probs),
            #        reduction_indices=(1,))
            t_kl = tf.reduce_sum(tf.exp(actor.t_probs) * (actor.t_probs - scratch_actor.t_probs))
            self.loss_grads[k] = tf.gradients(t_loss, scratch_actor.params)
            self.kls[k] = t_kl

    def update(self, data):
        grouped = defaultdict(list)
        for datum in data:
            grouped[datum.m1[0]].append(datum)

        def compute_search_direction():
            search_direction = {}
            for i_actor in grouped:
                m_data = grouped[i_actor]
                scratch_actor = self.scratch_actors[i_actor]
                states, mstates, actions, _, _, rewards = zip(*m_data)
                feats = [state.features() for state in states]
                args = [mstate[1] for mstate in mstates]
                action_mask = np.zeros((len(m_data), self.n_actions))
                for i, action in enumerate(actions):
                    action_mask[i, action] = 1
                feed_dict = {
                    self.inputs.t_arg: args,
                    self.inputs.t_feats: feats,
                    self.inputs.t_action_mask: action_mask,
                    self.inputs.t_reward: rewards
                }
                grads = self.session.run(self.loss_grads[i_actor], feed_dict=feed_dict)
                for param, grad in zip(scratch_actor.params, grads):
                    if param.name not in search_direction:
                        search_direction[param.name] = np.zeros(param.get_shape())
                    search_direction[param.name] += grad
            return search_direction

        def do_line_search(search_direction):
            step_size = 1.
            while True:
                total_kl = 0
                for i_actor in grouped:
                    m_data = grouped[i_actor]
                    actor = self.actors[i_actor]
                    scratch_actor = self.scratch_actors[i_actor]
                    states, mstates = zip(*m_data)[:2]
                    feats = [state.features() for state in states]
                    args = [mstate[1] for mstate in mstates]
                    feed_dict = {
                        self.inputs.t_arg: args,
                        self.inputs.t_feats: feats
                    }

                    param_values = self.session.run(actor.params)
                    for param, pv, scratch_param in zip(actor.params, param_values, scratch_actor.params):
                        assert scratch_param.name.replace("_scratch", "") == param.name
                        feed_dict[scratch_param.name] = pv - search_direction[scratch_param.name] * step_size
                    kl = self.session.run(self.kls[i_actor], feed_dict=feed_dict)
                    total_kl += kl

                if total_kl < DELTA:
                    return step_size
                step_size /= 2

        search_direction = compute_search_direction()
        step_size = do_line_search(search_direction)

        for i_actor in grouped:
            actor = self.actors[i_actor]
            scratch_actor = self.scratch_actors[i_actor]
            param_values = self.session.run(actor.params)
            for param, pv, scratch_param in zip(actor.params, param_values, scratch_actor.params):
                self.session.run(param.assign(scratch_param), 
                        feed_dict={
                            scratch_param: pv - search_direction[scratch_param.name] * step_size
                        })
