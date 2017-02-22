import numpy as np
import tensorflow as tf
import logging

from misc.experience import Transition
import net

DISCOUNT = 0.9
#BATCH_SIZE = 1000
#HISTORY_SIZE = 50000
BATCH_SIZE = 2000
HISTORY_SIZE = 50000

CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
SOLUTION = [4, 5, 2, 6, 0]

class ReflexMetaModel(object):
    def __init__(self, world, subtask_index, resource_index):
        self.world = world
        self.subtask_index = subtask_index
        #print self.subtask_index.contents
        #exit()
        self.resource_index = resource_index
        self.random = np.random.RandomState(0)
        self.good_experiences = []
        self.all_experiences = []
        self.experiences = []
        self.iter = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.t_feats = tf.placeholder(tf.float32, shape=(None,
                world.n_features))
                #len(self.world.cookbook.index)))
            #self.t_feats = tf.placeholder(tf.float32, shape=(None,
            #    1))
            #self.t_scores, self.v = net.mlp(self.t_feats, [128, len(CHOICES)])
            self.t_scores, self.v = net.mlp(self.t_feats, [len(CHOICES)])
            self.t_reward = tf.placeholder(tf.float32, shape=(None,))
            self.t_terminal = tf.placeholder(tf.float32, shape=(None,))
            with tf.variable_scope("target"):
                self.t_feats_target = tf.placeholder(tf.float32, shape=(None,
                    world.n_features))
                    #len(self.world.cookbook.index)))
                self.t_scores_target, self.v_target = net.mlp(self.t_feats_target,
                        #[128, len(CHOICES)])
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
            self.t_err = tf.reduce_mean(tf.square(self.t_reward + 0.8 *
                    self.t_terminal * self.t_score_next - self.t_score_now))
            self.opt = tf.train.RMSPropOptimizer(0.001)
            #self.opt = tf.train.AdamOptimizer()
            self.t_train_op = self.opt.minimize(self.t_err, var_list=self.v)
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())

    def act(self, state, init=False):
        #if init:
        #    self.counters = np.ones(len(state), dtype=int)
        #    return [(SOLUTION[0], None) for _ in state]
        ##print self.counters
        #return [
        #    (None if self.counters[i] >= len(SOLUTION) else SOLUTION[self.counters[i]], 
        #     None) 
        #    for i in range(len(state))]

        features = []
        for s in state:
            if s is None:
                features.append(np.zeros(self.world.n_features))
                #features.append(np.zeros(len(self.world.cookbook.index)))
            else:
                features.append(s.features())
                #features.append(np.clip(s.inventory, 0, 1))
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
                an = "none"
            elif np.random.random() < 0.2 * max(1. - self.iter / 2000., 0.5):
                action = np.random.choice(ch)
                an = "rand"
            else:
                action = CHOICES[np.argmax(sc)]
                an = "max"
            actions.append(action)
            args.append(an)

            #if s is not None and np.random.random() < 0.1:
            #    print 1. - self.iter / 50.
            #    print s.inventory[self.resource_index["wood"]]
            #    print sc
            #    print CHOICES[np.argmax(sc)], action
            #    print an
            #    print

        return zip(actions, args)

    def experience(self, episode):
        curr_state = episode[0].s1
        curr_mstate = episode[0].m1
        last_transition = episode[0]
        out = []
        for transition in episode:
            #print transition
            if transition.a >= self.world.n_actions:
                meta_transition = Transition(curr_state, curr_mstate,
                    None, transition.s1, None, 0)
                curr_state = transition.s1
                curr_mstate = transition.m2
                last_transition = transition
                #n_transition = n_transition._replace(s2=transition.s2,
                #        m2=transition.m2, r=transition.r)
                #out.append(n_transition)
                out.append(meta_transition)
                if transition.r > 0:
                    assert transition == episode[-1]
                #upd = True
            else:
                assert transition.r == 0
        #exit()

        meta_transition = Transition(last_transition.s1, last_transition.m2, 
                None, None, None, last_transition.r)
        out.append(meta_transition)

        ##wood, plank, grass, bed
        #if transition.r > 0:
        #    print "BEGIN"
        #    ing = ["wood", "plank", "grass", "bed"]
        #    for i, exp in enumerate(out):
        #        #if np.random.random() > 0.01:
        #        #    continue
        #        #print self.subtask_index.contents
        #        #if exp.m2.action != 0:
        #        #    continue
        #        for _ing in ing:
        #            print _ing, exp.s1.inventory[self.resource_index[_ing]]
        #        print exp.r
        #        print self.subtask_index.get(exp.m1.action)
        #        print exp.m1.arg
        #        print
        #    print "END"
        #    print

        self.all_experiences += out
        if last_transition.r > 0:
            self.good_experiences += out
            #for t in out:
            #    print t
            #    print t.s1.inventory[self.resource_index["wood"]]
            #    print t.s1.inventory[self.resource_index["plank"]]
            #    if t.s2 is not None:
            #        print t.s2.inventory[self.resource_index["wood"]]
            #        print t.s2.inventory[self.resource_index["plank"]]
            #    print
            #exit()

        #if len(good_out) > 0:
        #    for o in out:
        #        print o
        #    exit()

        #running_reward = 0
        #for transition in episode[::-1]:
        #    #running_reward = running_reward * DISCOUNT + transition.r
        #    #n_transition = transition._replace(r=running_reward)
        #    n_transition = transition
        #    if n_transition.a >= self.world.n_actions:
        #        self.all_experiences.append(n_transition)
        #        if n_transition.r > 0:
        #            self.good_experiences.append(n_transition)

    #@profile
    def train(self):
        self.iter += 1
        total_err = 0
        if len(self.all_experiences) < BATCH_SIZE:
            #return None
            return 0

        self.good_experiences = self.good_experiences[-HISTORY_SIZE/10:]
        self.all_experiences = self.all_experiences[-HISTORY_SIZE:]
        self.experiences = self.all_experiences + self.good_experiences
        #if len(self.good_experiences) > 0:
        #    for _ in range(HISTORY_SIZE / 10):
        #        self.experiences.append(self.good_experiences[np.random.randint(len(self.good_experiences))])
        #for _ in range(HISTORY_SIZE * 9 / 10):
        #    self.experiences.append(self.all_experiences[np.random.randint(len(self.all_experiences))])
        #self.good_experiences = self.good_experiences[-HISTORY_SIZE/10:]
        #self.all_experiences = self.all_experiences[-(HISTORY_SIZE-len(self.good_experiences)):]
        #self.experiences = self.good_experiences + self.all_experiences
        #for _iter in range(100):
        for _iter in range(1):
            batch_exp_i = [np.random.randint(len(self.experiences)) for i in
                    range(BATCH_SIZE)]
            batch_exp = [self.experiences[i] for i in batch_exp_i]


            action_mask = np.zeros((BATCH_SIZE, len(CHOICES)))
            for i, e in enumerate(batch_exp):
                action_mask[i, CHOICES.index(e.m1.action)] = 1

            feed_dict = {
                self.t_feats: [e.s1.features() for e in batch_exp],
                    #[np.clip(e.s1.inventory, 0, 1) for e in batch_exp],
                self.t_feats_target: [
                    np.zeros(self.world.n_features) if e.s2 is None else e.s2.features() 
                    #np.zeros(len(self.world.cookbook.index)) if e.s2 is None else np.clip(e.s2.inventory, 0, 1)
                    for e in batch_exp],
                self.t_terminal: [0 if e.s2 is None else 1 for e in batch_exp],
                #self.t_feats: [[0] for e in batch_exp],
                self.t_action_mask: action_mask,
                self.t_reward: [e.r for e in batch_exp]
            }

            #err, _, sc = self.session.run([self.t_err, self.t_train_op,
            #    self.t_scores], feed_dict=feed_dict)

            err, _ = self.session.run([self.t_err, self.t_train_op], feed_dict=feed_dict)

            ##wood, plank, grass, bed
            #ing = ["wood", "plank", "grass", "bed"]
            #for i, exp in enumerate(batch_exp):
            #    if np.random.random() > 0.01:
            #        continue
            #    #print self.subtask_index.contents
            #    #if exp.m2.action != 0:
            #    #    continue
            #    for _ing in ing:
            #        print _ing, exp.s1.inventory[self.resource_index[_ing]]
            #    print exp.r
            #    print self.subtask_index.get(exp.m1.action)
            #    print exp.m1.arg
            #    print sc[i, :]
            #    print sorted(self.subtask_index.contents.items(), key=lambda
            #            x: x[1])
            #    print

            total_err += err

        #logging.info(self.iter)
        if (self.iter + 1) % 100 == 0:
            logging.info(self.iter)
            self.session.run(self.t_assign_ops)
        #logging.info(len(self.good_experiences))
        #logging.info(self.iter)


        return total_err
