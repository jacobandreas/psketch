import numpy as np

class ReflexMetaModel(object):
    def __init__(self, subtask_index):
        self.subtask_index = subtask_index
        self.random = np.random.RandomState(0)

    def act(self, state):
        return zip(
                self.random.randint(len(self.subtask_index), size=len(state)),
                [None] * len(state)
        )

