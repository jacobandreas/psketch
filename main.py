#!/usr/bin/env python2

from misc.util import Struct
import models
import trainers
import worlds

import numpy as np
import random
import tensorflow as tf
import yaml

def main():
    config = configure()
    world = worlds.load(config)
    model = models.load(config)
    trainer = trainers.load(config)
    trainer.train(model, world)

def configure():
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))
    return config

if __name__ == "__main__":
    main()
