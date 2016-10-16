from misc.util import Struct, Index

import copy
import numpy as np
import yaml

class Cookbook(object):
    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.load(recipes_f)
        #self.environment = set(recipes["environment"])
        self.index = Index()
        self.environment = set(self.index.index(e) for e in recipes["environment"])
        self.primitives = set(self.index.index(p) for p in recipes["primitives"])
        self.recipes = {}
        for output, inputs in recipes["recipes"].items():
            d = {}
            for inp, count in inputs.items():
                # special keys
                if "_" in inp:
                    d[inp] = count
                else:
                    d[self.index.index(inp)] = count
            self.recipes[self.index.index(output)] = d
        kinds = self.environment | self.primitives | set(self.recipes.keys())
        self.n_kinds = len(self.index)

    def primitives_for(self, goal):
        out = {}

        def insert(kind, count):
            assert kind in self.primitives
            if kind not in out:
                out[kind] = count
            else:
                out[kind] += count

        for ingredient, count in self.recipes[goal].items():
            if not isinstance(ingredient, int):
                assert ingredient[0] == "_"
                continue
            elif ingredient in self.primitives:
                insert(ingredient, count)
            else:
                sub_recipe = self.recipes[ingredient]
                n_produce = sub_recipe["_yield"] if "_yield" in sub_recipe else 1
                n_needed = int(np.ceil(1. * count / n_produce))
                expanded = self.primitives_for(ingredient)
                for k, v in expanded.items():
                    insert(k, v * n_needed)

        return out
