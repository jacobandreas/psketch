from reflex import ReflexModel
from attentive import AttentiveModel
from modular import ModularModel
from modular_ac import ModularACModel
from modular_ac_interactive import ModularACInteractiveModel
from keyboard import KeyboardModel

def load(config):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
