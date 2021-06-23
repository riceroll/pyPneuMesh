# multi-objective optimization
import numpy as np
from model import Model

class MMOSetting:
    def __init__(self, setting):
        self.modelDir : str = ""
        self.numChannels : int = -1
        self.numActions : int = -1
        self.objectives : list = []
        self.model : Model = Model()
        
        self.load(setting)
    
    def load(self, setting: dict):
        for key in setting:
            assert (hasattr(self, key))
            assert (type(setting[key]) == type(getattr(self, key)))
            setattr(self, key, setting[key])

