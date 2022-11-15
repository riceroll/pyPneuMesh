import numpy as np

from utils.objectives.locomotion import Locomotion
from utils.truss import Truss


class MinEnergy(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vEnergys = self.truss.vEnergys
        return -np.sqrt((vEnergys ** 2).sum())
