import numpy as np

from utils.objectives.locomotion import Locomotion
from utils.truss import Truss


class GrabLobster(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        return -np.sqrt(((vs[:, 32] - vs[:, 29]) ** 2).sum(1).max())
