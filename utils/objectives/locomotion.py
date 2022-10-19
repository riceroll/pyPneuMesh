import numpy as np

from utils.geometry import getFrontDirection
from utils.objectives.objective import Objective
from utils.truss import Truss


class Locomotion(Objective):

    def __init__(self, truss: Truss):
        self.truss = truss

    def execute(self):
        pass


class MoveForward(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
        velX = dx
        return velX


class FaceForward(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        # 2d direction
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecX = np.array([1, 0, 0])
        alignment = (vecFront * vecX).sum()
        return alignment


class TurnLeft(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecL = np.array([0, 1, 0])
        alignment = (vecFront * vecL).sum()
        return alignment


class TurnRight(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecL = np.array([0, -1, 0])
        alignment = (vecFront * vecL).sum()
        return alignment


class LowerBodyMax(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        return -vs[:, :, 2].max()


class LowerBodyMean(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        return -vs[:, :, 2].max(1).mean()
