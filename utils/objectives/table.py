import numpy as np

from utils.geometry import getTopDirection3D
from utils.objectives.locomotion import Locomotion
from utils.truss import Truss


class TableUpright(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        indicesTop = [0, 1, 2, 3]
        vs = vs[:, indicesTop]
        nFrames = 5
        interval = len(vs) / nFrames
        vs = vs[0::interval] + [vs[-1]]
        vecZ = np.array([0, 0, 1])

        alignments = 0
        for v in vs[1:]:
            vecTop = getTopDirection3D(vs[0], v)
            alignments += (vecTop * vecZ).sum()

        alignment = alignments / (len(vs) - 1)
        return alignment


class TableTilted(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        indicesTop = [0, 1, 2, 3]
        vs = vs[:, indicesTop]
        nFrames = 5
        interval = len(vs) / nFrames
        vs = vs[0::interval] + [vs[-1]]
        vecZ = np.array([1 / 1.414, 0, 1 / 1.414])

        alignments = 0
        for v in vs[1:]:
            vecTop = getTopDirection3D(vs[0], v)
            alignments += (vecTop * vecZ).sum()

        alignment = alignments / (len(vs) - 1)
        return alignment


class TableHigh(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        indicesTop = [0, 1, 2, 3]
        zTarget = 5

        vs = vs[:, indicesTop]
        nFrames = 5
        interval = len(vs) / nFrames
        vs = vs[0::interval] + [vs[-1]]
        zMean = vs[:, :, 2].mean()

        return abs(zMean - zTarget)


class TableLow(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        indicesTop = [0, 1, 2, 3]
        zTarget = 3

        vs = vs[:, indicesTop]
        nFrames = 5
        interval = len(vs) / nFrames
        vs = vs[0::interval] + [vs[-1]]
        zMean = vs[:, :, 2].mean()

        zMax = vs[0].max(0)[-1]

        zDif = zTarget - zMean
        zDif /= zMax

        return min(0, zDif)
