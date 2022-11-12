from utils.objectives.locomotion import Locomotion
from utils.truss import Truss
import numpy as np


class CurvedBridge(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        # how much the top layer vertices are aligned with an arc

        xMin = vs[0, :, 0].min()
        xMax = vs[0, :, 0].max()
        zMax0 = vs[0, :, 2].max()  # the initial minimum z

        ivs = np.arange(len(vs[0]))[vs[0, :, 2] > zMax0 - 0.01]

        alpha = 60 / 180 * np.pi
        r = (xMax - xMin) / np.sin(alpha / 2)
        zC = zMax0 - np.cos(alpha / 2) * r
        xC = (xMin + xMax) / 2

        vsTop = vs[-1, ivs]  # len(ivs) x 3

        return -np.sqrt(((vsTop - xC) ** 2).sum(1) + ((vsTop - zC) ** 2).sum(1)).mean()


class FlatBridge(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        # how much the top layer vertices are aligned with a flat line

        # TODO: automate this
        zMax0 = 2.052018814086914  # the initial minimum z
        ivs = np.array([0, 1, 3, 4, 5, 6, 7, 8, 23, 24, 25, 26, 27,
                        29, 30, 31, 36, 38, 40, 43, 44, 45, 46, 47, 49, 52,
                        53, 57, 61, 71, 74, 75, 77, 79, 82, 83, 84, 86, 87,
                        88, 90, 91, 92, 93, 94, 95, 98, 100, 101, 102, 103, 104,
                        109, 111, 112, 114, 116, 122, 123, 125, 129, 131, 133, 138, 140,
                        141, 145, 146, 147, 148, 149, 150])

        vsTop = vs[-1, ivs]

        return -np.sqrt((vsTop[:, 2] - zMax0) ** 2).mean()
