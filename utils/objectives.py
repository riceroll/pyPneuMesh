import numpy as np
from utils.geometry import getFrontDirection

def objMoveForward(vs: np.ndarray):
    dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
    velX = dx / len(vs)
    return velX

def objFaceForward(vs: np.ndarray):
    # 2d direction
    vecFront = getFrontDirection(vs[0], vs[-1])     # unit Vector
    assert( abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecX = np.array([1, 0, 0])
    alignment = (vecFront * vecX).sum()
    return alignment

