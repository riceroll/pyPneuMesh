import numpy as np
from utils.geometry import getFrontDirection

def objMoveForward(vs: np.ndarray, es: np.ndarray):
    dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
    velX = dx
    return velX

def objFaceForward(vs: np.ndarray, es: np.ndarray):
    # 2d direction
    vecFront = getFrontDirection(vs[0], vs[-1])     # unit Vector
    assert( abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecX = np.array([1, 0, 0])
    alignment = (vecFront * vecX).sum()
    return alignment

def objTurnLeft(vs: np.ndarray, es: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, 1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment

def objTurnRight(vs: np.ndarray, es: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, -1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment

def objGrabLobster(vs: np.ndarray, es: np.ndarray):
    return -np.sqrt(((vs[:, 32] - vs[:, 29])**2).sum(1).max())

def objLowerBodyMax(vs: np.ndarray, es:np.ndarray):
    return -vs[:,:,2].max()

def objLowerBodyMean(vs: np.ndarray, es:np.ndarray):
    return -vs[:, :, 2].mean()

def testTurn(argv):
    vs = np.array([
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, -1, 0]
        ],
        [
            [-1, 0, 0],
            [0, 1, 0],
            [1, 0, 0]
        ]
    ])
    alignment = objTurnLeft(vs, np.zeros([]))
    assert(abs(alignment - 1) < 1e-6)

    vs = np.array([
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, -1, 0]
        ],
        [
            [1, 0, 0],
            [0, -1, 0],
            [-1, 0, 0]
        ]
    ])
    alignment = objTurnRight(vs, np.zeros([]))
    assert(abs(alignment - 1) < 1e-6)

tests = {
    'testTurn': testTurn,
}

if __name__ == "__main__":
    import sys
    testTurn(sys.argv)