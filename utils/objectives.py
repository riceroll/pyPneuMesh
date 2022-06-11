import numpy as np
from utils.geometry import getFrontDirection, getTopDirection3D

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

def objTableUpright(vs: np.ndarray, es: np.ndarray):
    indicesTop = [0,1,2,3]
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


def objTableTilted(vs: np.ndarray, es: np.ndarray):
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


def objTableHigh(vs: np.ndarray, es: np.ndarray):
    indicesTop = [0, 1, 2, 3]
    zTarget = 5
    
    vs = vs[:, indicesTop]
    nFrames = 5
    interval = len(vs) / nFrames
    vs = vs[0::interval] + [vs[-1]]
    zMean = vs[:, :, 2].mean()
    
    return abs(zMean - zTarget)

def objTableLow(vs: np.ndarray, es: np.ndarray):
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

def objLowerBodyMax(vs: np.ndarray, es: np.ndarray):
    return -vs[:, :, 2].max()

def objLowerBodyMean(vs: np.ndarray, es: np.ndarray):
    return -vs[:, :, 2].max(1).mean()

def objCurvedBridge(vs: np.ndarray, es: np.ndarray):
    # how much the top layer vertices are aligned with an arc
    
    xMin = vs[0, :, 0].min()
    xMax = vs[0, :, 0].max()
    zMax0 = vs[0, :, 2].max()      # the initial minimum z
    
    ivs = np.arange(len(vs[0]))[ vs[0, :, 2] > zMax0 - 0.01 ]
    
    alpha = 60 / 180 * np.pi
    r = (xMax - xMin) / np.sin(alpha / 2)
    zC = zMax0 - np.cos(alpha / 2) * r
    xC = (xMin + xMax) / 2
    
    vsTop = vs[-1, ivs]    # len(ivs) x 3
    
    return -np.sqrt( ((vsTop - xC) ** 2).sum(1) + ((vsTop - zC) ** 2).sum(1) ).mean()
    
    
def objFlatBridge(vs: np.ndarray, es: np.ndarray):
    # how much the top layer vertices are aligned with a flat line
    
    # TODO: automate this
    zMax0 = 2.052018814086914  # the initial minimum z
    ivs = np.array([  0,   1,   3,   4,   5,   6,   7,   8,  23,  24,  25,  26,  27,
        29,  30,  31,  36,  38,  40,  43,  44,  45,  46,  47,  49,  52,
        53,  57,  61,  71,  74,  75,  77,  79,  82,  83,  84,  86,  87,
        88,  90,  91,  92,  93,  94,  95,  98, 100, 101, 102, 103, 104,
       109, 111, 112, 114, 116, 122, 123, 125, 129, 131, 133, 138, 140,
       141, 145, 146, 147, 148, 149, 150])
    
    vsTop = vs[-1, ivs]
    
    return -np.sqrt((vsTop[:, 2] - zMax0) ** 2).mean()


# def virtualObjShape(ivs: np.ndarray, vsTargets: np.ndarray):
#     '''
#     virtual objective function for setting the key vertices of the target shape as the objectives
#     :param ivs: ids of target vertices
#     :param vsTargets: position of target vertices
#     :return: an objective function
#     '''
#
#     def objShape(vs: np.ndarray, es: np.ndarray):
#         # vs: shape: [nFrame, nVertex, nDim]
#         # es: shape: [nEdge, 2]
#         avgError = np.sqrt(((vs[-1][ivs] - vsTargets) ** 2).sum(1)).mean()
#         return avgError
#
#     return objShape


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

def testCurvedBridge(argv):
    v = np.array([
        [
        [0, 0, 0],
        [2, 0, 0],
        [5, 3, 0],
        [2, 2, -2],
        [2, 9, 0],
        ]
    ])
    
    alignment = objCurvedBridge(v, np.zeros([]))
    assert(abs(alignment - 18.98258662289415) < 1e-6)


def testFlatBridge(argv):
    v = np.array([
        [
            [0, 0, 0],
            [2, 0, 0],
            [5, 3, 0],
            [2, 2, -2],
            [2, 9, 0],
        ]
    ])
    
    alignment = objFlatBridge(v, np.zeros([]))
    # assert (abs(alignment - 18.98258662289415) < 1e-6)

tests = {
    'testTurn': testTurn,
    'testCurvedBridge': testCurvedBridge,
    'testFlatBridge': testFlatBridge,
}

if __name__ == "__main__":
    import sys
    testTurn(sys.argv)