import numpy as np
from src.geometry import getFrontDirection


def objMoveForward(vs: np.ndarray):
    dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
    velX = dx
    return velX

def objFaceForward(vs: np.ndarray):
    # 2d direction
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecX = np.array([1, 0, 0])
    alignment = (vecFront * vecX).sum()
    return alignment

def objTurnLeft(vs: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, 1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment

def objLowerBody(vs: np.ndarray):
    zMax = vs[:, :, 2].max()
    return -zMax

def objTableTilt(vs: np.ndarray):
    v45 = vs[-1, 45]
    v46 = vs[-1, 46]
    v47 = vs[-1, 47]
    v48 = vs[-1, 48]
    
    vec0 = v45 - v46
    vec1 = v47 - v48
    
    vecPerpendicular = np.cross(vec0, vec1)
    unitVecPerpendicular = vecPerpendicular / np.linalg.norm(vecPerpendicular)
    horizontal = np.linalg.norm(unitVecPerpendicular[:2])
    vertical = unitVecPerpendicular[2]
    unitVec = np.array([horizontal, vertical])
    unitVec45 = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

    return np.dot(unitVec, unitVec45)

def objTurnRight(vs: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, -1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment


def objTableHigh(vs: np.ndarray):
    indicesTop = [0, 1, 2, 3]
    zTarget = 5
    
    vs = vs[:, indicesTop]
    nFrames = 5
    interval = len(vs) / nFrames
    vs = vs[0::interval] + [vs[-1]]
    zMean = vs[:, :, 2].mean()
    
    return abs(zMean - zTarget)

def objGrabLobster(vs: np.ndarray):
    return -np.sqrt(((vs[:, 32] - vs[:, 29]) ** 2).sum(1).max())


def objLowerBodyMax(vs: np.ndarray):
    return -vs[:, :, 2].max()


def objLowerBodyMean(vs: np.ndarray):
    return -vs[:, :, 2].max(1).mean()

