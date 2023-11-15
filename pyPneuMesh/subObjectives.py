import numpy as np
from pyPneuMesh.geometry import getFrontDirection


def objMoveForward(vs: np.ndarray):
    # vs [time, index of vertex, xyz]
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

def objLowerFinalBody(vs: np.ndarray):
    zMax = vs[-1, :, 2].max()
    return -zMax

def objNarrowBody(vs: np.ndarray):
    horizontalMax = vs[:, :, 1].max() - vs[:, :, 1].min()
    return -horizontalMax

def objPillbugAsymmetrical(vs: np.ndarray):
    if vs[:, 3, 1].std() < 1e-5:
        return -1
    else:
        return 0

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

def objTableNoTilt(vs: np.ndarray):
    # minimize the z axis difference of the two pairs of the top four nodes
    return - abs(vs[-1, 45, 2]-vs[-1, 46, 2]) - abs(vs[-1, 47, 2] - vs[-1, 48, 2])

def objPillbugNoIntersection(vs: np.ndarray):
    minDistance = ((vs[:, 16] - vs[:, 17]) ** 2).sum(1).min()
    if minDistance < 1e-8:
        return - minDistance * 1e8
    else:
        return 0

def objTableAlwaysNoTilt(vs: np.ndarray):
    # minimize the z axis difference of the two pairs of the top four nodes
    return - ((vs[:, 45, 2]-vs[:, 46, 2])**2).sum() - ((vs[:, 47, 2] - vs[:, 48, 2])**2).sum()

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

def objTentaclePosition0(vs: np.ndarray):
    target = np.array([0.25661, -0.037002, 1.7182])
    v = vs[-1, 18] / 0.102 * 0.17
    d = ( (v - target ) ** 2).sum()
    return -d

def objTentaclePosition1(vs: np.ndarray):
    target = np.array([-0.3843, 0.13715, 1.8388])
    v = vs[-1, 18] / 0.102 * 0.17
    d = ( (v - target ) ** 2).sum()
    return -d

def objTentaclePosition2(vs: np.ndarray):
    # 0.102 is the simulation scale, 0.17 is the rendering scale
    target = np.array([-0.22215, -0.40834, 1.0739])     # target position in rendering scale (Blender)
    v = vs[-1, 18] / 0.102 * 0.17       # v in rendering scale
    d = ( (v - target ) ** 2).sum()
    return -d

def objGrabLobster(vs: np.ndarray):
    return -np.sqrt(((vs[:, 32] - vs[:, 29]) ** 2).sum(1).max())

def objLowerBodyMax(vs: np.ndarray):
    return -vs[:, :, 2].max()


def objLowerBodyMean(vs: np.ndarray):
    return -vs[:, :, 2].max(1).mean()

def objMaxPullingForce(vs, fs):
    return -fs.max()


# region helmet


helmetWeight = np.zeros(0)
iMiddle = 66
iLeft = 70
iRight = 68
middleTargetOffset = np.array([0.5, 0, -0.5])
leftTargetOffset = np.array([0, 0.1, -0.5])
rightTargetOffset = np.array([0, -0.1, -0.5])
def objHelmetMiddle(vs):
    middleOffset = vs[-1, iMiddle] - vs[0, iMiddle]
    leftOffset = vs[-1, iLeft] - vs[0, iLeft]
    rightOffset = vs[-1, iRight] - vs[0, iRight]
    
    dMiddle = ((middleOffset - middleTargetOffset) ** 2).sum()
    dLeft = (leftOffset ** 2).sum()
    dRight = (rightOffset ** 2).sum()

    global helmetWeight
    if len(helmetWeight) == 0:
        from pyPneuMesh.utils import getDistance, getWeightFromDistance
        
        trussparam = np.load('./scripts/trainHelmet4_static/data/helmet.trussparam.npy', allow_pickle=True).all()
        e = trussparam['e']
        distance = getDistance(e, [iMiddle, iLeft, iRight])
        helmetWeight = getWeightFromDistance(distance)
    
    
    allOff = (((vs[-1] - vs[0]) ** 2).sum(1) * helmetWeight).sum()
    
    return -(dMiddle * 2 + dLeft * 1 + dRight * 1 + allOff * 0.1)


def objHelmetSide(vs):
    middleOffset = vs[-1, iMiddle] - vs[0, iMiddle]
    leftOffset = vs[-1, iLeft] - vs[0, iLeft]
    rightOffset = vs[-1, iRight] - vs[0, iRight]
    
    dMiddle = (middleOffset ** 2).sum()
    dLeft = ((leftOffset - leftTargetOffset) ** 2).sum()
    dRight = ((rightOffset - rightTargetOffset) ** 2).sum()
    
    global helmetWeight
    if len(helmetWeight) == 0:
        from pyPneuMesh.utils import getDistance, getWeightFromDistance
        
        trussparam = np.load(
            '/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainHelmet4_static/data/helmet.trussparam.npy',
            allow_pickle=True).all()
        e = trussparam['e']
        distance = getDistance(e, [iMiddle, iLeft, iRight])
        helmetWeight = getWeightFromDistance(distance)
    
    allOff = (((vs[-1] - vs[0]) ** 2).sum(1) * helmetWeight).sum()
    
    return -(dMiddle * 2 + dLeft + dRight + allOff * 0.1)


# endregion


