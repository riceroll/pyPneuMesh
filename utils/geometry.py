import numpy as np
import igl


def getFrontDirection(v0, v, vf0=np.array([1, 0, 0])):
    """
    calculate the front direction of the robot projected to the horizontal plane

    :param v0: np.array, [nV, 3], initial positions of vertices
    :param v: np.array, [nV, 3], current positions of vertices
    :param vf0: np.array, [3, ] initial vector of the front, default along x axis
    :return: vf: np.array, [3, ] current unit vector of the front
    """
    v0 = v0[:, :2]
    v = v[:, :2]
    vf0 = vf0[:2]

    vf0 = vf0.reshape(-1, 1)

    center0 = v0.mean(0).reshape([1, -1])
    dv0 = v0 - center0
    dv0 = np.true_divide(dv0, (dv0 ** 2).sum(1, keepdims=True) ** 0.5)
    dots0 = dv0 @ vf0  # nv x 1

    center = v.mean(0).reshape([1, -1])
    dv = v - center
    dv = np.true_divide(dv, (dv ** 2).sum(1, keepdims=True) ** 0.5)

    def alignEnergy(vf):
        vf = np.true_divide(vf, (vf ** 2).sum() ** 0.5)
        vf = vf.reshape(-1, 1)
        dots = dv @ vf
        diff = ((dots - dots0) ** 2).sum()
        return diff

    from scipy import optimize

    vf0 = vf0.reshape(-1)
    sols = optimize.least_squares(alignEnergy, vf0.copy(), diff_step=np.ones_like(vf0) * 1e-7)

    vf = sols.x / (sols.x ** 2).sum() ** 0.5
    vf = np.concatenate([vf, np.array([0])])

    return vf


def getTopDirection3D(v0, v, vf0=np.array([0, 0, 1])):
    """
    calculate the front direction of the robot in 3D

    :param v0: np.array, [nV, 3], initial positions of vertices
    :param v: np.array, [nV, 3], current positions of vertices
    :param vf0: np.array, [3, ] initial vector of the front, default along x axis
    :return: vf: np.array, [3, ] current unit vector of the front
    """
    v0 = v0[:]
    v = v[:]
    vf0 = vf0

    vf0 = vf0.reshape(-1, 1)

    center0 = v0.mean(0).reshape([1, -1])
    dv0 = v0 - center0
    dv0 = np.true_divide(dv0, (dv0 ** 2).sum(1, keepdims=True) ** 0.5)
    dots0 = dv0 @ vf0  # nv x 1

    center = v.mean(0).reshape([1, -1])
    dv = v - center
    dv = np.true_divide(dv, (dv ** 2).sum(1, keepdims=True) ** 0.5)

    def alignEnergy(vf):
        vf = np.true_divide(vf, (vf ** 2).sum() ** 0.5)
        vf = vf.reshape(-1, 1)
        dots = dv @ vf
        diff = ((dots - dots0) ** 2).sum()
        return diff

    from scipy import optimize

    vf0 = vf0.reshape(-1)
    sols = optimize.least_squares(alignEnergy, vf0.copy(), diff_step=np.ones_like(vf0) * 1e-7)

    vf = sols.x / (sols.x ** 2).sum() ** 0.5
    # vf = np.concatenate([vf, np.array([0])])

    return vf


def getClosestVector(a, b, c, d):
    # return closest vector connecting ab to cd
    # return None if skew or parallel

    l1 = np.linalg.norm(b - a)
    l2 = np.linalg.norm(d - c)
    v1 = (b - a) / l1
    v2 = (d - c) / l2

    v1xv2 = np.cross(v1, v2)
    v1xv2normsqr = np.linalg.norm(v1xv2) ** 2

    if v1xv2normsqr == 0:
        # parallel
        return None


    else:
        m1 = np.zeros([3, 3])
        m1[0] = c - a
        m1[1] = v2
        m1[2] = v1xv2

        t = np.linalg.det(m1) / v1xv2normsqr

        m2 = m1.copy()
        m2[1] = v1
        s = np.linalg.det(m2) / v1xv2normsqr

        N1 = a + v1 * t
        N2 = c + v2 * s

        if t > l1 or s > l2:
            # skew
            return None

        return N2 - N1


def targetVnFnVolume(objFileDir):
    import pyvista as pv
    # objFileDir = "/Users/Roll/Desktop/1_Main/0_Projects/1_PneuMesh+/2_codes/pyPneuMesh/data/testTable.obj"
    pvmesh = pv.read(objFileDir)
    vTarget = pvmesh.points
    fTarget = pvmesh.faces.reshape(-1, 4)[:, 1:]

    import tetgen
    tet = tetgen.TetGen(pvmesh)
    tet.tetrahedralize(order=1)
    tet = tet.grid
    volumeTarget = tet.volume
    breakpoint()
    return vTarget, fTarget, volumeTarget


def shapeSimilarity(vTarget, fTarget, volumeTarget, model):
    signed_distances, _, _ = igl.signed_distance(model.v, vTarget, fTarget)
    distance = np.abs(signed_distances).max()

    trussVolume = model.volume()
    volumeDiff = np.abs(volumeTarget - trussVolume)

    return distance, volumeDiff


def boundingBox(vs):
    bv, _ = igl.bounding_box(vs)
    return bv


def bboxDiagonal(vs):
    return igl.bounding_box_diagonal(vs)


def center(vs):
    return np.mean(vs, axis=0)


def translationMatrix(dx, dy, dz):
    return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


def scaleMatrix(ax, by, cz):
    return np.array([[ax, 0, 0, 0], [0, by, 0, 0], [0, 0, cz, 0], [0, 0, 0, 1]])


def rotationMatrix(R):
    return np.array(
        [[R[0][0], R[0][1], R[0][2], 0], [R[1][0], R[1][1], R[1][2], 0], [R[2][0], R[2][1], R[2][2], 0], [0, 0, 0, 1]])


# Homogeneous Transformation
def transform3d(vs, transformations):
    vs4d = np.insert(vs, 3, 1, axis=1)

    for trans in transformations:
        vs4d = np.transpose(np.dot(trans, np.transpose(vs4d)))

    return np.delete(vs4d, 3, axis=1)


def rigid_align(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor
    # c = 1

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

    # homogeneous transformation
    m = P.shape[1]
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t
