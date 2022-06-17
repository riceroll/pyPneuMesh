import numpy as np

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
    m1 = np.zeros([3,3])
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

