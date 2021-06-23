import numpy as np

def getFrontDirection(v0, v, vf0=np.array([1, 0, 0])):
    """
    calculate the front direction of the robot

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


def testGetFrontDirection(argv):
    from utils.visualizer import _importVisualizer
    import numpy as np
    
    vs0 = np.array([
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
        [-1, -1, 0],
        [1, -1, 0]
    ])
    
    vs = np.array([
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-1, 1, 0],
        [-1, -1, 0],
    ])
    
    vf = getFrontDirection(vs0, vs)
    
    assert ((vf == np.array([0.0002064471523352829, -0.9999999786897864, 0.])).all())
    # return
    
    if 'plot' in argv:
        o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = _importVisualizer()
        viewer = o3.visualization.VisualizerWithKeyCallback()
        viewer.create_window()
        
        render_opt = viewer.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.mesh_show_wireframe = True
        render_opt.point_size = 8
        render_opt.line_width = 10
        render_opt.light_on = True
        
        vv = np.append(vs, np.array([[1, 0, 0]]), 0)
        print('vf', vf)
        pc = PointCloud(vv)
        
        # center = pc0.mean(0)
        viewer.add_geometry(pc)
        # viewer.add_geometry(pf)
        
        viewer.run()
        viewer.destroy_window()
    
tests = {
    'getFrontDirection': testGetFrontDirection,
}

if __name__ == "__main__":
    pass