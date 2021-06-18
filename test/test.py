
def testGetFrontDirection():
    from utils import getFrontDirection, import_visualizer
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
    
    # return
    o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = import_visualizer()
    viewer = o3.visualization.VisualizerWithKeyCallback()
    viewer.create_window()

    render_opt = viewer.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.mesh_show_wireframe = True
    render_opt.point_size = 8
    render_opt.line_width = 10
    render_opt.light_on = True
    
    vv = np.append(vs, np.array([[1,0,0]]), 0)
    print('vf', vf)
    pc = PointCloud(vv)
    
    # center = pc0.mean(0)
    viewer.add_geometry(pc)
    # viewer.add_geometry(pf)
    
    viewer.run()
    viewer.destroy_window()

def testModelStep():
    from model import Model
    from utils import getModel
    import json
    
    model = getModel("./test/data/lobsterIn.json")
    model.step(200)
    js = model.exportJSON(save=False)
    with open('./test/data/lobsterOut.json') as iFile:
        jsTrue = iFile.read()
        assert(js == jsTrue)
        print('testModelStep pass.')

if __name__ == "__main__":
    import sys
    if 'front' in sys.argv:
        testGetFrontDirection()
    
    if 'modelStep' in sys.argv:
        testModelStep()

