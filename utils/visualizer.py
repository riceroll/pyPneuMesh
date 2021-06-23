import numpy as np
from model import Model

def _importVisualizer():
    import open3d as o3
    
    # viewer
    vector3d = lambda v: o3.utility.Vector3dVector(v)
    vector3i = lambda v: o3.utility.Vector3iVector(v)
    vector2i = lambda v: o3.utility.Vector2iVector(v)
    LineSet = lambda v, e: o3.geometry.LineSet(points=vector3d(v), lines=vector2i(e))
    PointCloud = lambda v: o3.geometry.PointCloud(points=vector3d(v))
    
    def drawGround(viewer):
        n = 20
        vs = []
        es = []
        for i, x in enumerate(np.arange(1 - n, n)):
            vs.append([x, 1 - n, 0])
            vs.append([x, n - 1, 0])
            es.append([i * 2, i * 2 + 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        for i, x in enumerate(np.arange(1 - n, n)):
            vs.append([1 - n, x, 0])
            vs.append([n - 1, x, 0])
            es.append([i * 2, i * 2 + 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0.5])
        vs.append([1, 0, 0.5])
        es.append([0, 1])
        cs.append([1, 0, 0])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0.5])
        vs.append([0, 1, 0.5])
        es.append([0, 1])
        cs.append([0, 1, 0])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0.5])
        vs.append([0, 0, 1.5])
        es.append([0, 1])
        cs.append([0, 0, 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
    
    return o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround


def visualizeActions(model, actionSeq, nLoop=1):
    """

    :param model: the model
    :param actionSeq: the actions of the model, equals to the script of the model, np.array, [numChannel, numActions]
    :param loop: if true, the actions will be repeated
    """
    
    o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = _importVisualizer()
    viewer = o3.visualization.VisualizerWithKeyCallback()
    viewer.create_window()
    
    render_opt = viewer.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.mesh_show_wireframe = True
    render_opt.point_size = 8
    render_opt.line_width = 10
    render_opt.light_on = True
    
    ls = LineSet(model.v, model.e)
    viewer.add_geometry(ls)
    
    assert(actionSeq.ndim == 2)
    
    T = Model.numStepsPerActuation
    
    model.inflateChannel = actionSeq[:, -1]
    v = model.step(T)
    global iActionPrev, vs
    iActionPrev = -1
    vs = []
    model.numSteps = 0
    
    def timerCallback(vis):
        global iActionPrev, vs
        iAction = model.numSteps // T
        
        if iAction < len(actionSeq[0]) * nLoop:
            iAction = iAction % len(actionSeq[0])
            
        if iAction != iActionPrev:
            vs.append(model.v.copy())
        iActionPrev = iAction
        
        if iAction >= len(actionSeq[0]) * nLoop:
            vis.close()
            vis.destroy_window()
            return
        
        model.inflateChannel = actionSeq[:, iAction]
        
        model.step(25)
        ls.points = vector3d(model.v)
        viewer.update_geometry(ls)
        
    viewer.register_animation_callback(timerCallback)
    
    # def key_step(vis):
    #     pass
    # viewer.register_key_callback(65, key_step)
    
    drawGround(viewer)
    viewer.run()
    return vs


def testVisualizeActions(argv):
    if 'plot' not in argv:
        return
    
    from utils.modelInterface import getModel, getActionSeq
    modelDir = './test/data/pillBugIn.json'
    model = getModel(modelDir)
    # model.gravity *= 10
    # breakpoint()
    actionSeq = getActionSeq(modelDir)
    
    vs = visualizeActions(model, actionSeq, nLoop=1)
    vsTruth = np.load('./test/data/pillBugOutV.npy')
    assert(((vs - vsTruth) < 1e-5).all())

tests = {
    'visualizeActions': testVisualizeActions,
}

def testAll(argv):
    for key in tests:
        print('test{}{}():'.format(key[0].upper(), key[1:]))
        tests[key](argv)
        print('Pass.\n')


if __name__ == "__main__":
    import sys
    if 'test' in sys.argv:
        if 'all' in sys.argv:
            testAll(sys.argv)
        else:
            for key in tests:
                if key in sys.argv:
                    print('test{}{}():'.format(key[0].upper(), key[1:]))
                    tests[key](sys.argv)
                    print('Pass.\n')
    