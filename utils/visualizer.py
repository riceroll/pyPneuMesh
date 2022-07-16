import json
import numpy as np
from utils.model import Model
from matplotlib import cm

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
        vs.append([0, 0, 0])
        vs.append([1, 0, 0])
        es.append([0, 1])
        cs.append([1, 0, 0])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0])
        vs.append([0, 1, 0])
        es.append([0, 1])
        cs.append([0, 1, 0])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0])
        vs.append([0, 0, 1])
        es.append([0, 1])
        cs.append([0, 0, 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors = o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
    
    return o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround

def _create_window(o3):
    viewer = o3.visualization.VisualizerWithKeyCallback()
    viewer.create_window()

    render_opt = viewer.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.mesh_show_wireframe = True
    render_opt.point_size = 8
    render_opt.line_width = 10
    render_opt.light_on = True
    
    return viewer

def showFrames(frames, es, h):
    try:
        o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = _importVisualizer()
        viewer = _create_window(o3)
    except Exception as e:
        print(e)
        return
    
    ls = LineSet(frames[0], es)
    viewer.add_geometry(ls)
    
    global t0
    
    import time
    t0 = time.time()
    
    def callback(vis):
        
        t = time.time() - t0
        
        iFrame = int(t / h)
        
        if iFrame >= len(frames):
            vis.close()
            vis.destroy_window()
            return
        
        ls.points = vector3d(frames[iFrame])
        viewer.update_geometry(ls)

    viewer.register_animation_callback(callback)
    

    drawGround(viewer)
    viewer.run()
    

