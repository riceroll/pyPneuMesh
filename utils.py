import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
from optimizer import EvolutionAlgorithm
rootPath = os.path.split(os.path.realpath(__file__))[0]
tPrev = time.time()

def import_visualizer():
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
        lines.colors=o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0.5])
        vs.append([0, 1, 0.5])
        es.append([0, 1])
        cs.append([0, 1, 0])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors=o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)
        
        vs = []
        es = []
        cs = []
        vs.append([0, 0, 0.5])
        vs.append([0, 0, 1.5])
        es.append([0, 1])
        cs.append([0, 0, 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        lines.colors=o3.utility.Vector3dVector(cs)
        viewer.add_geometry(lines)

    return o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround

def visualizeActions(model, actions, loop=False):
    """
    
    :param model: the model
    :param actions: the actions of the model, equals to the script of the model, np.array, [numChannel, numActions]
    :param loop: if true, the actions will be repeated
    """
    
    o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = import_visualizer()
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
    
    model.reset()
    
    actions = actions.reshape(model.numChannels, -1)
    
    # print(Model.numStepsPerActuation)
    
    model.inflateChannel = actions[:, -1]
    model.numSteps = 0
    model.script = actions
    print(1)
    model.initializePos()
    print(2)
    
    def timerCallback(vis):
        iAction = model.numSteps // model.numStepsPerActuation
        
        if loop:
            iAction = iAction % len(actions)
        
        if iAction >= len(actions):
            return
        
        model.inflateChannel = actions[:, iAction]
        
        model.step(25)
        ls.points = vector3d(model.v)
        viewer.update_geometry(ls)
    
    viewer.register_animation_callback(timerCallback)
    
    # def key_step(vis):
    #     pass
    # viewer.register_key_callback(65, key_step)
    
    drawGround(viewer)
    viewer.run()
    viewer.destroy_window()
    
def getModel(inFileDir):
    """
    load a json file into a model
    :param inFileDir: the json file of the input configuration
    :return: the model with loaded json configuration
    """
    from model import Model
    model = Model()
    model.load(inFileDir)
    model.reset()
    return model

def getActions(model, inFileDir):
    with open(inFileDir) as inFile:
        content = inFile.read()
        data = json.loads(content)
        actions = np.array(data['genes'][0]).reshape(model.numChannels, -1)
    return actions

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", type=bool, default=False, help="whether to visualize the result")
    parser.add_argument("--testing", type=bool, default=False, help="whether in testing mode")
    parser.add_argument("--nWorkers", type=int, default=8, help="number of workers")
    parser.add_argument("--nGen", type=int, default=100, help="whether in testing mode")
    parser.add_argument("--nPop", type=int, default=8, help="size of population")
    parser.add_argument("--numActions", type=int, default=4, help="# of actions, -1: read the # from json")
    parser.add_argument("--numChannels", type=int, default=4, help="# of channels, -1: read the # from json")
    parser.add_argument("--targets", type=str, default="moveForward", help="type of target")
    parser.add_argument("--numStepsPerActionMultiplier", type=float, default=0.2, help="# of steps per action")
    parser.add_argument("--inFile", type=str, default="./data/lobster3.json", help="infile")
    args = parser.parse_args()
    return args

# min||np.dot(vecf, vecs) - np.dot(vecf0, vecs0)||

def getFrontDirection(vs0, vs, vf0=np.array([1, 0, 0])):
    """
    calculate the front direction of the robot
    
    :param vs0: np.array, [nV, 3], initial positions of vertices
    :param vs: np.array, [nV, 3], current positions of vertices
    :param vf0: np.array, [3, ] initial vector of the front, default along x axis
    :return: vf: np.array, [3, ] current vector of the front
    """
    vs0 = vs0[:, :2]
    vs = vs[:, :2]
    vf0 = vf0[:2]
    
    vf0 = vf0.reshape(-1, 1)
    center = vs0.mean(0).reshape([1, -1])
    vecs0 = vs0 - center
    dots0 = vecs0 @ vf0    # nv x 1
    # print(vs, vecs0, vf0)
    # print('dots0: ', dots0)
    
    def alignEnergy(vf):
        # print()
        # print('vf: ', vf)
        vf = vf.reshape(-1, 1)
        vf = np.true_divide(vf, (vf ** 2).sum() ** 0.5)
        dots = (vs - center) @ vf.reshape(-1, 1)
        # print('dots:', dots)
        diff = ((dots - dots0) ** 2).sum()
        # print('diff: ', diff)
        return diff

    from scipy import optimize
    
    vf0 = vf0.reshape(-1)
    sols = optimize.least_squares(alignEnergy, vf0.copy(), diff_step=np.ones_like(vf0)*1e-7)
    
    # print(sols)
    
    vf = sols.x / (sols.x ** 2).sum() ** 0.5
    vf = np.concatenate([vf, np.array([0])])
    
    return vf

def test(inFileDir, edgeChannel, maxContraction, actions):
    # model load inFile(v, e)
    # model set edgeChannel
    # model set maxContraction
    # model initpos
    # model play acitons and record vs
    # TODO
    pass

