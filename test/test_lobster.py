import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import json
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targets import Targets
from tqdm import tqdm

def getModel(inFileDir):
    model = Model()
    model.loadJson(inFileDir)
    model.scripting = False
    model.script = np.array([0])
    # model.testing = testing
    model.reset()
    return model

def getActions(model, inFileDir):
    with open(inFileDir) as inFile:
        content = inFile.read()
        data = json.loads(content)
        actions = np.array(data['genes'][0]).reshape(model.numChannels, -1)
    return actions


visualize = True
if visualize:
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


def visualizeActions(actions, loop=False):
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
    
    model.inflateChannel = actions[-1]
    model.initializePos()
    model.numSteps = 0
    
    def timerCallback(vis):
        iAction = model.numSteps // Model.numStepsPerActuation
        
        if loop:
            iAction = iAction % len(actions)
        
        if iAction >= len(actions):
            return
        
        model.inflateChannel = actions[iAction]
        
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


model = getModel("./data/lobster3.json")
actions = getActions(model, "./output/records/control_only/lobster2_g25_f111.78788591")
model.inflateChannel = np.array([0, 0, 0, 0])
model.numChannels = 4
model.script = actions
model.initializePos()

# model.script = np.array([[0, 0, 1, 0]])

# visualizeActions(np.array([[0,0,0,0]]), loop=True)

visualizeActions(actions, loop=True)
