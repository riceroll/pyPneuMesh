import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targets import Targets
from tqdm import tqdm

# consts
parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False, help="whether to visualize the result")
parser.add_argument("--testing", type=bool, default=False, help="whether in testing mode")
parser.add_argument("--iFile", type=str, default="fox", help="name without suffix of input file under ./data folder")
parser.add_argument("--nWorkers", type=int, default=8, help="number of workers")
parser.add_argument("--nGen", type=int, default=1000, help="whether in testing mode")
parser.add_argument("--nPop", type=int, default=100, help="size of population")
parser.add_argument("--numActions", type=int, default=-1, help="# of channels, -1: read the # from json")
parser.add_argument("--targets", type=str, default="moveForward", help="type of target")
parser.add_argument("--numStepsPerActionMultiplier", type=str, default="0.2", help="# of steps per action")
args = parser.parse_args()

scripting = True
visualize = args.visualize
inFileDir = args.iFile
testing = args.testing
numWorkers = args.nWorkers
numGeneration = args.nGen if not testing else 5
numPopulation = args.nPop if not testing else 8
numActions = args.numActions
numStepsPerActionMultiplier = float(args.numStepsPerActionMultiplier)

# visualize = True
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

# main
# inFileDir = './data/{}.json'.format(inFileName)
def getModel():
    model = Model()
    model.loadJson(inFileDir)
    model.scripting = False
    model.testing = testing
    model.reset(resetScript=True, numActions=20)
    return model

model = getModel()

def criterionMoveForward(actions, nStepsPerAction):
    m = getModel()
    actions = np.array(actions).reshape(m.numChannels, -1)
    
    m.inflateChannel = actions[-1]
    m.initializePos()
    v0 = m.v.copy()
    
    for action in actions:
        m.inflateChannel = action
        m.step(nStepsPerAction)
       
    v = m.v.copy()
    dx = v.mean(0)[0] - v0.mean(0)[0]
    
    v = v - v.mean(0)[0]
    dot = (v * v0).sum(1)
    
    print("criterion: ", dx * 100, dot.mean() )
    
    return dx * 100 + dot.mean()

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
        

        print(iAction, loop)
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

nDigits = model.numChannels * 2

criterion = lambda actions : criterionMoveForward(actions, Model.numStepsPerActuation)

ea = EvolutionAlgorithm(name="lobster2", lb=np.zeros(nDigits), ub=np.ones(nDigits), criterion=criterion,
                        nWorkers=args.nWorkers,
                        nPop=48,
                        mortality=0.2, pbCross=0.5, pbMut=0.05, pbCrossDig=0.05, pbMutDig=0.05, lenConverge=40)

ea.maximize(300, False)



