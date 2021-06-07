import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targetFuncs import Targets
from tqdm import tqdm

from sko.GA import GA
from sko.tools import set_run_mode

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

numStepsPerAction = int(Model.numStepsPerActuation * 0.5)
mutationPercent = 0.1
TEpisode = 10
nSamples = 8
horizon = 10

def criterionMoveForward(m, vPrev):
    v0 = m.v0
    v = m.v
    dx = v.mean(0)[0] - vPrev.mean(0)[0]
    
    v = v - v.mean(0)[0]
    dot = (v * v0).sum(1)
    
    # print("criterion: ", dx * 100, dot.mean() )
    
    # return dx
    return dx * 100 + dot.mean()

def sampleAction(actionSeqHorizon):
    actionSeqHorizon = actionSeqHorizon.reshape(-1)
    nHalfSamples = int(nSamples / 2)
    actionSeqSamples = []
    for iSample in range(nHalfSamples):
        actionSeqSamples.append(np.random.randint(np.ones([horizon, model.numChannels]) * 2))
    
    for iSample in range(nSamples - nHalfSamples):
        indices = np.random.choice(np.arange(len(actionSeqHorizon)))
        actionSeq = actionSeqHorizon.copy()
        actionSeq[indices] = (actionSeq[indices] + 1) % 2
        actionSeq = actionSeq.reshape(horizon, model.numChannels)
        actionSeq = np.array(actionSeq, dtype=np.int64)
        actionSeqSamples.append(actionSeq)
    return actionSeqSamples

# test
ash = np.zeros([horizon, model.numChannels])
ret = sampleAction(ash)

def runHorizon(actionSeq, state, criterion):
    model = getModel()
    # model.reset()
    model.v = state.copy()
    
    for i in range(len(actionSeq)):
        model.inflateChannel = actionSeq[i].copy()
        model.step(numStepsPerAction)
        
    return criterion(model, state)

def runStep(action, state):
    model = getModel()
    # model.reset()
    model.v = state
    
    model.inflateChannel = action
    model.step(numStepsPerAction)
    
    return model.v.copy()

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
    
    def timerCallback(vis):
        iAction = model.numSteps // numStepsPerAction
        
        if loop:
            iAction = iAction % len(actions)
        
        if iAction >= len(actions):
            return
        
        # print(iAction)
        
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


actionSeq = []    # [t, numChannel]
status = [model.v0]     # [t, V0: [numV, 3]]
returns = []    # [t, ]
actionSeqHorizon = np.zeros([horizon, model.numChannels])     # [horizon, numChannel]

for t in range(TEpisode):
    actionSeqSamples = sampleAction(actionSeqHorizon)   # [nSamples, horizon, numChannel]

    def evaluate(iSample):
        actionSeqSample = actionSeqSamples[iSample]
        ret = runHorizon(actionSeqSample, status[-1], criterionMoveForward)
        return ret
    
    iSamples = np.arange(nSamples).tolist()
    with Pool(multiprocessing.cpu_count()) as p:
        returnSamples = np.array(p.map(evaluate, iSamples))
        p.clear()
        p.restart()
    
    
    returnSamples = np.array([evaluate(iSample) for iSample in iSamples])

    weights = np.exp(nSamples * (returnSamples - returnSamples.max()))
    weights = weights.reshape(-1, 1, 1)    # [nSamples, 1]
    
    actionSeqHorizon = actionSeqSamples * weights / weights.sum()
    actionSeqHorizon = actionSeqHorizon.round().sum(0).round()

    ret = runHorizon(actionSeqHorizon, status[-1], criterionMoveForward)
    print('return: ', ret)
    if ret < returnSamples.max():
        print("ret not max:", ret, returnSamples)
        actionSeqHorizon = actionSeqSamples[returnSamples.argmax()]
    
    actionSeq.append(actionSeqHorizon[0])
    
    indices = np.arange(horizon) + 1
    indices[-1] = 0
    actionSeqHorizon = actionSeqHorizon[indices, :]
    actionSeqHorizon[-1] *= 0
    
    state = runStep(actionSeq[-1], status[-1])
    status.append(state)


