import os
import argparse
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targets import Targets

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

Model.numStepsPerActuation = int(numStepsPerActionMultiplier / Model.h)
model = Model()
model.loadJson(inFileDir)
model.scripting = scripting
model.testing = testing
model.reset(resetScript=True, numActions=20)


def criterion(scriptList):
    model = Model()
    model.loadJson(inFileDir)
    model.scripting = scripting
    model.testing = testing
    model.reset(resetScript=True, numActions=20)
    
    script = scriptList.reshape(model.script.shape)
    v = model.iterScript(script)
    x, y, z = v.mean(0)
    print(x)
    return -x

# set_run_mode(criterion, "multithreading")
# nDim = model.script.shape[0] * model.script.shape[1]
# ga = GA(func=criterion, size_pop=10, n_dim=nDim, max_iter=10, lb=np.zeros(nDim), ub=np.ones(nDim), precision=1)

# genes, fits = ga.run()

