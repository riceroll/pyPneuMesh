import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA


mode = "start"
# mode = "continue"
# mode = "load"
# mode = "configMOO"

# GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-51/output/gcp_2022-12-09_12-19-30/ElitePool_115.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 8,
        'nSurvivedMin': 4,     # actually is max
        'nGensPerPool': 1,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable_2channels/',

        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.1,
        'contractionCrossChance': 0.02,
        'actionCrossChance': 0.02,
        'crossChance': 0.5,
        'graphRandomInit': True,
        'contractionActionRandomInit': True,
        
        'randomInit': False
    }
    ga = GA(GASetting=GASetting)
    ga.run()
    
