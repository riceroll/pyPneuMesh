import numpy as np
import json
import multiprocessing

from utils.utils import readNpy, readMooDict
from utils.Model import Model
from utils.Graph import Graph
from utils.MOO import MOO
from utils.GA import GA


mode = "start"
mode = "continue"
mode = "load"
mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/examples/trainTable/output/2022-12-08_16-24-30/ElitePool_2.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 16,
        'nSurvivedMax': 8,
        'nGensPerPool': 2,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'examples/trainTable/',
        
        'contractionMutationChance': 0.1,
        'actionMutationChance': 0.2
    }
    ga = GA(GASetting=GASetting)
    ga.run()
    
elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    ga.run()

elif mode == "load":
    ga = GA(GACheckpointDir=GACheckpointDir)

elif mode == "configMOO":
    mooDict = readMooDict('./examples/trainTable/data')




