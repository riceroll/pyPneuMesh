import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA


mode = "start"
mode = "continue"
mode = "load"
# mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-05/output/2022-12-09_05-32-52/ElitePool_23.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 32,
        'nSurvivedMin': 16,
        'nGensPerPool': 2,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable_2022-12-09_5-05/',
        
        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.1,
        'randomInit': False
    }
    ga = GA(GASetting=GASetting)
    ga.run()
    
elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    ga.run()
    
elif mode == "load":
    ga = GA(GACheckpointDir=GACheckpointDir)
    print('genePool')
    ga.logPool(ga.genePool, printing=True, showAllGenes=True, showRValue=True)
    print('elitePool')
    ga.logPool(ga.elitePool, printing=True, showAllGenes=True, showRValue=True)
    
elif mode == "configMOO":
    mooDict = readMooDict('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-05/data')
    moo = MOO(mooDict=mooDict)
    moo.animate(0, 4, 3.0)
    
    


