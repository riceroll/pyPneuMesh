import numpy as np
import json
import multiprocessing

from src.utils import readNpy, readMooDict
from src.Model import Model
from src.Graph import Graph
from src.MOO import MOO
from src.GA import GA


mode = "start"
mode = "continue"
mode = "load"
# mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable/output/2022-12-09_07-07-13/ElitePool_55.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 64,
        'nSurvivedMin': 16,
        'nGensPerPool': 8,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable/',
        
        'contractionMutationChance': 0.1,
        'actionMutationChance': 0.2,
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
    mooDict = readMooDict('./scripts/trainTable/data')
    moo = MOO(mooDict=mooDict)
    
    


