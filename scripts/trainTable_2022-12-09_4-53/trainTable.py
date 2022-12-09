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

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_4-53/output/2022-12-09_04-53-16/ElitePool_15.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 32,
        'nSurvivedMin': 16,
        'nGensPerPool': 2,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable_2022-12-09_4-23/',
        
        'contractionMutationChance': 0.00,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.0,
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
    mooDict = readMooDict('./scripts/trainTable/data')
    moo = MOO(mooDict=mooDict)
    
    


