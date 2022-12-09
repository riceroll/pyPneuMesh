import numpy as np
import json
import multiprocessing

from src.utils import readNpy, readMooDict
from src.Model import Model
from src.Graph import Graph
from src.MOO import MOO
from src.GA import GA


mode = "start"
# mode = "continue"
# mode = "load"
# mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-51/output/2022-12-09_05-32-52/ElitePool_23.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 512,
        'nSurvivedMin': 256,
        'nGensPerPool': 16,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable_2022-12-09_5-51/',

        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.1,
        'contractionCrossChance': 0.02,
        'actionCrossChance': 0.02,
        'crossChance': 0.5,
        
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
    mooDict = readMooDict('scripts/trainTable_2022-12-09_5-51/data/')
    moo = MOO(mooDict=mooDict)
    
    


