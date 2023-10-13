import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

dict = readMooDict('scripts/trainTable_64_move/data')

# mode = "start"
mode = "continue"
# mode = "load"
# mode = "configMOO"

GACheckpointDir = "scripts/trainTable_64_move/output/2023-04-27_05-51-56/ElitePool_771.gacheckpoint.npy"
if mode == "start":
    GASetting = {
        'nGenesPerPool': 1024,
        'nSurvivedMin': 512,
        'nGensPerPool': 8,

        'nWorkers': multiprocessing.cpu_count(),

        'folderDir': 'scripts/trainTable_64_move/',

        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.1,
        'contractionCrossChance': 0.02,
        'actionCrossChance': 0.02,
        'crossChance': 0.5,
        'randomInit': True

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
    mooDict = readMooDict('./scripts/trainTable_64_move/data')


    mooDict['actionSeqs'] = {0 : mooDict["actionSeqs"][0]}
    mooDict['objectives'] = {0 : mooDict['objectives'][0]}


    moo = MOO(mooDict=mooDict)

    moo.saveAll('scripts/trainTable_64_move/data', 'table')
