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
# mode = "load"
# mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/output/2023-11-14_03-11-23/ElitePool_18.gacheckpoint.npy"

if mode == "start":
    # GASetting = {
    #     'nGenesPerPool': 512,
    #     'nSurvivedMin': 128,     # actually is max
    #     'nGensPerPool': 6,
    #
    #     'nWorkers': multiprocessing.cpu_count(),
    #
    #     'folderDir': 'scripts/trainPillbug_2023_10_21',
    #
    #     'contractionMutationChance': 0.01,
    #     'actionMutationChance': 0.01,
    #     'graphMutationChance': 0.1,
    #     'contractionCrossChance': 0.02,
    #     'actionCrossChance': 0.02,
    #     'crossChance': 0.5,
    #
    #     'randomInit': False
    # }
    
    GASetting = {
        'nGenesPerPool': 128,
        'nSurvivedMin': 32,  # actually is max
        'nGensPerPool': 4,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTentacle',
        
        'graphRandomInit': True,
        'contractionActionRandomInit': True,
        
        'contractionMutationChance': 0.1,
        'actionMutationChance': 0.1,
        'graphMutationChance': 0.8,
        
        'contractionCrossChance': 0.08,
        'actionCrossChance': 0.08,
        'crossChance': 0.4,
    }
    
    ga = GA(GASetting=GASetting)
    ga.run()
    
elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    # ga.run()
    
    
#     38 is good
#     moo = ga.elitePool[38]['moo']
#     moo.saveAll('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainPillBug_2023_10_23_no_intersection_forward_lower/data', 'pillbug')
    
    
    
elif mode == "load":
    ga = GA(GACheckpointDir=GACheckpointDir)
    print('genePool')
    ga.logPool(ga.genePool, printing=True, showAllGenes=True, showRValue=True)
    print('elitePool')
    ga.logPool(ga.elitePool, printing=True, showAllGenes=True, showRValue=True)

    genes = []
    for gene in ga.elitePool:
        if 2 < gene['score'][0] < 4 and gene['score'][1] > -200 and \
                gene['score'][2] > 0.5 and \
                gene['score'][3] > -1.0 and gene['score'][4] > -0.01:
                # gene['score'][5] > 0.7:
            genes.append(gene)
    print(genes)
    
    
elif mode == "configMOO":
    mooDict = readMooDict('scripts/trainTable_2022-12-09_5-51/data/')
    moo = MOO(mooDict=mooDict)
    
    


