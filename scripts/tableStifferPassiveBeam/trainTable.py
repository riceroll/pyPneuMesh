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
mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-51/output/gcp_2022-12-09_12-19-30/ElitePool_115.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 128,
        'nSurvivedMin': 32,     # actually is max
        'nGensPerPool': 6,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/tableStifferPassiveBeam',

        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.05,
        'graphMutationChance': 0.0,
        'contractionCrossChance': 0.02,
        'actionCrossChance': 0.02,
        'crossChance': 0.5,
        
        'graphRandomInit': False,
        'contractionActionRandomInit': True,
    }
    ga = GA(GASetting=GASetting)
    ga.run()
    
elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    # ga.run()
    
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
    mooDict = readMooDict('scripts/tableStifferPassiveBeam/data')
    moo = MOO(mooDict=mooDict)
    
    


