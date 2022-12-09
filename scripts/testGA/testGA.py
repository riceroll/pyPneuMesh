import numpy as np
import json
import multiprocessing

from utils.utils import readNpy
from utils.Model import Model
from utils.Graph import Graph
from utils.GA import GA

trussParam = readNpy('examples/testGraph/table/table.trussparam.npy')
simParam = readNpy('examples/testGraph/table/table.simparam.npy')
graphSetting = readNpy('examples/testGraph/table/table.graphsetting.npy')

nWorkers = multiprocessing.cpu_count()

GASetting = {
    'nGenesPerPool': 16,
    'nGensPerPool': 2,
    'nSurvivedMax': 8,
    
    'nWorkers': nWorkers,
    
    'folderDir': 'examples/testGA/table/',
    'name': 'table',
    
    'contractionMutationChance': 0.1,
    'actionMutationChance': 0.2
}

ga = GA(GASetting=GASetting)
ga.run()
