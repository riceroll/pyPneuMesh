import numpy as np
import json
import multiprocessing

from src.utils import readNpy
from src.Model import Model
from src.Graph import Graph
from src.GA import GA

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
