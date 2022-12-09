import pathlib
import copy
import numpy as np

from src.FullGraph import FullGraph
from src.HalfGraph import HalfGraph

class Graph(object):
    
    
    def __new__(cls, graphSetting, model):
        symmetric = graphSetting['symmetric']
        
        graphSetting = copy.deepcopy(graphSetting)
        
        if symmetric:
            return HalfGraph(model, graphSetting)
        else:
            assert(False)
            return FullGraph(model)
    
    def __init__(self, graphSetting, model):
        self.graphSetting = graphSetting.copy()
        self.model = model
        pass
    
    def saveGraphSetting(self, folderDir, name):
        pass
    
    def getGraphSetting(self):
        pass
    
    def mutate(self, chance):
        pass
    
    def randomize(self):
        pass
    
    def saveGraphSetting(self):
        pass
