import pathlib

from utils.utils import readNpy
from utils.Model import Model
from utils.MultiMotion import MultiMotion
from utils.MultiObjective import MultiObjective
from utils.Graph import Graph

class MOO(object):
    def __init__(self,
                 multiObjective: MultiObjective = None, graph: Graph = None,
                 mooDict: dict = None,
                 randomize=False):
        
        if multiObjective is not None:  # initialize with multiObjective and graph
            self.multiObjective = multiObjective
            self.graph = graph
            self.multiMotion = multiObjective.multiMotion
            self.model = self.graph.model
            
        else:
            trussParam = mooDict['trussParam'].copy()
            simParam = mooDict['simParam'].copy()
            actionSeqs = mooDict['actionSeqs'].copy()
            objectives = mooDict['objectives']
            graphSetting = mooDict['graphSetting']
            
            self.model = Model(trussParam, simParam)
            self.multiMotion = MultiMotion(actionSeqs, self.model)
            self.multiObjective = MultiObjective(objectives, self.multiMotion)
            self.graph = Graph(graphSetting, self.model)

        if randomize:
            self.randomize()
            
    def saveAll(self, folderDir, name):
        self.multiObjective.save(folderDir, name)
        self.graph.saveGraphSetting(folderDir, name)
        self.multiMotion.save(folderDir, name)
        self.model.save(folderDir, name)
    
    def getMooDict(self):
        mooDict = {
             'trussParam': self.model.getTrussParam(),
             'simParam': self.model.getSimParam(),
             'actionSeqs': self.multiMotion.getActionSeqs(),
             'objectives': self.multiObjective.getObjectives(),
             'graphSetting': self.graph.getGraphSetting(),
        }
        return mooDict
    
    def randomize(self):
        self.multiMotion.randomize()
        self.graph.randomize()
        
    def mutate(self, contractionMutationChance, actionMutationChance):
        self.graph.mutate(contractionMutationChance)
        self.multiMotion.mutate(actionMutationChance)
    
    def evaluate(self):
        return self.multiObjective.evaluate()
    
    def animate(self, *args, **kwargs):
        self.multiMotion.animate(*args, **kwargs)

