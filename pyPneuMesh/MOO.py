import pathlib

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective
from pyPneuMesh.Graph import Graph

class MOO(object):
    def __init__(self,
                 multiObjective: MultiObjective = None, graph: Graph = None,
                 mooDict: dict = None,
                 graphRandomize=False, contractionRandomize=False, actionRandomize=False):
        
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
            

        self.randomize(graph=graphRandomize, contraction=contractionRandomize, action=actionRandomize)
            
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
    
    def randomize(self, graph=True, contraction=True, action=True):
        
        self.graph.randomize(graphRandomize=graph, contractionRandomize=contraction)
        
        if action:
            self.multiMotion.randomize()
    
        
    def mutate(self, graphMutationChance, contractionMutationChance, actionMutationChance):
        self.graph.mutate(
            graphMutationChance=graphMutationChance,
            contractionMutationChance=contractionMutationChance
        )
        self.multiMotion.mutate(actionMutationChance)
    
    def cross(self, moo, contractionCrossChance, actionCrossChance):
        self.graph.cross(moo.graph, contractionCrossChance)
        self.multiMotion.cross(moo.multiMotion, actionCrossChance)
    
    def evaluate(self):
        return self.multiObjective.evaluate()
    
    def animate(self, *args, **kwargs):
        self.multiMotion.animate(*args, **kwargs)


