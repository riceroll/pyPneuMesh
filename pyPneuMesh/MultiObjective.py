import pathlib
import numpy as np
import copy
from pyPneuMesh import subObjectives

class MultiObjective(object):
    def __init__(self, objectives, multiMotion):
        self.objectives = copy.deepcopy(objectives)
        self.multiMotion = multiMotion
    
    def save(self, folderDir, name):
        folderPath = pathlib.Path(folderDir)
        objectivesPath = folderPath.joinpath("{}.objectives".format(name))
        np.save(str(objectivesPath), self.objectives)
    
    def getObjectives(self):
        return copy.deepcopy(self.objectives)
    
    def evaluate(self):
        assert(len(self.objectives) == len(self.multiMotion.actionSeqs))
        
        scores = []
        for i in range(len(self.objectives)):
            vs = self.multiMotion.simulate(i, self.objectives[i]['numLoop'])
            
            for j in range(len(self.objectives[i]['subObjectives'])):
                subObjectiveName = self.objectives[i]['subObjectives'][j]
                subObjective = getattr(subObjectives, 'obj{}{}'.format(subObjectiveName[0].upper(), subObjectiveName[1:]))
                score = subObjective(vs)
                scores.append(score)
        scores = np.array(scores, np.float64)
        return scores

