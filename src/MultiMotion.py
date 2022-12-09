import pathlib
import numpy as np
import copy
from src.Model import Model

class MultiMotion(object):
    def __init__(self, actionSeqs, model):
        actionSeqs = copy.deepcopy(actionSeqs)
        self.actionSeqs = [actionSeqs[key] for key in actionSeqs]
        self.model = model
    
    def save(self, folderDir, name):
        folderPath = pathlib.Path(folderDir)
        actionSeqsPath = folderPath.joinpath("{}.actionseqs".format(name))
        actionSeqs = self.getActionSeqs()
        np.save(str(actionSeqsPath), actionSeqs)
    
    def getActionSeqs(self):
        return { i: self.actionSeqs[i].copy() for i in range(len(self.actionSeqs))}
    
    def randomize(self):
        for i, actionSeq in enumerate(self.actionSeqs):
            self.actionSeqs[i] = np.random.randint(np.zeros_like(actionSeq), np.ones_like(actionSeq) * 2)
    
    def mutate(self, chance):
        for i, actionSeq in enumerate(self.actionSeqs):
            actionSeqRand = np.random.randint(np.zeros_like(actionSeq), np.ones_like(actionSeq) * 2)
            maskMutation = np.random.rand(actionSeq.shape[0], actionSeq.shape[1]) < chance
            self.actionSeqs[i][maskMutation] = actionSeqRand[maskMutation]
    
    def simulate(self, iAction, numLoop):
        actionSeq = self.actionSeqs[iAction]
        actionSeq = np.vstack([actionSeq] * numLoop)
        
        assert(actionSeq.shape[1] >= self.model.getNumChannel())
        
        times, lengths = self.model.actionSeq2timeNLength(actionSeq)
        totalTime = times[-1] + self.model.ACTION_TIME
        numSteps = int(totalTime / self.model.h)
        vs = self.model.step(numSteps, times, lengths)
        return vs
    
    def animate(self, iAction, numLoop, speed=1.0):
        vs = self.simulate(iAction, numLoop)
        self.model.animate(vs, speed=speed, singleColor=True)
        return vs

