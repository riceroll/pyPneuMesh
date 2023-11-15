import pathlib
import numpy as np
import copy
from pyPneuMesh.Model import Model

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
            
    def cross(self, multiMotion, chance):
        for i, actionSeq in enumerate(self.actionSeqs):
            maskMutation = np.random.rand(actionSeq.shape[0], actionSeq.shape[1]) < chance
            tmp = self.actionSeqs[i][maskMutation].copy()
            self.actionSeqs[i][maskMutation] = multiMotion.actionSeqs[i][maskMutation]
            multiMotion.actionSeqs[i][maskMutation] = tmp
        
    def simulate(self, iAction, numLoop, retForce=False):
        actionSeq = self.actionSeqs[iAction]
        actionSeq = np.vstack([actionSeq] * numLoop)
        # actionSeq = np.vstack([np.zeros(actionSeq.shape[1]), actionSeq])
        
        assert(actionSeq.shape[1] >= self.model.getNumChannel())
        
        times, lengths = self.model.actionSeq2timeNLength(actionSeq)
        totalTime = times[-1] + self.model.ACTION_TIME
        numSteps = int(totalTime / self.model.h)
        
        Vs, Fs = self.model.step(numSteps, times, lengths, retForce=retForce)
        
        Vs = Vs.reshape(numSteps + 1, -1, 3)
        Fs = Fs.reshape(numSteps + 1, -1)
        return Vs, Fs
    
        
    def animate(self, iAction, numLoop, speed=1.0, singleColor=True):
        vs, fs = self.simulate(iAction, numLoop)
        self.model.animate(vs, speed=speed, singleColor=singleColor)
        return vs
    
    def saveAnimation(self, folderDir, name, iAction, numLoop):
        Vs, Fs = self.simulate(iAction, numLoop, retForce=True)
        self.model.animate(Vs, speed=1, singleColor=True)
        
        data = {
            'Vs': Vs,
            'Fs': Fs,
            'E': self.model.e,
            'edgeChannel': self.model.edgeChannel,
            'h': self.model.h,
        }
        folderPath = pathlib.Path(folderDir)
        animationPath = folderPath.joinpath("{}.animation".format(name))
        np.save(str(animationPath), data)
        return data


