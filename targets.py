import numpy as np


class Targets(object):
    def __init__(self, model):
        self.targets = []       # [target1: [subtarget1, subtarget2], target2: [subtarget1]]
        self.model = model
    
    def numTargets(self):
        return len(self.targets)
    
    @staticmethod
    def locomotion(vs):
        v = vs[-1]
        x = v.mean(0)[0]
        y = abs(v.mean(0)[1])
        return x - y * 10
    
    @staticmethod
    def heightConstraint(vs):
        zMaxes = []
        for v in vs:
            zMaxes.append(v.max(0)[2])
        zMax = max(zMaxes)
        return -zMax
    
    def extractGene(self, geneSet, iTarget):
        [nEdgeChannel, nMaxContraction, nScript] = self.model.geneSetSize()
        edgeChannel = geneSet[0:nEdgeChannel]
        geneSet = geneSet[nEdgeChannel:]
        maxContraction = geneSet[:nMaxContraction]
        geneSet = geneSet[nMaxContraction:]
        for i in range(iTarget):
            geneSet = geneSet[nScript:]
        script = geneSet[:nScript]
        return np.concatenate([edgeChannel, maxContraction, script])
        
    def criterion(self, geneSet):
        fitness = 0
        for iTarget, target in enumerate(self.targets):
            gene = self.extractGene(geneSet, iTarget)
            vs = self.model.iter(gene)
            targetFitness = 0
            for iSubtarget, subtarget in enumerate(target):
                targetFitness += subtarget(vs)
            fitness += targetFitness
        return fitness

