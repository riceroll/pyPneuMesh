import os
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import json
rootPath = os.path.split(os.path.realpath(__file__))[0]


class Optimizer(object):
    def __init__(self, model=None):
        self.model = model

    def maximize(self):
        raise NotImplementedError('Please implement maximize function.')


class EvolutionAlgorithm(Optimizer):
    def __init__(self, model=None, criterion=None, nPop=100, nHero=1,
                 mortality=0.9, pbCross=0.5, pbMut=0.04, pbCrossDig=0.05, pbMutDig=0.05):
        super().__init__(model)
        
        self.criterion = criterion

        self.lb = self.model.lb()
        self.ub = self.model.ub()
        self.nStages = 4
        self.interval = (self.ub - self.lb) / self.nStages
        self.lbInt = self.lb
        self.ubInt = self.ub + 1

        self.nPop = nPop
        self.mortality = mortality
        self.nHero = nHero
        self.pbCross = pbCross
        self.pbMut = pbMut
        self.pbCrossDig = pbCrossDig
        self.pbMutDig = pbMutDig

        # variables
        self.oldPop = None
        self.pop = None
        self.fits = None
        self.nGen = None
        self.preTrained = False     # if True, a gene will be loaded into population under ``load`` function
        self.policyName = None
        
        # statistics
        self.history = {
            'gens': [],
            'fitss': [],
            'min': [],
            'max': [],
            'mean': []
        }

        self.reset()

    def reset(self):
        self.nGen = 0
        self.oldPop = None
        self.initPop()

        self.lb = self.model.lb()
        self.ub = self.model.ub()
        # self.interval = (self.ub - self.lb) / self.nStages
        self.lbInt = self.lb
        self.ubInt = self.ub + 1

    def survivor(self, i=0):
        return self.pop[i]

    def load(self, policyName):
        # load a pre-trained gene into population
        self.preTrained = True
        self.policyName = policyName
        name = os.path.join(rootPath, 'data/agent/', self.policyName + '.npy')
        p = np.load(name, allow_pickle=True)
        # p = self.popFloatToInt(p)
        assert(self.pop[0].shape == p.shape)
        self.pop[0] = p

    def randPop(self, n):
        return np.random.randint(self.lbInt, self.ubInt, size=(n, len(self.lbInt)))

    def initPop(self):
        self.pop = self.randPop(self.nPop)  # [nPop x len(lb)]

    # def popIntToFloat(self, pop):
    #     return pop * self.interval

    # def popFloatToInt(self, pop):
    #     pop = pop / self.interval
    #     assert( (pop == pop.astype(np.int)).all() )
    #     return pop

    def evaluate(self, disp=False):
        pops = [p for p in self.pop]
    
        with Pool(multiprocessing.cpu_count()) as p:
            self.fits = np.array(p.map(self.criterion, pops))
        
        # self.fits = np.array([self.criterion(pop) for pop in pops])

        meanFit = np.mean(self.fits)
        maxFit = np.max(self.fits)
        minFit = np.min(self.fits)
        self.sort()
        if disp:
            print('nGen: ', self.nGen)
            print('mean: ', meanFit)
            print('max: ', maxFit)
            print('min: ', minFit)
        
        self.history['min'].append(minFit)
        self.history['max'].append(maxFit)
        self.history['mean'].append(meanFit)
        self.history['gens'].append(self.pop.tolist())
        self.history['fitss'].append(self.fits.tolist())
        
        if self.nGen % 10 == 0:
            outFileName = "{}/output/gen-{}_fit-{:.8f}".format(rootPath, self.nGen, self.fits.max())
            with open(outFileName, 'w') as ofile:
                js = json.dumps(self.history)
                ofile.write(js)
        
    def sort(self):
        order = np.array(np.argsort(self.fits)[::-1], dtype=np.int64)
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def shuffle(self):
        order = np.array(np.random.permutation(len(self.pop)), dtype=np.int64)
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def select(self):
        fitMin = np.min(self.fits)
        fitMax = np.max(self.fits)
        fitInterval = fitMax - fitMin + 1e-8
        distance = (fitMax - self.fits) / fitInterval  # normalized distance of the fitness to the max fitness
        pbDie = distance ** 3 * self.mortality
        pbDie[:self.nHero] = 0

        dice = np.random.rand(len(self.pop))
        maskSurvived = np.invert(dice < pbDie)
        self.oldPop = np.copy(self.pop)
        self.pop = self.pop[maskSurvived]
        self.fits = self.fits[maskSurvived]

    def crossOver(self):
        self.shuffle()

        diceDig = np.random.rand(self.pop.shape[0] // 2, self.pop.shape[1])
        maskDig = diceDig < self.pbCrossDig
        dice = np.random.rand(self.pop.shape[0] // 2)
        mask = dice < self.pbCross
        maskDig = maskDig * mask[:, np.newaxis]
        maskDig = np.repeat(maskDig, 2, axis=0)
        maskDig0 = np.copy(maskDig)
        maskDig1 = np.copy(maskDig)
        maskDig0[np.arange(len(maskDig0))[::2]] = False
        maskDig1[np.arange(len(maskDig1))[1::2]] = False

        if len(maskDig0) < len(self.pop):
            maskDig0 = np.pad(maskDig0, ((0, 1), (0, 0)), 'constant', constant_values=((False, False), (False, False)))
            maskDig1 = np.pad(maskDig1, ((0, 1), (0, 0)), 'constant', constant_values=((False, False), (False, False)))

        self.pop[maskDig0], self.pop[maskDig1] = self.pop[maskDig1], self.pop[maskDig0]

    def mutate(self):
        diceDig = np.random.rand(self.pop.shape[0], self.pop.shape[1])
        maskDig = diceDig < self.pbMutDig
        dice = np.random.rand(self.pop.shape[0])
        mask = dice < self.pbMut
        maskDig = maskDig * mask[:, np.newaxis]

        newPop = self.randPop(self.pop.shape[0])
        self.pop[maskDig] = newPop[maskDig]

    def regenerate(self):
        nDead = self.nPop - len(self.pop)
        self.pop = np.append(self.pop, self.oldPop[:nDead], axis=0)
        self.fits = np.pad(self.fits, (0, nDead), 'wrap')
    
    def showHistory(self):
        plt.plot(np.arange(self.nGen + 1), self.history['min'])
        plt.plot(np.arange(self.nGen + 1), self.history['max'])
        plt.plot(np.arange(self.nGen + 1), self.history['mean'])
        plt.show()

    def maximize(self, nSteps=1, disp=True):
        self.initPop()
        if self.preTrained:
            self.load(self.policyName)

        self.history = {
            'gens': [],
            'fitss': [],
            'min': [],
            'max': [],
            'mean': []
        }
        
        self.evaluate(True)
        self.sort()
        for i in range(nSteps):
            self.nGen += 1
            self.select()
            self.crossOver()
            self.mutate()
            self.regenerate()
            self.evaluate(True)
        
        if disp:
            self.showHistory()
        
        return self.survivor(0)
