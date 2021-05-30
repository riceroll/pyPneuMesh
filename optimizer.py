import os
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import json
rootPath = os.path.split(os.path.realpath(__file__))[0]

class EvolutionAlgorithm:
    def __init__(self, name, lb, ub, criterion=None, nWorkers=None, nPop=100, nHero=1,
                 mortality=0.9, pbCross=0.5, pbMut=0.04, pbCrossDig=0.05, pbMutDig=0.05,
                 lenConverge=20):
        
        self.name = name
        self.nWorkers = nWorkers
        self.criterion = criterion

        self.lb = lb
        self.ub = ub
        self.nStages = 4
        self.interval = (self.ub - self.lb) / self.nStages

        self.nPop = nPop
        self.mortality = mortality
        self.nHero = nHero
        self.pbCross = pbCross
        self.pbMut = pbMut
        self.pbCrossDig = pbCrossDig
        self.pbMutDig = pbMutDig
        self.lenConverge = lenConverge

        # variables
        self.oldPop = None
        self.pop = None
        self.hero = []
        self.fits = None
        self.fitsHero = []
        self.nGen = None
        self.preTrained = False     # if True, a gene will be loaded into population under ``load`` function
        self.policyName = None
        
        # statistics
        self.history = {
            'genes': [],
            'fits': [],
            'min': [],
            'max': [],
            'mean': []
        }

        self.reset()

    def reset(self):
        self.nGen = 0
        self.oldPop = None
        self.initPop()
        
    def load(self, popDir):
        # load a pre-trained gene into population
        self.preTrained = True
        self.policyName = popDir
        p = np.load(popDir, allow_pickle=True)
        assert(self.pop[0].shape == p.shape)
        self.pop[0] = p

    def _generatePop(self, n):
        # both bounds are included
        return np.random.randint(self.lb, self.ub + 1, size=(n, len(self.lb)))

    # GA operations
    def initPop(self):
        self.pop = self._generatePop(self.nPop)  # [nPop x len(lb)]
        self.fits = np.zeros(self.nPop)

    def evaluate(self, disp=False):
        """
        use self.criterion to evaluate the fitness of all pops
        :param disp: if true, print out the fitness
        :return: self.fits: numpy.array [nPop, ] fitness of all pops
        """
        pops = [p for p in self.pop]
    
        with Pool(self.nWorkers if self.nWorkers else multiprocessing.cpu_count()) as p:
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
        self.history['genes'] = self.pop.tolist()
        self.history['fits'] = self.fits.tolist()
        
        if self.nGen % 5 == 0:
            outFileName = "{}/output/{}_g{}_f{:.8f}".format(rootPath, self.name, self.nGen, self.fits.max())
            with open(outFileName, 'w') as ofile:
                js = json.dumps(self.history)
                ofile.write(js)
        
        return self.fits
        
    def sort(self):
        # sort pops based on their fitness
        order = np.array(np.argsort(self.fits)[::-1], dtype=np.int64)
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def shuffle(self):
        # shuffle pops and their fitness
        order = np.array(np.random.permutation(len(self.pop)), dtype=np.int64)
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def select(self, extinction=True):
        """
        only preserve a part of the gene
        :param extinction: if true, kill and preserve hero if no progress for lenConverge generations
        :return:
        """
        
        fitMin = np.min(self.fits)
        fitMax = np.max(self.fits)
        fitInterval = fitMax - fitMin + 1e-8
        distance = (fitMax - self.fits) / fitInterval  # normalized distance of the fitness to the max fitness
        pbDie = distance ** 3 * self.mortality
        pbDie[:self.nHero] = 0
        
        dice = np.random.rand(len(self.pop))
        maskSurvived = np.invert(dice < pbDie)
        self.pop = self.pop[maskSurvived]
        self.oldPop = np.copy(self.pop)
        self.fits = self.fits[maskSurvived]

        noProgress = True
        recentMaxFits = self.history['max'][-self.lenConverge:]
        for m in recentMaxFits:
            if m != recentMaxFits[0]:
                noProgress = False

        if noProgress and extinction and len(recentMaxFits) >= self.lenConverge:
            print('kill')
            self.hero += self.pop[:self.nHero].tolist()
            self.fitsHero += self.fits[:self.nHero].tolist()
            self.initPop()

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

        newPop = self._generatePop(self.pop.shape[0])
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
        
    # GA maximize
    def maximize(self, nSteps=1, disp=True):
        self.initPop()
        if self.preTrained:
            self.load(self.policyName)
            
        self.history = {
            'genes': [],
            'fits': [],
            'min': [],
            'max': [],
            'mean': []
        }
        
        self.evaluate(True)
        self.sort()
        for i in range(nSteps):
            self.nGen += 1
            self.select(extinction=True)
            self.crossOver()
            self.mutate()
            self.regenerate()
            self.evaluate(True)
        
        if disp:
            self.showHistory()
        
        if len(self.hero) > 0:
            self.pop = np.vstack([self.pop, np.array(self.hero)])
            self.fits = np.hstack([self.fits, np.array(self.fitsHero)])
            self.sort()
            
        return self.pop[0]

if __name__ =="__main__":
    
    answer = "00000000000000000000"
    answer = np.array([0 for char in answer])
    answer_fake = "0001001000110101100111110"
    answer_fake = np.array([1 for char in answer])
    
    def criterion(guess):
        assert(len(guess) == len(answer))
        assert(len(guess) == len(answer_fake))
        fit = (guess == answer).sum() / len(answer)
        fit_fake = (guess == answer_fake).sum() / len(answer) * 0.9
        return max(fit, fit_fake)
    
    ea = EvolutionAlgorithm(name="test", lb=np.zeros(len(answer)), ub=np.ones(len(answer)), criterion=criterion,
                            nWorkers=8,
                            nPop=48,
                            mortality=0.2, pbCross=0.5, pbMut=0.05, pbCrossDig=0.05, pbMutDig=0.05, lenConverge=40)
    
    geneSet = ea.maximize(300, True)