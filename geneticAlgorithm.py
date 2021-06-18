import json
import datetime
from pathlib import Path
import numpy as np

def initPop(nPop, lb, ub):
    """
    initialize population
    :param nPop: int, size of population
    :param lb: np.ndarray, int, [n, ], lower bound (inclusive)
    :param ub: np.ndarray, int, [n, ], upper bound (non-inclusive)
    :return: pop: np.ndarray, int, [nPop, n]
    """
    assert(type(lb) == type(ub) == np.ndarray)
    assert(lb.dtype == ub.dtype == int)
    assert(lb.ndim == ub.ndim == 1)
    
    pop = np.random.randint(lb, ub, [nPop, len(lb)])
    return pop

def evaluate(pop, criterion, nWorkers):
    """
    evaluate
    :param pop: np.ndarray, int, [nPop, len(pop[0])]
    :param criterion: func(pop[0]), criterion function
    :param nWorkers: int
    :return:
        fits: np.ndarray, float, [nPop, ]
    """
    fits = []
    
    # TODO: write a parallel version
    
    for p in pop:
        fits.append(criterion(p))
    
    fits = np.array(fits)
    
    return fits

def sortPop(pop, fits):
    assert(type(pop) == type(fits) == np.ndarray)
    idsSorted = np.argsort(fits)
    idsSorted = idsSorted[::-1]
    newPop = pop[idsSorted]
    newFits = fits[idsSorted]
    return newPop, newFits

def select(pop, fits, surviveRatio, tournamentSize=2):
    """
    tournament selection
    :param pop:
    :param fits:
    :return:
    """
    pop = pop.copy()
    fits = fits.copy()
    newPop = []
    newFits = []
    nSurvive = int(np.ceil(len(pop) * surviveRatio))
    assert(nSurvive < len(pop))
    idsPop = list(np.arange(len(pop)))
    for iSurvive in range(nSurvive):
        ids = np.random.choice(idsPop, 2, replace=False)
        if iSurvive == 0 and 0 not in ids:  # keep the best one
            ids[0] = 0
        idMax = ids[fits[ids].argmax()]
        newPop.append(pop[idMax])
        newFits.append(fits[idMax])
        idsPop.remove(idMax)
    newPop = np.array(newPop, dtype=int)
    newFits = np.array(newFits)
    
    newPop, newFits = sortPop(newPop, newFits)
    return newPop, newFits

def cross(pop, nPop, crossRatio, crossGenePercent):
    """
    
    :param pop: np.ndarray, int, [nPop, n]
    :param nPop: int, size of initial population
    :param crossRatio: float, percentage of population to be generated by crossOver
    :param crossGenePercent:
    :return:
    """
    assert(nPop > 0)
    nCross = int(np.ceil(nPop * crossRatio))
    nCrossDig = int(np.ceil(len(pop[0]) * crossGenePercent))
    idsDad = np.random.choice(np.arange(0, len(pop)), [nCross, ], replace=False)
    idsMom = np.random.choice(np.arange(0, len(pop)), [nCross, ], replace=False)
    
    # remove duplicated parents
    idsIdentical = idsDad == idsMom
    idsDad[idsIdentical] += np.random.randint(1, len(pop) - 1, [idsIdentical.sum(), ])
    idsDad = idsDad % len(pop)
    
    iDigRow = np.arange(nCross).reshape(-1, 1)
    iDigCol = np.random.randint(0, len(pop[0]), [nCross, nCrossDig])    # duplicate possible
    
    sons = pop[idsDad].copy()
    sons[iDigRow, iDigCol] = pop[idsMom][iDigRow, iDigCol]
    daughters = pop[idsMom].copy()
    daughters[iDigRow, iDigCol] = pop[idsDad][iDigRow, iDigCol]
    
    newPop = np.vstack([pop, sons, daughters])
    return newPop

def mutate(pop, lb, ub, mutateRatio, mutateGenePercent):
    nPop = len(pop)
    nMutate = int(np.ceil(nPop * mutateRatio))
    nMutateDig = int(np.ceil(nPop * mutateGenePercent))
    
    nNoMutation = 1     # top n will not be mutated
    if nMutate > len(pop) - nNoMutation:
        nMutate = len(pop) - nNoMutation
    ids = np.random.choice(np.arange(nNoMutation, len(pop)), [nMutate], replace=False)    # don't mutate the best
    
    iDigRow = np.arange(nMutate).reshape(-1, 1)
    iDigCol = np.random.randint(0, len(pop[0]), [nMutate, nMutateDig])      # duplicate possible
    
    newPop = pop.copy()
    genesMutate = newPop[ids]
    genesMutate[iDigRow, iDigCol] = np.random.randint(lb[iDigCol], ub[iDigCol])
    newPop[ids] = genesMutate
    return newPop
    
def regenerate(pop, nPop, lb, ub):
    """
    
    :param pop: np.ndarray, int, [len(pop), len(pop[0])]
    :param nPop: int, expected size of population
    :return: newPop: np.ndarray, int, [nPop, len(pop[0])]
    """
    assert(type(pop) is np.ndarray and pop.ndim == 2)
    assert(nPop > len(pop))
    generatedPop = initPop(nPop - len(pop), lb, ub)
    newPop = np.vstack([pop, generatedPop])
    return newPop

def plot(history):
    import matplotlib.pyplot as plt
    ids = np.arange(len(history.fitsBest))
    plt.plot(ids, history.fitsBestHero)
    plt.plot(ids, history.fitsBest)
    plt.plot(ids, history.fitsMean)

    fitsBest = np.array(history.fitsBest)
    fitsBestExtinctions = fitsBest[history.iExtinctions]
    fitsBestRevivals = fitsBest[history.iRevivals]
    plt.plot(history.iExtinctions, fitsBestExtinctions, 'o', color='red')
    plt.plot(history.iRevivals, fitsBestRevivals, 'o', color='green')
    plt.show()


class GeneticAlgorithm(object):
    class Setting:
        def __init__(self):
            self.data = dict()
            self.nPop = None
            self.lenEra = None
            self.nEraRevive = None
            self.surviveRatio = None
            self.crossRatio = None
            self.crossGenePercent = None
            self.mutateRatio = None
            self.mutateGenePercent = None
            
            self.nGenMax = None
            self.nWorkers = None
            self.mute = None
            self.plot = None
            
        def load(self, newSetting):
            assert(type(newSetting) is dict)
            for key in newSetting:
                assert(hasattr(self, key))
                setattr(self, key, newSetting[key])
            self.data = newSetting
        
        @staticmethod
        def getDefaultSetting():
            setting = {
                'nPop': 48,
                'surviveRatio': 0.6,
                'crossRatio': 0.4,
                'crossGenePercent': 0.05,
                'mutateRatio': 0.05,
                'mutateGenePercent': 0.05,
                'lenEra': 20,
                'nEraRevive': 4,
        
                'nGenMax': 200,
                'nWorkers': 8,
                'plot': True,
                'mute': False,
            }
            return setting
            
    class History:
        def __init__(self):
            self.fitsBestHero = []
            self.fitsBest = []
            self.fitsMean = []
            self.genesBest = []
            self.iExtinctions = []
            self.iRevivals = []
        
        def toJSON(self):
            history = {
                'fitsBestHero': np.array(self.fitsBestHero).tolist(),
                'fitsBest': np.array(self.fitsBest).tolist(),  # best fitness at the current generation
                'fitsMean': np.array(self.fitsMean).tolist(),  # mean fitness at the current generation
                'genesBest': np.array(self.genesBest).tolist(),  # best gene at the current generation
                'iExtinctions': np.array(self.iExtinctions).tolist(),     # ids of generation of extinctions
                'iRevivals': np.array(self.iRevivals).tolist()     # ids of generation of revivals
            }
            js = json.dumps(history)
            return js
        
        def loadJSON(self, js):
            history = json.loads(js)
            for key in history:
                assert(hasattr(self, key))
                value = [np.array(v) for v in history[key]]
                setattr(self, key, value)
                
    def __init__(self, criterion=None, lb=None, ub=None, setting=None):
        # load default setting
        self.setting = self.Setting()
        self.loadSetting(self.getDefaultSetting())
        if setting is not None:
            self.loadSetting(setting)
            
        self.pop = []
        self.fits = []
        self.heroes = []
        self.fitsHero = []
        self.criterion = None

        self.lb = lb
        self.ub = ub
        self.criterion = criterion
        
        self.history = self.History()
        
        assert (self.lb.shape == self.ub.shape)
        assert (self.lb.ndim == self.ub.ndim == 1)
        assert (self.lb.dtype == self.ub.dtype == int)
        
    @staticmethod
    def getDefaultSetting():
        return GeneticAlgorithm.Setting.getDefaultSetting()
    
    def loadSetting(self, setting):
        self.setting.load(setting)
    
    def log(self, extinct, reviving):
        fitBest = max(self.fits)
        
        popBest = self.pop[np.argmax(self.fits)]
        fitBestHero = np.max(self.fitsHero)
        fitMean = np.mean(self.fits)
        self.history.fitsBest.append(fitBest)
        self.history.genesBest.append(popBest)
        if fitBest > fitBestHero:
            self.heroes.append(popBest)
            self.fitsHero.append(fitBest)
        self.history.fitsBestHero.append(fitBestHero)
        self.history.fitsMean.append(fitMean)
        
        iGen = len(self.history.fitsBest) - 1
        if extinct:
            self.history.iExtinctions.append(iGen)
        if reviving:
            self.history.iRevivals.append(iGen)
        
        if not self.setting.mute:
            print('gen: {}\tfbh: {:.2f}\tfb: {:.2f}'.format(iGen, fitBestHero, fitBest))
        
    def saveHistory(self, startTime, iGen, appendix):
        t = startTime
        folderName = 'GA_{}{}-{}:{}:{}'.format(t.month, t.day, t.hour, t.minute, t.second)
        folderPath = './output/' + folderName
        Path(folderPath).mkdir(parents=True, exist_ok=True)
        with open(folderPath + '/g{}_{:.2f}'.format(iGen, appendix), 'w') as oFile:
            js = self.history.toJSON()
            oFile.write(js)
            
    def getBest(self, n=1):
        allPop = np.vstack([self.pop, self.heroes, self.history.genesBest])
        allFits = np.hstack([self.fits, self.fitsHero, self.history.fitsBest])
        return allPop[:n], allFits[:n]
    
    def disaster(self, nExtinctions):
        """
        :param nExtinctions:
        :return:
            extinct: true if there is an extinction
            reviving: true if there is a revival after extinction
            nExtinctions: number of extinctions happened by the end of this disaster
        """
        
        extinct = False
        reviving = False
        fitsBest = self.history.fitsBest
        iGen = len(fitsBest)
        if len(fitsBest) >= self.setting.lenEra:
            if (fitsBest[-self.setting.lenEra:] == fitsBest[-1]).all():
                if len(self.history.iRevivals) == 0 or \
                        len(self.history.iRevivals) > 0 and \
                        iGen - self.history.iRevivals[-1] > self.setting.lenEra:
                    extinct = True
                    nExtinctions += 1
                    if nExtinctions % self.setting.nEraRevive == 0:
                        reviving = True
        
        if extinct:
            self.heroes.append(self.pop[0])
            self.fitsHero.append(self.fits[0])
            self.pop = initPop(nPop=self.setting.nPop, lb=self.lb, ub=self.ub)
            self.fits = evaluate(pop=self.pop, criterion=self.criterion, nWorkers=self.setting.nWorkers)
            if reviving:
                self.pop[:len(self.heroes)] = self.heroes
                self.fits[:len(self.heroes)] = self.fitsHero

        self.pop, self.fits = sortPop(self.pop, self.fits)
        
        return extinct, reviving, nExtinctions
        
    def run(self):
        startTime = datetime.datetime.now()
        
        self.pop = initPop(nPop=self.setting.nPop, lb=self.lb, ub=self.ub)
        self.fits = evaluate(pop=self.pop, criterion=self.criterion, nWorkers=self.setting.nWorkers)
        self.pop, self.fits = sortPop(pop=self.pop, fits=self.fits)
        self.heroes.append(self.pop[0])
        self.fitsHero.append(self.fits[0])
        
        nExtinctions = 0
        for iGen in range(self.setting.nGenMax):
            extinct, reviving, nExtinctions = self.disaster(nExtinctions=nExtinctions)

            pop = self.pop.copy()
            fits = self.fits.copy()
            self.pop, self.fits = select(pop=self.pop, fits=self.fits, surviveRatio=self.setting.surviveRatio)
            self.pop = cross(   pop=self.pop, nPop=self.setting.nPop - len(self.pop),
                                crossRatio=self.setting.crossRatio, crossGenePercent=self.setting.crossGenePercent)

            self.pop = mutate(  pop=self.pop, lb=self.lb, ub=self.ub,
                                mutateRatio=self.setting.mutateRatio, mutateGenePercent=self.setting.mutateGenePercent)
            self.pop = regenerate(  pop=self.pop, nPop=self.setting.nPop, lb=self.lb, ub=self.ub)

            self.fits = evaluate(   pop=self.pop, criterion=self.criterion, nWorkers=self.setting.nWorkers)

            self.pop, self.fits = sortPop(  pop=self.pop, fits=self.fits)
            
            self.log(  extinct=extinct, reviving=reviving)
            if iGen % 5 == 0:
                self.saveHistory(startTime=startTime, iGen=iGen, appendix=self.history.fitsBestHero[-1])
        
        if self.setting.plot:
            plot(self.history)
        
        return self.getBest()

        
if __name__ == "__main__":
    def testCross():
        pop = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])
        
        newPop = cross(pop=pop, nPop=10, crossRatio=0.1, crossGenePercent=0.66)
        print(newPop)
        assert(newPop.shape == (6, 3))
        assert((newPop[:4] == pop).all())
        assert(not (newPop[4] == newPop[5]).any())
        assert(newPop[4, 0] in pop[:, 0] and newPop[5, 0] in pop[:, 0] and newPop[4, 1] in pop[:, 1])
        assert(newPop[4].tolist() not in pop.tolist())
        
    def testMutate():
        pop = np.zeros([5, 3])
        lb = np.array([2, 2, 2])
        ub = np.array([3, 3, 3])
        
        newPop = mutate(pop, lb, ub, mutateRatio=1.0, mutateGenePercent=1.0)
        print(newPop)
        assert(newPop.shape == (5, 3))
        assert (((newPop == 0) + (newPop == 2)).all())

        pop = np.zeros([20, 10])
        lb = np.ones(10, dtype=int) * 2
        ub = np.ones(10, dtype=int) * 3
        newPop = mutate(pop, lb, ub, mutateRatio=0.5, mutateGenePercent=0.66)
        
        assert (newPop.shape == (20, 10))
        assert (((newPop == 0) + (newPop == 2)).all())
        assert (not ((newPop == 0) * (newPop == 2)).all())

    def testRegenerate():
        pop = np.zeros([5, 3])
        lb = np.zeros(3, dtype=int)
        ub = np.ones(3, dtype=int)
    
        newPop = regenerate(pop, 10, lb, ub)
        assert(newPop.shape == (10, 3))

    def testSelect():
        pop = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
        ])
        fits = np.array([0.5, 0.4, 0.3, 0.1, 0])
        for i in range(50):
            newPop, newFits = select(pop, fits, surviveRatio=0.6)
            assert (newPop.shape == (3, 3))
            assert(newFits[0] == fits[0])

    def testGA():
        nDigs = 16
        z = np.zeros(nDigs)
        zFake = np.ones(nDigs)
        criterion = lambda x: max((x == z).sum() * 2 * ((x == z).sum() > 13), (x==zFake).sum())
        # criterion = lambda x : (x == z).sum()
        lb = np.zeros(nDigs, dtype=int)
        ub = np.ones(nDigs, dtype=int) * 2
        
        ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)
        defaultSetting = ga.getDefaultSetting()
        setting = {
            'nPop': 24,
            'surviveRatio': 0.6,
            'crossRatio': 0.4,
            'crossGenePercent': 0.05,
            'mutateRatio': 0.05,
            'mutateGenePercent': 0.05,
            'lenEra': 200,
            'nEraRevive': 2,
    
            'nGenMax': 200,
            'nWorkers': 8,
            'plot': False,
            'mute': True,
        }
        assert(setting.keys() == defaultSetting.keys())
        
        ga.loadSetting(setting)
        pop, fits = ga.run()
        assert(len(fits) == len(pop) == 1)
        assert(len(pop[0]) == len(lb) == len(ub))
        
    
    import sys
    if 'test' in sys.argv:
        tests = {
            'select': testSelect,
            'cross': testCross,
            'mutate': testMutate,
            'regenerate': testRegenerate,
            'ga': testGA,
        }
        
        if 'all' in sys.argv:
            for key in tests:
                print('test{}{}():'.format(key[0].upper(), key[1:]))
                tests[key]()
                print('Pass.\n')
        else:
            for key in tests:
                if key in sys.argv:
                    print('test{}{}():'.format(key[0].upper(), key[1:]))
                    tests[key]()
                    print('Pass.\n')
