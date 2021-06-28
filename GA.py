import os
import json
import datetime
from pathlib import Path
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
from typing import List

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


def getR(ratings: np.ndarray) -> (np.ndarray):
    ratings = ratings.reshape(len(ratings), -1)
    ratingsCol = ratings.reshape([ratings.shape[0], 1, ratings.shape[1]])
    ratingsRow = ratings.reshape([1, ratings.shape[0], ratings.shape[1]])
    dominatedMatrix = (ratingsCol <= ratingsRow).all(2) * (ratingsCol < ratingsRow).any(2)

    Rs = np.ones(len(ratings), dtype=int) * -1
    R = 0
    while -1 in Rs:
        nonDominated = (~dominatedMatrix[:, np.arange(len(Rs))[Rs == -1]]).all(1) * (Rs == -1)
        Rs[nonDominated] = R
        R += 1
        
    return Rs
    
    
def getCD(ratings: np.ndarray, Rs: np.ndarray) -> (np.ndarray):
    ratings = ratings.reshape(len(ratings), -1)
    CDMatrix = np.zeros_like(ratings, dtype=np.float)
    sortedIds = np.argsort(ratings, axis=0)
    CDMatrix[sortedIds[0], np.arange(len(ratings[0]))] = np.inf
    CDMatrix[sortedIds[-1], np.arange(len(ratings[0]))] = np.inf
    ids0 = sortedIds[:-1, :]
    ids1 = sortedIds[1:, :]
    distances = ratings[ids1, np.arange(len(ratings[0]))] - ratings[ids0, np.arange(len(ratings[0]))]
    if ((ratings.max(0) - ratings.min(0)) > 0).all():
        CDMatrix[sortedIds[1:-1, :], np.arange(len(ratings[0]))] = \
            (distances[1:] + distances[:-1]) / (ratings.max(0) - ratings.min(0))
    else:
        CDMatrix[sortedIds[1:-1, :], np.arange(len(ratings[0]))] = np.inf
    CDs = CDMatrix.mean(1)
    
    return CDs


def getRCD(ratings: np.ndarray) -> (np.ndarray, np.ndarray):
    ratings = ratings.reshape(len(ratings), -1)
    Rs = getR(ratings)
    CDs = getCD(ratings, Rs)
    return Rs, CDs

def evaluate(pop, criterion, nWorkers):
    """
    evaluate
    :param pop: np.ndarray, int, [nPop, len(pop[0])]
    :param criterion: func(pop[0]), criterion function
    :param nWorkers: int
    :return:
        ratings: np.ndarray, float, [nPop, ]
    """
    
    if nWorkers == 1:
        ratings = []
        for p in pop:
            ratings.append(criterion(p))
        ratings = np.array(ratings)
    else:
        with Pool(nWorkers if nWorkers != -1 else multiprocessing.cpu_count()) as p:
            ratings = np.array(p.map(criterion, pop))
            
    Rs, CDs = getRCD(ratings)
    
    return ratings, Rs, CDs

def sortPop(pop, ratings, Rs, CDs):
    assert(type(pop) == type(ratings) == type(Rs) == type(CDs) == np.ndarray)
    Cs = 1 / (CDs + 1e-3)       # crowding
    idsSorted = np.lexsort([Cs, Rs])
    newPop = pop[idsSorted]
    newRatings = ratings[idsSorted]
    newRs = Rs[idsSorted]
    newCDs = CDs[idsSorted]
    return newPop, newRatings, newRs, newCDs

def select(pop, ratings, Rs, CDs, surviveRatio, tournamentSize=2):
    """
    tournament selection, assuming sorting is done
    :param pop:
    :param ratings:
    :return:
    """
    nSurvive = int(np.ceil(len(pop) * surviveRatio))
    assert(nSurvive < len(pop))
    idsPop = list(np.arange(len(pop)))  # numbering all population
    idsSurvived = []                    # ids of survived pops
    for iSurvive in range(nSurvive):
        idsGroup = np.random.choice(idsPop, tournamentSize, replace=False)   # select a group
        
        if iSurvive == 0 and 0 not in idsGroup and ratings.ndim == 1:  # make sure the best one is selected
            idsGroup[0] = 0
        
        idWinner = idsGroup.min()
        idsSurvived.append(idWinner)
        idsPop.remove(idWinner)
    idsSurvived = np.array(idsSurvived)
    idsSurvived.sort()
    
    newPop = pop[idsSurvived]
    newRatings = ratings[idsSurvived]
    newRs = Rs[idsSurvived]
    newCDs = CDs[idsSurvived]
    
    return newPop, newRatings, newRs, newCDs

def cross(pop, nPop, crossRatio, crossGenePercent):
    """
    
    :param pop: np.ndarray, int, [nPop, n]
    :param nPop: int, size of initial population
    :param crossRatio: float, percentage of population to be generated by crossOver
    :param crossGenePercent:
    :return:
    """
    assert(nPop > 0)
    nCross = max(int((int(nPop * crossRatio) / 2)), 1)
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
    assert(newPop.shape[0] <= nPop)
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

def loadHistory(historyDir):
    """
    used with `plot(history)`
    """
    with open(historyDir) as iFile:
        js = iFile.read()
        history = GeneticAlgorithm.History()
        history.loadJSON(js)
        return history

def plot(history):
    import matplotlib.pyplot as plt
    
    if np.array(history.ratingsBest).ndim == 1:
        ids = np.arange(len(history.ratingsBest))
        plt.plot(ids, history.ratingsBestHero)
        plt.plot(ids, history.ratingsBest)
        plt.plot(ids, history.ratingsMean)
        
        fitsBest = np.array(history.ratingsBest)
        fitsBestExtinctions = fitsBest[history.iExtinctions]
        fitsBestRevivals = fitsBest[history.iRevivals]
        plt.plot(history.iExtinctions, fitsBestExtinctions, 'o', color='red')
        plt.plot(history.iRevivals, fitsBestRevivals, 'o', color='green')
        plt.show()
    
    if np.array(history.ratingsBest).ndim == 2:
        ax = plt.axes(projection='3d')
        ids = np.arange(len(history.ratingsBest))
        # ax.scatter3D(np.array(history.ratingsBestHero)[:, 0], np.array(history.ratingsBestHero)[:, 1], ids, color='red', marker='.')
        ax.scatter3D(np.array(history.ratingsBest)[:, 0], np.array(history.ratingsBest)[:, 1], ids, color='green', marker='.')
        ax.scatter3D(np.array(history.ratingsMean)[:, 0], np.array(history.ratingsMean)[:, 1], ids, color='blue', marker='.')
        # ax.plot(ga.ratings[:, 0], ga.ratings[:, 1], 'o', color='purple' )
        ax.scatter3D(np.zeros(len(history.iExtinctions)), np.zeros(len(history.iExtinctions)), history.iExtinctions, color='pink', marker='o')
        # ax.scatter3D(0, 0, history.iExtinctions, color='pink', marker='o')
        
        for i in range(len(history.heroes)):
            if i % 100 == 0:
                ax.scatter3D(history.ratingsHero[i][:, 0], history.ratingsHero[i][:, 1], i, color='black', marker='o')
                
        
        plt.show()

class GeneticAlgorithm(object):
    class Setting:
        def __init__(self):
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
            self.saveHistory = None
            
        def load(self, newSetting):
            assert(type(newSetting) is dict)
            for key in newSetting:
                assert(hasattr(self, key))
                setattr(self, key, newSetting[key])
        
        def data(self):
            keys = [attr for attr in dir(self) if
                    not callable(getattr(self, attr)) and not attr.startswith("__")]
            
            return {key: getattr(self, key) for key in keys}
        
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
                'nWorkers': -1,
                'plot': True,
                'mute': False,
                'saveHistory': True,
            }
            return setting
            
    class History:
        def __init__(self):
            self.ratingsBestHero = []
            self.ratingsBest = []
            self.ratingsMean = []
            self.genesBest = []
            self.iExtinctions = []
            self.iRevivals = []
            self.heroes = []
            self.ratingsHero = []
        
        def toJSON(self):
            history = {
                'ratingsBestHero': np.array(self.ratingsBestHero).tolist(),
                'ratingsBest': np.array(self.ratingsBest).tolist(),  # best fitness at the current generation
                'ratingsMean': np.array(self.ratingsMean).tolist(),  # mean fitness at the current generation
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
        self.ratings = []
        self.Rs = []
        self.CDs = []
        
        self.heroes = []
        self.ratingsHero = []
        self.criterion = None

        self.lb = lb
        self.ub = ub
        self.criterion = criterion
        
        self.history = self.History()
        self.startTime = None
        self.folderDir = None
        
        assert (self.lb.shape == self.ub.shape)
        assert (self.lb.ndim == self.ub.ndim == 1)
        assert (self.lb.dtype == self.ub.dtype == int)
        
    @staticmethod
    def getDefaultSetting():
        return GeneticAlgorithm.Setting.getDefaultSetting()
    
    def loadSetting(self, setting):
        self.setting.load(setting)
    
    def log(self, extinct, reviving):
        ratingBest = self.ratings[0]
        popBest = self.pop[0]
        
        Rs, CDs = getRCD(np.array(self.ratingsHero))
        heroes, rs, _, _ = sortPop(np.array(self.heroes), np.array(self.ratingsHero), Rs, CDs)
        ratingBestHero = rs[0].copy()
        ratingMean = np.mean(self.ratings, 0)
        
        self.history.ratingsBest.append(ratingBest)
        self.history.genesBest.append(popBest)
        
        self.heroes += self.pop[self.Rs == 0].tolist()
        self.ratingsHero += self.ratings[self.Rs == 0].tolist()
        self.disasterHero()
        
        self.history.ratingsBestHero.append(ratingBestHero)
        self.history.ratingsMean.append(ratingMean)
        
        self.history.ratingsHero.append(np.array(self.ratingsHero).copy())
        self.history.heroes.append(np.array(self.heroes).copy())
        
        iGen = len(self.history.ratingsBest) - 1
        if extinct:
            self.history.iExtinctions.append(iGen)
        if reviving:
            self.history.iRevivals.append(iGen)
        
        if not self.setting.mute:
            appendix = ratingBestHero
            if type(appendix) is np.ndarray:
                appendix = ', '.join(["{:.2f}".format(i) for i in appendix.tolist()])
            ratingBestHero = appendix

            appendix = ratingBest
            if type(appendix) is np.ndarray:
                appendix = ', '.join(["{:.2f}".format(i) for i in appendix.tolist()])
            ratingBest = appendix
            
            print('gen: {}\tfbh: {}\tfb: {}'.format(iGen, ratingBestHero, ratingBest))
        
    def saveHistory(self, iGen, appendix:np.ndarray):
        folderPath = self.folderDir
        Path(folderPath).mkdir(parents=True, exist_ok=True)
        if type(appendix) is np.ndarray:
            appendix = ','.join(["{:.2f}".format(i) for i in appendix.tolist()])
        with open(folderPath + '/g{}_{}.hs'.format(iGen, appendix), 'w') as oFile:
            js = self.history.toJSON()
            oFile.write(js)
            
    def loadHistory(self, historyDir: str = ""):
        if historyDir == "":
            fileNames = os.listdir(self.folderDir)
            historyDir = os.path.join(self.folderDir, sorted(fileNames)[-1])

        with open(historyDir) as iFile:
            js = iFile.read()
            self.history.loadJSON(js)
            
    def getHeroes(self, historyDir: str = ""):
        self.loadHistory(historyDir)
        heroes = self.history.heroes
        ratingsHero = self.history.ratingsHero
        Path(historyDir).mkdir(parents=True, exist_ok=True)
        fileDirs = []
        for i in range(len(heroes)):
            fileNameSeqs = []
            for score in ratingsHero[i]:
                fileNameSeqs.append("{.2f}".format(float(score)))
            fileName = "-".join(fileNameSeqs) + "_tg{}.json".format(i)
            fileDir = os.path.join(historyDir, fileName)
            fileDirs.append(fileDir)
        return heroes, fileDirs
            
    def sort(self):
        self.pop, self.ratings, self.Rs, self.CDs = \
            sortPop(pop=self.pop, ratings=self.ratings, Rs=self.Rs, CDs=self.CDs)
    
    def select(self):
        self.pop, self.ratings, self.Rs, self.CDs = \
            select(pop=self.pop, ratings=self.ratings, Rs=self.Rs,
                   CDs=self.CDs, surviveRatio=self.setting.surviveRatio)
            
    def cross(self):
        self.pop = \
            cross(pop=self.pop, nPop=self.setting.nPop,
                  crossRatio=self.setting.crossRatio, crossGenePercent=self.setting.crossGenePercent)

    def mutate(self):
        self.pop = \
            mutate(pop=self.pop, lb=self.lb, ub=self.ub,
                   mutateRatio=self.setting.mutateRatio, mutateGenePercent=self.setting.mutateGenePercent)

    def regenerate(self):
        self.pop = \
            regenerate(pop=self.pop, nPop=self.setting.nPop, lb=self.lb, ub=self.ub)

    def evaluate(self):
        self.ratings, self.Rs, self.CDs = \
            evaluate(pop=self.pop, criterion=self.criterion,
                     nWorkers=self.setting.nWorkers)
    
    def disasterHero(self):
        Rs, CDs = getRCD(np.array(self.ratingsHero))
        self.heroes, self.ratingsHero, Rs, CDs = sortPop(np.array(self.heroes), np.array(self.ratingsHero), Rs, CDs)
        
        iSurvived = Rs == 0
        self.heroes = np.array(self.heroes)[iSurvived][:10]
        self.ratingsHero = np.array(self.ratingsHero)[iSurvived][:10]
        
        self.heroes = [np.array(h) for h in self.heroes]
        self.ratingsHero = self.ratingsHero.tolist()
    
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
        ratingsBest = np.array(self.history.ratingsBest)
        iGen = len(ratingsBest)
        if len(ratingsBest) >= self.setting.lenEra:
            if (ratingsBest[-self.setting.lenEra:] == ratingsBest[-1]).all():
                if len(self.history.iRevivals) == 0 or \
                        len(self.history.iRevivals) > 0 and \
                        iGen - self.history.iRevivals[-1] > self.setting.lenEra:
                    extinct = True
                    nExtinctions += 1
                    if nExtinctions % self.setting.nEraRevive == 0:
                        reviving = True
        
        if extinct:
            if not (self.pop[0] == np.array(self.heroes)).all(1).any():
                self.heroes.append(self.pop[0])
                self.ratingsHero.append(self.ratings[0])
                self.disasterHero()
            
            self.pop = initPop(nPop=self.setting.nPop, lb=self.lb, ub=self.ub)
            self.ratings, self.Rs, self.CDs = evaluate(pop=self.pop, criterion=self.criterion,
                                                       nWorkers=self.setting.nWorkers)
                       
            if reviving:
                self.pop[:len(self.heroes)] = self.heroes
                self.ratings[:len(self.heroes)] = self.ratingsHero

        self.sort()

        if not self.setting.mute:
            string = ""
            if extinct:
                string += "Extinction "
            if reviving:
                string += "Revival"
            print(string)
        
        return extinct, reviving, nExtinctions
        
    def getBest(self):
        return self.heroes, self.ratingsHero
    
    def getFront(self):
        ids = self.Rs == 0
        return self.pop[ids][0], self.ratings[ids][0]
        
    def run(self):
        self.startTime = t = datetime.datetime.now()
        t = self.startTime
        folderName = 'GA_{}{}-{}:{}:{}'.format(t.month, t.day, t.hour, t.minute, t.second)
        self.folderDir = './output/' + folderName
        
        if not self.setting.mute:
            print("\nBegin at {}.".format(self.startTime))
            data = self.setting.data()
            for key in data:
                print("{}: {}".format(key, data[key]))
        
        self.pop = initPop(nPop=self.setting.nPop, lb=self.lb, ub=self.ub)
        self.ratings, self.Rs, self.CDs = evaluate(pop=self.pop, criterion=self.criterion,
                                                   nWorkers=self.setting.nWorkers)
        self.sort()
        self.heroes.append(self.pop[0])
        self.ratingsHero.append(self.ratings[0])
        
        nExtinctions = 0
        for iGen in range(self.setting.nGenMax):
            extinct, reviving, nExtinctions = self.disaster(nExtinctions=nExtinctions)

            self.select()
            self.cross()
            self.mutate()
            self.regenerate()
            self.evaluate()
            self.sort()
            
            self.log(  extinct=extinct, reviving=reviving)
            if iGen % 5 == 0 and self.setting.saveHistory:
                self.saveHistory(iGen=iGen, appendix=self.history.ratingsBestHero[-1])
        
        if self.setting.plot:
            try:
                plot(self.history)
            except Exception as e:
                print(e)
        
        return self.getBest()

# testings ========================
def testCross(argv):
    pop = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ])

    newPop = cross(pop=pop, nPop=10, crossRatio=0.1, crossGenePercent=0.66)
    # print(newPop)
    assert (newPop.shape == (6, 3))
    assert ((newPop[:4] == pop).all())
    assert (not (newPop[4] == newPop[5]).any())
    assert (newPop[4, 0] in pop[:, 0] and newPop[5, 0] in pop[:, 0] and newPop[4, 1] in pop[:, 1])
    assert (newPop[4].tolist() not in pop.tolist())

def testMutate(argv):
    pop = np.zeros([5, 3])
    lb = np.array([2, 2, 2])
    ub = np.array([3, 3, 3])

    newPop = mutate(pop, lb, ub, mutateRatio=1.0, mutateGenePercent=1.0)
    # print(newPop)
    assert (newPop.shape == (5, 3))
    assert (((newPop == 0) + (newPop == 2)).all())

    pop = np.zeros([20, 10])
    lb = np.ones(10, dtype=int) * 2
    ub = np.ones(10, dtype=int) * 3
    newPop = mutate(pop, lb, ub, mutateRatio=0.5, mutateGenePercent=0.66)

    assert (newPop.shape == (20, 10))
    assert (((newPop == 0) + (newPop == 2)).all())
    assert (not ((newPop == 0) * (newPop == 2)).all())

def testRegenerate(argv):
    pop = np.zeros([5, 3])
    lb = np.zeros(3, dtype=int)
    ub = np.ones(3, dtype=int)

    newPop = regenerate(pop, 10, lb, ub)
    assert (newPop.shape == (10, 3))

def testGetR(argv):
    ratings = np.array([
        [0, 5, 0],
        [5, 5, 4],
        [4, 5, 5],
        [5, 4, 4],
        [2, 2, 3],
        [0, 0, 0],
        [1, 5, 0],
        [4, 4, 3],
    ])
    RsTrue = np.array([2, 0, 0, 1, 3, 4, 1, 2], dtype=int)
    Rs = getR(ratings)
    assert((Rs == RsTrue).all())

def testGetCD(argv):
    F = np.load('./test/data/F.npy')
    cdsTruth = np.load('./test/data/cds.npy')
    Rs, CDs = getRCD(F)
    assert((Rs == 0).all())
    assert((CDs == cdsTruth).all())

def testSortNSelect(argv):
    # 1
    pop = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ])  # random
    ratings = np.array([0.5, 0.3, 0.1, 0, 0.4]) # random
    Rs = np.array([0, 0, 0, 1, 0])
    CDs = np.array([1, 0.4, 0.3, 1, 0.8])
    
    pop, ratings, Rs, CDs = sortPop(pop, ratings, Rs, CDs)
    assert( (pop == np.array([
        [0, 1, 2],
        [12, 13, 14],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ])).all())
    assert ((ratings == [0.5, 0.4, 0.3, 0.1, 0]).all())
    assert ((Rs == [0, 0, 0, 0, 1]).all())
    assert ((CDs == [1, 0.8, 0.4, 0.3, 1]).all())
    for i in range(50):
        newPop, newRatings, newRs, newCDs = select(pop=pop, ratings=ratings, Rs=Rs, CDs=CDs,
                                                    surviveRatio=0.6)
        assert (newPop.shape == (3, 3))
    
    # 2
    ratings = np.load('./test/data/F.npy')
    cdsTruth = np.load('./test/data/cds.npy')
    pop = np.zeros(len(ratings))    # random
    Rs, CDs = getRCD(ratings)
    newPop, newRatings, newRs, newCDs = sortPop(pop, ratings, Rs, CDs)
    idsSorted = np.argsort(cdsTruth)[::-1]
    assert ((newRatings == ratings[idsSorted]).all())
    
    # 3
    pop, ratings, Rs, CDs = newPop, newRatings, newRs, newCDs
    for i in range(50):
        newPop, newRatings, newRs, newCDs = select(pop=pop, ratings=ratings, Rs=Rs, CDs=CDs,
                                                   surviveRatio=0.99)
        assert(len(newPop) == 99)
        assert((newRatings == ratings[:-1]).all())
        
def testGA1D(argv):
    nDigs = 16
    z = np.zeros(nDigs)
    zFake = np.ones(nDigs)
    criterion = lambda x: max((x == z).sum() * 2 * ((x == z).sum() > 10), (x == zFake).sum())
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
        'mutateRatio': 0.1,
        'mutateGenePercent': 0.05,
        'lenEra': 40,
        'nEraRevive': 2,
    
        'nGenMax': 200,
        'nWorkers': 1,
        'plot': 'plot' in argv,
        'mute': 'unmute' not in argv,
        'saveHistory': False,
    }
    assert (setting.keys() == defaultSetting.keys())

    ga.loadSetting(setting)
    pop, ratings = ga.run()
    assert((pop[0] == 0).all())


def testGA2D(argv):
    nDigs = 16
    z = np.zeros([nDigs, 2])
    zFake = np.ones([nDigs, 2])
    
    def criterion(x):
        x = x.reshape(-1, 2)
        return (((x == zFake).sum(0) > 10) * (x == zFake).sum(0) * 10 + (x == z).sum(0))
        
    # criterion = lambda x : (x == z).sum()
    lb = np.zeros(nDigs * 2, dtype=int)
    ub = np.ones(nDigs * 2, dtype=int) * 2
    
    ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)
    defaultSetting = ga.getDefaultSetting()
    setting = {
        'nPop': 48,
        'surviveRatio': 0.6,
        'crossRatio': 0.4,
        'crossGenePercent': 0.05,
        'mutateRatio': 0.2,
        'mutateGenePercent': 0.05,
        'lenEra': 100,
        'nEraRevive': 4,
        
        'nGenMax': 800,
        'nWorkers': 1,
        'plot': 'plot' in argv,
        'mute': 'unmute' not in argv,
        'saveHistory': False,
    }
    assert (setting.keys() == defaultSetting.keys())
    
    ga.loadSetting(setting)
    pop, ratings = ga.run()
    if 'unmute' in argv:
        print(pop[0].reshape(-1, 2), criterion(pop[0]))
    assert((pop[0] == 1).all())
    

tests = {
    'cross': testCross,
    'mutate': testMutate,
    'regenerate': testRegenerate,
    'getR': testGetR,
    'getCD': testGetCD,
    'sortNSelect': testSortNSelect,
    'ga1d': testGA1D,
    'ga2d': testGA2D,
    }

def testAll(argv):
    for key in tests:
        print('test{}{}():'.format(key[0].upper(), key[1:]))
        tests[key](argv)
        print('Pass.\n')

def test():
    import sys
    if 'test' in sys.argv:
        if 'all' in sys.argv:
            testAll(sys.argv)
        else:
            for key in tests:
                if key in sys.argv:
                    print('test{}{}():'.format(key[0].upper(), key[1:]))
                    tests[key](sys.argv)
                    print('Pass.\n')
  
if __name__ == "__main__":
    # eg. run GA test all plot unmute
    
    test()