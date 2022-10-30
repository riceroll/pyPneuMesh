import os
import json
import datetime
from pathlib import Path
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
from typing import List
import logging
from utils.moo import MOO
from utils.getCriterion import getCriterion
import pickle
import copy


# region functions: multi-objective genetic algorithm NSGA-II
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


# endregion

class GeneticAlgorithm(object):
    class Setting:
        def __init__(self):
            self.nGenesPerPool = None
            self.nGensPerPool = None
            self.nSurvivedMax = None

            self.nWorkers = None
            self.mute = None
            self.plot = None
            self.saveHistory = None

        def load(self, newSetting):
            assert (type(newSetting) is dict)
            for key in newSetting:
                assert (hasattr(self, key))
                setattr(self, key, newSetting[key])

        def data(self):
            keys = [attr for attr in dir(self) if
                    not callable(getattr(self, attr)) and not attr.startswith("__")]

            return {key: getattr(self, key) for key in keys}

        @staticmethod
        def getDefaultSetting():
            setting = {
                'nGenesPerPool': 8,
                'nGensPerPool': 5,
                'nSurvivedMax': 2,

                'nWorkers': -1,
                'plot': True,
                'mute': False,
                'saveHistory': True,
            }
            return setting

    def __init__(self, MOOSetting, GASetting=None):
        # load default setting
        self.setting = self.Setting()
        self.MOOSetting = MOOSetting
        self.loadSetting(self.getDefaultSetting())
        setting = GASetting
        if setting is not None:
            self.loadSetting(setting)

        self.pop = []
        self.ratings = []
        self.Rs = []
        self.CDs = []

        self.genePool = []
        self.elitePool = []

        self.heroes = []
        self.ratingsHero = []
        self.criterion = None

        self.startTime = None
        self.folderDir = None
        self.iPool = None

    @staticmethod
    def getDefaultSetting():
        return GeneticAlgorithm.Setting.getDefaultSetting()

    def loadSetting(self, setting):
        self.setting.load(setting)

    def initPoolFromScratch(self, sizePool):
        genePool = [{'moo': MOO(self.MOOSetting, randInit=True), 'score': None} for _ in range(sizePool)]
        return genePool

    def evaluate(self, genePool, nWorkers=-1):
        # for i, gene in enumerate(genePool):
        #     moo = gene['moo']
        #     criterion = getCriterion(moo)
        #     score = criterion(moo)
        #     genePool[i]['score'] = score

        def criterion(gene):
            if gene['score'] is not None:
                return gene['score']
            else:
                return getCriterion(gene['moo'])(gene['moo'])

        with Pool(nWorkers if nWorkers != -1 else multiprocessing.cpu_count()) as p:
            scores = np.array(p.map(criterion, genePool))

        for i in range(len(genePool)):
            genePool[i]['score'] = scores[i]

        return genePool

    def select(self, genePool, nSurvivedMax):
        scores = [gene['score'] for gene in genePool]
        Rs = getR(np.array(scores))
        CDs = getCD(np.array(scores), Rs)

        indices_sorted = np.lexsort((-CDs, Rs))
        genePool = [genePool[i] for i in indices_sorted]
        genePool = genePool[:nSurvivedMax]

        logging.info("{:<10} {:<15} {:<40} {:<10} {:<10}".format('i', 'address', 'score', 'R', 'CD'))

        for i in range(len(genePool)):
            logging.info(
                "{:<10} {:<15} {:<40} {:<10} {:<10}".format(i, str(genePool[i]['moo'])[-15:], str(genePool[i]['score']),
                                                            Rs[i], CDs[i]))

        return genePool

    # def select(self, genePool, nSurvivedMax):
    #     scores = [gene['score'] for gene in genePool]
    #     Rs = getR(np.array(scores))
    #     CDs = getCD(np.array(scores), Rs)
    #     idsSorted = np.lexsort((CDs, Rs))[::-1]
    #     idsSurvived = idsSorted[:nSurvivedMax]
    #     genePool = [genePool[i] for i in idsSurvived]
    #
    #     logging.info("{:<10} {:<15} {:<40} {:<10} {:<10}".format('i', 'address', 'score', 'R', 'CD'))
    #
    #     for i in range(len(genePool)):
    #         logging.info(
    #             "{:<10} {:<15} {:<40} {:<10} {:<10}".format(i, str(genePool[i]['moo'])[-15:], str(genePool[i]['score']),
    #                                                         Rs[i], CDs[i]))
    #
    #     return genePool

    def mutateAndRegenerate(self, genePool, sizePool):
        nGeneration = sizePool - len(genePool)
        while len(genePool) < sizePool:
            genePoolNew = [{'moo': copy.deepcopy(gene['moo']).mutate(), 'score': None} for gene in genePool]
            genePool += genePoolNew

        genePool = genePool[:sizePool]
        return genePool

    def addElites(self, genePoolSurvived, elitePool):
        elitePool += genePoolSurvived
        return elitePool

    def elitePoolFull(self, sizePool, elitePool):
        if len(elitePool) >= sizePool:
            return True
        return False

    def initPoolFromElites(self, elitePool, sizePool):
        genePool = elitePool[:sizePool]
        elitePool = []
        return genePool, elitePool

    def log(self, iPool, elitePool, GASetting, MOOSetting):
        fileDir = os.path.join(self.folderDir, 'iPool_{}'.format(iPool))
        data = {
            'elitePool': elitePool,
            'GASetting': GASetting,
            'MOOSetting': MOOSetting
        }

        with open(fileDir, 'wb') as oFile:
            pickle.dump(data, oFile, pickle.HIGHEST_PROTOCOL)

    def loadCheckpoint(self, checkpointDir):
        cpDir = Path(checkpointDir)
        self.folderDir = str(cpDir.parent)
        iPool = int(cpDir.name.split('_')[-1])
        self.iPool = iPool
        import pickle5
        data = pickle5.load(open(str(cpDir), 'rb'))
        elitePool = data['elitePool']
        [gene['moo'].model.configure(gene['moo'].MOOSetting.modelConfigDir) for gene in elitePool]
        self.elitePool = elitePool

    def run(self):
        self.startTime = t = datetime.datetime.now()
        t = self.startTime
        folderName = 'GA_{}{}-{}:{}:{}'.format(t.month, t.day, t.hour, t.minute, t.second)

        self.folderDir = './output/' + folderName if self.folderDir is None else self.folderDir
        if self.setting.saveHistory:
            Path(self.folderDir).mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename=os.path.join(self.folderDir, 'history.log'),
                                filemode='w', format='%(message)s',
                                level=logging.WARNING if not self.setting.saveHistory else logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING if not self.setting.saveHistory else logging.INFO)

        console = logging.StreamHandler()
        logging.getLogger().addHandler(console)

        if not self.setting.mute:
            logging.info("\nBegin at {}.".format(self.startTime))
            data = self.setting.data()
            for key in data:
                logging.info("{}: {}".format(key, data[key]))

        sizePool = self.setting.nGenesPerPool
        nGenPerPool = self.setting.nGensPerPool
        nSurvivedMax = self.setting.nSurvivedMax

        if len(self.elitePool) != 0:
            self.genePool, self.elitePool = self.initPoolFromElites(self.elitePool, sizePool)
        else:
            self.genePool = self.initPoolFromScratch(sizePool)
            self.genePool = self.evaluate(self.genePool, self.setting.nWorkers)
            self.genePool = self.select(self.genePool, nSurvivedMax)

        iPool = 0 if self.iPool is None else self.iPool

        while True:
            iPool += 1
            print('iPool: ', iPool)
            for iGen in range(nGenPerPool):
                print('iGen: ', iGen)
                self.genePool = self.mutateAndRegenerate(self.genePool, sizePool)
                self.genePool = self.evaluate(self.genePool, self.setting.nWorkers)
                self.genePool = self.select(self.genePool, nSurvivedMax)

            self.elitePool = self.addElites(self.genePool, self.elitePool)

            self.log(iPool, self.elitePool, self.setting, self.MOOSetting)

            if self.elitePoolFull(sizePool, self.elitePool):
                print('elitePoolFull')
                self.genePool, self.elitePool = self.initPoolFromElites(self.elitePool, sizePool)
            else:
                self.genePool = self.initPoolFromScratch(sizePool)

        #
        # self.pop = initPop(nPop=self.setting.nPop, lb=self.lb, ub=self.ub)
        #
        # self.ratings, self.Rs, self.CDs = evaluate(pop=self.pop, criterion=self.criterion,
        #                                            nWorkers=self.setting.nWorkers)
        #
        # self.sort()
        # self.heroes.append(self.pop[0])
        # self.ratingsHero.append(self.ratings[0])
        #
        # nExtinctions = 0
        # for iGen in range(self.setting.nGenMax):
        #     extinct, reviving, nExtinctions = self.disaster(nExtinctions=nExtinctions)
        #
        #     self.select()
        #     self.cross()
        #     self.mutate()
        #     self.regenerate()
        #     self.evaluate()
        #     self.sort()
        #
        #     self.log(  extinct=extinct, reviving=reviving)
        #     if iGen % 5 == 0 and self.setting.saveHistory:
        #         self.saveHistory(iGen=iGen, appendix=self.history.ratingsBestHero[-1])
        #
        # if self.setting.plot:
        #     # try:
        #     self.history.plot()
        #     # except Exception as e:
        #     #     print('plot', e)
        #
        #
        # return self.getBest()
