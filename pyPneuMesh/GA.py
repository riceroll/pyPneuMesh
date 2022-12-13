import os
import json
import datetime
import pathlib
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
from typing import List
import logging
from pyPneuMesh.MOO import MOO
import pickle
import copy
import time
from pyPneuMesh.utils import readMooDict, readNpy


# region functions: multi-objective genetic algorithm NSGA-II

# endregion

class GA(object):
    def __init__(self, GASetting=None, GACheckpointDir=None):
        if GASetting:
            GASetting, genePool, elitePool, nPoolsTrained, secondsPassed, checkpointFolderPath \
                = self.__initFromGASetting(GASetting)
        
        else:   # use GACheckpointDir
            GASetting, genePool, elitePool, nPoolsTrained, secondsPassed, checkpointFolderPath \
                = self.__initFromGACheckpointDir(GACheckpointDir)
        
        # load GASetting
        self.GASetting = GASetting
        self.nGenesPerPool = GASetting['nGenesPerPool']
        self.nGensPerPool = GASetting['nGensPerPool']
        self.nSurvivedMin = GASetting['nSurvivedMin']
        self.contractionMutationChance = GASetting['contractionMutationChance']
        self.actionMutationChance = GASetting['actionMutationChance']
        self.graphMutationChance = GASetting['graphMutationChance']
        self.contractionCrossChance = GASetting['contractionCrossChance']
        self.actionCrossChance = GASetting['actionCrossChance']
        self.crossChance = GASetting['crossChance']
        self.randomInit = GASetting['randomInit']       # randomize the init truss or follow the init truss
        
        self.nWorkers = GASetting['nWorkers']
        self.folderPath = pathlib.Path(GASetting['folderDir'])
        
        # init constants
        self.dataFolderPath = self.folderPath.joinpath('data')
        self.mooDict = readMooDict(str(self.dataFolderPath))
        
        self.checkpointFolderPath = checkpointFolderPath
        
        # init variables
        self.nPoolsTrained = nPoolsTrained
        self.secondsPassed = secondsPassed
        
        self.genePool = genePool
        self.elitePool = elitePool


    @staticmethod
    def __initFromGASetting(GASetting):
        GASetting = GASetting.copy()
        nPoolsTrained = 0
        secondsPassed = 0
        genePool = []
        elitePool = []
    
        folderPath = pathlib.Path(GASetting['folderDir'])
        startTime = datetime.datetime.now()
        nowStr = startTime.strftime("%Y-%m-%d_%H-%M-%S")
        checkpointFolderPath = folderPath.joinpath('output').joinpath(nowStr)
        checkpointFolderPath.mkdir(parents=True, exist_ok=True)
        return GASetting, genePool, elitePool, nPoolsTrained, secondsPassed, checkpointFolderPath
    
    @staticmethod
    def __initFromGACheckpointDir(GACheckpointDir):
        gaCheckpoint = readNpy(GACheckpointDir)
        if 'nSurvivedMax' in gaCheckpoint['GASetting']:
            print('nSurvivedMax')
            gaCheckpoint['GASetting']['nSurvivedMin'] = gaCheckpoint['GASetting']['nSurvivedMax']
            del gaCheckpoint['GASetting']['nSurvivedMax']
        
        GASetting = gaCheckpoint['GASetting']
        nPoolsTrained = gaCheckpoint['nPoolsTrained']
        secondsPassed = gaCheckpoint['secondsPassed']
        
        genePoolMOODict = gaCheckpoint['genePoolMOODict']
        elitePoolMOODict = gaCheckpoint['elitePoolMOODict']
        
        genePool = [
            {'moo': MOO(mooDict=geneMooDict['mooDict'], randomize=False),
             'score': geneMooDict['score']}
            for geneMooDict in genePoolMOODict]
        elitePool = [
            {'moo': MOO(mooDict=eliteMOODict['mooDict'], randomize=False),
             'score': eliteMOODict['score']}
            for eliteMOODict in elitePoolMOODict]
        
        checkpointFolderPath = pathlib.Path(GACheckpointDir).parent
        return GASetting, genePool, elitePool, nPoolsTrained, secondsPassed, checkpointFolderPath
        
    def __initializeLogger(self):
        logging.getLogger().handlers = []
        
        logging.basicConfig(
            filename=str(self.checkpointFolderPath.joinpath('log.txt')),
            filemode='a', format='%(message)s',
            level=logging.INFO
        )
        console = logging.StreamHandler()
        logging.getLogger().addHandler(console)
        
        now = datetime.datetime.now()
        logging.info(now.strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def logLine(character):
        terminal_width = os.get_terminal_size().columns
        logging.info(character * terminal_width)

    def logTrainingTime(self):
        m, s = divmod(int(self.secondsPassed), 60)
        h, m = divmod(m, 60)
        logging.info('Training time: {}:{}:{}'.format(h, m, s))

    def logObjectives(self):
        objectives = self.mooDict['objectives']
        objectiveStrings = []
        for key in objectives:
            objective = objectives[key]
            subObjectiveStrings = ['{:^20.20s}'.format(subObjective) for subObjective in objective['subObjectives']]
            objectiveString = ''.join(subObjectiveStrings)
            objectiveStrings.append(objectiveString[:-1])
        objectivesString = '|'.join(objectiveStrings)

        logging.info("{:<20.20s}{}".format('', objectivesString))
        
    def logPool(self, pool, name=None,
                printing=False,
                showAllGenes=False,
                showRValue=False):
        lines = []
        if showAllGenes:
            if showRValue:
                Rs, _ = self.getRCD(np.array([gene['score'] for gene in pool]))
            else:
                Rs = None
                
            for iGene, gene in enumerate(pool):
                scoresStr = ['{:<20.6s}'.format(str(score)) for score in gene['score']]
                if Rs is not None:
                    scoresStr.append('{:<20d}'.format(Rs[iGene]))
                scoreStr = "".join(scoresStr)
                
                outputString = "{:<8d}{}".format(iGene, scoreStr)
                lines.append(outputString)

        # mean / max values
        meanScore, maxScore = self.getMeanMaxScore(pool)
        scoresStr = [
            '{:^20}'.format(
                '{:.6} / {:.6}'.format(str(meanScore[i]), str(maxScore[i]))
            )
            for i in range(len(meanScore))]
        scoreStr = "".join(scoresStr)
        scoreStr = "{:<20s}{}".format(name if name is not None else "", scoreStr)
        lines.append(scoreStr)
        
        if printing:
            for line in lines:
                print(line)
        else:
            for line in lines:
                logging.info(line)

    # def plotPool(self, pool, idsObjective):
    #     pass

    def saveCheckPoint(self):
        genePoolMOODict = [
            {
                'mooDict': gene['moo'].getMooDict(),
                'score': gene['score']
            }
            for gene in self.genePool
        ]
        elitePoolMOODict = [
            {
                'mooDict': gene['moo'].getMooDict(),
                'score': gene['score']
            }
            for gene in self.elitePool
        ]
        gaCheckPoint = {
            'GASetting': self.GASetting,
            'genePoolMOODict': genePoolMOODict,
            'elitePoolMOODict': elitePoolMOODict,
            'nPoolsTrained': self.nPoolsTrained,
            'secondsPassed': self.secondsPassed
        }

        np.save(
            str(self.checkpointFolderPath.joinpath('ElitePool_{}.gacheckpoint'.format(self.nPoolsTrained - 1))),
            gaCheckPoint
        )

    @staticmethod
    def getMeanMaxScore(pool):
        scores = []
        for gene in pool:
            if gene['score'] is not None:
                scores.append(gene['score'])
        scores = np.array(scores, dtype=float)
        if len(scores):
            maxScores = np.max(scores, axis=0)
            meanScores = np.mean(scores, axis=0)
        else:
            maxScores = []
            meanScores = []

        return meanScores, maxScores
    
    def evaluate(self):
        
        def criterion(gene):
            if gene['score'] is not None:   # already calculated
                moo: MOO = gene['moo']
                score = moo.evaluate()
                
                if np.linalg.norm(score - gene['score']) > 0.001:
                    breakpoint()
                # score = gene['score']
            else:
                moo: MOO = gene['moo']
                score = moo.evaluate()
            return score

        if self.nWorkers > 1:
            with Pool(self.nWorkers) as p:
                scores = np.array(p.map(criterion, self.genePool))
        else:
            scores = np.array([criterion(gene) for gene in self.genePool])

        for i in range(len(self.genePool)):
            self.genePool[i]['score'] = scores[i]

    @staticmethod
    def getRCD(ratings: np.ndarray) -> (np.ndarray, np.ndarray):
        if len(ratings) == 0:
            return np.array([]), np.array([])
        # get R
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
        
        # get CD
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

        return Rs, CDs

    def select(self):
        scores = [gene['score'] for gene in self.genePool]
        Rs, CDs = self.getRCD(np.array(scores))

        indices_sorted = np.lexsort((-CDs, Rs))
        self.genePool = [self.genePool[i] for i in indices_sorted]
        # nGenesR0 = (Rs == 0).sum()
        self.genePool = self.genePool[:self.nSurvivedMin]

    def refillGenePoolByMutationCrossRegeneration(self):
        while len(self.genePool) < self.nGenesPerPool:
            nGenesSurviving = len(self.genePool)
            nGenesToClone = min(self.nGenesPerPool - nGenesSurviving, nGenesSurviving)
            nGenesToInitialize = self.nGenesPerPool - (nGenesToClone + nGenesSurviving)
            
            # mutation or cross
            for i in range(nGenesToClone):
                if np.random.random() < self.crossChance:
                    iMoo0 = np.random.randint(nGenesToClone)
                    iMoo1 = np.random.randint(nGenesToClone)
                    mooDict0 = self.genePool[iMoo0]['moo'].getMooDict()
                    mooDict1 = self.genePool[iMoo1]['moo'].getMooDict()
                    moo0 = MOO(mooDict=mooDict0, randomize=False)
                    moo1 = MOO(mooDict=mooDict1, randomize=False)
                    moo0.cross(moo1,
                               contractionCrossChance=self.contractionCrossChance,
                               actionCrossChance=self.actionCrossChance
                               )
                    geneNew = {'moo': moo0, 'score': None}
                else:
                    iMoo = np.random.randint(nGenesToClone)
                    mooExisting: MOO = self.genePool[iMoo]['moo']
                    moo = MOO(mooDict=mooExisting.getMooDict(), randomize=False)
                    
                    moo.mutate(
                        graphMutationChance=self.graphMutationChance,
                        actionMutationChance=self.actionMutationChance,
                        contractionMutationChance=self.contractionMutationChance
                    )
                    geneNew = {'moo': moo, 'score': None}
                self.genePool.append(geneNew)
                
            
            # initialize
            for i in range(nGenesToInitialize):
                if self.randomInit:
                    moo = MOO(mooDict=self.mooDict, randomize=True)
                else:
                    moo = MOO(mooDict=self.mooDict, randomize=False)
                    moo.mutate(
                        graphMutationChance=self.graphMutationChance,
                        actionMutationChance=self.actionMutationChance,
                        contractionMutationChance=self.contractionMutationChance
                    )
                geneNew = {'moo': moo, 'score': None}
                self.genePool.append(geneNew)

    def collectElites(self):
        nEmptySlotsInElitePool = self.nGenesPerPool - len(self.elitePool)
        nMove = min(nEmptySlotsInElitePool, len(self.genePool))
        self.elitePool += self.genePool[:nMove]
        self.genePool = self.genePool[nMove:]

    def elitePoolFull(self):
        if len(self.elitePool) >= self.nGenesPerPool:
            return True
        return False

    def elitePool2genePool(self):
        self.genePool = self.elitePool[:self.nGenesPerPool]
        self.elitePool = []
        
        
    def run(self):
        self.__initializeLogger()
        
        while True:
            t0 = time.time()
            self.logLine("=")
            logging.info("Training ElitePool: {}".format(self.nPoolsTrained))
            self.logLine("=")
            self.logObjectives()
            
            for iGen in range(self.nGensPerPool):
                self.refillGenePoolByMutationCrossRegeneration()
                self.evaluate()
                self.select()
                
                print(len(self.genePool))
                self.logPool(self.genePool, "gen:{:>4d}".format(iGen), showAllGenes=True, showRValue=True)
            
            self.collectElites()
            if self.elitePoolFull():
                logging.info('elitePoolFull')
                self.elitePool2genePool()

            self.nPoolsTrained += 1
            self.secondsPassed += time.time() - t0
            self.saveCheckPoint()
            logging.info('')
            self.logTrainingTime()
            self.logPool(self.genePool, "GenePool:{:>4d}".format(self.nPoolsTrained - 1), showRValue=True)
            self.logPool(self.elitePool, 'ElitePool:{:<4d}'.format(self.nPoolsTrained - 1))

            
            
            
            


