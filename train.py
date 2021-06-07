import os

import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm, EvolutionConfig
from tqdm import tqdm
from utils import visualizeActions, getModel, getActions, parseArgs

# consts
args = parseArgs()
visualize = args.visualize
testing = args.testing
numWorkers = args.nWorkers
numGeneration = args.nGen
numPopulation = args.nPop
numChannels = args.numChannels
numActions = args.numActions
numStepsPerActionMultiplier = args.numStepsPerActionMultiplier
inFile = args.inFile

model = getModel(inFile)

def criterionMoveForward(actions, nStepsPerAction):
    m = getModel(inFile)
    actions = np.array(actions).reshape(m.numChannels, -1)
    
    m.inflateChannel = actions[-1]
    m.initializePos()
    v0 = m.v.copy()
    m.v0 = v0
    
    for action in actions:
        m.inflateChannel = action
        m.step(nStepsPerAction)
       
    v = m.v.copy()
    dx = v.mean(0)[0] - v0.mean(0)[0]
    
    v = v - v.mean(0)[0]
    dot = (v * v0).sum(1)
    
    print("criterion: ", dx * 100, dot.mean() )
    
    return dx * 100 + dot.mean()

def loadConfig(model, config):
    """
    load the configuration (edgeChannel, maxContractionLevel) to the model
    :param model: the model
    :param config: np.array, [numEdge + numEdge, ], concatenate([edgeChannel, maxContractionLevel]),
        edgeChannel: {0, 1, ... model.numChannels - 1}
        maxContraction: {0, 1, ... Model.contractionLevels - 1}
    """
    edgeChannel = np.array(config[:int(len(config) / 2)], dtype=int)
    maxContractionLevel = np.array(config[int(len(config) / 2):], dtype=int)
    
    model.edgeChannel = edgeChannel
    model.maxContraction = maxContractionLevel * Model.contractionInterval
    
def simulate(config, actionSeq):
    """
    simulate a trajectory given one configuration and one sequence of actions
    
    :param config: np.array, [numEdge + numEdge, ], concatenate([edgeChannel, maxContractionLevel]),
        edgeChannel: {0, 1, ... model.numChannels - 1}
        maxContraction: {0, 1, ... Model.contractionLevels - 1}
    :param actionSeq: np.array int, [numChannels, numActions], {0, 1}
    :return vs: np.array float, [numActions + 1, numVs, 3], trajectory of vertex locations at different actuation time
    """
    m = getModel(inFile)
    m.numChannels = numChannels
    m.numActions = numActions
    actions = np.array(actionSeq).reshape(numChannels, -1)
    
    loadConfig(m, config)
    m.inflateChannel = actions[-1]
    m.initializePos()
    m.v0 = m.v.copy()
    
    vs = list()
    vs.append(m.v.copy())
    for action in actionSeq:
        m.inflateChannel = action
        v = m.step(Model.numStepsPerActuation)
        vs.append(v)
    return vs


def criterionActionSeqs(config, actionSeqs, targets, weights, subWeightss):
    """
    :param config: np.array, [numEdge + numEdge, ], concatenate([edgeChannel, maxContractionLevel]),
        edgeChannel: {0, 1, ... model.numChannels - 1}
        maxContraction: {0, 1, ... Model.contractionLevels - 1}:
    :param actionSeqs: np.array, [numActionSeqs, numChannel, numActions]  action sequences
    :param targets: list of functions, [numTargets == numActionSeqs, ]
    :param weights: np.array float, [numTargets, ], weight of targets
    :param subWeightss: list of float, [numTargets, numSubTargets]
    :return: sum of fitness scores of all targets
    """
    
    # optimize actions (targets)
    fitness = 0
    for i in range(len(actionSeqs)):
        vs = simulate(config, actionSeqs[i])
        target = targets[i]
        targetFitness = 0
        for j in range(len(target)):
            subtarget = target[j]
            targetFitness += subWeightss[i][j] * subtarget(vs)
        
        fitness += weights[i] * targetFitness
    return fitness

def optimizeActionSeqs(config, actionSeqs, targets, weights, subWeightss):
    criterion = lambda ass: criterionActionSeqs(config, ass, targets, weights, subWeightss)

    eaass = EvolutionAlgorithm(name="lobster2", lb=np.zeros(nDigits), ub=np.ones(nDigits), criterion=criterion,
                            nWorkers=args.nWorkers,
                            nPop=numPopulation, pop=[actionSeqs],
                            mortality=0.2, pbCross=0.5, pbMut=0.05, pbCrossDig=0.05, pbMutDig=0.05, lenConverge=40)
    ea.maximize(3, False)


eaconf = EvolutionAlgorithm(name="test", lb=np.zeros(nDigits), ub=np.ones(nDigits), criterion=criterion,
                        nWorkers=args.nWorkers,
                        nPop=numPopulation,
                        mortality=0.2, pbCross=0.5, pbMut=0.05, pbCrossDig=0.05, pbMutDig=0.05, lenConverge=40)

nDigits = model.numChannels * numActions
criterion = lambda actions: criterionMoveForward(actions, Model.numStepsPerActuation)

ea = EvolutionConfig(name="lobster2", lb=np.zeros(nDigits), ub=np.ones(nDigits), criterion=criterion,
                        nWorkers=args.nWorkers,
                        nPop=numPopulation,
                        mortality=0.2, pbCross=0.5, pbMut=0.05, pbCrossDig=0.05, pbMutDig=0.05, lenConverge=40)

ea.maximize(numGeneration, False)

if visualize:
    ea.showHistory()
    visualizeActions(model, ea.pop[0], True)
    model.exportJSON()

