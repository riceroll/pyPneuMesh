import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
# from targetFuncs import Targets
from tqdm import tqdm
from utils import visualizeActions, getModel, getActions

# consts
parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False, help="whether to visualize the result")
parser.add_argument("--testing", type=bool, default=False, help="whether in testing mode")
parser.add_argument("--iFile", type=str, default="fox", help="name without suffix of input file under ./data folder")
parser.add_argument("--nWorkers", type=int, default=8, help="number of workers")
parser.add_argument("--nGen", type=int, default=30, help="whether in testing mode")
parser.add_argument("--nPop", type=int, default=24, help="size of population")
parser.add_argument("--numActions", type=int, default=4, help="# of channels, -1: read the # from json")
parser.add_argument("--targets", type=str, default="moveForward", help="type of target")
parser.add_argument("--numStepsPerActionMultiplier", type=str, default="0.2", help="# of steps per action")
parser.add_argument("--inFile", type=str, default="./data/apps/lobster/lobster_sym_walk_grab.json", help="infile")
args = parser.parse_args()

scripting = True
visualize = args.visualize
inFileDir = args.iFile
testing = args.testing
numWorkers = args.nWorkers
numGeneration = args.nGen if not testing else 5
numPopulation = args.nPop if not testing else 8
numActions = args.numActions
numStepsPerActionMultiplier = float(args.numStepsPerActionMultiplier)
inFile = args.inFile

model = getModel(inFile)

print("inFile: ", inFile)
print("numChannels: ", model.numChannels)
print("numActions: ", model.numActions)
print("numGenerations: ", numGeneration)
print("numPopulation: ", numPopulation)
print("numWorkers: ", numWorkers)

visualizeActions(model, model.script, True)

