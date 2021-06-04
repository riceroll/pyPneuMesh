import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import json
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targets import Targets
from tqdm import tqdm
from utils import visualizeActions, getModel, getActions


m = getModel("./data/lobster3.json")
actions = getActions(m, "./output/records/control_only/lobster2_g25_f111.78788591")
m.inflateChannel = np.array([0, 0, 0, 0])
m.numChannels = 4
# model.initializePos()

# model.script = np.array([[0, 0, 1, 0]])

# visualizeActions(np.array([[0,0,0,0]]), loop=True)

visualizeActions(m, actions, loop=True)
