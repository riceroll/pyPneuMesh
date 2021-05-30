import os
import argparse
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np
from model import Model
from optimizer import EvolutionAlgorithm
from targets import Targets
from tqdm import tqdm

def getModel(inFileDir):
    model = Model()
    model.loadJson(inFileDir)
    model.scripting = False
    model.script = np.array([0])
    # model.testing = testing
    # model.reset()
    return model


model = getModel("./data/lobster3.json")
model.script = np.array([0,0,0,0])

#
# #   step 1
for i in range(20000):
    model.inflateChannel[0] = 1
    model.step(1)
    print(model.l[1])
    print(model.f[0])
    print(model.v[0])


# #   step 2
# model.inflateChannel[0] = 0
# model.step(20000)
# print(model.v[0])
