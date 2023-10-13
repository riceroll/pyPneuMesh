import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_64/output/date/ElitePool_970.gacheckpoint.npy"



ga = GA(GACheckpointDir=GACheckpointDir)

pool = ga.elitePool
if len(ga.elitePool) == 0:
    pool = ga.genePool


moo = pool[0]['moo']
model = moo.model

moo.multiMotion.saveAnimation(folderDir='/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_64/data', name='table64', iAction=0, numLoop=1)