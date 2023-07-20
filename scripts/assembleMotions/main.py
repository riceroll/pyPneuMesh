import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA


mode = "start"
mode = "continue"
mode = "load"
# mode = "configMOO"

mooFolder = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/assembleMotions/data/"
mooDict = readMooDict(mooFolder)
moo = MOO(mooDict=mooDict)

a0 = moo.multiMotion.actionSeqs[0]
a1 = moo.multiMotion.actionSeqs[1]
a2 = moo.multiMotion.actionSeqs[2]
a3 = moo.multiMotion.actionSeqs[3]

actionSeq = np.vstack([a0, a0, a1, a1, a1, a0, a2, a2, a2, a2, a0, a3, a3, a3, a3])

moo.multiMotion.actionSeqs.append(actionSeq)
moo.multiMotion.animate(4, 1, 4)


