import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

mooDict = readMooDict('./scripts/tableExample/data/')
moo = MOO(mooDict=mooDict)


def compileActionSeqs(actionseqs, indices):
    
    newList = []
    for id in indices:
        if id != -1:
            actionSeq = actionseqs[id]
            
        else:
            actionSeq = np.zeros([1, 6])

        newList.append(actionSeq)
    return np.vstack(newList)
    
a = compileActionSeqs(moo.multiMotion.actionSeqs, [0, -1, -1, 1, 1, -1, -1,0, -1,-1, 2, 2, 2, 2, -1, -1, 3, 3, 3, 3,-1, -1])

moo.multiMotion.actionSeqs = [a]
# moo.animate(0, 1, 3)


