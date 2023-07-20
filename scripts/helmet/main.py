import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model

trussParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/helmet/data/helmet.trussparam.npy')
simParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/helmet/data/helmet.simparam.npy')

model = Model(trussParam, simParam)

actionSeq = np.zeros([5, 4])
actionSeq[0, :] += 1
actionSeq[2, :] += 1
actionSeq[4, :] += 1

times, lengths = model.actionSeq2timeNLength(actionSeq)

vs = model.step(30000, times, lengths)


# model.show()
model.animate(vs, speed=50.0, singleColor=True)
