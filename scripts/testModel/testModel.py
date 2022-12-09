import numpy as np

from src.utils import readNpy
from src.Model import Model

trussParam = readNpy('examples/testModel/table/table.trussparam.npy')
simParam = readNpy('examples/testModel/table/table.simparam.npy')

model = Model(trussParam, simParam)

actionSeq = np.ones([3, 3])
actionSeq[1] *= 0
actionSeq[2, 1] *= 0

times, lengths = model.actionSeq2timeNLength(actionSeq)

vs = model.step(30000, times, lengths)



# model.show()
model.animate(vs, speed=1.0, singleColor=True)


