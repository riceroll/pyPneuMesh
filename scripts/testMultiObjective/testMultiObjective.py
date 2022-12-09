import numpy as np

from utils.utils import readNpy
from utils.Model import Model
from utils.MultiMotion import MultiMotion
from utils.MultiObjective import MultiObjective

trussParam = readNpy('examples/testMultiObjective/table/table.trussparam.npy')
simParam = readNpy('examples/testMultiObjective/table/table.simparam.npy')
actionSeqs = readNpy('examples/testMultiObjective/table/table.actionseqs.npy')
objectives = readNpy('examples/testMultiObjective/table/table.objectives.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)
mo = MultiObjective(objectives, mm)

# mm.actionSeqs[0] *= 0
mm.actionSeqs[1] *= 0
mm.actionSeqs[2] *= 0
mm.actionSeqs[3] *= 0

scores = mo.evaluate()



