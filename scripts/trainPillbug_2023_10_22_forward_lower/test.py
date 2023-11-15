from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.JointGenerator import JointGenerator
from pyPneuMesh.MOO import MOO
import numpy as np


dataFolderDir = '/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainPillbug_2023_10_21/data/'
mooDict = readMooDict(dataFolderDir)
moo = MOO(mooDict=mooDict)
#
# moo.model.NUM_CONTRACTION_LEVEL = 4
# moo.model.CONTRACTION_PER_LEVEL = 0.011
# moo.model.MAX_ACTIVE_BEAM_LENGTH = 0.135
# moo.model.CONTRACTION_SPEED = 0.010
# moo.model.ACTION_TIME = 5.0
#
# v0 = moo.model.v0
# v00 = v0[moo.model.e[:, 0]]
# v01 = v0[moo.model.e[:, 1]]
# meanLength = np.sqrt(((v00 - v01) ** 2 ).sum(1)).mean()
#
# moo.model.v0 = moo.model.v0 / meanLength * (moo.model.MAX_ACTIVE_BEAM_LENGTH - moo.model.CONTRACTION_PER_LEVEL * 3)
# moo.model.edgeChannel *= 0
# moo.model.edgeActive[:] = True
# moo.model.contractionLevel *= 0
# moo.model.contractionLevel += 3
#
# moo.model.save('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainPillbug_2023_10_21/data', 'pillbug')

