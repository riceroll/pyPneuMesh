from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.MOO import MOO
from pyPneuMesh.MultiMotion import MultiMotion

mooDict = readMooDict('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data')
trussParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data/lobster_grab.trussparam.npy')
simParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data/lobster.simparam.npy')

v0 = mooDict['trussParam']['v0']
v0_new = v0 - v0.mean(0)





mooDict['trussParam']['v0'] = v0_new

trussParam['v0'] = v0_new

model = Model(trussParam, simParam)


# moo = MOO(mooDict=mooDict)
