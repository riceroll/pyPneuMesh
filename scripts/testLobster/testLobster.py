from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.MOO import MOO
from pyPneuMesh.MultiMotion import MultiMotion

mooDict = readMooDict('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data')
trussParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data/Lobster_0.trussparam.npy')
simParam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/testLobster/data/lobster.simparam.npy')

v0 = mooDict['trussParam']['v0']
v0_new = v0 - v0.mean(0)

a = 0
np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])



mooDict['trussParam']['v0'] = v0_new

trussParam['v0'] = v0_new

model = Model(trussParam, simParam)


moo = MOO(mooDict=mooDict)
