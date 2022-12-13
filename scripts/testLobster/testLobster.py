from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion

trussParam = readNpy('examples/testLobster/lobster/Lobster_0.trussparam.npy')
simParam = readNpy('examples/testLobster/lobster/lobster.simparam.npy')
actionSeqs = readNpy('examples/testLobster/lobster/Lobster_0.actionseqs.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)

mm.animate(0, 10, 10)


trussParam = readNpy('examples/testLobster/lobster/lobster_grabgo.trussparam.npy')
simParam = readNpy('examples/testLobster/lobster/lobster.simparam.npy')
actionSeqs = readNpy('examples/testLobster/lobster/lobster_grabgo.actionseqs.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)

mm.animate(0, 10, 10)


trussParam = readNpy('examples/testLobster/lobster/lobster_walk.trussparam.npy')
simParam = readNpy('examples/testLobster/lobster/lobster.simparam.npy')
actionSeqs = readNpy('examples/testLobster/lobster/lobster_walk.actionseqs.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)

mm.animate(0, 10, 10)
