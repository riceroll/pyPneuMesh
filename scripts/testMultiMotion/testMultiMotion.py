from src.utils import readNpy
from src.Model import Model
from src.MultiMotion import MultiMotion

trussParam = readNpy('examples/testMultiMotion/table/table.trussparam.npy')
simParam = readNpy('examples/testMultiMotion/table/table.simparam.npy')
actionSeqs = readNpy('examples/testMultiMotion/table/table.actionseqs.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)

mm.mutate(0.1)

vs = mm.simulate(0, 2)
vs = mm.animate(0, 2)

