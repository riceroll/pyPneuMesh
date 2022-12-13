import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph

trussParam = readNpy('examples/testGraph/table/table.trussparam.npy')
simParam = readNpy('examples/testGraph/table/table.simparam.npy')
graphSetting = readNpy('examples/testGraph/table/table.graphsetting.npy')

m = Model(trussParam, simParam)

g = Graph(graphSetting, m)
g.randomize()
m.show()

for i in range(5):
    g.mutate(0.2)
    m.show()


