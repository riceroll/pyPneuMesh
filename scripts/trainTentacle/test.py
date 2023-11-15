from pyPneuMesh.Model import Model
from pyPneuMesh.MOO import MOO
from pyPneuMesh.Graph import Graph
from pyPneuMesh.FullGraph import FullGraph
from pyPneuMesh.utils import readNpy, readMooDict

trussparam = readNpy("/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/tentacle.trussparam.npy")
simparam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/tentacle.simparam.npy')

model = Model(trussParam=trussparam, simParam=simparam)

graphSetting = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/tentacle.graphsetting.npy')

graph = Graph(graphSetting=graphSetting, model=model)
graph.randomize()

contractions = []
for i in range(10):
    contractions.append(graph.contractions.copy())
    graph.mutate(0.0, 0.6)



graph1 = Graph(graphSetting=graphSetting, model=model)
graph1.randomize()

graph.cross(graph1, 0.5)

