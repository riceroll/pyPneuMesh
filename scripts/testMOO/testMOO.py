from utils.utils import readNpy, readMooDict
from utils.Model import Model
from utils.MultiMotion import MultiMotion
from utils.MultiObjective import MultiObjective
from utils.Graph import Graph
from utils.MOO import MOO

trussParam = readNpy('examples/testMOO/table/table.trussparam.npy')
simParam = readNpy('examples/testMOO/table/table.simparam.npy')
actionSeqs = readNpy('examples/testMOO/table/table.actionseqs.npy')
objectives = readNpy('examples/testMOO/table/table.objectives.npy')
graphSetting = readNpy('examples/testMOO/table/table.graphsetting.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)
mo = MultiObjective(objectives, mm)
g = Graph(graphSetting, m)
moo = MOO(multiObjective=mo, graph=g, randomize=False)

print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.0)
print(moo.evaluate())
g.randomize()
print(moo.evaluate())
moo.randomize()
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.1)
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.1, actionMutationChance=0.0)
print(moo.evaluate())

print(' ')

mooDict = readMooDict('examples/testMOO/table')

moo = MOO(mooDict=mooDict, randomize=False)

print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.0)
print(moo.evaluate())
moo.graph.randomize()
print(moo.evaluate())
moo.randomize()
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.1)
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.1, actionMutationChance=0.0)
print(moo.evaluate())




