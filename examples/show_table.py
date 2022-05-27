from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/table.json',
    'numChannels': 6,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: 1,
        2: -1,
        3: -1,
        4: 5,
    },
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]]
}

moo = MOO(setting)
model = moo.model




# test
import copy
mc = copy.copy(model.maxContraction)
ec = copy.copy(model.edgeChannel)
model.toHalfGraph(reset=True)
model.maxContraction *= 20
model.edgeChannel *= 20
model.fromHalfGraph()
# assert((ec == model.edgeChannel).all())

model.initHalfGraph()

model.show()

for i in range(200):
    model.mutateHalfGraph()
    model.show()
    