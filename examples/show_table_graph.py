from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax

MOOsetting = {
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

moo = MOO(MOOsetting)
model = moo.model


# test
model.toHalfGraph(reset=True)
model.fromHalfGraph()
model.initHalfGraph()

model.show()

for i in range(5):
    model.mutateHalfGraph()
    model.show()
