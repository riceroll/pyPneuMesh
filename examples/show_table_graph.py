from utils.moo import MOO
from utils.objectives.locomotion import MoveForward, FaceForward, TurnLeft, LowerBodyMax

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
    'objectives': [[MoveForward, FaceForward], [TurnLeft], [LowerBodyMax]]
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
