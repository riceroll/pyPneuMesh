from utils.moo import MOO
from utils.objectives.locomotion import MoveForward, FaceForward, TurnLeft, LowerBodyMax

MOOsetting = {
    'modelDir': './data/half_helmet.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: -1,
        1: -1,
        2: -1,
        3: -1,
    },
    'objectives': [[MoveForward, FaceForward], [TurnLeft], [LowerBodyMax]]
}

moo = MOO(MOOsetting, randInit=True)
model = moo.model

model.show()

for i in range(5):
    model.mutateGraph()
    model.fromGraph()
    # model.show()
