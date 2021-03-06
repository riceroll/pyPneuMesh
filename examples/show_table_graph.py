from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax

MOOSetting = {
    'modelDir': './data/table.json',
    'modelSettingDir': './data/model_setting_large.json',
    'nActions': 4,
    'nLoopPerSim': 2,
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]],
    'randInit': True
}

moo = MOO(MOOSetting)
model = moo.model

# test
model.toHalfGraph(reset=True)
model.fromHalfGraph()
model.initHalfGraph()

# model.show()

# for i in range(5):
#     model.mutateHalfGraph()
#     model.show()
