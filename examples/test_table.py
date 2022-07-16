from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
import pickle5


MOOSetting = {
    'modelDir': './data/table.json',
    'modelSettingDir': './data/model_setting_large.json',
    'nActions': 4,
    'nLoopPerSim': 4,
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]],
    'randInit': True
}

moo = MOO(MOOSetting)
model = moo.model

# test
model.toHalfGraph(reset=True)
model.fromHalfGraph()
model.initHalfGraph()

model.truss.eLengthLevel *= 0
model.truss.eLengthLevel += 3

result = pickle5.load(open('./output/GA_531-8-36-53/iPool_580', 'rb'))

mooOld = result['elitePool'][3]['moo']
moo.model.truss.eChannel = mooOld.model.edgeChannel
moo.actionSeqs = mooOld.actionSeqs
moo.model.truss.eLengthLevel = ((mooOld.model.maxContraction/0.05625) / 4 * 3).astype(int)

moo.simulateOpenLoop(1, visualize=True)
# model.show()







# for i in range(5):
#     model.mutateHalfGraph()
#     model.show()
