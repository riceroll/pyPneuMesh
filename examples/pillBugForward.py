from utils.modelInterface import getModel, encodeGeneSpace, decodeGene, simulate
from utils.mmoCriterion import getCriterion
from utils.mmoSetting import MMOSetting
from utils.objectives import objMoveForward, objFaceForward
from GA import GeneticAlgorithm

setting = {
    'modelDir': './test/data/pillBugIn.json',
    'numChannels': 4,
    'numActions': 4,
    'objectives': [[objMoveForward, objFaceForward]]
}

mmoSetting = MMOSetting(setting)
lb, ub = encodeGeneSpace(mmoSetting=mmoSetting)

criterion = getCriterion(mmoSetting)
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 8
settingGA['nGenMax'] = 80
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

model, actionSeqs = decodeGene(mmoSetting, heroes[0])

from utils.visualizer import visualizeActions

visualizeActions(model, actionSeqs[0])
