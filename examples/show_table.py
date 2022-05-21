from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/table.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: -1,
        3: -1
    },
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]]
}

mmo = MOO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 8
settingGA['nGenMax'] = 1
settingGA['lenEra'] = 25
settingGA['nEraRevive'] = 3
settingGA['nWorkers'] = 32
ga.loadSetting(settingGA)
# heroes, ratingsHero = ga.run()


# print("ratingsHero: ")
# print(ga.ratingsHero)


# genes, fileDirs = ga.getHeroes()
# for i in range(len(genes)):
# gene = genes[i]

import sys
from utils.GA import GeneticAlgorithm, loadHistory

hs_dir = './output/GA_1116-20:55:18/g2615_9.99,0.94,0.79,-2.59.hs'

history = loadHistory(hs_dir)
# history.plot(False)

i_hero = 7
gene = history.heroes[i_hero]

model, actionSeqs = mmo.loadGene(gene)


from utils.visualizer import visualizeActions

vs = visualizeActions(model, actionSeqs[1], nLoop=2, exportFrames=False)