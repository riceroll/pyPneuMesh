from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/table.json',
    'numChannels': 8,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: -1,
        5: -1,
        6: -1,
        7: -1
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
#
# print("ratingsHero: ")
# print(ga.ratingsHero)
#
#
# genes, fileDirs = ga.getHeroes()
# for i in range(len(genes)):
# gene = genes[i]

import sys
from utils.GA import GeneticAlgorithm, loadHistory

hs_dir = '/Users/Roll/Desktop/pyPneuMesh/output/GA_1116-20:55:18/g2615_9.99,0.94,0.79,-2.59.hs'


history = loadHistory(hs_dir)
# history.plot(False)

i_hero = 0
gene = history.heroes[i_hero]

_, actionSeqs = mmo.loadGene(gene)
mmo.refreshModel()
for iActionSeq in range(mmo.numObjectives):
    mmo.model.exportJSON(actionSeq=actionSeqs[iActionSeq], saveDir=None, appendix="c16_h"+str(i_hero)+"_a"+str(iActionSeq))