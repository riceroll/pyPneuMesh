from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './test/data/pillBugIn.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: -1,
        3: -1,
    },
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objTurnRight]]
}

mmo = MOO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 48
settingGA['nGenMax'] = 2000
settingGA['lenEra'] = 40
settingGA['nEraRevive'] = 2
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

print("ratingsHero: ")
print(ga.ratingsHero)


genes, fileDirs = ga.getHeroes()
for i in range(len(genes)):
    gene = genes[i]

    _, actionSeqs = mmo.loadGene(gene)
    mmo.refreshModel()
    for iActionSeq in range(mmo.numObjectives):
        mmo.model.exportJSON(actionSeq=actionSeqs[iActionSeq], saveDir=fileDirs[i], appendix=iActionSeq)