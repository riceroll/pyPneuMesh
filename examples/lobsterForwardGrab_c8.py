from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objGrabLobster
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/lobsterIn_all_active.json',
    'numChannels': 8,
    'numActions': 4,
    'numObjectives': 1,
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
    'objectives': [[objMoveForward, objFaceForward, objGrabLobster]]
}

mmo = MOO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 48
settingGA['nGenMax'] = 10000
settingGA['lenEra'] = 10
settingGA['nEraRevive'] = 2
settingGA['nWorkers'] = 8
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
