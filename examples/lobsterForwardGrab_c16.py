from utils.mmoCriterion import getCriterion
from utils.mmo import MMO
from utils.objectives import objMoveForward, objFaceForward, objGrabLobster
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/lobsterIn_all_active.json',
    'numChannels': 16,
    'numActions': 4,
    'numObjectives': 1,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 4,
        6: 7,
        7: 6,
        8: -1,
        9: -1,
        10:-1,
        11: -1,
        12: -1,
        13: -1,
        14: -1,
        15: -1
    },
    'objectives': [[objMoveForward, objFaceForward, objGrabLobster]]
}

mmo = MMO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 48
settingGA['nGenMax'] = 2000
settingGA['lenEra'] = 40
settingGA['nEraRevive'] = 2
settingGA['nWorkers'] = 12
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
