from utils.mmoCriterion import getCriterion
from utils.mmo import MMO
from utils.objectives import objMoveForward, objFaceForward
from GA import GeneticAlgorithm

setting = {
    'modelDir': './test/data/pillBugIn.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 1,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: -1,
        3: -1,
    },
    'objectives': [(objMoveForward, objFaceForward)]
}

mmo = MMO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 8
settingGA['nGenMax'] = 2
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

print("ratingsHero: ")
print(ga.ratingsHero)

genes, fileDirs = ga.getHeroes()

for i in range(len(genes)):
    gene = genes[i]

    _, actionSeqs = mmo.loadGene(gene)
    mmo.refreshModel()
    for iActionSeq in range(mmo.numActions):
        mmo.model.exportJSON(actionSeq=actionSeqs[iActionSeq], saveDir=fileDirs[i])

    
    # from utils.visualizer import visualizeActions
    # visualizeActions(mmo.model, actionSeqs[iActionSeq], nLoop=5)
#
# iHero = 4
# iActionSeq = 1
#
# iHeroes = [1,2,7,8,9]
# iActionSeqs = [0, 1]
# import time
# for iHero in iHeroes:
#     _, actionSeqs = mmo.loadGene(heroes[iHero])
#     mmo.refreshModel()
#     for iActionSeq in iActionSeqs:
#         mmo.model.exportJSON(actionSeq=actionSeqs[iActionSeq])
#         time.sleep(1.1)

