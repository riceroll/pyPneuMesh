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
settingGA['nGenMax'] = 40
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

print("ratingsHero: ")
print(ga.ratingsHero)

if False:
    iHero = 1
    iActionSeq = 0
    
    _, actionSeqs = mmo.loadGene(heroes[iHero])
    mmo.refreshModel()
    mmo.model.exportJSON(actionSeq=actionSeqs[0])
    from utils.visualizer import visualizeActions
    visualizeActions(mmo.model, actionSeqs[iActionSeq], nLoop=5)

