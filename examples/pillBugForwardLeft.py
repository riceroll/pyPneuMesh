from utils.mmoCriterion import getCriterion
from utils.mmo import MMO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft
from GA import GeneticAlgorithm

setting = {
    'modelDir': './test/data/pillBugIn.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 2,
    "channelMirrorMap": {
        0: 1,
        1: 0,
        2: -1,
        3: -1,
    },
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft]]
}

mmo = MMO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 48
settingGA['nGenMax'] = 2500
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

print("ratingsHero: ")
print(ga.ratingsHero)

if False:
    iHero = 2
    iActionSeq = 0
    
    model, actionSeqs = decodeGene(mmo, heroes[iHero])
    model.exportJSON(actionSeq=actionSeqs[0])
    from utils.visualizer import visualizeActions
    visualizeActions(model, actionSeqs[iActionSeq])
