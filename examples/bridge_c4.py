from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objCurvedBridge, objFlatBridge
from utils.GA import GeneticAlgorithm

setting = {
    'modelDir': './data/bridge.json',
    'numChannels': 4,
    'numActions': 1,
    'numObjectives': 2,
    "channelMirrorMap": {
        0: -1,
        1: -1,
        2: -1,
        3: -1,
    },
    'objectives': [[objFlatBridge], [objCurvedBridge]]
}

mmo = MOO(setting)
lb, ub = mmo.getGeneSpace()

criterion = getCriterion(mmo)
mmo.check()
ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)

settingGA = ga.getDefaultSetting()
settingGA['nPop'] = 48
settingGA['nGenMax'] = 2000
settingGA['lenEra'] = 25
settingGA['nEraRevive'] = 2
settingGA['nWorkers'] = -1
ga.loadSetting(settingGA)
heroes, ratingsHero = ga.run()

print("ratingsHero: ")
print(ga.ratingsHero)


if False:
    from utils.GA import GeneticAlgorithm, loadHistory
    
    hs_dir = '/Users/Roll/Desktop/pyPneuMesh/output/GA_128-13:23:23/g935_-0.00,-61.60.hs'
    
    history = loadHistory(hs_dir)
    # history.plot(False)
    
    i_hero = 0
    gene = history.heroes[i_hero]
    
    _, actionSeqs = mmo.loadGene(gene)
    mmo.refreshModel()
    for iActionSeq in range(mmo.numObjectives):
        mmo.model.exportJSON(actionSeq=actionSeqs[iActionSeq], saveDir=None, appendix="c4_h"+str(i_hero)+"_a"+str(iActionSeq))