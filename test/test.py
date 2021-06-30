
def testMMO(argv):
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
    settingGA['nGenMax'] = 1
    settingGA['saveHistory'] = False
    settingGA['nWorkers'] = -1
    settingGA['mute'] = "unmute" not in argv
    settingGA['plot'] = False
    ga.loadSetting(settingGA)
    heroes, ratingsHero = ga.run()
    
    mmo.loadGene(heroes[0])
    
def testGetCriterion(argv):
    from utils.mmoCriterion import getCriterion
    from utils.mmo import MMO
    from utils.objectives import objMoveForward, objFaceForward
    from GA import GeneticAlgorithm
    import numpy as np
    
    def assertCriterion(mmo: MMO, criterion, ratingTruth):
        rating = criterion(mmo.getGene())
        assert ((rating == ratingTruth).all())
    
    # 1
    modelDir = './test/data/pillBugIn.json'

    objectives1 = [[objMoveForward], [objFaceForward]]
    objectives2 = [[objMoveForward, objFaceForward]]
    
    setting = {
        'modelDir': modelDir,
        'numChannels': 4,
        'numActions': 4,
        'numObjectives': 2,
        "channelMirrorMap": {
            0: 2,
            1: -1,
            2: 0,
            3: -1,
        },
        'objectives': objectives1,
        "modelConfigDir": "./data/config_0.json",
    }
    mmo = MMO(setting)
    
    actionSeq3 = mmo.actionSeqs[0].copy()
    actionSeq3[0, 1] = (actionSeq3[0, 1] + 1) % 2
    actionSeqs3 = np.vstack([np.expand_dims(actionSeq3, 0), np.expand_dims(actionSeq3, 0)])
    mmo.actionSeqs = actionSeqs3
    
    criterion = getCriterion(mmo)
    assertCriterion(mmo, criterion, (1.010703784566089, 0.9974059986505434))

    setting = {
        'modelDir': modelDir,
        'numChannels': 4,
        'numActions': 4,
        'numObjectives': 1,
        "channelMirrorMap": {
            0: 2,
            1: -1,
            2: 0,
            3: -1,
        },
        'objectives': objectives2,
        "modelConfigDir": "./data/config_0.json",
    }
    mmo = MMO(setting)
    
    actionSeq33 = mmo.actionSeqs[0].copy()
    actionSeq33[0, 1] = (actionSeq33[0, 1] + 1) % 2
    actionSeqs33 = np.expand_dims(actionSeq33, 0)
    mmo.actionSeqs = actionSeqs33

    criterion = getCriterion(mmo)
    assertCriterion(mmo, criterion, (1.010703784566089, 0.9974059986505434))
    
tests = {
    'getCriterion': testGetCriterion,
    'mmo': testMMO,
}

if __name__ == "__main__":
    import sys

    from utils.visualizer import tests as testsUtilsVisualizer
    from utils.geometry import tests as testsUtilsGeometry
    from model import tests as testsModel
    from GA import tests as testsGA
    from utils.objectives import tests as testUtilsObjectives
    from utils.mmo import tests as testUtilsMMO
    
    testsDict = {
        'model.py': testsModel,
        'utils/visualizer.py': testsUtilsVisualizer,
        'utils/geometry.py': testsUtilsGeometry,
        'utils/objective.py': testUtilsObjectives,
        'GA.py': testsGA,
        'utils/mmo.py': testUtilsMMO,
        'test': tests,
    }
    
    for testsName in testsDict:
        if "all" not in sys.argv and testsName not in sys.argv:
            continue
        print(testsName, flush=True)
        tests = testsDict[testsName]
        for testName in tests:
            print('\t'+testName + '()', end="..........", flush=True)
            tests[testName](sys.argv)
            print('P')
        print('\n')
