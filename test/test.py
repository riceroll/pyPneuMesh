
def testMMO(argv):
    from utils.modelInterface import getModel, encodeGeneSpace, decodeGene, simulate
    from utils.mmoCriterion import getCriterion
    from utils.mmoSetting import MMOSetting
    from utils.objectives import objMoveForward, objFaceForward
    from GA import GeneticAlgorithm

    setting = {
        'modelDir': './test/data/pillBugIn.json',
        'numChannels': 4,
        'numActions': 4,
        'objectives': [(objMoveForward, objFaceForward)]
    }
    
    mmoSetting = MMOSetting(setting)
    lb, ub = encodeGeneSpace(mmoSetting=mmoSetting)
    
    criterion = getCriterion(mmoSetting)
    ga = GeneticAlgorithm(criterion=criterion, lb=lb, ub=ub)
    
    settingGA = ga.getDefaultSetting()
    settingGA['nPop'] = 8
    settingGA['nGenMax'] = 1
    settingGA['saveHistory'] = False
    settingGA['nWorkers'] = -1
    settingGA['mute'] = True
    settingGA['plot'] = False
    ga.loadSetting(settingGA)
    heroes, ratingsHero = ga.run()
    
def testGetCriterion(argv):
    from utils.modelInterface import getModel, getActionSeq, encodeGene, encodeGeneSpace, decodeGene, simulate
    from utils.mmoCriterion import getCriterion
    from utils.mmoSetting import MMOSetting
    from utils.objectives import objMoveForward, objFaceForward
    from GA import GeneticAlgorithm
    import numpy as np
    
    def assertCriterion(model, actionSeqs, criterion, ratingTruth):
        gene = encodeGene(model, actionSeqs)
        rating = criterion(gene)
        assert ((rating == ratingTruth).all())
        
    def subTestGetCriterion(objectives, actionSeqs, truth):
        modelDir = './test/data/pillBugIn.json'
        
        setting = {
            'modelDir': modelDir,
            'numChannels': actionSeqs[0].shape[0],
            'numActions': actionSeqs[0].shape[1],
            'objectives': objectives
        }
        
        mmoSetting = MMOSetting(setting)
        model = getModel(modelDir)
        criterion = getCriterion(mmoSetting)
        
        assertCriterion(model, actionSeqs, criterion, truth)
        
    # 1
    modelDir = './test/data/pillBugIn.json'

    objectives1 = [[objMoveForward], [objFaceForward]]
    objectives2 = [[objMoveForward, objFaceForward]]

    actionSeq3 = getActionSeq(modelDir)
    actionSeq3[0, 1] = (actionSeq3[0, 1] + 1) % 2
    actionSeqs3 = np.vstack([np.expand_dims(actionSeq3, 0), np.expand_dims(actionSeq3, 0)])
    subTestGetCriterion(objectives1, actionSeqs3, (0.20214075691321778, 0.9974059986505434))

    actionSeq33 = getActionSeq(modelDir)
    actionSeq33[0, 1] = (actionSeq33[0, 1] + 1) % 2
    actionSeqs33 = np.expand_dims(actionSeq33, 0)
    subTestGetCriterion(objectives2, actionSeqs33, (0.20214075691321778, 0.9974059986505434))
    
    
tests = {
    'getCriterion': testGetCriterion,
    'mmo': testMMO,
}

if __name__ == "__main__":
    import sys

    from utils.modelInterface import tests as testsUtilsModelInterface
    from utils.visualizer import tests as testsUtilsVisualizer
    from utils.geometry import tests as testsUtilsGeometry
    from model import tests as testsModel
    from GA import tests as testsGA
    
    testsDict = {
        'model.py': testsModel,
        'utils/modelInterface.py': testsUtilsModelInterface,
        'utils/visualizer.py': testsUtilsVisualizer,
        'utils/geometry.py': testsUtilsGeometry,
        'GA.py': testsGA,
        'test/test.py': tests,
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
