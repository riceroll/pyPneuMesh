import numpy as np
from utils.model import Model
from utils.mmo import MMO

def getModel(inFileDir):
    """
    load a json file into a model
    :param inFileDir: the json file of the input configuration
    :return: the model with loaded json configuration
    """
    model = Model()
    model.load(inFileDir)
    return model


def getActionSeq(inFileDir):
    with open(inFileDir) as iFile:
        content = iFile.read()
        import json
        data = json.loads(content)
        return np.array(data['script'])


def _encodeLensSpaces(mmoSetting: MMO):
    ms = mmoSetting
    model = getModel(ms.modelDir)
    
    lens = []
    spaces = []
    
    lens.append(len(model.edgeChannel))
    spaces.append((0, ms.numChannels))
    
    lens.append(model.edgeActive.sum())
    spaces.append((0, Model.contractionLevels))
    
    lens.append(len(ms.objectives) * ms.numChannels * ms.numActions)
    spaces.append((0, 2))
    return lens, spaces

def encodeGeneSpace(mmoSetting: MMO) -> (np.ndarray, np.ndarray):
    lens, spaces = _encodeLensSpaces(mmoSetting)
    
    lb = np.hstack([np.ones(lens[i], dtype=int) * spaces[i][0] for i in range(len(lens))])
    ub = np.hstack([np.ones(lens[i], dtype=int) * spaces[i][1] for i in range(len(lens))])
    
    return lb, ub

def encodeGene(model: Model, actionSeqs: np.ndarray) -> np.ndarray:
    assert(actionSeqs.ndim == 3)
    
    geneSegs = list()
    geneSegs.append(model.edgeChannel)
    geneSegs.append(model.maxContraction[model.edgeActive] / Model.contractionInterval)
    geneSegs.append(actionSeqs.reshape(-1))
    
    gene = np.hstack(geneSegs)
    geneInt = np.array(gene, dtype=int)
    assert((gene == geneInt).all())
    return geneInt


def decodeGene(mmoSetting: MMO, gene: np.ndarray) -> (Model, list):
    lb, ub = encodeGeneSpace(mmoSetting)
    assert((lb <= gene).all() and (gene < ub).all())
    
    lens, spaces = _encodeLensSpaces(mmoSetting)
    
    ms = mmoSetting
    model = getModel(ms.modelDir)
    model.gene = gene.copy()
    model.mmoSetting = mmoSetting
    
    i = 0
    j = lens[0]
    model.edgeChannel = gene[i: j]
    i = j
    j += lens[1]
    model.maxContraction *= 0
    model.maxContraction[model.edgeActive] = gene[i: j] * Model.contractionInterval
    i = j
    j += lens[2]
    actionSeqs = gene[i: j].reshape(len(ms.objectives), ms.numChannels, ms.numActions)
    
    return model, actionSeqs

def simulate(model: Model, actionSeq, nLoops=1, visualize=False, testing=False) -> np.ndarray:
    assert(actionSeq.ndim == 2)
    assert(actionSeq.shape[0] >= 1)
    assert(actionSeq.shape[1] >= 1)
    
    T = Model.numStepsPerActuation
    
    #  initialize with the last action
    model, _ = decodeGene(model.mmoSetting, model.gene)
    
    model.inflateChannel = actionSeq[:, -1]
    v = model.step(T)
    vs = [v]
    
    for iLoop in range(nLoops):
        for iAction in range(len(actionSeq[0])):
            model.inflateChannel = actionSeq[:, iAction]
            v = model.step(T)
            vs.append(v)
    vs = np.array(vs)
    assert(vs.shape == (nLoops * len(actionSeq[0]) + 1, len(model.v), 3))
    
    return vs

# testing

def testEncodeGeneSpace(argv):
    from utils.mmo import MMO
    from utils.objectives import objMoveForward, objFaceForward
    
    mmoSetting = {
        'modelDir': './test/data/lobsterIn.json',
        'numChannels': 4,
        'numActions': 5,
        'objectives': [objMoveForward, objFaceForward]
    }
    setting = MMO(mmoSetting)
    lb, ub = encodeGeneSpace(setting)
    
    assert((ub == np.array(np.hstack([np.ones(140) * 4, np.ones(44)*5, np.ones(20 * 2 )*2]), dtype=int)).all())
    assert(lb.shape == ub.shape and (lb == 0).all())

def testDecodeGene(argv):
    from utils.mmo import MMO
    from utils.objectives import objMoveForward, objFaceForward
    
    ub = np.array(np.hstack([np.ones(140) * 4, np.ones(44)*5, np.ones(20 * 2)*2]), dtype=int)
    gene = ub - 1
    mmoSetting = {
        'modelDir': './test/data/lobsterIn.json',
        'numChannels': 4,
        'numActions': 5,
        'objectives': [objMoveForward, objFaceForward]
    }
    setting = MMO(mmoSetting)
    model, actionSeqs = decodeGene(setting, gene)
    assert(model.edgeChannel.shape == model.maxContraction.shape == (140, ))
    assert((model.edgeChannel == 3).all())
    assert ((model.maxContraction == np.array([0.3, 0. , 0. , 0. , 0. , 0.3, 0. , 0. , 0.3, 0. , 0. , 0.3, 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0.3, 0. ,
       0. , 0.3, 0. , 0. , 0.3, 0. , 0.3, 0.3, 0.3, 0. , 0. , 0.3, 0.3,
       0. , 0.3, 0. , 0.3, 0.3, 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0. ,
       0. , 0. , 0. , 0.3, 0.3, 0. , 0.3, 0. , 0. , 0.3, 0.3, 0.3, 0.3,
       0. , 0.3, 0. , 0.3, 0.3, 0. , 0. , 0.3, 0.3, 0. , 0. , 0.3, 0.3,
       0. , 0. , 0.3, 0. , 0. , 0. , 0.3, 0. , 0.3, 0. , 0. , 0.3, 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. ,
       0. , 0. , 0.3, 0.3, 0. , 0. , 0. , 0. , 0.3, 0.3, 0.3, 0. , 0.3,
       0.3, 0. , 0.3, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])).all())
    assert(actionSeqs.shape == (2, 4, 5))
    assert((actionSeqs == 1).all())

def testEncodeGene(argv):
    from utils.mmo import MMO
    from utils.objectives import objMoveForward, objFaceForward
    
    ub = np.array(np.hstack([np.ones(140) * 4, np.ones(44)*5, np.ones(20 * 2)*2]), dtype=int)
    gene = ub - 1
    mmoSetting = {
        'modelDir': './test/data/lobsterIn.json',
        'numChannels': 4,
        'numActions': 5,
        'objectives': [objMoveForward, objFaceForward]
    }
    setting = MMO(mmoSetting)
    model, actionSeqs = decodeGene(setting, gene)
    
    geneOut = encodeGene(model, actionSeqs)
    assert((geneOut == gene).all())

def testSimulate(argv):
    modelDir = './test/data/pillBugIn.json'
    model = getModel(modelDir)
    actionSeq = getActionSeq(modelDir)
    vs = simulate(model, actionSeq, testing=False)
    
    vsTruth = np.load('./test/data/pillBugOutV.npy')
    assert(((vs - vsTruth) < 1e-5).all())
    
def testGetActionSeq(argv):
    actionSeq = getActionSeq('./test/data/lobsterIn.json')
    assert(actionSeq.shape == (4, 1))
    assert((actionSeq == False).all())

tests = {
    "encodeGeneSpace": testEncodeGeneSpace,
    'decodeGene': testDecodeGene,
    'encodeGene': testEncodeGene,
    'simulate': testSimulate,
}

def testAll(argv):
    for key in tests:
        print('test{}{}():'.format(key[0].upper(), key[1:]))
        tests[key](argv)
        print('Pass.\n')


if __name__ == "__main__":
    import sys
    if 'test' in sys.argv:
        if 'all' in sys.argv:
            testAll(sys.argv)
        else:
            for key in tests:
                if key in sys.argv:
                    print('test{}{}():'.format(key[0].upper(), key[1:]))
                    tests[key](sys.argv)
                    print('Pass.\n')