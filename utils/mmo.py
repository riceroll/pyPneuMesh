# multi-objective optimization
import copy
import numpy as np
from typing import List, Dict, Tuple

from model import Model

class MMO:
    def __init__(self, setting):
        self.modelDir: str = ""
        self.numChannels: int = -1
        self.numActions: int = -1
        self.numObjectives: int = -1
        self.channelMirrorMap: dict = dict()
        
        self.objectives: List = []
        self.actionSeqs: np.ndarray = np.zeros([])
        self.gene: np.ndarray = np.zeros([])
        
        self.model: Model = Model()
        
        self._loadSetting(setting)
    
    def _loadSetting(self, setting: Dict):
        for key in setting:
            assert (hasattr(self, key))
            assert (type(setting[key])==type(getattr(self, key)))
            setattr(self, key, setting[key])
        
        keysRequired = ['modelDir', 'numChannels', 'numActions', 'numObjectives']
        for key in keysRequired:
            assert(key in setting)
        
        if "channelMirrorMap" not in setting:
            self.channelMirrorMap = {ic: -1 for ic in range(self.numChannels)}
        assert(len(self.channelMirrorMap) == self.numChannels)
        
        if "actionSeqs" not in setting:
            self._loadActionSeqs()
        
        if "model" not in setting:
            self._loadModel()

        if "gene" not in setting:
            pass

        if "objectives" not in setting:
            pass
        
        # try getGene for channelMirrorMap
        # try:
        #     self.getGene()
        # except Exception as e:
        #     print("getGene failed. channelMirrorMap is likely inalid. ")
        #     self.channelMirrorMap = {ic: -1 for ic in range(self.numChannels)}
        
    def _loadActionSeqs(self):
        assert (isinstance(self.modelDir, str) and len(self.modelDir) != 0)
        with open(self.modelDir) as iFile:
            content = iFile.read()
            import json
            data = json.loads(content)
            actionSeqs = np.array(data['script'])
        if actionSeqs.ndim == 2:
            actionSeqs = np.expand_dims(actionSeqs, 0)
        self.actionSeqs = actionSeqs
    
    def _loadModel(self):   # load model from modelDir
        assert (isinstance(self.modelDir, str) and len(self.modelDir) != 0)
        self.model = Model()
        self.model.load(self.modelDir)
    
    def refreshModel(self):
        self._loadModel()
        self.loadGene(self.gene)
    
    def _getGeneSpaces(self) -> (Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray]):
        nEdgeChannelMiddle = (np.array(list(self.model.edgeMirrorMap.values())) == -1).sum()
        nEdgeChannelHalf = (len(self.model.e) - nEdgeChannelMiddle) / 2
        assert(nEdgeChannelHalf == int(nEdgeChannelHalf))
        nEdgeChannelHalf = int(nEdgeChannelHalf)
        
        nChannelMiddle = 0      # number of channels self-mirrored
        for ic in self.channelMirrorMap:
            if self.channelMirrorMap[ic] == -1:
                nChannelMiddle += 1
        
        ubEdgeChannelHalf = self.numChannels
        ubEdgeChannelMiddle = nChannelMiddle
        edgeChannelHalfSpace = (np.zeros(nEdgeChannelHalf), np.ones(nEdgeChannelHalf, dtype=int) * ubEdgeChannelHalf)
        edgeChannelMiddleSpace = (np.zeros(nEdgeChannelMiddle), np.ones(nEdgeChannelMiddle, dtype=int) * ubEdgeChannelMiddle)
        
        edgeMirrorArray = np.array(list(self.model.edgeMirrorMap.items()))      # convert edgeMirrorMap to n x 2 array
        edgeMirrorArray = edgeMirrorArray[np.argsort(edgeMirrorArray[:, 0])]
        edgeMirrorActive = edgeMirrorArray[self.model.edgeActive]               # get the active edges from edgeMirrorArray
        nEdgeMirrorActiveMiddle = (edgeMirrorActive[:, 1] == -1).sum()          # number of active edges in the middle
        nEdgeMirrorActiveHalf = (len(edgeMirrorActive) - nEdgeMirrorActiveMiddle) / 2
        assert(nEdgeMirrorActiveHalf == int(nEdgeMirrorActiveHalf))
        nContractionLevel = int(nEdgeMirrorActiveHalf) + nEdgeMirrorActiveMiddle
        ubContractionLevel = Model.contractionLevels
        contractionLevelSpace = (np.zeros(nContractionLevel), np.ones(nContractionLevel, dtype=int) * ubContractionLevel)
        
        nActionSeqs = self.numObjectives * self.numChannels * self.numActions
        ubActionSeqs = 2
        actionSeqsSpace = (np.zeros(nActionSeqs), np.ones(nActionSeqs) * ubActionSeqs)
        
        return edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace
    
    def getGeneSpace(self) -> (np.ndarray, np.ndarray):
        spaces = self._getGeneSpaces()
        lb = np.hstack([space[0] for space in spaces]).astype(int)
        ub = np.hstack([space[1] for space in spaces]).astype(int)
        return lb, ub
    
    def getGene(self) -> (np.ndarray):
        # TODO: test
        edgeChannelHalf = []
        edgeChannelMiddle = []
        maxContraction = []
        actionSeqs = self.actionSeqs.reshape(-1).tolist()

        iChannelsMiddle = []  # id of channels in the middle
        for ic in self.channelMirrorMap:
            if self.channelMirrorMap[ic] == -1:
                iChannelsMiddle.append(ic)
                
        ieVisited = set()
        for ie in range(len(self.model.e)):
            if ie in ieVisited:
                continue
            ieMirrored = self.model.edgeMirrorMap[ie]
            if ieMirrored == -1:  # self-mirrored edge
                ic = iChannelsMiddle.index(self.model.edgeChannel[ie])
                edgeChannelMiddle.append(ic)
                if self.model.edgeActive[ie]:
                    maxContraction.append(self.model.maxContraction[ie])
                ieVisited.add(ie)
            else:
                edgeChannelHalf.append(self.model.edgeChannel[ie])
                if self.model.edgeActive[ie]:
                    maxContraction.append(self.model.maxContraction[ie])
                ieVisited.add(ie)
                ieVisited.add(ieMirrored)
        
        maxContractionLevel = (np.array(maxContraction) / Model.contractionInterval).astype(int).tolist()
        gene = np.array(edgeChannelHalf + edgeChannelMiddle + maxContractionLevel + actionSeqs, dtype=int)
        return gene
    
    def loadGene(self, gene: np.ndarray) -> (Model, List):  # load gene into model and actionSeqs
        if gene.shape == ():
            return self.model, self.actionSeqs
        edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = self._getGeneSpaces()
        nEdgeChannelHalf = len(edgeChannelHalfSpace[0])
        nEdgeChannelMiddle = len(edgeChannelMiddleSpace[0])
        nContractionLevel = len(contractionLevelSpace[0])
        nActionSeqs = len(actionSeqsSpace[0])
        
        n = 0
        m = nEdgeChannelHalf
        edgeChannelHalf = gene[n:m]
        n = m
        m += nEdgeChannelMiddle
        edgeChannelMiddle = gene[n:m]
        n = m
        m += nContractionLevel
        contractionLevel = gene[n:m]
        n = m
        m += nActionSeqs
        actionSeqs = gene[n: m]
        assert(m == len(gene))

        # load edgeChannel
        iChannelsMiddle = []        # id of channels in the middle
        for ic in self.channelMirrorMap:
            if self.channelMirrorMap[ic] == -1:
                iChannelsMiddle.append(ic)
        
        self.model.edgeChannel = np.ones_like(self.model.edgeChannel, dtype=int)    # reset
        self.model.edgeChannel *= -1
        iEdgeChannelHalf = iEdgeChannelMiddle = 0
        for ie in range(len(self.model.e)):
            if self.model.edgeChannel[ie] != -1:     # already assigned
                continue
            
            ieMirror = self.model.edgeMirrorMap[ie]
            if ieMirror == -1:  # self mirror
                ic = iChannelsMiddle[edgeChannelMiddle[iEdgeChannelMiddle]]
                iEdgeChannelMiddle += 1
                self.model.edgeChannel[ie] = ic
            else:   # on the half
                ic = edgeChannelHalf[iEdgeChannelHalf]
                iEdgeChannelHalf += 1
                icMirror = self.channelMirrorMap[ic]
                
                if icMirror != -1:  # channel not mirrored
                    self.model.edgeChannel[ie] = ic
                    self.model.edgeChannel[ieMirror] = icMirror
                else:       # channel mirrored
                    self.model.edgeChannel[ie] = ic
                    self.model.edgeChannel[ieMirror] = ic
        
        assert (iEdgeChannelHalf == len(edgeChannelHalf))
        assert (iEdgeChannelMiddle == len(edgeChannelMiddle))
        assert ((self.model.edgeChannel == -1).sum() == 0)
        
        # load contractionLevel
        maxContractionGene = contractionLevel * Model.contractionInterval
        self.model.maxContraction = np.ones_like(self.model.maxContraction)
        self.model.maxContraction *= -1
        iMaxContractionGene = 0
        for ie in range(len(self.model.e)):
            if self.model.maxContraction[ie] != -1:    # already assigned
                continue
            if not self.model.edgeActive[ie]:       # passive Beam
                continue

            maxContractionRatio = maxContractionGene[iMaxContractionGene]
            iMaxContractionGene += 1
            self.model.maxContraction[ie] = maxContractionRatio
            
            ieMirror = self.model.edgeMirrorMap[ie]
            if ieMirror != -1:  # self mirror
                self.model.maxContraction[ieMirror] = maxContractionRatio
        
        assert (iMaxContractionGene == len(maxContractionGene))
        assert ((self.model.maxContraction == -1).sum() == (~self.model.edgeActive).sum())
        
        # load actionSeqs
        self.actionSeqs = actionSeqs.reshape(self.numObjectives, self.numChannels, self.numActions)
        
        self.gene = gene.copy()
        
        return self.model, self.actionSeqs
    
    def simulate(self, actionSeq, nLoops=1, visualize=False, testing=False) -> np.ndarray:
        assert (actionSeq.ndim == 2)
        assert (actionSeq.shape[0] >= 1)
        assert (actionSeq.shape[1] >= 1)
    
        T = Model.numStepsPerActuation
    
        #  initialize with the last action
        self.refreshModel()
        model = self.model
        
        model.inflateChannel = actionSeq[:, -1]
        v = model.step(T)
        vs = [v]
    
        for iLoop in range(nLoops):
            for iAction in range(len(actionSeq[0])):
                model.inflateChannel = actionSeq[:, iAction]
                v = model.step(T)
                vs.append(v)
        vs = np.array(vs)
        assert (vs.shape == (nLoops * len(actionSeq[0]) + 1, len(model.v), 3))
    
        return vs
    
    def check(self):
        if len(self.channelMirrorMap) != 0:
            assert(self.numChannels == len(self.channelMirrorMap))
            assert(self.numChannels == self.model.edgeChannel.max() + 1)
        if len(self.objectives) != 0:
            assert(len(self.objectives) == self.numObjectives)
        if self.actionSeqs.shape != ():
            assert(self.actionSeqs.shape == (self.numObjectives, self.numChannels, self.numActions))
        
    
def testMMO(argv):
    # 1
    setting = dict({
        "modelDir": "./test/data/testTetIn.json",
        "numChannels": 2,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
        },
    })
    
    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()

    # assert False
    assert( (np.array(edgeChannelHalfSpace) == np.array([[0., 0.], [2., 2.]])).all() )
    assert( (np.array(edgeChannelMiddleSpace) == np.array([[0., 0.], [2., 2.]])).all() )
    assert( (np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0.], [5, 5, 5, 5]]) ).all() )
    assert( (np.array(actionSeqsSpace) ==
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]] )).all() )
    
    gene = np.hstack([edgeChannelHalfSpace[1]-1, edgeChannelMiddleSpace[1]-1,
                      contractionLevelSpace[1]-1, actionSeqsSpace[1]-1]).astype(int)
    mmo.loadGene(gene)
    assert( (mmo.model.edgeChannel == np.array([1, 1, 1, 1, 1, 1])).all())
    assert ((mmo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(mmo.actionSeqs) == np.array([[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]])).all())
    
    # 2
    setting = dict({
        "modelDir": "./test/data/testTetIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    
    assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0.], [4, 4]])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0.], [2., 2.]])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0.], [5, 5, 5, 5]])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0]*32,
                       [2]*32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    assert ((mmo.model.edgeChannel == np.array([3, 1, 2, 1, 3, 2])).all())
    assert ((mmo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(mmo.actionSeqs) == np.ones([2, 4, 4])).all())

    # 3
    setting = dict({
        "modelDir": "./test/data/testDoubleTetIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    
    assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0., 0.], [4, 4, 4]])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0., 0.], [2., 2., 2.]])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0., 0.], [5, 5, 5, 5, 5]])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0] * 32,
                       [2] * 32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    assert ((mmo.model.edgeChannel == np.array([3, 2, 1, 3, 1, 2, 3, 2, 1])).all())
    assert ((mmo.model.maxContraction == np.array([ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3, -1. , -1. ,  0.3])).all())
    assert ((np.array(mmo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    from utils.visualizer import visualizeSymmetry
    if "plot" in argv:
        visualizeSymmetry(mmo.model)

    # 4
    setting = dict({
        "modelDir": "./test/data/lobsterIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: 1,
            1: 0,
            2: -1,
            3: -1,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    #
    # assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0., 0.], [4, 4, 4]])).all())
    # assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0., 0.], [2., 2., 2.]])).all())
    # assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0., 0.], [5, 5, 5, 5, 5]])).all())
    # assert ((np.array(actionSeqsSpace) ==
    #          np.array([[0] * 32,
    #                    [2] * 32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    assert ((mmo.model.edgeChannel == np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3])).all())
    assert ((mmo.model.maxContraction == np.array([ 0.3, -1. , -1. , -1. , -1. ,  0.3, -1. , -1. ,  0.3, -1. , -1. ,
        0.3, -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,
        0.3, -1. ,  0.3, -1. , -1. ,  0.3, -1. , -1. ,  0.3, -1. ,  0.3,
        0.3,  0.3, -1. , -1. ,  0.3,  0.3, -1. ,  0.3, -1. ,  0.3,  0.3,
       -1. , -1. , -1. , -1. , -1. ,  0.3, -1. , -1. , -1. , -1. , -1. ,
        0.3,  0.3, -1. ,  0.3, -1. , -1. ,  0.3,  0.3,  0.3,  0.3, -1. ,
        0.3, -1. ,  0.3,  0.3, -1. , -1. ,  0.3,  0.3, -1. , -1. ,  0.3,
        0.3, -1. , -1. ,  0.3, -1. , -1. , -1. ,  0.3, -1. ,  0.3, -1. ,
       -1. ,  0.3, -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,
       -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,
       -1. , -1. , -1. , -1. , -1. ,  0.3, -1. , -1. , -1. ,  0.3,  0.3,
       -1. , -1. , -1. , -1. ,  0.3,  0.3,  0.3, -1. ,  0.3,  0.3, -1. ,
        0.3, -1. , -1. , -1. , -1. , -1. , -1. , -1. ])).all())
    assert ((np.array(mmo.actionSeqs) == np.ones([2, 4, 4])).all())

    from utils.visualizer import visualizeSymmetry
    if "plot" in argv:
        visualizeSymmetry(mmo.model)

    # 5
    setting = dict({
        "modelDir": "./test/data/lobsterIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    #
    # assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0., 0.], [4, 4, 4]])).all())
    # assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0., 0.], [2., 2., 2.]])).all())
    # assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0., 0.], [5, 5, 5, 5, 5]])).all())
    # assert ((np.array(actionSeqsSpace) ==
    #          np.array([[0] * 32,
    #                    [2] * 32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    assert ((mmo.model.edgeChannel == np.array([1, 3, 2, 3, 2, 1, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3,
       3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 1, 3, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 3, 2, 2, 2, 3, 3, 3, 2, 2, 1, 3, 1,
       2, 3, 2, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2,
       2, 2, 2, 2, 2, 1, 3, 2])).all())
    assert ((mmo.model.maxContraction == np.array([0.3, -1., -1., -1., -1., 0.3, -1., -1., 0.3, -1., -1.,
                                                   0.3, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                                   0.3, -1., 0.3, -1., -1., 0.3, -1., -1., 0.3, -1., 0.3,
                                                   0.3, 0.3, -1., -1., 0.3, 0.3, -1., 0.3, -1., 0.3, 0.3,
                                                   -1., -1., -1., -1., -1., 0.3, -1., -1., -1., -1., -1.,
                                                   0.3, 0.3, -1., 0.3, -1., -1., 0.3, 0.3, 0.3, 0.3, -1.,
                                                   0.3, -1., 0.3, 0.3, -1., -1., 0.3, 0.3, -1., -1., 0.3,
                                                   0.3, -1., -1., 0.3, -1., -1., -1., 0.3, -1., 0.3, -1.,
                                                   -1., 0.3, -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                                   -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                                   -1., -1., -1., -1., -1., 0.3, -1., -1., -1., 0.3, 0.3,
                                                   -1., -1., -1., -1., 0.3, 0.3, 0.3, -1., 0.3, 0.3, -1.,
                                                   0.3, -1., -1., -1., -1., -1., -1., -1.])).all())
    assert ((np.array(mmo.actionSeqs) == np.ones([2, 4, 4])).all())

    from utils.visualizer import visualizeSymmetry
    if "plot" in argv:
        visualizeSymmetry(mmo.model)
        
def testGetGene(argv):
    # 0
    setting = dict({
        "modelDir": "./test/data/testTetIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })
    
    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    geneOut = mmo.getGene()
    assert (gene == geneOut).all()

    # 1
    setting = dict({
        "modelDir": "./test/data/testDoubleTetIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    geneOut = mmo.getGene()
    assert((gene == geneOut).all())
    
    # 2
    setting = dict({
        "modelDir": "./test/data/lobsterIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    geneOut = mmo.getGene()
    assert((gene == geneOut).all())
    
    # 3
    setting = dict({
        "modelDir": "./test/data/lobsterIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: 1,
            1: 0,
            2: -1,
            3: -1,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    geneOut = mmo.getGene()
    assert ((gene == geneOut).all())
    
    # 4
    setting = dict({
        "modelDir": "./test/data/pillBugIn.json",
        "numChannels": 4,
        "numActions": 4,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: 1,
            1: 0,
            2: -1,
            3: -1,
        },
    })

    mmo = MMO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = mmo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    mmo.loadGene(gene)
    geneOut = mmo.getGene()
    assert ((gene == geneOut).all())

tests = {
    'mmo': testMMO,
    'getGene': testGetGene,
}
