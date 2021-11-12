# multi-objective optimization
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import ray

from utils.model import Model

class MOO:
    class Setting:
        def __init__(self):
            # hyper parameter settings, unrelated to the specific model
            self.nStepsPerCapture = 400  # number of steps to capture one frame of v
            self.modelConfigDir = './data/config.json'
            self.nLoopPreSimulate = 1
            self.nLoopSimulate = 1
    
    class GeneHandler:
        def __init__(self, moo):
            self.moo = moo
            self.model = moo.model
            self.gene: np.ndarray = np.array([])
            
            self.numChannels: int = moo.numChannels
            self.numActions: int = moo.numActions
            self.numObjectives: int = moo.numObjectives
            
            self.channelMirrorMap: Dict[int:int] = moo.channelMirrorMap
            self.invChannelMirrorMap: Dict[int:int] = {v: k for k, v in self.channelMirrorMap.items()}
            self.idsHalfMirrorChannel: List[int] = []
            self.idsMiddleChannel: List[int] = []
            
            self.edgeMirrorMap: Dict[int:int] = moo.model.edgeMirrorMap
            self.invEdgeMirrorMap: Dict[int:int] = {v: k for k, v in self.edgeMirrorMap.items()}
            self.idsHalfMirroredEdge: List[int] = []
            self.idsMiddleEdge: List[int] = []
            
            self.edgeActive: List[bool] = moo.model.edgeActive
            self.idsActiveEdge: List[int] = moo.model.e[self.edgeActive]
            self.idsHalfActiveMirroredEdge: List[int] = []
            self.idsActiveMiddleEdge: List[int] = []
            
            self._precompute()
            
        def _precompute(self):
            # idsHalfMirroredChannel, idsMiddleChannel
            unvisitedChannels = set(self.channelMirrorMap.keys())
            for iChannel in self.channelMirrorMap:
                if iChannel not in unvisitedChannels:
                    continue
                unvisitedChannels.remove(iChannel)
                iChannelMirror = self.channelMirrorMap[iChannel]
                if iChannelMirror == -1:
                    self.idsMiddleChannel.append(iChannel)
                else:
                    self.idsHalfMirrorChannel.append(iChannel)
                    unvisitedChannels.remove(iChannelMirror)
            
            # idsHalfActiveMirroredEdge, idsActiveMiddleEdge
            unvisitedEdges = set(self.edgeMirrorMap.keys())
            for iEdge in self.edgeMirrorMap:
                if iEdge not in unvisitedEdges:
                    continue
                unvisitedEdges.remove(iEdge)
                iEdgeMirror = self.edgeMirrorMap[iEdge]
                if iEdgeMirror == -1:
                    self.idsMiddleEdge.append(iEdge)
                    if self.edgeActive[iEdge]:
                        self.idsActiveMiddleEdge.append(iEdge)
                else:
                    self.idsHalfMirroredEdge.append(iEdge)
                    if self.edgeActive[iEdge]:
                        assert(self.edgeActive[iEdgeMirror])
                        self.idsHalfActiveMirroredEdge.append(iEdge)
                    else:
                        assert(not self.edgeActive[iEdgeMirror])
                    unvisitedEdges.remove(iEdgeMirror)
            
        def _getGeneBounds(self):
            lbChannelMirrorEdge = np.zeros(len(self.idsHalfMirroredEdge))
            ubChannelMirrorEdge = np.ones(len(self.idsHalfMirroredEdge)) * (len(self.channelMirrorMap))
            lbChannelMiddleEdge = np.zeros(len(self.idsMiddleEdge))
            ubChannelMiddleEdge = np.ones(len(self.idsMiddleEdge)) * len(self.idsMiddleChannel)
            
            lbContractActiveMirrorEdge = np.zeros(len(self.idsHalfActiveMirroredEdge))
            ubContractActiveMirrorEdge = np.ones(len(self.idsHalfActiveMirroredEdge)) * self.model.contractionLevels
            lbContractActiveMiddleEdge = np.zeros(len(self.idsActiveMiddleEdge))
            ubContractActiveMiddleEdge = np.ones(len(self.idsActiveMiddleEdge)) * self.model.contractionLevels
            
            lenActionSeqs = self.numObjectives * self.numChannels * self.numActions
            lbActionSeqs = np.zeros(lenActionSeqs)
            ubActionSeqs = np.ones(lenActionSeqs) * 2
            
            lens = [len(ubChannelMirrorEdge), len(ubChannelMiddleEdge),
                    len(ubContractActiveMirrorEdge), len(ubContractActiveMiddleEdge),
                    len(ubActionSeqs)]
            
            return lens, \
                   (lbChannelMirrorEdge, ubChannelMirrorEdge), \
                   (lbChannelMiddleEdge, ubChannelMiddleEdge),\
                   (lbContractActiveMirrorEdge, ubContractActiveMirrorEdge), \
                   (lbContractActiveMiddleEdge, ubContractActiveMiddleEdge), \
                   (lbActionSeqs, ubActionSeqs)
            
        def getGeneSpaces(self):
            ret = self._getGeneBounds()
            lens = ret[0]
            bounds = ret[1:]
            lb = np.hstack([bound[0] for bound in bounds]).astype(int)
            ub = np.hstack([bound[1] for bound in bounds]).astype(int)
            return lb, ub
        
        def getGene(self):
            channelHalfMirrorEdge = []
            channelMiddleEdge = []
            contractActiveHalfMirrorEdge = []
            contractActiveMiddleEdge = []
            actionSeqs = self.moo.actionSeqs.reshape(-1).tolist()
            
            ieVisited = set()
            for ie in range(len(self.model.e)):
                if ie in ieVisited:
                    continue
                ieVisited.add(ie)
                ic = self.model.edgeChannel[ie]
                iContractLevel = self.model.maxContraction[ie] / self.model.contractionInterval
                assert(iContractLevel < 0 or iContractLevel == int(iContractLevel))
                iContractLevel = int(iContractLevel)
                
                ieMirror = self.edgeMirrorMap[ie]
                if ieMirror == -1:
                    assert(ie in self.idsActiveMiddleEdge or not self.model.edgeActive[ie])
                    ic0 = self.idsMiddleChannel.index(ic)
                    channelMiddleEdge.append(ic0)
                    if self.edgeActive[ie]:
                        contractActiveMiddleEdge.append(iContractLevel)
                else:
                    ieVisited.add(ieMirror)
                    assert(ie in self.idsHalfActiveMirroredEdge or not self.model.edgeActive[ie])
                    # ic = self.idsHalfMirrorChannel[ic0]
                    ic0 = ic
                    channelHalfMirrorEdge.append(ic0)
                    if self.edgeActive[ie]:
                        contractActiveHalfMirrorEdge.append(iContractLevel)
            
            gene = np.array(channelHalfMirrorEdge + channelMiddleEdge +
                            contractActiveHalfMirrorEdge + contractActiveMiddleEdge +
                            actionSeqs, dtype=int)
            
            return gene
            
        def loadGene(self, gene: np.ndarray) -> (Model, np.ndarray):     # load gene into model and actionSeqs
            if gene.shape == ():
                return self.model, self.moo.actionSeqs
            
            ret = self._getGeneBounds()
            lens = ret[0]
            bounds = ret[1:]
            
            i = 0
            channelHalfMirrorEdge = gene[i: i + lens[0]]
            i += lens[0]
            channelMiddleEdge = gene[i: i + lens[1]]
            i += lens[1]
            contractActiveHalfMirrorEdge = gene[i: i + lens[2]]
            i += lens[2]
            contractActiveMiddleEdge = gene[i: i + lens[3]]
            i += lens[3]
            actionSeqs = gene[i: i + lens[4]]
            i += lens[4]
            assert(i == len(gene))

            self.model.edgeChannel *= 0
            self.model.edgeChannel += -1
            self.model.maxContraction *= 0
            self.model.maxContraction += -1

            # actionSeqs
            self.moo.actionSeqs = actionSeqs.reshape(self.numObjectives, self.numChannels, self.numActions)
            
            # channelHalfMirrorEdge
            for ie0, ic0 in enumerate(channelHalfMirrorEdge):
                ie = self.idsHalfMirroredEdge[ie0]
                ieMirror = self.edgeMirrorMap[ie]
                assert(ieMirror != -1)
                # ic = self.idsHalfMirrorChannel[ic0]
                ic = ic0
                icMirror = self.channelMirrorMap[ic]
                if icMirror == -1:
                    icMirror = ic
                self.model.edgeChannel[ie] = ic
                self.model.edgeChannel[ieMirror] = icMirror
                
            # channelMiddleEdge
            for ie0, ic0 in enumerate(channelMiddleEdge):
                ie = self.idsMiddleEdge[ie0]
                assert (self.edgeMirrorMap[ie] == -1)
                ic = self.idsMiddleChannel[ic0]
                assert (self.channelMirrorMap[ic] == -1)
                self.model.edgeChannel[ie] = ic
            
            # contractActiveHalfMirrorEdge
            for ie0, contLev in enumerate(contractActiveHalfMirrorEdge):
                ie = self.idsHalfActiveMirroredEdge[ie0]
                ieMirror = self.edgeMirrorMap[ie]
                assert(ieMirror != -1)
                cont = contLev * self.model.contractionInterval
                self.model.maxContraction[ie] = cont
                self.model.maxContraction[ieMirror] = cont
            
            # contractActiveMiddleEdge
            for ie0, contLev in enumerate(contractActiveMiddleEdge):
                ie = self.idsActiveMiddleEdge[ie0]
                assert(self.edgeMirrorMap[ie] == -1)
                cont = contLev * self.model.contractionInterval
                self.model.maxContraction[ie] = cont

            assert ((self.model.maxContraction == -1).sum() == (~self.model.edgeActive).sum())
            
            self.gene = gene.copy()
            self.moo.gene = gene.copy()
            
            return self.model, self.moo.actionSeqs
        
        def initChannel(self):
            # assign the channels for the model
            incidenceMat = self.moo.incidenceMatrix     # row: v, col: e
            
            edgeChannel = np.ones(len(self.model.e)) * -1
            
            # assign one edge for each channel
            channelUnvisited = set([i for i in range(len(self.channelMirrorMap))])
            while len(channelUnvisited) != 0:
                ic = channelUnvisited.pop()
                icMirror = self.channelMirrorMap[ic]
                
                if icMirror == -1:  # middle channel
                    # pick one middle edge
                    idsAvailableMiddleEdge = np.array(self.idsMiddleEdge)[np.where((edgeChannel[self.idsMiddleEdge]) == -1)[0]]
                    ie = np.random.choice(idsAvailableMiddleEdge)
                    edgeChannel[ie] = ic
                    
                else:   # mirrored channel
                    channelUnvisited.remove(icMirror)
                    # pick one mirror edge
                    idsAvailableHalfMirroredEdge = np.array(self.idsHalfMirroredEdge)[np.where((edgeChannel[self.idsHalfMirroredEdge]) == -1)[0]]
                    ie = np.random.choice(idsAvailableHalfMirroredEdge)
                    ieMirror = self.edgeMirrorMap[ie]
                    assert(edgeChannel[ie] == edgeChannel[ieMirror] == -1)
                    edgeChannel[ie] = ic
                    edgeChannel[ieMirror] = icMirror
            print(edgeChannel)
            edgeUnvisited = set(np.where(edgeChannel == -1)[0])
            
            # assign all edges to channels
            while len(edgeUnvisited) != 0:
                # pick a channel
                ic = np.random.choice(len(self.channelMirrorMap))
                icMirror = self.channelMirrorMap[ic]
                
                # iesAvailable
                iesOfChannel = np.where(edgeChannel == ic)[0]     # ids of edge in channel ic
                ivsOfChannel = np.array(list(set(self.model.e[iesOfChannel].reshape(-1))))
                subIncidenceMat = incidenceMat[ivsOfChannel]    # row: v of channel ic, col: edge
                iesConnected = set(np.where(subIncidenceMat == 1)[1])    # id of es connected to vs in channel
                iesAvailable = []
                for ie in iesConnected:
                    ieMirror = self.edgeMirrorMap[ie]
                    if ie in edgeUnvisited:
                        if icMirror != -1:  # mirror channel
                            if ieMirror == -1:  # middle edge
                                continue
                        iesAvailable.append(ie)
                if len(iesAvailable) == 0:
                    print('channel {} not available'.format(ic))
                    continue
                print(ic, iesConnected, iesAvailable)
                
                # pick an edge and assign
                ie = np.random.choice(iesAvailable)
                edgeChannel[ie] = ic
                edgeUnvisited.remove(ie)
                print(ie)
                
                ieMirror = self.edgeMirrorMap[ie]
                if ieMirror != -1:  # mirror edge
                    edgeChannel[ieMirror] = icMirror
                    print('mirror', ieMirror)
                    assert(icMirror != -1)
                    edgeUnvisited.remove(ieMirror)
            
            assert((edgeChannel == -1).sum() == 0)
            return edgeChannel
            
    def __init__(self, setting):
        self.modelDir: str = ""
        self.numChannels: int = -1
        self.numActions: int = -1
        self.numObjectives: int = -1
        self.channelMirrorMap: dict = dict()
        
        self.objectives: List = []
        self.actionSeqs: np.ndarray = np.zeros([])
        self.gene: np.ndarray = np.zeros([])

        self.incidenceMatrix = None
        self.model: Model = Model()
        self.setting = MOO.Setting()
        self._loadSetting(setting)
        self.geneHandler = self.GeneHandler(self)
    
    def _loadSetting(self, setting: Dict):
        for key in setting:
            if hasattr(self, key):
                assert (type(setting[key]) == type(getattr(self, key)))
                setattr(self, key, setting[key])
            else:
                assert (hasattr(self.setting, key))
                assert (type(setting[key]) == type(getattr(self.setting, key)))
                setattr(self.setting, key, setting[key])
        
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
        self.model = Model(self.setting.modelConfigDir)
        self.model.load(self.modelDir)
        # TODO: quick fix
        self.model.numChannels = self.numChannels
        if self.numChannels < 4:
            self.model.setToSingleChannel()
            
        self.model._reset()
        
        # compute incidenceMatrix
        G = nx.Graph()
        edgeMap = dict()
        for ie, e in enumerate(self.model.e):
            G.add_edge(e[0], e[1])
            edgeMap[(e[0], e[1])] = ie
            edgeMap[(e[1], e[0])] = ie
        self.incidenceMatrix = nx.linalg.graphmatrix.incidence_matrix(G).toarray()
        
        # reorder incidenceMatrix
        im = np.zeros_like(self.incidenceMatrix)
        for icol, e in enumerate(G.edges()):
            icolNext = edgeMap[e]
            im[:, icolNext] = self.incidenceMatrix[:, icol]
        self.incidenceMatrix = np.array(im, dtype=bool)
        
    def refreshModel(self):
        self._loadModel()
        self.loadGene(self.gene)
    
    # gene
    
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

    def loadGene(self, gene: np.ndarray) -> (Model, np.ndarray):  # load gene into model and actionSeqs
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
        assert ((self.model.edgeChannel == -1).sum() == 0)
        assert (iEdgeChannelMiddle == len(edgeChannelMiddle))
        
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
    
    # init Gene
    def _getAvailableEdges(self, edgeChannel, iChannel):
        iesOfChannel = np.where(edgeChannel == iChannel)[0]  # id of edges belonging to the channel
        ivsOfChannel = np.array(list(set(self.model.e[iesOfChannel].reshape(-1))))  # id of vertices belonging to the channel
        incidenceMatrix = self.incidenceMatrix[ivsOfChannel]
        assert(incidenceMatrix.ndim == 2)
        boolEdgesIncidence = incidenceMatrix.sum(0) > 0
        # breakpoint()
        return boolEdgesIncidence
        
    def _initEdgeChannel(self, test=False):
        boolEdgeMiddle = np.array([self.model.edgeMirrorMap[i] for i in range(len(self.model.edgeMirrorMap))]) == -1
        boolEdgeHalf = ~boolEdgeMiddle
        idsEdge = np.arange(len(self.model.e))
        edgeChannel = -np.ones(len(self.model.e))
        edgeChannels = []
        
        idsEdgesMiddle = lambda: idsEdge[boolEdgeMiddle * (edgeChannel == -1)]     # middle and active and not assigned
        idsEdgesHalf = lambda: idsEdge[boolEdgeHalf * (edgeChannel == -1)]         # half and active and not assigned
        anyEdgeNotAssigned = lambda: (edgeChannel == -1).any()
        
        # assign the first edge for each channel
        iCVisited = set()
        for iChannel in range(len(self.channelMirrorMap)):
            if iChannel in iCVisited:
                continue
            iCVisited.add(iChannel)
            iChannelMirrored = self.channelMirrorMap[iChannel]
            
            if iChannelMirrored == -1:   # self-mirrored
                ie = np.random.choice(idsEdgesMiddle())
                edgeChannel[ie] = iChannel
            else:   # mirrored
                iCVisited.add(iChannelMirrored)
                ie = np.random.choice(idsEdgesHalf())
                edgeChannel[ie] = iChannel
                ieMirrored = self.model.edgeMirrorMap[ie]
                assert(edgeChannel[ieMirrored] == -1)
                edgeChannel[ieMirrored] = iChannelMirrored  # assign mirrored channel value to the mirrored edge
            
            if test:
                assert (isinstance(edgeChannel, np.ndarray))
                edgeChannels.append(edgeChannel.copy())
                
        # assign edges
        while anyEdgeNotAssigned():
            iChannel = np.random.choice(len(self.channelMirrorMap))
            iChannelMirrored = self.channelMirrorMap[iChannel]
            
            boolEdgesConnected = self._getAvailableEdges(edgeChannel, iChannel)
            
            if iChannelMirrored == -1:  # self-mirrored
                boolEdgesAvailable = boolEdgesConnected * (edgeChannel == -1)
                
                try:
                    ie = np.random.choice(idsEdge[boolEdgesAvailable])
                except:
                    print('noooo')
                    continue
                    # breakpoint()
                ieMirrored = self.model.edgeMirrorMap[ie]
                edgeChannel[ie] = iChannel
                if ieMirrored != -1:    # mirrored edge
                    assert(edgeChannel[ieMirrored] == -1)
                    edgeChannel[ieMirrored] = iChannel
                
            else:   # mirrored edge
                boolEdgesAvailable = boolEdgesConnected * boolEdgeHalf * (edgeChannel == -1)
                
                try:
                    ie = np.random.choice(idsEdge[boolEdgesAvailable])
                except:
                    print('nooo')
                    continue
                    # breakpoint()
                ieMirrored = self.model.edgeMirrorMap[ie]
                edgeChannel[ie] = iChannel
                assert(ieMirrored != -1)
                assert(edgeChannel[ieMirrored] == -1)
                edgeChannel[ieMirrored] = iChannelMirrored
            
            if test:
                assert(isinstance(edgeChannel, np.ndarray))
                edgeChannels.append(edgeChannel.copy())
        
        def edgeChannelToHalfNMiddle(edgeChannel):
            iChannelMiddle = np.where(np.array(list(self.channelMirrorMap.values())) == -1)[0]
            mapChannelToMiddleChannel = {ic: list(self.channelMirrorMap.keys()).index(ic) for ic in iChannelMiddle} # e.g. 1, 3, 4 -> 0, 1 2
            if test:
                mapChannelToMiddleChannel[-1] = -1
            
            edgeChannelMiddle = np.array(list(map(lambda c: mapChannelToMiddleChannel[c], edgeChannel[boolEdgeMiddle])), dtype=int)
            
            boolEdgeHalfMirrored = boolEdgeHalf.copy()  # the mirrored edge is not required
            for ie in range(len(self.model.edgeMirrorMap)):
                if boolEdgeHalfMirrored[ie] == False:
                    continue
                ieMirrored = self.model.edgeMirrorMap[ie]
                assert(boolEdgeHalfMirrored[ieMirrored] == True)
                boolEdgeHalfMirrored[ieMirrored] = False
            
            edgeChannelHalf = np.array(edgeChannel[boolEdgeHalfMirrored], dtype=int)
            return edgeChannelMiddle.copy(), edgeChannelHalf.copy()
            
        if False:
            edgeChannelMiddles = []
            edgeChannelHalfs = []
            for edgeChannel in edgeChannels:
                edgeChannelMiddlee, edgeChannelHalff = edgeChannelToHalfNMiddle(edgeChannel)
                edgeChannelMiddles.append(edgeChannelMiddlee.copy())
                edgeChannelHalfs.append(edgeChannelHalff.copy())
            return edgeChannelMiddles, edgeChannelHalfs
        else:
            edgeChannelMiddlee, edgeChannelHalff = edgeChannelToHalfNMiddle(edgeChannel)
            return edgeChannelHalff, edgeChannelMiddlee
    
    def initGene(self, test=False):
        edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = self._getGeneSpaces()
        
        lbHalf = np.hstack([contractionLevelSpace[0], actionSeqsSpace[0]])
        ubHalf = np.hstack([contractionLevelSpace[1], actionSeqsSpace[1]])
        popHalf = np.random.randint(lbHalf, ubHalf)
        pops = []
        
        # randomize edge channels
        if True:
            edgeChannelHalves, edgeChannelMiddles = self._initEdgeChannel(test=True)
            edgeChannelHalf = edgeChannelHalves[0]
            edgeChannelMiddle = edgeChannelMiddles[0]
            edgeChannelHalf, edgeChannelMiddle = self._initEdgeChannel(test=True)
            assert ((edgeChannelHalf < edgeChannelHalfSpace[1]).all() and (edgeChannelHalf >= edgeChannelHalfSpace[0]).all())
            assert ((edgeChannelMiddle < edgeChannelMiddleSpace[1]).all() and (edgeChannelMiddle >= edgeChannelMiddleSpace[0]).all())
            # for i in range(len(edgeChannelHalves)):
            if True:
                i = -1
                pop = np.hstack([edgeChannelHalves[i], edgeChannelMiddles[i], popHalf.copy()])
                pops.append(pop.copy())
            # breakpoint()
            pop = np.hstack([edgeChannelHalf, edgeChannelMiddle, popHalf])
            return pop
        else:
            edgeChannelHalf, edgeChannelMiddle = self._initEdgeChannel(test=True)
            assert ((edgeChannelHalf < edgeChannelHalfSpace[1]).all() and (edgeChannelHalf >= edgeChannelHalfSpace[0]).all())
            assert ((edgeChannelMiddle < edgeChannelMiddleSpace[1]).all() and (edgeChannelMiddle >= edgeChannelMiddleSpace[0]).all())
        
            pop = np.hstack([edgeChannelHalf, edgeChannelMiddle, popHalf])
            return pop
    
    def simulate(self, actionSeq, nLoops=1, visualize=False, testing=False) -> (np.ndarray, np.ndarray):
        assert (actionSeq.ndim == 2)
        assert (actionSeq.shape[0] >= 1)
        assert (actionSeq.shape[1] >= 1)
    
        T = Model.numStepsPerActuation
        nStepsPerCapture = self.setting.nStepsPerCapture
    
        #  initialize with the last action
        self.refreshModel()
        model = self.model
    
        model.inflateChannel = actionSeq[:, -1]
        v = model.step(T * self.setting.nLoopPreSimulate, ret=True)
        vs = [v]
    
        for iLoop in range(self.setting.nLoopSimulate):
            for iAction in range(len(actionSeq[0])):
                model.inflateChannel = actionSeq[:, iAction]
                for iStep in range(T):
                    # append at the beginning of every nStepsPerCapture including frame 0
                    if model.numSteps % nStepsPerCapture == 0:
                        v = model.step(ret=True)
                        vs.append(v)
                    else:
                        model.step()
        vs.append(model.v.copy())   # last frame
        vs = np.array(vs)
        assert (vs.shape == (vs.shape[0], len(model.v), 3))
    
        return vs, self.model.e.copy()

    def check(self):
        if len(self.channelMirrorMap) != 0:
            assert(self.numChannels == len(self.channelMirrorMap))
            assert(self.numChannels >= self.model.edgeChannel.max() + 1)
        if len(self.objectives) != 0:
            assert(len(self.objectives) == self.numObjectives)
        # if self.actionSeqs.shape != ():
        #     assert(self.actionSeqs.shape == (self.numObjectives, self.numChannels, self.numActions))
        
    
def testMOO(argv):
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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    
    assert ((np.array(spaceChannelMirror) == np.array([[0., 0.], [2., 2.]])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0., 0.], [2., 2.]])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0., 0.], [5, 5]])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0., 0.], [5, 5]])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]])).all())
    
    # assert False
    assert( (np.array(edgeChannelHalfSpace) == np.array([[0., 0.], [2., 2.]])).all() )
    assert( (np.array(edgeChannelMiddleSpace) == np.array([[0., 0.], [2., 2.]])).all() )
    assert( (np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0.], [5, 5, 5, 5]]) ).all() )
    assert( (np.array(actionSeqsSpace) ==
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]] )).all() )
    
    gene = np.hstack([edgeChannelHalfSpace[1]-1, edgeChannelMiddleSpace[1]-1,
                      contractionLevelSpace[1]-1, actionSeqsSpace[1]-1]).astype(int)
    moo.geneHandler.loadGene(gene)
    assert( (moo.model.edgeChannel == np.array([1, 1, 1, 1, 1, 1])).all())
    assert ((moo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.array([[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]])).all())
    
    moo.loadGene(gene)
    assert( (moo.model.edgeChannel == np.array([1, 1, 1, 1, 1, 1])).all())
    assert ((moo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.array([[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]])).all())
    
    
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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()

    assert ((np.array(spaceChannelMirror) == np.array([[0., 0.], [4., 4.]])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0., 0.], [2., 2.]])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0., 0.], [5, 5]])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0., 0.], [5, 5]])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0]*32,
                       [2]*32])).all())

    assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0.], [4, 4]])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0.], [2., 2.]])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0.], [5, 5, 5, 5]])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0]*32,
                       [2]*32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([3, 1, 2, 1, 3, 2])).all())
    assert ((moo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    moo.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([3, 1, 2, 1, 3, 2])).all())
    assert ((moo.model.maxContraction == np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())

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
        "modelConfigDir": "./data/config_0.json",
    })

    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    assert ((np.array(edgeChannelHalfSpace) == np.array([[0., 0., 0.], [4, 4, 4]])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0., 0., 0.], [2., 2., 2.]])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0., 0., 0., 0., 0.], [5, 5, 5, 5, 5]])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0] * 32,
                       [2] * 32])).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    
    assert ((np.array(spaceChannelMirror) == np.array([[0., 0., 0.], [4, 4, 4]])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0., 0., 0.], [2., 2., 2.]])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0., 0.], [5, 5]])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0., 0., 0.], [5, 5, 5]])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0]*32,
                       [2]*32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    
    assert ((moo.model.edgeChannel == np.array([3, 2, 1, 3, 1, 2, 3, 2, 1])).all())
    assert ((moo.model.maxContraction == np.array([ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3, -1. , -1. ,  0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())

    moo.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([3, 2, 1, 3, 1, 2, 3, 2, 1])).all())
    assert ((moo.model.maxContraction == np.array([ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3, -1. , -1. ,  0.3])).all())
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    from utils.visualizer import visualizeChannel
    if "plot" in argv:
        visualizeChannel(moo.model)

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
        "modelConfigDir": "./data/config_0.json",
    })

    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()

    assert ((np.array(edgeChannelHalfSpace) == np.array([[0] * 64, [4]*64])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0]*12, [2.]*12])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0.]*23, [5]*23])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0] * 32,
                       [2] * 32])).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()

    assert ((np.array(spaceChannelMirror) == np.array([[0] * 64, [4]*64])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0]*12, [2.]*12])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0.]*21, [5]*21])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0.]*2, [5]*2])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0] * 32,
                       [2] * 32])).all())

    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    
    moo.geneHandler.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3])).all())
    assert ((moo.model.maxContraction == np.array([ 0.3, -1. , -1. , -1. , -1. ,  0.3, -1. , -1. ,  0.3, -1. , -1. ,
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
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    moo.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3])).all())
    assert ((moo.model.maxContraction == np.array([0.3, -1., -1., -1., -1., 0.3, -1., -1., 0.3, -1., -1.,
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
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())

    from utils.visualizer import visualizeChannel
    if "plot" in argv:
        visualizeChannel(moo.model)

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
        "modelConfigDir": "./data/config_0.json",
    })

    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()

    assert ((np.array(edgeChannelHalfSpace) == np.array([[0.]*64, [4]*64])).all())
    assert ((np.array(edgeChannelMiddleSpace) == np.array([[0]*12, [2]*12])).all())
    assert ((np.array(contractionLevelSpace) == np.array([[0]*23, [5]*23])).all())
    assert ((np.array(actionSeqsSpace) ==
             np.array([[0] * 32,
                       [2] * 32])).all())
    
    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()

    assert ((np.array(spaceChannelMirror) == np.array([[0] * 64, [4]*64])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0]*12, [2.]*12])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0.]*21, [5]*21])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0.]*2, [5]*2])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0] * 32,
                       [2] * 32])).all())
    
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([1, 3, 2, 3, 2, 1, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3,
       3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 1, 3, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 3, 2, 2, 2, 3, 3, 3, 2, 2, 1, 3, 1,
       2, 3, 2, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2,
       2, 2, 2, 2, 2, 1, 3, 2])).all())
    assert ((moo.model.maxContraction == np.array([0.3, -1., -1., -1., -1., 0.3, -1., -1., 0.3, -1., -1.,
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
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([1, 3, 2, 3, 2, 1, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3,
       3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 1, 3, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 3, 2, 2, 2, 3, 3, 3, 2, 2, 1, 3, 1,
       2, 3, 2, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2,
       2, 2, 2, 2, 2, 1, 3, 2])).all())
    assert ((moo.model.maxContraction == np.array([0.3, -1., -1., -1., -1., 0.3, -1., -1., 0.3, -1., -1.,
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
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    from utils.visualizer import visualizeChannel
    if "plot" in argv:
        visualizeChannel(moo.model)

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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert (gene == geneOut).all()
    
    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                     spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    geneOut = moo.geneHandler.getGene()
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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert ((gene == geneOut).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    geneOut = moo.geneHandler.getGene()
    assert (gene == geneOut).all()
    
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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert ((gene == geneOut).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    geneOut = moo.geneHandler.getGene()
    assert (gene == geneOut).all()

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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert ((gene == geneOut).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    geneOut = moo.geneHandler.getGene()
    assert (gene == geneOut).all()

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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    edgeChannelHalfSpace, edgeChannelMiddleSpace, contractionLevelSpace, actionSeqsSpace = moo._getGeneSpaces()
    
    gene = np.hstack([edgeChannelHalfSpace[1] - 1, edgeChannelMiddleSpace[1] - 1,
                      contractionLevelSpace[1] - 1, actionSeqsSpace[1] - 1]).astype(int)
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert ((gene == geneOut).all())

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    geneOut = moo.geneHandler.getGene()
    assert (gene == geneOut).all()

def testSimulate(argv):
    # 6
    setting = dict({
        "modelDir": "./test/data/lobsterIn.json",
        "numChannels": 4,
        "numActions": 4,
        "nLoopPreSimulate": 2,
        "nLoopSimulate": 2,
        "numObjectives": 2,
        "channelMirrorMap": {
            0: -1,
            1: -1,
            2: 3,
            3: 2,
        },
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)

    lens, spaceChannelMirror, spaceChannelMiddle, spaceContractMirror, spaceContractMiddle, spaceActionSeqs \
        = moo.geneHandler._getGeneBounds()
    assert ((np.array(spaceChannelMirror) == np.array([[0] * 64, [4] * 64])).all())
    assert ((np.array(spaceChannelMiddle) == np.array([[0] * 12, [2.] * 12])).all())
    assert ((np.array(spaceContractMirror) == np.array([[0.] * 21, [5] * 21])).all())
    assert ((np.array(spaceContractMiddle) == np.array([[0.] * 2, [5] * 2])).all())
    assert ((np.array(spaceActionSeqs) ==
             np.array([[0] * 32,
                       [2] * 32])).all())
    
    gene = np.hstack([spaceChannelMirror[1] - 1, spaceChannelMiddle[1] - 1, spaceContractMirror[1] - 1,
                      spaceContractMiddle[1] - 1, spaceActionSeqs[1] - 1]).astype(int)
    moo.geneHandler.loadGene(gene)
    assert ((moo.model.edgeChannel == np.array([1, 3, 2, 3, 2, 1, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3,
                                                3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 1, 3, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
                                                2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 3, 2, 2, 2, 3, 3, 3, 2, 2, 1, 3, 1,
                                                2, 3, 2, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2,
                                                2, 2, 2, 2, 2, 1, 3, 2])).all())
    assert ((moo.model.maxContraction == np.array([0.3, -1., -1., -1., -1., 0.3, -1., -1., 0.3, -1., -1.,
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
    assert ((np.array(moo.actionSeqs) == np.ones([2, 4, 4])).all())
    
    vs, e = moo.simulate(moo.actionSeqs[0])
    assert((vs[40][40:] - np.array([[-5.01474011, -2.53568316,  0.84041267],
       [-6.04074005, -1.25597253,  0.34191512],
       [-5.17598253,  0.3160106 ,  0.84041532],
       [-6.05118767, -1.07119833,  0.34191533],
       [-3.35309915, -2.44564473,  0.        ],
       [-3.51477444,  0.41385203,  0.        ],
       [ 0.34776737, -0.80208073,  0.24659735],
       [ 0.78374118,  1.46975112,  0.        ],
       [ 0.74655325, -1.19672941,  0.        ]])).mean() ** 2 < 1e-21)

def testInitChannel(argv):
    from utils.visualizer import visualizeChannel

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
        "modelConfigDir": "./data/config_0.json",
    })

    moo = MOO(setting)
    edgeChannel = moo.geneHandler.initChannel()
    moo.model.edgeChannel = edgeChannel

    if "plot" in argv:
        visualizeChannel(moo.model)

#TODO test
def testInitGene(argv):
    from utils.visualizer import visualizeChannel
    
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
        "modelConfigDir": "./data/config_0.json",
    })

    moo = MOO(setting)
    gene = moo.initGene()
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert((gene == geneOut).all())

    if "plot" in argv:
        visualizeChannel(moo.model)

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
        "modelConfigDir": "./data/config_0.json",
    })
    
    moo = MOO(setting)
    gene = moo.initGene(test=False)
    moo.loadGene(gene)
    geneOut = moo.getGene()
    assert ((gene == geneOut).all())

    if "plot" in argv:
        visualizeChannel(moo.model)

    # # 2
    # setting = dict({
    #     "modelDir": "./test/data/lobsterIn.json",
    #     "numChannels": 4,
    #     "numActions": 4,
    #     "numObjectives": 2,
    #     "channelMirrorMap": {
    #         0: -1,
    #         1: -1,
    #         2: 3,
    #         3: 2,
    #     },
    #     "modelConfigDir": "./data/config_0.json",
    # })
    #
    # moo = moo(setting)
    # gene = moo.initGene(test=False)
    # moo.loadGene(gene)
    # geneOut = moo.getGene()
    # assert ((gene == geneOut).all())
    #
    # if "plot" in argv:
    #     visualizeSymmetry(moo.model)
    #
    
    # 3
    # setting = dict({
    #     "modelDir": "./test/data/pillBugIn.json",
    #     "numChannels": 4,
    #     "numActions": 4,
    #     "numObjectives": 2,
    #     "channelMirrorMap": {
    #         0: -1,
    #         1: -1,
    #         2: 3,
    #         3: 2,
    #     },
    #     "modelConfigDir": "./data/config_0.json",
    # })
    #
    # moo = moo(setting)
    # gene = moo.initGene(test=False)
    # moo.loadGene(gene)
    # geneOut = moo.getGene()
    # assert ((gene == geneOut).all())
    #
    # if "plot" in argv:
    #     visualizeSymmetry(moo.model)
    #
    # moo = moo(setting)
    # gene = moo.initGene(test=True)
    # # breakpoint()
    # # gene = genes[-1]
    # moo.loadGene(gene)
    # geneOut = moo.getGene()
    # assert ((gene == geneOut).all())
    #
    # if "plot" in argv:
    #     visualizeSymmetry(moo.model)
    #
    # # for gene in genes:
    # if True:
    #     for i in range(len(gene)):
    #         gene[i] = 0 if gene[i] == -1 else gene[i]
    #
    #     moo = moo(setting)
    #     moo.loadGene(gene)
    #     geneOut = moo.getGene()
    #     assert ((gene == geneOut).all())
    #
    #     if "plot" in argv:
    #         visualizeSymmetry(moo.model)

    # # 4
    # setting = dict({
    #     "modelDir": "./test/data/lobsterIn.json",
    #     "numChannels": 4,
    #     "numActions": 4,
    #     "numObjectives": 2,
    #     "channelMirrorMap": {
    #         0: -1,
    #         1: -1,
    #         2: 3,
    #         3: 2,
    #     },
    #     "modelConfigDir": "./data/config_0.json",
    # })
    #
    # moo = moo(setting)
    # genes = moo.initGene(test=True)
    # for gene in genes:
    #     for i in range(len(gene)):
    #         gene[i] = 0 if gene[i] == -1 else gene[i]
    #
    #     moo.loadGene(gene)
    #     geneOut = moo.getGene()
    #     assert ((gene == geneOut).all())
    #
    #     if "plot" in argv:
    #         visualizeSymmetry(moo.model)
    
    ## 5
    # setting = dict({
    #     "modelDir": "./test/data/peacock.json",
    #     "numChannels": 4,
    #     "numActions": 4,
    #     "numObjectives": 2,
    #     "channelMirrorMap": {
    #         0: -1,
    #         1: -1,
    #         2: 3,
    #         3: 2,
    #     },
    #     "modelConfigDir": "./data/config_0.json",
    # })
    #
    # moo = moo(setting)
    # gene = moo.initGene()
    # moo.loadGene(gene)
    # geneOut = moo.getGene()
    # assert ((gene == geneOut).all())
    #
    # if "plot" in argv:
    #     visualizeSymmetry(moo.model)


    
tests = {
    'moo': testMOO,
    'getGene': testGetGene,
    'simulate': testSimulate,
    'initChannel': testInitChannel,
    'initGene': testInitGene,
}
