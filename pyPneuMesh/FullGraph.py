import numpy as np
import copy

from pyPneuMesh.Model import Model

class FullGraph(object):
    eAdjacencyList = None
    
    def __init__(self, model:Model, graphSetting):
        self.channelMirrorMap = graphSetting['channelMirrorMap'].copy()
        self.numChannels = len(self.channelMirrorMap)
        
        self.model = model
        
        self.contractions = model.contractionLevel.copy()
        self.channels = model.edgeChannel.copy()
        
        
        incM = np.zeros([len(self.model.v0), len(self.model.e)])
        for ie, e in enumerate(self.model.e):
            incM[e[0], ie] = 1
            incM[e[1], ie] = 1
        
        if FullGraph.eAdjacencyList is None:
            FullGraph.eAdjacencyList = []
            for ie in range(len(self.model.e)):
                iv0 = self.model.e[ie, 0]
                iv1 = self.model.e[ie, 1]
                iesConnected0 = np.where(incM[iv0] == 1)[0]
                iesConnected1 = np.where(incM[iv1] == 1)[0]
                iesConnected = np.concatenate([iesConnected0, iesConnected1])
                s = set(iesConnected.tolist())
                s.remove(ie)
                iesConnected = np.array(list(s))
                FullGraph.eAdjacencyList.append(iesConnected)
        
        # create
        
        # self.numChannels = model.numChannel
        # self.nV = model.nV
        # self.nE = model.nE
        #
        # self.ivsSub = ivsSub  # mapping from original vertex indices to new indices of the subGraph
        # self.iesSub = iesSub  # mapping from original edge indices to new indices of the subGraph
        # self.esSub = esSub  # ne x 2, edges of the subGraph
        # self.channels = np.ones(self.nE) * -1  # ne, indices of channels of edges
        # self.contractions = np.zeros(
        #     self.nE)  # ne, int value of contractions, Model.contractionLevels number of types of contractions
        #
        # self.esChannels = []
        #
        # self.incM = np.zeros([self.nV, self.nE])  # vertex-edge incidence matrix
        # for ie, e in enumerate(self.esSub):
        #     self.incM[e[0], ie] = 1
        #     self.incM[e[1], ie] = 1
        #
        # self.ieAdjList = []  # nE x X, each row includes indices of adjacent edges of ie, np.array
        # for ie in range(self.nE):
        #     iv0 = self.esSub[ie, 0]
        #     iv1 = self.esSub[ie, 1]
        #     iesConnected0 = np.where(self.incM[iv0] == 1)[0]
        #     iesConnected1 = np.where(self.incM[iv1] == 1)[0]
        #     iesConnected = np.concatenate([iesConnected0, iesConnected1])
        #     self.ieAdjList.append(iesConnected)
        #
        # self.init()
        pass
    
    def saveGraphSetting(self, folderDir, name):
        #TODO: implement this
        
        
        pass
    
    def getGraphSetting(self):
        graphSetting = {
            'symmetric': False,
            'channelMirrorMap': self.channelMirrorMap
        }
        return copy.deepcopy(graphSetting)
    
    def toModel(self):
        self.model.contractionLevel = self.contractions.copy()
        self.model.edgeChannel = self.channels.copy()
    
    def randomize(self, graphRandomize=True, contractionRandomize=True):
        if contractionRandomize:
            self.contractions = np.random.randint(0, self.model.NUM_CONTRACTION_LEVEL, self.contractions.shape)
        
        if graphRandomize:
            stuck = True
            while stuck:
                stuck = False
                
                self.channels *= 0
                self.channels += -1
                
                iesUnassigned = set(np.arange(len(self.contractions)))
                
                # init one edge for channels
                for iChannel in range(self.numChannels):
                    ie = np.random.choice(list(iesUnassigned))
                    iesUnassigned.remove(ie)
                    self.channels[ie] = iChannel
                
                # grow channels to fill the entire graph
                numNotUpdate = 0
                while iesUnassigned:
                    numNotUpdate += 1
                    
                    if numNotUpdate > 100:
                        stuck = True
                        break
                    
                    ic = np.random.choice(self.numChannels)
                    iesUnassignedAroundChannel = self.iesAroundChannel(ic, unassigned=True)
                    if iesUnassignedAroundChannel is not None:
                        ie = np.random.choice(iesUnassignedAroundChannel)
                        iesUnassigned.remove(ie)
                        self.channels[ie] = ic
                        numNotUpdate = 0
                    else:
                        continue
        
        self.toModel()
    
    def iesAroundChannel(self, ic, unassigned=True):
        # boolEsChannel: np, numEdges, bool, True if edge is in channel ic
        boolEsChannel = self.channels == ic
        # iEdges: np, numEdges, int, indices of edges in channel ic
        iEdgesInChannel = np.arange(len(self.contractions))[boolEsChannel]
        
        al = FullGraph.eAdjacencyList
        
        iesConnectedAll = []
        for ie in iEdgesInChannel:
            iesConnected = al[ie]
            iesConnectedAll.append(iesConnected)
        # print(iEdgesInChannel)
        iesConnectedAll = np.concatenate(iesConnectedAll)
        
        iesConnectedAll = list(set(iesConnectedAll.tolist()))
        
        if unassigned:
            iesConnectedAll = [ie for ie in iesConnectedAll if self.channels[ie] == -1]
        
        if len(iesConnectedAll) == 0:
            return None
        
        return np.array(iesConnectedAll)
        
    def mutate(self, graphMutationChance, contractionMutationChance):
        if np.random.random() < graphMutationChance:
            self.mutateGraph()
        
        self.mutateContraction(contractionMutationChance)
        self.toModel()

    def mutateGraph(self):
        # randomly choose one channel
        
        iess = []
        for ic in range(self.numChannels):
            iess.append(self.iesAroundChannel(ic, unassigned=False))
        ies = np.array(list(set(np.concatenate(iess))))
        
        succeeded = False
        while not succeeded:
            ie = np.random.choice(ies)
            icOriginal = self.channels[ie]
            
            iesAdjacent = FullGraph.eAdjacencyList[ie]
            icsAdjacent = set(self.channels[iesAdjacent].tolist())
            if icOriginal in icsAdjacent:
                icsAdjacent.remove(icOriginal)
            icsAdjacent = list(icsAdjacent)
            
            while len(icsAdjacent) > 0:
                ic = np.random.choice(icsAdjacent)
                icsAdjacent.remove(ic)
                self.channels[ie] = ic
                
                channelsConnected = self.channelsConnected()
                
                if channelsConnected:
                    succeeded = True
                    break
                else:
                    self.channels[ie] = icOriginal
            
                    
    def mutateContraction(self, contractionMutationChance):
        maskMutation = np.random.rand(len(self.contractions))
        contraction = np.random.randint(
            np.zeros(len(self.contractions)), self.model.NUM_CONTRACTION_LEVEL * np.ones(len(self.contractions)))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < contractionMutationChance:
                self.contractions[ie] = contraction[ie]

    def cross(self, graph, chance):
        maskMutation = np.random.rand(len(self.contractions))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < chance:
                tmp = self.contractions[ie]
                self.contractions[ie] = graph.contractions[ie]
                graph.contractions[ie] = tmp
            
            if maskMutation[ie] < chance:
                channel = self.channels[ie]
                self.channels[ie] = graph.channels[ie]
                graph.channels[ie] = channel
                if graph.channelsConnected() and self.channelsConnected():
                    pass
                else:
                    # revert
                    channel = self.channels[ie]
                    self.channels[ie] = graph.channels[ie]
                    graph.channels[ie] = channel
        
        self.toModel()
    
    # def init(self):
    #     self.contractions = np.random.randint(0, Model.contractionLevels, self.contractions.shape)
    #
    #     # randomly choose numChannel edges and assign channels
    #     dice = np.arange(self.nE)
    #     np.random.shuffle(dice)
    #     ies = dice[:self.numChannels]
    #     for ic, ie in enumerate(ies):
    #         self.channels[ie] = ic
    #
    #     # grow channels to fill the entire graph
    #     while (self.channels == -1).any():
    #         iesToGrow = []
    #         for ie in range(self.nE):
    #             iesConnected = self.ieAdjList[ie]
    #             if (self.channels[iesConnected] == -1).any():
    #                 iesToGrow.append(ie)
    #
    #         ie = np.random.choice(iesToGrow)
    #         iesConnected = self.ieAdjList[ie]
    #         np.random.shuffle(iesConnected)
    #         for ieConnected in iesConnected:
    #             if self.channels[ieConnected] == -1:
    #                 self.channels[ieConnected] = self.channels[ie]


    def channelsConnected(self):
        # check if all channels are connected
        
        channelsConnected = np.zeros(self.numChannels)
        
        iEdgesUnvisited = set(np.arange(len(self.model.e)).tolist())
        
        while len(iEdgesUnvisited) > 0:
            ie = iEdgesUnvisited.pop()
            iesQueue = [ie]
            ic = self.channels[ie]
            if channelsConnected[ic] == 1:
                return False
            
            while len(iesQueue) != 0:
                
                iEdge = iesQueue.pop(0)
                
                ies = FullGraph.eAdjacencyList[iEdge]
                for iEdge in ies:
                    if iEdge in iEdgesUnvisited and self.channels[iEdge] == ic:
                        iesQueue.append(iEdge)
                        iEdgesUnvisited.remove(iEdge)
            
            channelsConnected[ic] = 1
            
        if channelsConnected.sum() == self.numChannels:
            return True
        
        else:
            return False
            
            
        
    
    #
    # def channelConnected(self, ic):
    #     # check if channel ic is interconnected
    #     nEic = (self.channels == ic).sum()  # number of edges of channel ic
    #
    #     for ie in range(len(self.model.e)):
    #         if self.channels[ie] == ic:
    #             break
    #
    #     queue = [ie]  # ies in the queue
    #     visited = set()  # ies visited
    #
    #     while queue:
    #         ie = queue.pop(0)
    #         visited.add(ie)
    #
    #         iv0 = self.esSub[ie, 0]
    #         iv1 = self.esSub[ie, 1]
    #         iesConnected0 = np.where(self.incM[iv0] == 1)[0]
    #         iesConnected1 = np.where(self.incM[iv1] == 1)[0]
    #         iesConnected = np.concatenate([iesConnected0, iesConnected1])
    #         for ie in iesConnected:
    #             if ie not in visited and self.channels[ie] == ic:
    #                 queue.append(ie)
    #
    #     return nEic == len(visited)
    #
    
    # def mutate(self):
    #     # mutate one digit of contractions and one edge channel
    #     self.contractions[np.random.choice(len(self.contractions))] = np.random.randint(Model.contractionLevels)
    #
    #     ies = np.arange(self.nE)
    #     np.random.shuffle(ies)
    #     for ie in ies:
    #         iesConnected = self.ieAdjList[ie]
    #
    #         icOld = self.channels[ie]
    #         icsConnected = self.channels[iesConnected].tolist()
    #         icsConnected = set(icsConnected)
    #         icsConnected.remove(icOld)
    #         icsConnected = np.array(list(icsConnected))
    #         if len(icsConnected):
    #             np.random.shuffle(icsConnected)
    #
    #         for icNew in icsConnected:
    #             self.channels[ie] = icNew  # change the channel of the edge
    #             if self.channelConnected(icOld):  # if the changed channel is still connected
    #                 return True  # mutation finished
    #             else:
    #                 self.channels[ie] = icOld  # revert channel change
    #     print('mutation failed')
    #     return False
