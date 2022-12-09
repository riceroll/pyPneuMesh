import numpy as np
import copy
import pathlib

from src.Model import Model

class HalfGraph(object):
    def __init__(self, model: Model, graphSetting):
        self.channelMirrorMap = graphSetting['channelMirrorMap'].copy()
        self.model = model
        self.edgeMirrorMap = self.__getEdgeMirrorMap()
        
        self.ins_o = []  # nn, original indices of nodes in a halfgraph
        self.edges = []  # indices of two incident nodes, ne x 2, ne is the number of edges in a halfgraph
        self.ies_o = []  # ne, original indices of edges in a halfgraph
        self.channels = []  # ne, indices of channels
        self.contractions = []  # ne, int value of contractions
        self.esOnMirror = []  # ne, bool, if the edge in on the mirror plane
        
        # region assigning eLeft, eRight, eMiddle
        eLeft = []
        eRight = []
        eMiddle = []
        edgeMirrorMap = self.__getEdgeMirrorMap()

        while len(edgeMirrorMap):
            ie, ieMirror = edgeMirrorMap.popitem()
            if ieMirror == -1:
                eMiddle.append(ie)
            else:
                if model.v0[model.e[ie][0], 1] < 0 or model.v0[model.e[ie][1], 1] < 0:
                    eLeft.append(ie)
                    eRight.append(ieMirror)
                else:
                    eLeft.append(ieMirror)
                    eRight.append(ie)
                    
                edgeMirrorMap.pop(ieMirror)
        self.eLeft = eLeft
        self.eRight = eRight
        self.eMiddle = eMiddle
        # endregion

        esInfo = []
        for ie in eLeft + eMiddle:
            contraction = self.model.contractionLevel[ie]
    
            esInfo.append((self.model.e[ie][0], self.model.e[ie][1],
                           {
                               'ie': ie,
                               'onMirror': ie in eMiddle,
                               'channel': self.model.edgeChannel[ie],
                               'contraction': contraction
                           }
                           )
                          )
        self.__add_edges_from(esInfo)

    def __getEdgeMirrorMap(self):
        threshold = 0.01
        
        vertexMirrorMap = dict()
        for iv, v in enumerate(self.model.v0):
            if iv not in vertexMirrorMap:
                if abs(v[1]) < threshold:
                    vertexMirrorMap[iv] = -1  # on the mirror plane
                else:  # mirrored with another vertex
                    for ivMirror, vMirror in enumerate(self.model.v0):
                        if ivMirror == iv:
                            continue
                        
                        if abs(vMirror[0] - v[0]) < threshold and \
                            abs(vMirror[2] - v[2]) < threshold and \
                                abs(-vMirror[1] - v[1]) < threshold:
                            assert(iv not in vertexMirrorMap)
                            assert (ivMirror not in vertexMirrorMap)
                            vertexMirrorMap[iv] = ivMirror
                            vertexMirrorMap[ivMirror] = iv
            
            if iv not in vertexMirrorMap:
                assert(False)
                
        edgeMirrorMap = dict()
    
        for ie, e in enumerate(self.model.e):
            if ie in edgeMirrorMap:
                continue
        
            iv0 = e[0]
            iv1 = e[1]
            if vertexMirrorMap[iv0] == -1:
                ivM0 = iv0  # itself
            else:
                ivM0 = vertexMirrorMap[iv0]
            if vertexMirrorMap[iv1] == -1:
                ivM1 = iv1
            else:
                ivM1 = vertexMirrorMap[iv1]
            eM = [ivM0, ivM1]
            if ivM0 == iv0 and ivM1 == iv1:  # edge on the mirror plane
                edgeMirrorMap[ie] = -1
            else:
                iesMirrored = (eM == self.model.e).all(1) + (eM[::-1] == self.model.e).all(1)
                assert (iesMirrored.sum() == 1)
                ieMirrored = np.where(iesMirrored)[0][0]
                assert (ieMirrored not in edgeMirrorMap)
                if ieMirrored == ie:  # edge rides across the mirror plane
                    edgeMirrorMap[ie] = -1
                else:
                    edgeMirrorMap[ie] = ieMirrored
                    edgeMirrorMap[ieMirrored] = ie
    
        return edgeMirrorMap

    def saveGraphSetting(self, folderDir, name):
        graphSetting = self.getGraphSetting()
    
        folderPath = pathlib.Path(folderDir)
        graphSettingPath = folderPath.joinpath("{}.graphsetting".format(name))
        np.save(str(graphSettingPath), graphSetting)

    def getGraphSetting(self):
        graphSetting = {
            'symmetric': True,
            'channelMirrorMap': self.channelMirrorMap
        }
        return copy.deepcopy(graphSetting)

    def __add_edges_from(self, esInfo):
        # input:
        # esInfo: [(iv0, iv1, {'ie': , 'channel': , 'contraction': })]

        ne = len(esInfo)
        self.ies_o = np.zeros(ne, dtype=int)
        self.edges = np.zeros([ne, 2], dtype=int)
        self.channels = np.zeros(ne, dtype=int)
        self.contractions = np.zeros(ne, dtype=int)
        self.esOnMirror = np.zeros(ne, dtype=bool)

        for i, eInfo in enumerate(esInfo):
            ie = eInfo[2]['ie']  # original index of the edge
            self.ies_o[i] = ie

            self.ins_o.append(eInfo[0])
            self.ins_o.append(eInfo[1])
            self.channels[i] = int(eInfo[2]['channel'])
            self.contractions[i] = int(eInfo[2]['contraction'])
            self.esOnMirror[i] = bool(eInfo[2]['onMirror'])

        self.ins_o = sorted(list(set(self.ins_o)))

        for i, eInfo in enumerate(esInfo):
            in0 = np.where(self.ins_o == eInfo[0])[0][0]
            in1 = np.where(self.ins_o == eInfo[1])[0][0]
            self.edges[i, 0] = in0
            self.edges[i, 1] = in1
            
    def toModel(self):
        for i, edge in enumerate(self.edges):
        
            ie = self.ies_o[i]
        
            contractionLevel = self.contractions[i]
            ic = self.channels[i]
        
            # region set edgeChannel
            self.model.edgeChannel[ie] = ic
            ieMirror = self.edgeMirrorMap[ie]
        
            if ieMirror == -1:
                pass
            else:
                if ic == -1:
                    self.model.edgeChannel[ieMirror] = -1
                else:
                    icMirror = self.channelMirrorMap[ic]
                    if icMirror == -1:
                        icMirror = ic
                    self.model.edgeChannel[ieMirror] = icMirror
            # endregion
        
            # set maxContraction
            self.model.contractionLevel[ie] = contractionLevel
            if ieMirror != -1:
                self.model.contractionLevel[ieMirror] = contractionLevel

    def randomize(self):
        stuck = True
        while stuck:
            stuck = False
            
            self.channels *= 0
            self.channels += -1
            self.contractions *= 0
            
            # init channels
            iesUnassigned = set(np.arange(len(self.edges)))  # halfgraph indices of edges
            iesIncidentMirrorUnassigned = set(self.iesIncidentMirror())
            iesNotMirrorUnassigned = set(self.iesNotMirror())

            for iChannel in self.channelMirrorMap.keys():
                # breakpoint()
                icMirror = self.channelMirrorMap[iChannel]
                if icMirror == -1:
                    ie = np.random.choice(list(iesIncidentMirrorUnassigned))
                    iesIncidentMirrorUnassigned.remove(ie)
                    iesUnassigned.remove(ie)
                    if ie in iesNotMirrorUnassigned:
                        iesNotMirrorUnassigned.remove(ie)
                else:
                    ie = np.random.choice(list(iesNotMirrorUnassigned))
                    iesUnassigned.remove(ie)
                    if ie in iesIncidentMirrorUnassigned:
                        iesIncidentMirrorUnassigned.remove(ie)
                    iesNotMirrorUnassigned.remove(ie)
                self.channels[ie] = iChannel

            numNotUpdate = 0
            while iesUnassigned:
                numNotUpdate += 1

                if numNotUpdate > 40:
                    stuck = True
                    break
                #
                # self.fromHalfGraph()
                # self.show(show=False)

                ic = np.random.choice(list(self.channelMirrorMap.keys()))
                iesUnassignedAroundChannel = self.iesAroundChannel(ic)
                if self.iesNotMirror() is not None and iesUnassignedAroundChannel is not None:
                    iesUnassignedAroundChannelNotMirror = np.intersect1d(iesUnassignedAroundChannel, self.iesNotMirror())
                else:
                    iesUnassignedAroundChannelNotMirror = None

                icMirror = self.channelMirrorMap[ic]
                if icMirror != -1:
                    if iesUnassignedAroundChannelNotMirror is None or len(iesUnassignedAroundChannelNotMirror) == 0:
                        continue
                    ieToAssign = np.random.choice(iesUnassignedAroundChannelNotMirror)
                else:
                    if iesUnassignedAroundChannel is None or len(iesUnassignedAroundChannel) == 0:
                        continue
                    ieToAssign = np.random.choice(iesUnassignedAroundChannel)

                numNotUpdate = 0
                iesUnassigned.remove(ieToAssign)
                self.channels[ieToAssign] = ic
                
        self.contractions = np.random.randint(0, self.model.NUM_CONTRACTION_LEVEL, self.contractions.shape)
        
        self.toModel()
        
    def mutate(self, graphMutationChance, contractionMutationChance):
        if np.random.random() < graphMutationChance:
            self.mutateGraph()
        self.mutateContraction(contractionMutationChance)
        self.toModel()
        
    def mutateGraph(self):
        # choose a random edge and change its channel
    
        # find all edges that can be changed
        # randomly pick one edge
        # change its channel to one of the available channels incident to the edge
        iess = []
        for ic in self.channelMirrorMap.keys():
            iess.append(self.iesAroundChannel(ic, unassigned=False))
        ies = np.array(list(set(np.concatenate(iess))))
    
        succeeded = False
        while not succeeded:
            ie = np.random.choice(ies)
            ic_original = self.channels[ie]
        
            ies_incident = self.incidentEdges(self.edges[ie])
            ics_available = []
            for ie_incident in ies_incident:
                ic = self.channels[ie_incident]
                icMirror = self.channelMirrorMap[ic]
                if not (icMirror != -1 and self.esOnMirror[ie_incident]) and ic != ic_original:
                    ics_available.append(ic)
        
            while len(ics_available) > 0:
                ic = np.random.choice(ics_available)
                self.channels[ie] = ic
            
                channelsConnected = True
            
                for iChannel in self.channelMirrorMap.keys():
                    channelsConnected *= self.channelConnected(iChannel)
            
                if channelsConnected:
                    succeeded = True
                    break
                else:
                    self.channels[ie] = ic_original
                    ics_available.remove(ic)
                    
    def mutateContraction(self, chance):
        maskMutation = np.random.rand(len(self.contractions))
        contraction = np.random.randint(
            np.zeros(len(self.contractions)), self.model.NUM_CONTRACTION_LEVEL * np.ones(len(self.contractions)))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < chance:
                self.contractions[ie] = contraction[ie]

    def cross(self, graph, chance):
        maskMutation = np.random.rand(len(self.contractions))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < chance:
                tmp = self.contractions[ie]
                self.contractions[ie] = graph.contractions[ie]
                graph.contractions[ie] = tmp
        
    def iesIncidentMirror(self):
        # indices of edges on / connecting to the mirror plane
        ies = np.arange(len(self.edges))[self.esOnMirror.copy()]
        ins_onMirror = self.incidentNodes(ies)
        ies_IncidentMirror = self.incidentEdges(ins_onMirror)
        return ies_IncidentMirror

    def iesNotMirror(self):
        ies = np.arange(len(self.edges))[~self.esOnMirror]
        return ies

    def incidentNodes(self, ies):
        # get the incident nodes of a set of edges
        ins = np.array(list(set(self.edges[ies].reshape(-1))))
        return ins

    def incidentEdges(self, ins):
        # get the incident edges of a set of noes
        isin = np.isin(self.edges, ins)
        ifEdgesIncident = isin[:, 0] + isin[:, 1]
        ies = np.arange(len(self.edges))[ifEdgesIncident]
        return ies

    def iesAroundChannel(self, ic, unassigned=True):
        # breakpoint()
        # get ies incident but not belonged to the channel ic and not assigned
        boolEsChannel = self.channels == ic  # bool, es in the channel
        ins = self.incidentNodes(np.arange(len(self.edges))[boolEsChannel])

        isin = np.isin(self.edges, ins)
        boolEsIncidentChannel = isin[:, 0] + isin[:, 1]
        boolEsAroundChannel = boolEsIncidentChannel * (~boolEsChannel)
        boolEsUnassigned = self.channels == -1
        if unassigned:
            bools = boolEsAroundChannel * boolEsUnassigned
        else:
            bools = boolEsAroundChannel
        if True in bools:
            return np.arange(len(self.edges))[bools]
        else:
            return None

    def channelConnected(self, ic):
        iesUnvisited = np.arange(len(self.edges))[self.channels == ic].tolist()  # ies of the channel

        if len(iesUnvisited) == 0:  # no graph exist at all
            return False

        ie = iesUnvisited.pop()

        queue = [ie]

        while len(queue) != 0:
            ie = queue.pop(0)

            ins = self.incidentNodes([ie])
            iesAdjacent = self.incidentEdges(ins)

            for ieAdjacent in iesAdjacent:
                if ieAdjacent in iesUnvisited:
                    iesUnvisited.remove(ieAdjacent)
                    queue.append(ieAdjacent)

        if len(iesUnvisited) != 0:
            return False
        else:
            return True
