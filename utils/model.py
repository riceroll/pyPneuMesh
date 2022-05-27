import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
import networkx as nx
import copy

import utils
from utils.geometry import getFrontDirection
rootPath = os.path.split(os.path.realpath(__file__))[0]
rootPath = os.path.split(rootPath)[0]
tPrev = time.time()

class HalfGraph(object):
    def __init__(self):
        self.ins_o = []             # nn, original indices of nodes in a halfgraph
        self.edges = []             # indices of two incident nodes, ne x 2, ne is the number of edges in a halfgraph
        self.ies_o = []               # ne, original indices of edges in a halfgraph
        self.channels = []          # ne, indices of channels
        self.contractions = []      # ne, int value of contractions
        self.esOnMirror = []          # ne, bool, if the edge in on the mirror plane
        
    def add_edges_from(self, esInfo):
        # input:
        # esInfo: [(iv0, iv1, {'ie': , 'channel': , 'contraction': })]
        
        ne = len(esInfo)
        self.ies_o = np.zeros(ne, dtype=int)
        self.edges = np.zeros([ne, 2], dtype=int)
        self.channels = np.zeros(ne, dtype=int)
        self.contractions = np.zeros(ne, dtype=int)
        self.esOnMirror = np.zeros(ne, dtype=bool)
        
        for i, eInfo in enumerate(esInfo):
            ie = eInfo[2]['ie']    # original index of the edge
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
        boolEsChannel = self.channels == ic     # bool, es in the channel
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
        iesUnvisited = np.arange(len(self.edges))[self.channels == ic].tolist()     # ies of the channel
        
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
        
        
class Model(object):
    # k = 200000
    # h = 0.001
    # dampingRatio = 0.999
    # contractionInterval = 0.1
    # contractionLevels = 4
    # maxMaxContraction = round(contractionInterval * (contractionLevels - 1) * 100) / 100
    # contractionPercentRate = 1e-3
    # gravityFactor = 9.8 * 10
    # gravity = 1
    # defaultMinLength = 1.2
    # defaultMaxLength = defaultMinLength / (1 - maxMaxContraction)
    # frictionFactor = 0.8
    # numStepsPerActuation = int(2 / h)
    # defaultNumActions = 1
    # defaultNumChannels = 4
    # angleThreshold = np.pi / 2
    # angleCheckFrequency = numStepsPerActuation / 20

    @staticmethod
    def configure(configDir):
        with open(configDir) as ifile:
            content = ifile.read()
            data = json.loads(content)
            Model.k = data['k']
            Model.h = data['h']
            Model.dampingRatio = data['dampingRatio']
            Model.contractionInterval = data['contractionInterval']
            Model.contractionLevels = data['contractionLevels']
            Model.maxMaxContraction = round(Model.contractionInterval * (Model.contractionLevels - 1) * 100) / 100
            Model.contractionPercentRate = data['contractionPercentRate']
            Model.gravityFactor = data['gravityFactor']
            Model.gravity = data['gravity']
            Model.defaultMinLength = data['defaultMinLength']
            Model.defaultMaxLength = Model.defaultMinLength / (1 - Model.maxMaxContraction)
            Model.frictionFactor = data['frictionFactor']
            Model.numStepsPerActuation = int(data['numStepsActionMultiplier'] / Model.h)
            Model.defaultNumActions = data['defaultNumActions']
            Model.defaultNumChannels = data['defaultNumChannels']
            Model.angleThreshold = data['angleThreshold']
            Model.angleCheckFrequency = Model.numStepsPerActuation * data['angleCheckFrequencyMultiplier']
            Model.directionalFriction = bool(data['directionalFriction'])
    
    def __init__(self, configDir="./data/config.json"):
        
        # ======= variable =======
        self.v = None       # vertices locations    [nv x 3]
        self.e = None       # edge                  [ne x 2] int
        self.v0 = None
        self.c = None       # indices of v at corners [nc x 3] e.g. <AOB -> [iO, iA, iB]
        self.a0 = None      # rest angle of corresponding corner [nc]
        self.fixedVs = None
        self.lMax = None
        self.edgeActive = None
        self.edgeChannel = None
        self.script = None
        
        self.maxContraction = None
        self.vel = None     # velocity              [nv x 3]
        self.f = None
        self.l = None
        
        self.frontVec = None
        
        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        self.inflateChannel = None
        self.contractionPercent = None
        self.numChannels = None
        self.numActions = None
        
        self.testing = False    # testing mode
        self.vertexMirrorMap = dict()
        self.edgeMirrorMap = dict()
        self.channelMirrorMap = dict()
        
        self.showHistory = []   # edges at every update of channel growing
        
        Model.configure(configDir)
    
    # initialization =======================
    def load(self, modelDir):
        """
        load parameters from a json file into the model
        :param modelDir: dir of the json file
        :return: data as a dictionary
        """
        self.modelDir = modelDir
        with open(modelDir) as ifile:
            content = ifile.read()
        data = json.loads(content)

        self.v = np.array(data['v'])
        self.v0 = self.v.copy()
        self.e = np.array(data['e'], dtype=np.int64)
        
        self.lMax = np.array(data['lMax'])
        self.edgeActive = np.array(data['edgeActive'], dtype=bool)
        self.edgeChannel = np.array(data['edgeChannel'], dtype=np.int64)
        self.maxContraction = np.array(data['maxContraction'])
        
        self.fixedVs = np.array(data['fixedVs'], dtype=np.int64)
        
        assert((self.lMax[self.edgeActive] == self.lMax[self.edgeActive][0]).all())
        assert(len(self.e) == len(self.lMax) == len(self.edgeChannel) == len(self.edgeActive) and len(self.maxContraction))
        self._reset()
        
    def reload(self):
        modelDir = self.modelDir
        self.__init__()
        self.load(modelDir)
    
    def _reset(self, resetScript=False):
        """
        reset the model to the initial state with input options
        :param resetScript: if True, reset script to zero with numChannels and numActions
        :param numChannels: if -1, use the default value from self.numChannels or self.edgeChannel
        :param numActions: if -1, use the default value from self.numActions or self.script[0]
        """
        
        self.v = self.v0.copy()
        self.vel = np.zeros_like(self.v)
        self.frontVec = None

        # TODO: quick fix
        try:
            self.numChannels = max(self.edgeChannel.max() + 1, self.numChannels)
        except:     # if self.numChannels not defined
            self.numChannels = self.edgeChannel.max() + 1
        
        self.inflateChannel = np.ones(self.numChannels)
        self.contractionPercent = np.zeros(self.numChannels)
        

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        # self.updateCornerAngles()
        
        # TODO: quick fix
        try:
            self.computeSymmetry()
            self.symmetric = True
        except:
            self.symmetric = False
        
    def setToSingleChannel(self):
        # set all channels to the same channel
        self.edgeChannel *= 0
        self.edgeChannel += 1
    
    # end initialization
    
    # stepping =======================
    def step(self, n=1, ret=False):
        """
        step simulation of the model if self.simulate is True
        :param n: number of steps
        """
        
        if not self.simulate:
            return True
        
        for i in range(n):
            self.numSteps += 1
        
            # contractionPercent
            for iChannel in range(len(self.inflateChannel)):
                if self.inflateChannel[iChannel]:
                    self.contractionPercent[iChannel] -= Model.contractionPercentRate
                    if self.contractionPercent[iChannel] < 0:
                        self.contractionPercent[iChannel] = 0
                    
                else:
                    self.contractionPercent[iChannel] += Model.contractionPercentRate
                    if self.contractionPercent[iChannel] > 1:
                        self.contractionPercent[iChannel] = 1
                    
        
            f = np.zeros_like(self.v)
        
            iv0 = self.e[:, 0]
            iv1 = self.e[:, 1]
            v = self.v
            vec = v[iv1] - v[iv0]
            self.l = l = np.sqrt((vec ** 2).sum(1))
            l0 = np.copy(self.lMax)
            lMax = np.copy(self.lMax)
            lMin = lMax * (1 - (Model.contractionLevels - 1) * self.contractionInterval )
            
            # edge strain
            l0[self.edgeActive] = (lMax - self.contractionPercent[self.edgeChannel] * (lMax - lMin))[self.edgeActive]
            fMagnitude = (l - l0) * Model.k
            fEdge = vec / l.reshape(-1, 1) * fMagnitude.reshape(-1, 1)
            np.add.at(f, iv0, fEdge)
            np.add.at(f, iv1, -fEdge)
            
            # gravity
            f[:, 2] -= Model.gravityFactor * Model.gravity
            self.f = f
            
            self.vel += Model.h * f

            boolUnderground = self.v[:, 2] <= 0
            self.vel[boolUnderground, :2] *= 1 - Model.frictionFactor       # uniform friction
            # damping
            self.vel *= Model.dampingRatio
            velMag = np.sqrt((self.vel ** 2).sum(1))
            if (velMag > 5).any():
                self.vel[velMag > 5] *= np.power(0.9, np.ceil(np.log(5 / velMag[velMag > 5]) / np.log(0.9))).reshape(-1, 1)

            # directional surface
            if Model.directionalFriction:
                if self.numSteps % (Model.numStepsPerActuation / 20) == 0 or self.frontVec is None:
                    self.frontVec = getFrontDirection(self.v0, self.v).reshape(-1, 1)
                # self.frontVec = np.array([1, 0, 0]).reshape(-1, 1)
                vel = self.vel.copy()
                vel[2] *= 0
    
                dot = (vel @ self.frontVec)
                ids = (dot < 0).reshape(-1) * boolUnderground
                self.vel[ids] -= dot[ids] * self.frontVec.reshape(1, -1)
                
            self.v += Model.h * self.vel
            self.v[np.where(np.array(self.fixedVs) == 1)] -= Model.h * self.vel[np.where(np.array(self.fixedVs) == 1)]

            boolUnderground = self.v[:, 2] <= 0
            self.vel[boolUnderground, 2] *= -1
            self.vel[boolUnderground, 2] *= 0
            self.v[boolUnderground, 2] = 0
        
            # angle check
            # if False and self.numSteps % Model.angleCheckFrequency == 0:
            #     a = self.getCornerAngles()
            #     if (a - self.a0 > self.angleThreshold).any():
            #         print("exceed angle")
            #         return np.ones_like(self.v) * -1e6
            
            # if self.numSteps % 100 == 0:
            #     vs.append(self.v)
        
        if ret:
            return self.v.copy()
    
    # end stepping
    
    # checks =========================
    def updateCornerAngles(self):
        c = []
        
        for iv in range(len(self.v)):
            # get indices of neighbor vertices
            ids = np.concatenate([self.e[self.e[:, 0] == iv][:, 1], self.e[self.e[:, 1] == iv][:, 0]])
            for i in range(len(ids) - 1):
                id0 = ids[i]
                id1 = ids[i+1]
                c.append([iv, id0, id1])
        
        c = np.array(c)
        self.a0 = self.getCornerAngles(c)
    
    def getCornerAngles(self, c):
        # return angles of self.c
        O = self.v[c[:, 0]]
        A = self.v[c[:, 1]]
        B = self.v[c[:, 2]]
        vec0 = A - O
        vec1 = B - O
        l0 = np.sqrt((vec0 ** 2).sum(1))
        l1 = np.sqrt((vec1 ** 2).sum(1))
        cos = (vec0 * vec1).sum(1) / l0 / l1
        angles = np.arccos(cos)
        return angles
    
    # end checks
    
    # utility ===========================
    def centroid(self):
        return self.v.sum(0) / self.v.shape[0]
    
    def initializePos(self):
        """
        put the model's centroid back to original point and wait for a while to stabilize the model
        reset v0
        :return:
        """
        self.v -= self.centroid()
        self.v[:, 2] -= self.v[:, 2].min()
        self.v[:, 2] += self.v[:, 2].max()
        self.step(int(Model.numStepsPerActuation * 1.5))
        self.v0 = self.v
        
    def computeSymmetry(self):
        """
        assuming that the model is centered and aligned with +x
        """
        vertexMirrorMap = dict()
        ys = self.v[:, 1]
        xzs = self.v[:, [0, 2]]
        for iv, v in enumerate(self.v):
            if iv not in vertexMirrorMap:
                if abs(v[1]) < 0.1:
                    vertexMirrorMap[iv] = -1    # on the mirror plane
                else:       # mirrored with another vertex
                    boolMirrored = ((v[[0, 2]] - xzs) ** 2).sum(1) < 1
                    ivsMirrored = np.where(boolMirrored)[0]
                    ivMirrored = ivsMirrored[np.argmin(abs(v[1] + self.v[ivsMirrored][:, 1]))]
                    assert(ivMirrored not in vertexMirrorMap)
                    vertexMirrorMap[iv] = ivMirrored
                    vertexMirrorMap[ivMirrored] = iv

        edgeMirrorMap = dict()
        
        for ie, e in enumerate(self.e):
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
            if ivM0 == iv0 and ivM1 == iv1:     # edge on the mirror plane
                edgeMirrorMap[ie] = -1
            else:
                iesMirrored = (eM == self.e).all(1) + (eM[::-1] == self.e).all(1)
                assert(iesMirrored.sum() == 1)
                ieMirrored = np.where(iesMirrored)[0][0]
                assert(ieMirrored not in edgeMirrorMap)
                if ieMirrored == ie:            # edge rides across the mirror plane
                    edgeMirrorMap[ie] = -1
                else:
                    edgeMirrorMap[ie] = ieMirrored
                    edgeMirrorMap[ieMirrored] = ie
        
        self.vertexMirrorMap = vertexMirrorMap
        self.edgeMirrorMap = edgeMirrorMap
        return vertexMirrorMap, edgeMirrorMap
    
    def toHalfGraph(self, reset=False):
        G = HalfGraph()
        
        # region assigning eLeft, eRight, eMiddle
        eLeft = []
        eRight = []
        eMiddle = []
        edgeMirrorMap = copy.copy(self.edgeMirrorMap)
        
        while len(edgeMirrorMap):
            ie, ieMirror = edgeMirrorMap.popitem()
            if ieMirror == -1:
                eMiddle.append(ie)
            else:
                if self.v[self.e[ie][0], 1] < 0 or self.v[self.e[ie][1], 1] < 0:
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
        
        # for iv in range(len(self.v)):
        #     G.add_node(iv)
            
        # region G.add_edges_from(...)
        esInfo = []
        for ie in eLeft + eMiddle:
            contraction = (int(round(self.maxContraction[ie] / Model.contractionInterval))) if not reset else 0
            if not contraction in [0, 1, 2, 3]:
                contraction = 0
            
            esInfo.append((self.e[ie][0], self.e[ie][1],
                       {
                           'ie': ie,
                           'onMirror': ie in eMiddle,
                           'channel': self.edgeChannel[ie],
                           'contraction': contraction
                        }
                            )
                       )
        G.add_edges_from(esInfo)
        #end region
        self.G = G
        return G
    
    def fromHalfGraph(self):
        # set values for self.maxContraction, self.edgeChannel
        G = self.G
        
        for i, edge in enumerate(G.edges):
            
            ie = G.ies_o[i]
            
            contractionLevel = G.contractions[i]
            ic = G.channels[i]
            
            # region set edgeChannel
            self.edgeChannel[ie] = ic
            ieMirror = self.edgeMirrorMap[ie]
            
            if ieMirror == -1:
                pass
            else:
                if ic == -1:
                    self.edgeChannel[ieMirror] = -1
                else:
                    icMirror = self.channelMirrorMap[ic]
                    if icMirror == -1:
                        icMirror = ic
                    self.edgeChannel[ieMirror] = icMirror
            # endregion

            # set maxContraction
            self.maxContraction[ie] = contractionLevel * Model.contractionInterval
            if ieMirror != -1:
                self.maxContraction[ieMirror] = contractionLevel * Model.contractionInterval
    
    def initHalfGraph(self):
        stuck = True
        while stuck:
            stuck = False
            
            G = self.G
            G.channels *= 0
            G.channels += -1
            G.contractions *= 0
            
            # init channels
            iesUnassigned = set(np.arange(len(G.edges)))     # halfgraph indices of edges
            iesIncidentMirrorUnassigned = set(G.iesIncidentMirror())
            iesNotMirrorUnassigned = set(G.iesNotMirror())
            
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
                G.channels[ie] = iChannel
            
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
                iesUnassignedAroundChannel = G.iesAroundChannel(ic)
                if G.iesNotMirror() is not None and iesUnassignedAroundChannel is not None:
                    iesUnassignedAroundChannelNotMirror = np.intersect1d(iesUnassignedAroundChannel, G.iesNotMirror())
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
                G.channels[ieToAssign] = ic
                
        self.G.contractions = np.random.randint(0, Model.contractionLevels, self.G.contractions.shape)
        
        self.fromHalfGraph()
        
    def mutateHalfGraph(self):
        # choose a random edge and change its channel
        
        # find all edges that can be changed
        # randomly pick one edge
        # change its channel to one of the available channels incident to the edge
        
        G = self.G
        
    
        iess = []
        for ic in self.channelMirrorMap.keys():
            iess.append(G.iesAroundChannel(ic, unassigned=False))
        ies = np.array(list(set(np.concatenate(iess))))
        
        succeeded = False
        while not succeeded:
            ie = np.random.choice(ies)
            ic_original = G.channels[ie]
        
            ies_incident = G.incidentEdges(G.edges[ie])
            ics_available = []
            for ie_incident in ies_incident:
                ic = G.channels[ie_incident]
                icMirror = self.channelMirrorMap[ic]
                if not(icMirror != -1 and G.esOnMirror[ie_incident]) and ic != ic_original:
                    ics_available.append(ic)
            
            while len(ics_available) > 0:
                ic = np.random.choice(ics_available)
                G.channels[ie] = ic
                
                channelsConnected = True
                
                for iChannel in self.channelMirrorMap.keys():
                    channelsConnected *= G.channelConnected(iChannel)
                    
                if channelsConnected:
                    succeeded = True
                    break
                else:
                    G.channels[ie] = ic_original
                    ics_available.remove(ic)

        if np.random.rand() > 0.5:  # mutate contraction
            i = np.random.randint(len(G.contractions))
            G.contractions[i] = np.random.randint(Model.contractionLevels)
            
        self.fromHalfGraph()
        
    
    def show(self, show=True, essPrev=None):
        import polyscope as ps
        try:
            ps.init()
        except:
            pass
        
        G = self.G
        for ic in list(np.arange(self.numChannels)):
            ies = np.arange(len(self.e))[self.edgeChannel == ic]
            
            # if ic == -1:
            #     assert(len(ies) == 0)
            # else:
            #     assert(len(ies) != 0)
            
            es = self.e[ies]
            
            if show:
                if essPrev is not None:
                    es = essPrev[ic]
                
                ps.register_curve_network(str(ic), self.v, es)
            else:
                self.showHistory.append(es.copy())
            
            
        if show:
            ps.show()
        
            
    def loadEdgeChannel(self, edgeChannel):
        """
        load edgeChannel
        :param edgeChannel: np.array int [numEdge, ]
        """
        assert(type(edgeChannel) == np.ndarray)
        assert(edgeChannel.dtype == int)
        assert(edgeChannel.shape == (len(self.e),) )
        self.edgeChannel = edgeChannel
    
    def loadMaxContraction(self, maxContraction):
        """
        load maxContraction
        :param maxContraction: np.array float [numEdge, ]
        """
        exactDivisible = lambda dividend, divisor, threshold=1e-6: \
            (dividend - (dividend / divisor).round() * divisor).mean() < threshold
        
        assert(type(maxContraction) == np.ndarray)
        assert(maxContraction.shape == (len(self.e),))
        assert(exactDivisible(maxContraction, Model.contractionInterval))
        assert(maxContraction / Model.contractionInterval + 1e-6 < Model.contractionLevels - 1)
        self.maxContraction = maxContraction
        
    def exportJSON(self, modelDir=None,
                   actionSeq=np.zeros([4, 1]),
                   edgeChannel=None,
                   maxContraction=None,
                   saveDir=None,
                   appendix="",
                   save=True):
        """
        export the model into JSON, with original model from the JSON as name

        :param modelDir: name of the imported json file
        :param edgeChannel: np.array int [numEdge, ]
        :param maxContraction: np.array float [numEdgeActive, ]
        :param actionSeq: np.array [numChannel, numActions]
        """
        modelDir = modelDir if modelDir else self.modelDir
        with open(modelDir) as iFile:
            content = iFile.read()
            data = json.loads(content)

        if edgeChannel is not None:
            self.loadEdgeChannel(edgeChannel)
        data['edgeChannel'] = self.edgeChannel.tolist()
        
        if maxContraction is not None:
            self.maxContraction(maxContraction)
        data['maxContraction'] = self.maxContraction.tolist()
        
        if actionSeq is not None:
            data['script'] = actionSeq.tolist()
            data['numChannels'] = actionSeq.shape[0]
            data['numActions'] = actionSeq.shape[1]
        js = json.dumps(data)

        name = modelDir.split('/')[-1].split('.')[0]
        now = datetime.datetime.now()
        timeStr = "{}{}-{}:{}:{}".format(now.month, now.day, now.hour, now.minute, now.second)
        if saveDir is None:
            saveDir = '{}/output/{}_{}.json'.format(rootPath, name, str(appendix))
        
        print(rootPath, name, saveDir)
        if save:
            with open(saveDir, 'w') as oFile:
                oFile.write(js)
                print('Save to {}_{}.json'.format(saveDir, str(appendix)))
                
        return js

    # end optimization


def testModelStep(argv):
    model = Model()
    model.load("./test/data/lobsterIn.json")
    v = model.step(200, ret=True)
    js = model.exportJSON(save=False)
    with open('./test/data/lobsterOut.json') as iFile:
        jsTrue = iFile.read()
        assert (js == jsTrue)
        
    vTrue = np.load('./test/data/lobsterOutV.npy')
    assert ((v == vTrue).all())

def testComputeSymmetry(argv):
    model = Model()
    model.load("./test/data/pillBugIn.json")
    model.computeSymmetry()
    assert(model.vertexMirrorMap == {0: -1, 1: 2, 2: 1, 3: -1, 4: 5, 5: 4, 6: 7, 7: 6, 8: -1, 9: -1, 10: 11, 11: 10, 12: 13, 13: 12, 14: 15, 15: 14, 16: 17, 17: 16})
    assert(model.edgeMirrorMap == {0: -1, 1: 2, 2: 1, 3: 4, 4: 3, 5: -1, 6: 10, 10: 6, 7: 9, 9: 7, 8: 11, 11: 8, 12: 16, 16: 12, 13: 15, 15: 13, 14: 17, 17: 14, 18: 19, 19: 18, 20: -1, 21: 22, 22: 21, 23: -1, 24: 28, 28: 24, 25: 27, 27: 25, 26: 29, 29: 26, 30: 34, 34: 30, 31: 33, 33: 31, 32: 35, 35: 32, 36: 40, 40: 36, 37: 39, 39: 37, 38: 41, 41: 38, 42: 46, 46: 42, 43: 45, 45: 43, 44: 47, 47: 44})

    model = Model()
    model.load("./test/data/lobsterIn.json")
    model.computeSymmetry()
    assert(model.vertexMirrorMap == {0: -1, 1: 2, 2: 1, 3: -1, 4: 5, 5: 4, 6: 7, 7: 6, 8: -1, 9: 10, 10: 9, 11: 21, 21: 11, 12: 22, 22: 12, 13: 23, 23: 13, 14: 24, 24: 14, 15: 25, 25: 15, 16: 26, 26: 16, 17: 19, 19: 17, 18: -1, 20: -1, 27: 30, 30: 27, 28: 31, 31: 28, 29: 32, 32: 29, 33: 35, 35: 33, 34: -1, 36: 37, 37: 36, 38: -1, 39: -1, 40: 42, 42: 40, 41: 43, 43: 41, 44: 45, 45: 44, 46: -1, 47: 48, 48: 47})
    assert(model.edgeMirrorMap == {0: -1, 1: 2, 2: 1, 3: 4, 4: 3, 5: -1, 6: 10, 10: 6, 7: 9, 9: 7, 8: 11, 11: 8, 12: 16, 16: 12, 13: 15, 15: 13, 14: 17, 17: 14, 18: -1, 19: 20, 20: 19, 21: 25, 25: 21, 22: 24, 24: 22, 23: 26, 26: 23, 27: 58, 58: 27, 28: 57, 57: 28, 29: 59, 59: 29, 30: 61, 61: 30, 31: 60, 60: 31, 32: 62, 62: 32, 33: 64, 64: 33, 34: 63, 63: 34, 35: 65, 65: 35, 36: 67, 67: 36, 37: 66, 66: 37, 38: 68, 68: 38, 39: 70, 70: 39, 40: 69, 69: 40, 41: 71, 71: 41, 42: 73, 73: 42, 43: 72, 72: 43, 44: 74, 74: 44, 45: 133, 133: 45, 46: 52, 52: 46, 47: 134, 134: 47, 48: -1, 49: 132, 132: 49, 50: 53, 53: 50, 51: -1, 54: -1, 55: 56, 56: 55, 75: 85, 85: 75, 76: 84, 84: 76, 77: 86, 86: 77, 78: 88, 88: 78, 79: 87, 87: 79, 80: 89, 89: 80, 81: 91, 91: 81, 82: 90, 90: 82, 83: 92, 92: 83, 93: 100, 100: 93, 94: 135, 135: 94, 95: 99, 99: 95, 96: -1, 97: -1, 98: 101, 101: 98, 102: 105, 105: 102, 103: 106, 106: 103, 104: 136, 136: 104, 107: -1, 108: 110, 110: 108, 109: -1, 111: 112, 112: 111, 113: -1, 114: 121, 121: 114, 115: 120, 120: 115, 116: 122, 122: 116, 117: 124, 124: 117, 118: 123, 123: 118, 119: 125, 125: 119, 126: 130, 130: 126, 127: 129, 129: 127, 128: 131, 131: 128, 137: -1, 138: 139, 139: 138})
    
tests = {
    'step': testModelStep,
    'computeSymmetry': testComputeSymmetry,
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
