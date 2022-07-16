import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
import networkx as nx
import copy
from gym import Env
from gym.spaces import Discrete, Box

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
        self.contractions = np.array([])     # ne, int value of contractions
        self.esOnMirror = np.array([])          # ne, bool, if the edge in on the mirror plane
        
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
    class Setting:
        def __init__(self, settingDir):
            self.k = None
            self.h = None
            self.damping = None
            self.gravity = None
            self.friction = None
            self.actuationTime = None
            self.jointWeight = None
            
            self.minLengths = []
            
            self.nChannels = None
            self.channelMirrorMap = None
            
            # calculated values
            self.nStepsPerActuation = None
            
            self.loadJSON(settingDir)
            
        def loadJSON(self, settingDir):
            # dir to the json configuration file
            with open(settingDir) as iFile:
                content = iFile.read()
                setting = json.loads(content)
            
            for key in setting:
                assert(hasattr(self, key))
                setattr(self, key, setting[key])
            
            self.channelMirrorMap = {
                int(key): self.channelMirrorMap[key]
                for key in self.channelMirrorMap
            }
            self.minLengths = np.array(self.minLengths, dtype=float)
            self.nStepsPerActuation = int(self.actuationTime / self.h)
            dampingPerMs = self.damping
            self.damping = dampingPerMs ** (1.0 / (0.001 / self.h))
            
    class Truss:
        def __init__(self, modelDir=None):
            self.v0 = None
            self.e = None       # edge                  [ne x 2] int
            self.eLengthLevel = None
            self.eActive = None
            self.eChannel = None
            self.vFixed = None
            
            self.loadJson(modelDir)

        def loadJson(self, settingDir):
            # dir to the json configuration file
            with open(settingDir) as iFile:
                content = iFile.read()
                data = json.loads(content)
            
            self.v0 = np.array(data['v'], dtype=float)
            self.e = np.array(data['e'], dtype=int)
            self.eLengthLevel = np.array(data['eLengthLevel'], dtype=int)
            self.eActive = np.array(data['eActive'], dtype=bool)
            self.eChannel = np.array(data['eChannel'], dtype=int)
            self.vFixed = np.array(data['vFixed'], dtype=bool)
            
    def __init__(self, settingDir="./data/config_model_large.json", modelDir="./data/table.json"):
        self.settingDir = settingDir
        self.modelDir = modelDir
        self.setting = self.Setting(self.settingDir)
        self.truss = self.Truss(self.modelDir)
        
        self.inflateChannel = None
        self.contractionPercent = None

        self.v = None  # vertices locations    [nv x 3]
        self.vel = None     # velocity              [nv x 3]
        self.f = None
        self.l = None
        
        self.c = None  # indices of v at corners [nc x 3] e.g. <AOB -> [iO, iA, iB]
        self.a0 = None  # rest angle of corresponding corner [nc]
        self.frontVec = None
        
        self.vertexMirrorMap = dict()
        self.edgeMirrorMap = dict()
        self.tetList = None       # #T by 4, vertices of tetrahedrons

        self.recordEveryFrame = False
        self.nSteps = 0
        self.vs = []     # record vertex positions with certain interval

        self.reset()
        
    def reset(self):
        """
        reset the model to the initial state with input options
        :param numChannels: if -1, use the default value from self.numChannels or self.edgeChannel
        :param numActions: if -1, use the default value from self.numActions or self.script[0]
        """
        
        self.v = self.truss.v0.copy()
        self.vel = np.zeros_like(self.v)
        
        self.inflateChannel = np.ones(self.setting.nChannels)
        self.contractionPercent = np.zeros(self.setting.nChannels)
        
        self.recordEveryFrame = False
        self.nSteps = 0
        self.vs = [self.v.copy()]   # record the first frame
        
        try:
            self.computeSymmetry()
            self.symmetric = True
        except:
            self.symmetric = False
            print('Truss not symmetric.')
        
        self.updateInitAngle()
            
    def recordEveryFrameOn(self):
        self.recordEveryFrame = True
        
    def recordEveryFrameOff(self):
        self.recordEveryFrame = False
        
    def step(self, nSteps=1):
        for i in range(nSteps):
            self.nSteps += 1

            # contractionPercent
            actuationPercentRate = 1.0 / self.setting.nStepsPerActuation
            for iChannel in range(len(self.inflateChannel)):
                if self.inflateChannel[iChannel]:
                    self.contractionPercent[iChannel] -= actuationPercentRate
                    if self.contractionPercent[iChannel] < 0:
                        self.contractionPercent[iChannel] = 0
                else:
                    self.contractionPercent[iChannel] += actuationPercentRate
                    if self.contractionPercent[iChannel] > 1:
                        self.contractionPercent[iChannel] = 1
                    
            f = np.zeros_like(self.v)
        
            iv0 = self.truss.e[:, 0]
            iv1 = self.truss.e[:, 1]
            
            v = self.v
            vec = v[iv1] - v[iv0]
            self.l = l = np.sqrt((vec ** 2).sum(1))
            
            minLengths = self.setting.minLengths[self.truss.eLengthLevel]
            maxLength = max(self.setting.minLengths)
            
            # edge strain
            l0 = maxLength - self.contractionPercent[self.truss.eChannel] * (maxLength - minLengths)
            l0[~self.truss.eActive] = max(self.setting.minLengths)
            fMagnitude = (l - l0) * self.setting.k
            fEdge = vec / l.reshape(-1, 1) * fMagnitude.reshape(-1, 1)
            
            np.add.at(f, iv0, fEdge)
            np.add.at(f, iv1, -fEdge)
            
            # gravity
            f[:, 2] -= self.setting.gravity * self.setting.jointWeight
            self.f = f
            
            # integrate velocity
            a = f / self.setting.jointWeight
            self.vel += self.setting.h * a

            # friction
            boolUnderground = self.v[:, 2] <= 0
            self.vel[boolUnderground, :2] *= 1 - self.setting.friction       # uniform friction
            self.vel[boolUnderground, :2] *= 0
            
            # damping
            self.vel *= self.setting.damping
            velMag = np.sqrt((self.vel ** 2).sum(1))
            maxVel = 100.0
            # if (velMag > maxVel).any():
            #     # self.vel[velMag > maxVel] = maxVel
            #     print(1)
            #     self.vel[velMag > maxVel] *= np.power(0.9, np.ceil(np.log(maxVel / velMag[velMag > maxVel]) / np.log(0.9))).reshape(-1,1)
            
            # conflict with ground
            boolUnderground = self.v[:, 2] <= 0
            self.vel[boolUnderground, 2][self.vel[boolUnderground, 2] < 0] *= 0

            # integrate position
            self.v += self.setting.h * self.vel

            # conflict with ground
            self.v[boolUnderground, 2] = 0

            
            # fixed points
            self.v[np.where(np.array(self.truss.vFixed) == 1)] -= self.setting.h * self.vel[np.where(np.array(self.truss.vFixed) == 1)]
            
            # record trajectory
            if self.recordEveryFrame:
                self.vs.append(self.v.copy())
                
            elif self.nSteps % self.setting.nStepsPerActuation == 0 or self.nSteps == 1:    # record between every actuation
                self.vs.append(self.v.copy())
            
            # angle check
            # if (self.getBendingAngle() > np.pi / 2).any():
            #     print('exceed')
            # else:
            #     print('not exceed')
            
            
    # region angles
    
    def updateInitAngle(self):
        # update the initial angle a0

        def initAngleIndices():
            # update the indices of the corners
            c = []
    
            for iv in range(len(self.v)):
                # get indices of neighbor vertices
                ids = np.concatenate(
                    [self.truss.e[self.truss.e[:, 0] == iv][:, 1], self.truss.e[self.truss.e[:, 1] == iv][:, 0]])
                for i in range(len(ids) - 1):
                    id0 = ids[i]
                    id1 = ids[i + 1]
                    c.append([iv, id0, id1])
    
            self.c = np.array(c)
        
        initAngleIndices()
        
        self.a0 = self.getCurrentAngle(self.c)
    
    def getCurrentAngle(self, c):
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
    
    def getBendingAngle(self):
        return np.abs(self.getCurrentAngle(self.c) - self.a0)
    
    # endregion
    
    # region utility
    def centroid(self):
        return self.v.sum(0) / self.v.shape[0]
        
    def computeSymmetry(self):
        """
        assuming that the model is centered and aligned with +x
        """
        vertexMirrorMap = dict()
        ys = self.v[:, 1]
        xzs = self.v[:, [0, 2]]
        for iv, v in enumerate(self.v):
            if iv not in vertexMirrorMap:
                if abs(v[1]) < 0.001:
                    vertexMirrorMap[iv] = -1    # on the mirror plane
                else:       # mirrored with another vertex
                    boolMirrored = ((v[[0, 2]] - xzs) ** 2).sum(1) < 0.01
                    ivsMirrored = np.where(boolMirrored)[0]
                    ivMirrored = ivsMirrored[np.argmin(abs(v[1] + self.v[ivsMirrored][:, 1]))]
                    assert(ivMirrored not in vertexMirrorMap)
                    vertexMirrorMap[iv] = ivMirrored
                    vertexMirrorMap[ivMirrored] = iv
                    
        edgeMirrorMap = dict()
        
        for ie, e in enumerate(self.truss.e):
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
                iesMirrored = (eM == self.truss.e).all(1) + (eM[::-1] == self.truss.e).all(1)
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
        # reset: if True, reset all contraction to 0
        
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
                if self.v[self.truss.e[ie][0], 1] < 0 or self.v[self.truss.e[ie][1], 1] < 0:
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
            contraction = self.truss.eLengthLevel[ie] if not reset else 0
            
            esInfo.append((self.truss.e[ie][0], self.truss.e[ie][1],
                       {
                           'ie': ie,
                           'onMirror': ie in eMiddle,
                           'channel': self.truss.eChannel[ie],
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
            self.truss.eChannel[ie] = ic
            ieMirror = self.edgeMirrorMap[ie]
            
            if ieMirror == -1:
                pass
            else:
                if ic == -1:
                    self.truss.eChannel[ieMirror] = -1
                else:
                    icMirror = self.setting.channelMirrorMap[ic]
                    if icMirror == -1:
                        icMirror = ic
                    self.truss.eChannel[ieMirror] = icMirror
            # endregion

            # set maxContraction
            self.truss.eLengthLevel[ie] = contractionLevel
            if ieMirror != -1:
                self.truss.eLengthLevel[ieMirror] = contractionLevel
    
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
            
            for iChannel in self.setting.channelMirrorMap.keys():
                # breakpoint()
                icMirror = self.setting.channelMirrorMap[iChannel]
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
                
                ic = np.random.choice(list(self.setting.channelMirrorMap.keys()))
                iesUnassignedAroundChannel = G.iesAroundChannel(ic)
                if G.iesNotMirror() is not None and iesUnassignedAroundChannel is not None:
                    iesUnassignedAroundChannelNotMirror = np.intersect1d(iesUnassignedAroundChannel, G.iesNotMirror())
                else:
                    iesUnassignedAroundChannelNotMirror = None
                
                icMirror = self.setting.channelMirrorMap[ic]
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
        
        self.G.contractions = np.random.randint(0, len(self.setting.minLengths), self.G.contractions.shape)
        
        self.fromHalfGraph()
        
    def mutateHalfGraph(self):
        # choose a random edge and change its channel
        
        # find all edges that can be changed
        # randomly pick one edge
        # change its channel to one of the available channels incident to the edge
        
        G = self.G
    
        iess = []
        for ic in self.setting.channelMirrorMap.keys():
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
                icMirror = self.setting.channelMirrorMap[ic]
                if not(icMirror != -1 and G.esOnMirror[ie_incident]) and ic != ic_original:
                    ics_available.append(ic)
            
            while len(ics_available) > 0:
                ic = np.random.choice(ics_available)
                G.channels[ie] = ic
                
                channelsConnected = True
                
                for iChannel in self.setting.channelMirrorMap.keys():
                    channelsConnected *= G.channelConnected(iChannel)
                    
                if channelsConnected:
                    succeeded = True
                    break
                else:
                    G.channels[ie] = ic_original
                    ics_available.remove(ic)

        if np.random.rand() > 0.5:  # mutate contraction
            i = np.random.randint(len(G.contractions))
            G.contractions[i] = np.random.randint(len(self.setting.minLengths))
            
        self.fromHalfGraph()
        
    def frontDirection(self):
        return getFrontDirection(self.truss.v0, self.v).reshape(-1, 1)
        
    def relativePositions(self, requiresVelocity=False):

        v = self.v.copy()
        v -= self.v.mean(0)
        
        front = self.frontDirection()
        angle = -np.arctan(front[1], front[0])
        
        M = np.array([
            [np.cos(angle)[0], -np.sin(angle)[0]],
            [np.sin(angle)[0], np.cos(angle)[0]]
        ])
        
        v = np.hstack([M.dot(v.T[:2, :]).T, self.v[:, -1:]])
        
        vel = self.vel.copy()
        vel = np.hstack([M.dot(vel.T[:2, :]).T, self.vel[:, -1:]])
        
        if requiresVelocity:
            return v, vel
        else:
            return v
    
    def _updateTetList(self):
        self.tetList = []
        A = np.eye(len(self.v))
        
        for e in self.truss.e:
            A[e[0], e[1]] = 1
            A[e[1], e[0]] = 1
        print(A)
        
        for i in range(len(self.v)):
            for j in range(len(self.v)):
                if j == i:
                    continue
                
                if A[i, j] == 0:
                    continue
                
                for k in range(len(self.v)):
                    if k in {i, j}:
                        continue

                    if A[i, k] == 0 or A[j, k] == 0:
                        continue
                    
                    for l in range(len(self.v)):
                        if l in {i, j, k}:
                            continue
                        
                        if A[i, l] == A[j, l] == A[k, l] == 1:
                            tet = tuple(sorted([i, j, k, l]))
                            if tet in self.tetList:
                                continue
                            self.tetList.append( tet )
            
        self.tetList = np.array(self.tetList)
        
    def volume(self):
        try:
            self.tetList
        except:
            self.tetList = None
        
        if self.tetList is None:
            self._updateTetList()
        
        tetVertices = self.v[self.tetList]
        v0 = tetVertices[:, 0, :]
        v1 = tetVertices[:, 1, :]
        v2 = tetVertices[:, 2, :]
        v3 = tetVertices[:, 3, :]
        
        volumes = np.abs( ( np.cross(v1 - v0, v2 - v0) * ( v3 - v0 )).sum(1) / 6 )
        
        return volumes.sum()
    
    # region stabilize & save model
    def rescale(self):
        # scale the input model to a proper size

        iv0 = self.truss.e[:, 0]
        iv1 = self.truss.e[:, 1]
        v = self.truss.v0
        vec = v[iv1] - v[iv0]
        l = np.sqrt((vec ** 2).sum(1))
        
        scale = max(self.setting.minLengths) / l.mean()
        self.truss.v0 *= scale * 1.2
    
    def stabilize(self):
        self.reset()
        self.contractionPercent *= 0
        self.inflateChannel *= 0
        self.inflateChannel += 1
        
        self.recordEveryFrameOn()
        self.step(10000)
    
    def save(self, relativeJSONDir):
        data = dict()
        data['v'] = self.v.tolist()
        data['e'] = self.truss.e.tolist()
        data['eLengthLevel'] = self.truss.eLengthLevel.tolist()
        data['eActive'] = self.truss.eActive.tolist()
        data['eChannel'] = self.truss.eChannel.tolist()
        data['vFixed'] = self.truss.vFixed.tolist()
        
        with open(self.modelDir) as iFile:
            content = iFile.read()
            import json
            dataIn = json.loads(content)
            keys = dataIn.keys()
        
        assert(tuple(data.keys()) == tuple(keys))
        
        with open(relativeJSONDir, 'w') as oFile:
            c = json.dumps(data)
            oFile.write(c)
    
    def stabilizeNSave(self):
        self.rescale()
        self.stabilize()
        self.save(self.modelDir[:-5] + '_stabilized.json')
    
    # end region
    
    def show(self, show=True, essPrev=None):
        import polyscope as ps
        try:
            ps.init()
        except:
            pass
        
        G = self.G
        
        for ic in list(np.arange(self.setting.nChannels)):
            ies = np.arange(len(self.truss.e))[self.truss.eChannel == ic]
            
            # if ic == -1:
            #     assert(len(ies) == 0)
            # else:
            #     assert(len(ies) != 0)
            
            es = self.truss.e[ies]
            
            if essPrev is not None:
                es = essPrev[ic]
            
            ps.register_curve_network(str(ic), self.v, es)
        
            
        if show:
            ps.show()
    
    # endregion
