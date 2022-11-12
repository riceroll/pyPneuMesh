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

import numpy as np


class HalfGraph(object):
    def __init__(self):
        self.ins_o = []  # nn, original indices of nodes in a halfgraph
        self.edges = []  # indices of two incident nodes, ne x 2, ne is the number of edges in a halfgraph
        self.ies_o = []  # ne, original indices of edges in a halfgraph
        self.channels = []  # ne, indices of channels
        self.contractions = []  # ne, int value of contractions
        self.esOnMirror = []  # ne, bool, if the edge in on the mirror plane

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


class Model(object):

    # MOO setting
    # objectives
    # model
    # GA

    # h = 0.001

    # gravity
    # dampingRatio = 0.999

    # k = 200000    # mass spring vs yongs modulus

    # nodeWeight

    # lengths
    # 101, 124, 154, 174

    # numChannels

    # angleThreshold = np.pi / 2

    # GA control
    # numActions
    # numStepsPerAction
    # nLoopsSimulate

    # GA parameters

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
    def configure(configDir="./data/config.json"):
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
        self.v = None  # vertices locations    [nv x 3]
        self.e = None  # edge                  [ne x 2] int
        self.v0 = None
        self.c = None  # indices of v at corners [nc x 3] e.g. <AOB -> [iO, iA, iB]
        self.a0 = None  # rest angle of corresponding corner [nc]
        self.fixedVs = None
        self.lMax = None
        self.edgeActive = None
        self.edgeChannel = None
        self.script = None

        self.maxContraction = None
        self.vel = None  # velocity              [nv x 3]
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

        self.testing = False  # testing mode
        self.vertexMirrorMap = dict()
        self.edgeMirrorMap = dict()
        self.channelMirrorMap = dict()

        self.tetList = None  # #T by 4, vertices of tetrahedrons

        self.showHistory = []  # edges at every update of channel growing

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

        # assert ((self.lMax[self.edgeActive] == self.lMax[self.edgeActive][0]).all())
        assert (len(self.e) == len(self.lMax) == len(self.edgeChannel) == len(self.edgeActive) and len(
            self.maxContraction))
        self._reset()

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
        except:  # if self.numChannels not defined
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
            lMin = lMax * (1 - (Model.contractionLevels - 1) * self.contractionInterval)

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
            self.vel[boolUnderground, :2] *= 1 - Model.frictionFactor  # uniform friction
            # damping
            self.vel *= Model.dampingRatio
            velMag = np.sqrt((self.vel ** 2).sum(1))
            if (velMag > 5).any():
                self.vel[velMag > 5] *= np.power(0.9, np.ceil(np.log(5 / velMag[velMag > 5]) / np.log(0.9))).reshape(-1,
                                                                                                                     1)

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
                id1 = ids[i + 1]
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
                    vertexMirrorMap[iv] = -1  # on the mirror plane
                else:  # mirrored with another vertex
                    boolMirrored = ((v[[0, 2]] - xzs) ** 2).sum(1) < 1
                    ivsMirrored = np.where(boolMirrored)[0]
                    ivMirrored = ivsMirrored[np.argmin(abs(v[1] + self.v[ivsMirrored][:, 1]))]
                    assert (ivMirrored not in vertexMirrorMap)
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
            if ivM0 == iv0 and ivM1 == iv1:  # edge on the mirror plane
                edgeMirrorMap[ie] = -1
            else:
                iesMirrored = (eM == self.e).all(1) + (eM[::-1] == self.e).all(1)
                assert (iesMirrored.sum() == 1)
                ieMirrored = np.where(iesMirrored)[0][0]
                assert (ieMirrored not in edgeMirrorMap)
                if ieMirrored == ie:  # edge rides across the mirror plane
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
        # end region
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
            iesUnassigned = set(np.arange(len(G.edges)))  # halfgraph indices of edges
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
                if not (icMirror != -1 and G.esOnMirror[ie_incident]) and ic != ic_original:
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

    def frontDirection(self):
        return getFrontDirection(self.v0, self.v).reshape(-1, 1)

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

        for e in self.e:
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
                            self.tetList.append(tet)

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

        volumes = np.abs((np.cross(v1 - v0, v2 - v0) * (v3 - v0)).sum(1) / 6)

        return volumes.sum()

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

    def _loadEdgeChannel(self, edgeChannel):
        """
        load edgeChannel
        :param edgeChannel: np.array int [numEdge, ]
        """
        assert (type(edgeChannel) == np.ndarray)
        assert (edgeChannel.dtype == int)
        assert (edgeChannel.shape == (len(self.e),))
        self.edgeChannel = edgeChannel

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
            self._loadEdgeChannel(edgeChannel)
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
