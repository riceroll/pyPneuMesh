# multi-objective optimization
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import ray
import copy
from gym import Env
from gym.spaces import Discrete, Box

from utils.model import Model
from utils.visualizer import showFrames


class TrussEnv(Env):
    def __init__(self, model, moo, objective):
        self.model = model
        self.moo = moo
        self.model.configure(moo.setting.modelConfigDir)
        
        self.action_space = Discrete(2 ** model.numChannels)
        self.observation_space = Box(low=np.ones([model.v.shape[0] * 3 * 3]) * -np.inf,
                                     high=np.ones([model.v.shape[0] * 3 * 3]) * np.inf)

        vRelative, vel = self.model.relativePositions(requiresVelocity=True)
        
        self.state = np.vstack([self.model.v.copy(), vRelative, self.model.vel.copy()]).reshape(-1)
        
        self.numStepsPerAction = Model.numStepsPerActuation
        self.numStepsPerSample = moo.setting.nStepsPerCapture
        self.numStepsTotal = self.numStepsPerAction * 16
        
        self.objective = objective
        self.vs = [self.model.v.copy()]
    
    def step(self, action):
        assert (action in self.action_space)
        self.model.inflateChannel = np.array([int(dig) for dig in bin(action)[2:]])
        self.model.step(self.numStepsPerAction)
        # v = self.model.v.copy()
        # vel = self.model.vel.copy()
        
        vRelative, vel = self.model.relativePositions(requiresVelocity=True)
        v = self.model.v.copy()
        self.state = np.vstack([v, vRelative, vel]).reshape(-1)
        
        # if self.model.numSteps % self.numStepsPerSample == 0:
        self.vs.append(v)
        
        done = False
        reward = 0
        info = {}

        if self.model.numSteps > self.numStepsTotal:
            done = True
            for subObjective in self.objective:
                reward += subObjective(self.vs, self.model.e)
        
        return self.state, reward, done, info
    
    def render(self, mode="human"):
        
        import polyscope as ps
        try:
            ps.init()
        except:
            pass

        ps.register_curve_network('curves', self.model.v, self.model.e)

        ps.show()

    
    def reset(self):
            self.model._reset()
            self.model.configure(self.moo.setting.modelConfigDir)

            vRelative, vel = self.model.relativePositions(requiresVelocity=True)
            
            self.state = np.vstack([self.model.v.copy(), vRelative, vel]).reshape(-1)
            self.vs = []
            return self.state

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
            unvisitedChannels = set(list(self.channelMirrorMap.keys()) + list(self.channelMirrorMap.values()))
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
                print(ic, icMirror)
                
                # iesAvailable
                iesOfChannel = np.where(edgeChannel == ic)[0]     # ids of edge in channel ic
                ivsOfChannel = np.array(list(set(self.model.e[iesOfChannel].reshape(-1))))
                subIncidenceMat = incidenceMat[ivsOfChannel]    # row: v of channel ic, col: edge
                iesConnected = set(np.where(subIncidenceMat == 1)[1])    # id of es connected to vs in channel
                iesAvailable = []
                print(iesConnected)
                
                for ie in iesConnected:
                    ieMirror = self.edgeMirrorMap[ie]
                    if ie in edgeUnvisited:
                        if not ((icMirror == -1 and ieMirror == -1) or (icMirror != -1 and ieMirror != -1)):  # mirror channel and mirror edge
                            continue
                        iesAvailable.append(ie)
                if len(iesAvailable) == 0:
                    print('channel {} not available'.format(ic))
                    continue
                print(iesAvailable)
                
                # pick an edge and assign
                ie = np.random.choice(iesAvailable)
                edgeChannel[ie] = ic
                edgeUnvisited.remove(ie)
                print(ie)
                
                ieMirror = self.edgeMirrorMap[ie]
                if ieMirror != -1:  # mirror edge
                    edgeChannel[ieMirror] = icMirror
                    print('mirror', ieMirror, icMirror)
                    assert(icMirror != -1)
                    edgeUnvisited.remove(ieMirror)
            
            assert((edgeChannel == -1).sum() == 0)
            return edgeChannel
            
    def __init__(self, setting, randInit=False):
        self.randInit = randInit
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
        # self.geneHandler = self.GeneHandler(self)
    
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
            
        if "actionSeqs" not in setting:
            self._loadActionSeqs()
        
        if "model" not in setting:
            self._loadModel()

        if "gene" not in setting:
            pass

        if "objectives" not in setting:
            pass

        if "channelMirrorMap" not in setting:
            self.channelMirrorMap = {ic: -1 for ic in range(self.numChannels)}
        assert (len(set(list(self.channelMirrorMap.keys()) + list(self.channelMirrorMap.values()))) - 1 == self.numChannels)
        
        
        
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
        
        if self.randInit:
            self.actionSeqs = np.random.randint(0, 2, [self.numObjectives, self.numChannels, self.numActions])
    
    def _loadModel(self):   # load model from modelDir
        assert (isinstance(self.modelDir, str) and len(self.modelDir) != 0)
        self.model = Model(self.setting.modelConfigDir)
        self.model.load(self.modelDir)
        # TODO: quick fix

        self.model.channelMirrorMap = self.channelMirrorMap
        self.model.numChannels = self.numChannels
        if self.numChannels < 4:
            self.model.setToSingleChannel()
        
        if self.randInit:
            self.model.toHalfGraph()
            self.model.initHalfGraph()
            self.model.fromHalfGraph()
        
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
        self.model._reset()
        self.model.configure(self.setting.modelConfigDir)
        # self.loadGene(self.gene)
    
    def loadGene(self, gene: np.ndarray) -> (Model, np.ndarray):  # load gene into model and actionSeqs
        return self.model, self.actionSeqs
        
    def simulate(self, actionSeq, nLoops=1, visualize=False, export=True) -> (np.ndarray, np.ndarray):
        assert (actionSeq.ndim == 2)
        assert (actionSeq.shape[0] >= 1)
        assert (actionSeq.shape[1] >= 1)
        
        T = Model.numStepsPerActuation
        nStepsPerCapture = self.setting.nStepsPerCapture
        
        self.refreshModel()
        
        model = self.model
        
        model.inflateChannel = actionSeq[:, -1]
        
        vs = []
        frames = []
          
        # v = model.step(T * self.setting.nLoopPreSimulate, ret=True)
        # vs = [v]
    
        for iLoop in range(self.setting.nLoopSimulate):
            for iAction in range(len(actionSeq[0])):
                model.inflateChannel = actionSeq[:, iAction]
                
                for iStep in range(T):
                    # append at the beginning of every nStepsPerCapture including frame 0
                    if model.numSteps % nStepsPerCapture == 0:
                        v = model.step(ret=True)
                        vs.append(v)
                    else:
                        v = model.step(ret=True)
                    # print(model.contractionPercent)
                    frames.append(v)
                            
        vs.append(model.v.copy())   # last frame
        vs = np.array(vs)

        assert (vs.shape == (vs.shape[0], len(model.v), 3))
        
        if visualize:
            print(len(frames))
            showFrames(frames, model.e)
            
        if export:
            path = './output/frames_.json'
            data = {
                'vs': vs.tolist(),
                'e': self.model.e.tolist(),
                'channels': self.model.edgeChannel.tolist()
            }
            
            import json
            js = json.dumps(data)
            with open(path, 'w') as oFile:
                oFile.write(js)
                
            
        return vs, self.model.e.copy()
    
    def train(self):
        # PPO
        pass
    
    
    def mutate(self):
        self.model.mutateHalfGraph()
        self.model.fromHalfGraph()
        
        shape = self.actionSeqs.shape
        tp = self.actionSeqs.dtype
        actionSeqs = self.actionSeqs.reshape(-1)
        i = np.random.randint(len(actionSeqs))
        actionSeqs[i] = np.random.randint(2)
        self.actionSeqs = actionSeqs.reshape(shape).astype(tp)
        return self
        
    def check(self):
        if len(self.channelMirrorMap) != 0:
            assert(self.numChannels == len(self.channelMirrorMap))
            assert(self.numChannels >= self.model.edgeChannel.max() + 1)
        if len(self.objectives) != 0:
            assert(len(self.objectives) == self.numObjectives)
        # if self.actionSeqs.shape != ():
        #     assert(self.actionSeqs.shape == (self.numObjectives, self.numChannels, self.numActions))
    
    
    def make_env(self, iObjective):
        newModel = copy.deepcopy(self.model)
        env = TrussEnv(newModel, self, self.objectives[iObjective])
        return env
        
    
    # region220522
    def initModel(self):
        # initialize the model's parameters (channel, contraction, actions)

        pass
    
    def mutateModel(self):
        # update the model's parameters (channel, contraction, actions)
        
        pass
    
    def evaluateModel(self) -> (np.ndarray):
        # return:
        #   a list of scores corresponding to the list of combined objectives
        
        pass
    #endregion


