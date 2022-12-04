# multi-objective optimization
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import ray
import copy
from gym import Env
from gym.spaces import Discrete, Box

from utils.geometry import boundingBox
from utils.mesh import Mesh
from utils.model import Model
from utils.truss import Truss
from utils.trussEnv import TrussEnv
from utils.visualizer import showFrames


class MOO:
    class Setting:
        def __init__(self):
            # hyper parameter settings, unrelated to the specific model
            self.nStepsPerCapture = 400  # number of steps to capture one frame of v
            self.modelConfigDir = './data/config.json'
            self.nLoopPreSimulate = 1
            self.nLoopSimulate = 1

    def __init__(self, setting, randInit=False):
        self.randInit = randInit
        self.modelDir: str = ""
        self.numChannels: int = -1
        self.numActions: int = -1
        self.numObjectives: int = -1
        self.numTargets: int = -1
        self.channelMirrorMap: dict = dict()
        
        self.objectives: List = []
        self.actionSeqs: np.ndarray = np.zeros([])
        self.gene: np.ndarray = np.zeros([])

        self.incidenceMatrix = None
        self.model: Model = Model()
        self.setting = MOO.Setting()
        self.targetMeshes: List = []
        self.keyPointsIndices: List = []
        self.meshDirs: List = []
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
            assert (key in setting)

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

        if "meshDirs" in setting:
            self._loadMesh()
        assert (len(set(
            list(self.channelMirrorMap.keys()) + list(self.channelMirrorMap.values()))) - 1 == self.numChannels)


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

    def _loadMesh(self):
        assert (self.numTargets != -1)  # there are target meshes
        for i in range(self.numTargets):
            assert (isinstance(self.meshDirs[i], str) and len(self.meshDirs[i]) != 0)
            self.targetMeshes.append(Mesh(self.meshDirs[i], boundingBox(self.model.v)))

    def _loadModel(self):  # load model from modelDir
        assert (isinstance(self.modelDir, str) and len(self.modelDir) != 0)
        self.model = Model(self.setting.modelConfigDir)
        self.model.load(self.modelDir)
        # TODO: quick fix

        self.model.channelMirrorMap = self.channelMirrorMap
        self.model.numChannels = self.numChannels

        if self.randInit:
            if self.model.symmetric:
                self.model.toHalfGraph()
                self.model.initHalfGraph()
                self.model.fromHalfGraph()
            else:
                self.model.initGraph()
                self.model.fromGraph()

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

    def simulate(self, actionSeq, nLoops=1, visualize=False, export=True, mesh: Mesh = None) -> (
            np.ndarray, np.ndarray):
        assert (actionSeq.ndim == 2)
        assert (actionSeq.shape[0] >= 1)
        assert (actionSeq.shape[1] >= 1)
        
        modelConfigDir = self.setting.modelConfigDir
        self.model.configure(modelConfigDir)
        T = self.model.numStepsPerActuation
        nStepsPerCapture = self.setting.nStepsPerCapture

        self.refreshModel()

        model = self.model

        model.inflateChannel = actionSeq[:, -1]

        vs = []
        frames = []
        mesh_frames = []

        # v = model.step(T * self.setting.nLoopPreSimulate, ret=True)
        # vs = [v]

        for iLoop in range(self.setting.nLoopSimulate):
            for iAction in range(len(actionSeq[0])):
                model.inflateChannel = actionSeq[:, iAction]

                for iStep in range(T):
                    # append at the beginning of every nStepsPerCapture including frame

                    if model.numSteps % nStepsPerCapture == 0:
                        v = model.step(ret=True)
                        vs.append(v)
                    else:
                        v = model.step(ret=True)

                    if mesh != None:
                        if vs:
                            mesh.rigid_affine(vs[-1], v)
                        else:
                            mesh.rigid_affine(v, v)
                        mesh_frames.append(mesh.v)

                    frames.append(v)
        vs.append(model.v.copy())  # last frame
        vs = np.array(vs)

        assert (vs.shape == (vs.shape[0], len(model.v), 3))

        if visualize:
            print(len(frames))
            print(len(mesh_frames))
            showFrames(frames, model.e, mesh=mesh, mesh_frames=mesh_frames)

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

    def mutate(self):
        self.model.mutate()
        
        shape = self.actionSeqs.shape
        tp = self.actionSeqs.dtype
        actionSeqs = self.actionSeqs.reshape(-1)
        i = np.random.randint(len(actionSeqs))
        actionSeqs[i] = np.random.randint(2)
        self.actionSeqs = actionSeqs.reshape(shape).astype(tp)
        
        return self

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
    # endregion
