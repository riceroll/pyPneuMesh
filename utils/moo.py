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
        
        self.model.reset()
        
        self.action_space = Discrete(2 ** model.setting.nChannels)
        self.observation_space = Box(low=np.ones([model.v.shape[0] * 3 * 3]) * -np.inf,
                                     high=np.ones([model.v.shape[0] * 3 * 3]) * np.inf)
        
        vRelative, vel = self.model.relativePositions(requiresVelocity=True)
        
        self.state = np.vstack([self.model.v.copy(), vRelative, self.model.vel.copy()]).reshape(-1)
        
        self.nStepsDone = moo.nStepsPerActuation * moo.setting.nActions
        
        self.objective = objective
        self.vs = [self.model.v.copy()]
    
    def step(self, action: int):
        # assign action
        assert (action in self.action_space)
        self.model.inflateChannel = np.array([int(dig) for dig in bin(action)[2:]])
        
        # step
        self.model.step(self.moo.nStepsPerActuation)
        
        # get state & trajectory
        vRelative, vel = self.model.relativePositions(requiresVelocity=True)
        v = self.model.v.copy()
        self.state = np.vstack([v, vRelative, vel]).reshape(-1)
        self.vs.append(v)
        
        done = False
        reward = 0
        info = {}

        if self.model.nSteps > self.nStepsDone:
            done = True
            
            # evaluate objective functions by summing up
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
        self.model.reset()

        vRelative, vel = self.model.relativePositions(requiresVelocity=True)
        
        self.state = np.vstack([self.model.v.copy(), vRelative, vel]).reshape(-1)
        self.vs = []
        return self.state


class MOO:
    
    class Setting:
        def __init__(self, settingDict):
            # hyper parameter settings, unrelated to the specific model
            self.modelDir = None
            self.modelSettingDir = None
            self.nActions = None
            self.nLoopPerSim = None

            self.randInit = True
            self.nStepsPerCapture = 400     # number of steps to capture one frame of v
            self.objectives = []
            
            self.load(settingDict)
        
        def load(self, settingDict):
            for key in settingDict:
                assert(hasattr(self, key))
                setattr(self, key, settingDict[key])
                
    def __init__(self, settingDict):
        self.setting = MOO.Setting(settingDict)
        self.model = Model(settingDir=self.setting.modelSettingDir, modelDir=self.setting.modelDir)
        self.actionSeqs = np.ones([
            len(self.setting.objectives),
            self.model.setting.nChannels,
            self.setting.nActions
        ])
        
        if self.setting.randInit and self.model.symmetric:
            self.model.toHalfGraph()
            self.model.initHalfGraph()
            self.model.fromHalfGraph()
        
        self.model.reset()
        self.mutate()
    
    def simulateOpenLoop(self, iActionSeq=0, visualize=False, export=False):
        actionSeq = self.actionSeqs[iActionSeq]
        assert (actionSeq.ndim == 2)
        
        # reset model
        self.model.reset()
        model = self.model
        if visualize:
            model.recordEveryFrameOn()
        else:
            model.recordEveryFrameOff()
    
        for iLoop in range(self.setting.nLoopPerSim):
            for iAction in range(len(actionSeq[0])):
                model.inflateChannel = actionSeq[:, iAction]
                
                model.step(self.model.setting.nStepsPerActuation)
        
        if visualize:
            showFrames(model.vs, model.truss.e, model.setting.h)
        
        vs = np.array(model.vs)
        assert (vs.shape == (vs.shape[0], len(model.v), 3))
        
        if export:
            path = './output/frames_.json'
            data = {
                'vs': vs.tolist(),
                'e': self.model.truss.e.tolist(),
                'channels': self.model.truss.eChannel.tolist()
            }
            
            import json
            js = json.dumps(data)
            with open(path, 'w') as oFile:
                oFile.write(js)
                
        return vs, self.model.truss.e.copy()
    
    def mutate(self):
        self.model.mutateHalfGraph()
        self.model.fromHalfGraph()
        
        # mutate one digit of actionSeq
        shape = self.actionSeqs.shape
        tp = self.actionSeqs.dtype
        actionSeqs = self.actionSeqs.reshape(-1)
        i = np.random.randint(len(actionSeqs))
        actionSeqs[i] = np.random.randint(2)
        self.actionSeqs = actionSeqs.reshape(shape).astype(tp)
        
        return self
        
    def check(self):
        if len(self.model.setting.channelMirrorMap) != 0:
            assert(self.model.setting.nChannels == len(self.model.setting.channelMirrorMap))
            assert(self.model.setting.nChannels >= self.model.truss.eChannel.max() + 1)
        
    def make_env(self, iObjective):
        newModel = copy.deepcopy(self.model)
        env = TrussEnv(newModel, self, self.setting.objectives[iObjective])
        return env
