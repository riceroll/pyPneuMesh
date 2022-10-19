# multi-objective optimization
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

from utils.model import Model


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
