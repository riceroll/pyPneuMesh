import os
import sys
import time
import datetime
import json
import argparse
import numpy as np

import utils
from utils.geometry import getFrontDirection
rootPath = os.path.split(os.path.realpath(__file__))[0]
tPrev = time.time()

class Model(object):
    # k = 200000
    # h = 0.001
    # dampingRatio = 0.999
    # contractionInterval = 0.075
    # contractionLevels = 5
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
    def configure():
        with open("./data/config.json") as ifile:
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
    
    def __init__(self):
        
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
        
        self.targets = None
        self.testing = False    # testing mode
        self.modelDir = ""     # input json file directory
        
        Model.configure()
    
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
        self.load(self.modelDir)
        
    def _reset(self, resetScript=False):
        """
        reset the model to the initial state with input options
        :param resetScript: if True, reset script to zero with numChannels and numActions
        :param numChannels: if -1, use the default value from self.numChannels or self.edgeChannel
        :param numActions: if -1, use the default value from self.numActions or self.script[0]
        """
        
        self.v = self.v0.copy()
        self.vel = np.zeros_like(self.v)

        self.numChannels = self.edgeChannel.max() + 1
        self.inflateChannel = np.zeros(self.numChannels)
        self.contractionPercent = np.ones(self.numChannels)

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        self.updateCornerAngles()
    # end initialization
    
    # stepping =======================
    def step(self, n=1):
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
            lMin = lMax * (1 - self.maxContraction)
            
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
            if True:
                if self.numSteps % (Model.numStepsPerActuation / 20) == 0 or self.frontVec is None:
                    self.frontVec = getFrontDirection(self.v0, self.v).reshape(-1, 1)
                # self.frontVec = np.array([1, 0, 0]).reshape(-1, 1)
                vel = self.vel.copy()
                vel[2] *= 0
    
                dot = (vel @ self.frontVec)
                ids = (dot < 0).reshape(-1) * boolUnderground
                self.vel[ids] -= dot[ids] * self.frontVec.reshape(1, -1)
                
            self.v += Model.h * self.vel

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
                   save=True):
        """
        export the model into JSON, with original model from the JSON as name

        :param modelDir: name of the imported json file
        :param edgeChannel: np.array int [numEdge, ]
        :param maxContraction: np.array float [numEdgeActive, ]
        :param actionSeq: np.array [numChannel, numActions]
        """
        modelDir = modelDir if modelDir else self.modelDir
        self.load(modelDir)
        
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
        
        if save:
            with open('{}/output/{}_{}.json'.format(rootPath, name, timeStr), 'w') as oFile:
                oFile.write(js)
                print('Save to {}/output/{}_{}.json'.format(rootPath, name, timeStr))
                
        return js

    # end optimization


def testModelStep(argv):
    from utils.modelInterface import getModel
    
    model = getModel("./test/data/lobsterIn.json")
    v = model.step(200)
    js = model.exportJSON(save=False)
    with open('./test/data/lobsterOut.json') as iFile:
        jsTrue = iFile.read()
        assert (js == jsTrue)
    
    vTrue = np.load('./test/data/lobsterOutV.npy')
    assert ((v == vTrue).all())

    
tests = {
    'step': testModelStep,
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
