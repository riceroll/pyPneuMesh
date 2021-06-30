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

        self.numChannels = self.edgeChannel.max() + 1
        self.inflateChannel = np.zeros(self.numChannels)
        self.contractionPercent = np.ones(self.numChannels)

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        self.updateCornerAngles()
        self.computeSymmetry()
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
            saveDir = '{}/output/{}_{}.json'.format(rootPath, name, timeStr)
        
        if save:
            with open(saveDir, 'w') as oFile:
                oFile.write(js)
                print('Save to {}/output/{}_{}.json'.format(rootPath, name, timeStr))
                
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
