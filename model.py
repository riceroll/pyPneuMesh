import os
import sys
import time
import json
import argparse
import numpy as np
import open3d as o3
from optimizer import EvolutionAlgorithm
rootPath = os.path.split(os.path.realpath(__file__))[0]
tPrev = time.time()

# viewer
vector3d = lambda v: o3.utility.Vector3dVector(v)
vector3i = lambda v: o3.utility.Vector3iVector(v)
vector2i = lambda v: o3.utility.Vector2iVector(v)
LineSet = lambda v, e: o3.geometry.LineSet(points=vector3d(v), lines=vector2i(e))
PointCloud = lambda v: o3.geometry.PointCloud(points=vector3d(v))

def drawGround(viewer):
    n = 20
    vs = []
    es = []
    for i, x in enumerate(np.arange(1 - n, n)):
        vs.append([x, 1 - n, 0])
        vs.append([x, n - 1, 0])
        es.append([i * 2, i * 2 + 1])
    lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
    viewer.add_geometry(lines)
    
    vs = []
    es = []
    for i, x in enumerate(np.arange(1 - n, n)):
        vs.append([1 - n, x, 0])
        vs.append([n - 1, x, 0])
        es.append([i * 2, i * 2 + 1])
    lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
    viewer.add_geometry(lines)

# end of viewer ==================================

class Model(object):
    k = 200000
    h = 0.001
    dampingRatio = 0.999
    contractionInterval = 0.075
    contractionLevels = 5
    maxMaxContraction = round(contractionInterval * (contractionLevels - 1) * 100) / 100
    contractionPercentRate = 1e-3
    gravityFactor = 9.8 * 10
    gravity = 1
    defaultMinLength = 1.2
    defaultMaxLength = defaultMinLength / (1 - maxMaxContraction)
    frictionFactor = 0.8
    numStepsPerAction = int(2 / h)
    defaultNumActions = 1
    defaultNumChannels = 4
    angleThreshold = np.pi / 2
    cornerCheckFrequency = numStepsPerAction / 20

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

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        self.scripting = True
        
        self.inflateChannel = None
        self.contractionPercent = None
        self.numChannels = None
        self.numActions = None
        
        self.targets = None
        self.testing = False    # testing mode
        
    # initialization =======================
    def loadJson(self, name):
        """
        load parameters from a json file into the model
        :param name: dir of the json file
        :return: data as a dictionary
        """
        
        with open(name) as ifile:
            content = ifile.read()
        data = json.loads(content)
        self.loadDict(data)
        return data
        
    def loadDict(self, data):
        """
        load the data into the model as the initial state
        :param data: data as a dictionary (from self.loadJSON)
        """
        self.v = np.array(data['v'])
        # self.v -= self.v.mean(0)    # center the model
        self.v0 = np.array(self.v)
        self.e = np.array(data['e'], dtype=np.int64)
        
        self.lMax = np.array(data['lMax'])
        self.maxContraction = np.array(data['maxContraction'])
        if len(self.maxContraction) == 0:
            self.maxContraction = np.zeros_like(self.lMax)
        self.fixedVs = np.array(data['fixedVs'], dtype=np.int64)
        self.edgeChannel = np.array(data['edgeChannel'], dtype=np.int64)
        self.edgeActive = np.array(data['edgeActive'], dtype=bool)
        self.script = np.array([0 for c in range(np.max(self.edgeChannel) + 1)], dtype=np.int64).reshape(-1, 1)
        
        self.reset()
        
    def reset(self, resetScript=False, numChannels=None, numActions=None):
        """
        reset the model to the initial state with input options
        :param resetScript: if True, reset script to zero with numChannels and numActions
        :param numChannels: if -1, use the default value from self.numChannels or self.edgeChannel
        :param numActions: if -1, use the default value from self.numActions or self.script[0]
        """
        
        self.v = np.array(self.v0)
        self.vel = np.zeros_like(self.v)

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        self.numChannels = numChannels if numChannels and numChannels != -1 \
            else self.numChannels if self.numChannels \
            else int(np.max(self.edgeChannel) + 1)
        
        self.numActions = numActions if numActions and numActions != -1 \
            else self.numActions if self.numActions \
            else len(self.script[0])
        self.inflateChannel = np.zeros(self.numChannels)
        self.contractionPercent = np.ones(self.numChannels)
        if resetScript:
            self.script = np.zeros([self.numChannels, self.numActions])
        self.updateCorner()
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
        
            # script
            if self.scripting:
                if self.numSteps > ((self.iAction + 1) % self.numActions) * Model.numStepsPerAction:
                    self.iAction = int(np.floor(self.numSteps / Model.numStepsPerAction) % self.numActions)
                
                    for iChannel in range(self.numChannels):
                        self.inflateChannel[iChannel] = self.script[iChannel, self.iAction]
        
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

            l0[self.edgeActive] = (lMax - self.contractionPercent[self.edgeChannel] * (lMax - lMin))[self.edgeActive]
            
            
            fMagnitude = (l - l0) * Model.k
            fEdge = vec / l.reshape(-1, 1) * fMagnitude.reshape(-1, 1)
            np.add.at(f, iv0, fEdge)
            np.add.at(f, iv1, -fEdge)
        
            f[:, 2] -= Model.gravityFactor * Model.gravity
            self.f = f
        
            self.vel += Model.h * f
        
            boolUnderground = self.v[:, 2] <= 0
            self.vel[boolUnderground, :2] *= 1 - Model.frictionFactor
        
            self.vel *= Model.dampingRatio
            velMag = np.sqrt((self.vel ** 2).sum(1))
            while (velMag > 5).any():
                self.vel[(velMag > 5)] *= 0.9
                velMag = np.sqrt((self.vel ** 2).sum(1))
        
            self.v += Model.h * self.vel
        
            self.vel[boolUnderground, 2] *= -1
            self.v[boolUnderground, 2] = 0
        
            if self.numSteps % Model.cornerCheckFrequency == 0:
                a = self.getCornerAngles()
                if (a - self.a0 > self.angleThreshold).any():
                    print("exceed angle")
                    return np.ones_like(self.v) * -1e6
        return self.v

    def iter(self, gene=None, visualize=False):
        """
        load the gene and run a series of steps
        :param gene: the gene to load into the model, gene format is
            edgeChannel : (ne, ) int, [0, self.numChannels)
            contractionPercentLevel : (ne, ) int, [0, Model.contractionLevels)
            script : (na, nc) int, {0, 1}
        :param visualize: if True, visualize the iteration with open3D
        :return: the vertices location of the model
        """
        if gene is not None:
            self.loadGene(gene)
        self.reset()
        
        vs = []
        if not visualize:
            if self.testing:
                for i in range(5):
                    ret = self.step(2)
                    vs.append(ret)
            else:
                for i in range(4):
                    ret = self.step(int(Model.numStepsPerAction * self.numActions))
                    vs.append(ret)
        else:
            viewer = o3.visualization.VisualizerWithKeyCallback()
            viewer.create_window()
        
            render_opt = viewer.get_render_option()
            render_opt.mesh_show_back_face = True
            render_opt.mesh_show_wireframe = True
            render_opt.point_size = 8
            render_opt.line_width = 10
            render_opt.light_on = True
        
            ls = LineSet(self.v, self.e)
            viewer.add_geometry(ls)
        
            def timerCallback(vis):
                self.step(25)
                ls.points = vector3d(self.v)
                viewer.update_geometry(ls)
        
            viewer.register_animation_callback(timerCallback)
        
            # def key_step(vis):
            #     pass
        
            # viewer.register_key_callback(65, key_step)
        
            drawGround(viewer)
            viewer.run()
            viewer.destroy_window()
    
        return vs
    # end stepping
    
    # checks =========================
    def updateCorner(self):
        self.c = []
        
        for iv in range(len(self.v)):
            # get indices of neighbor vertices
            ids = np.concatenate([self.e[self.e[:, 0] == iv][:, 1], self.e[self.e[:, 1] == iv][:, 0]])
            for i in range(len(ids) - 1):
                id0 = ids[i]
                id1 = ids[i+1]
                self.c.append([iv, id0, id1])
        
        self.c = np.array(self.c)
        
        self.a0 = self.getCornerAngles()
    
    def getCornerAngles(self):
        # return angles of self.c
        O = self.v[self.c[:, 0]]
        A = self.v[self.c[:, 1]]
        B = self.v[self.c[:, 2]]
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
        self.step(Model.numStepsPerAction * 2)
        self.v0 = self.v
    
    # end utility

    # optimization ===========================
    def geneSetSize(self):
        nEdgeChannel = len(self.edgeChannel)
        nMaxContraction = len(self.maxContraction)
        nScript = self.script.shape[0] * self.script.shape[1]
        return nEdgeChannel, nMaxContraction, nScript
    
    def setTargets(self, targets):
        self.targets = targets
    
    def lb(self):
        [nEdgeChannel, nMaxContraction, nScript] = self.geneSetSize()
        nTargets = self.targets.numTargets()
        return np.zeros(nEdgeChannel + nMaxContraction + nTargets * nScript, dtype=np.int64)
    
    def ub(self):
        # caution: the upper bound here is noninclusive
        [nEdgeChannel, nMaxContraction, nScript] = self.geneSetSize()
        
        ubEdgeChannel = np.ones(nEdgeChannel) * self.numChannels
        ubMaxContraction = np.ones(nMaxContraction) * Model.contractionLevels
        ubScript = np.ones(nScript * self.targets.numTargets()) * 2
        return np.concatenate([ubEdgeChannel, ubMaxContraction, ubScript])
    
    def loadGene(self, gene):
        nEdgeChannel = len(self.edgeChannel)
        nMaxContraction = len(self.maxContraction)
        nScript = len(self.script.reshape(-1))
        
        self.edgeChannel = np.array(gene[:nEdgeChannel], dtype=np.int64)
        gene = gene[nEdgeChannel:]
        
        self.maxContraction = np.array(gene[:nMaxContraction] * Model.contractionInterval)
        gene = gene[nMaxContraction:]
        
        self.script = np.array(gene[:], dtype=bool).reshape(self.numChannels, self.numActions)
        
        self.edgeActive = np.ones_like(self.edgeActive, dtype=bool)
    
    def exportJSON(self, gene, inDir, appendix=None):
        """
        export the gene into JSON, with original model from the JSON with name
        :param gene: name of the gene
        :param inDir: name of the imported json file
        :param appendix: appendix to the output filename
        """
        
        with open(inDir) as iFile:
            content = iFile.read()
        data = json.loads(content)
        self.loadGene(gene)
        data['edgeChannel'] = self.edgeChannel.tolist()
        data['edgeActive'] = self.edgeActive.tolist()
        data['maxContraction'] = self.maxContraction.tolist()
        data['script'] = self.script.tolist()
        data['numChannels'] = self.numChannels
        data['numActions'] = self.numActions
        
        name = inDir.split('/')[-1]
        with open('{}/output/{}_{}.json'.format(rootPath, name, appendix), 'w') as oFile:
            js = json.dumps(data)
            oFile.write(js)
            
    def exportJSONs(self, geneSet, targets, inDir):
        for iTarget in range(targets.numTargets()):
            gene = targets.extractGene(geneSet, iTarget)
            self.exportJSON(gene, inDir, str(iTarget))
    # end optimization
    

# if __name__ == "__main__":
#     inFileDir = '{}/data/{}.json'.format(rootPath, inFileName)
#
#     model = Model()
#     model.loadJson(inFileDir)
#     model.scripting = scripting
#     model.reset(resetScript=True, numChannels=numChannels, numActions=numActions)
#
#     def locomotion(gene, direction='x'):
#         v = model.iter(gene)
#         centroid = v.mean(0)
#         if direction == 'x':
#             return centroid[0] - abs(centroid[1]) * 10
#         elif direction == 'y':
#             return centroid[1] - abs(centroid[0]) * 10
#
#     locomotion_x = lambda g : locomotion(g, 'x')
#     locomotion_y = lambda g: locomotion(g, 'y')
#     criterion = locomotion_x if direction == "x" else locomotion_x
#
#     ea = EvolutionAlgorithm(name=inFileName, model=model, criterion=criterion,
#                             nWorkers=numWorkers,
#                             nPop=numPopulation)
#     gene = ea.maximize(10 if testing else numGeneration)
#
#     model.loadGene(gene)
#     model.exportJSON(gene, inFileDir)
#
#     if visualize:
#         model.iter(gene, True)
