import os
import sys
import time
import json
import argparse
import numpy as np
from optimizer import EvolutionAlgorithm
rootPath = os.path.split(os.path.realpath(__file__))[0]
tPrev = time.time()

# consts
parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False, help="whether to visualize the result")
parser.add_argument("--testing", type=bool, default=False, help="whether in testing mode")
parser.add_argument("--iFile", type=str, default="trussformer", help="name of input file under ./data folder")
parser.add_argument("--nGen", type=int, default=1000, help="whether in testing mode")
parser.add_argument("--nPop", type=int, default=100, help="size of population")
parser.add_argument("--direction", type=str, default="x", help="direction of locomotion")
args = parser.parse_args()

scripting = True
visualize = args.visualize
testing = args.testing
inFileName = args.iFile
numGeneration = args.nGen
numPopulation = args.nPop
direction = args.direction

if visualize:
    import open3d as o3
    
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
    dampingRatio = 0.992
    contractionInterval = 0.075
    contractionLevels = 5
    maxMaxContraction = round(contractionInterval * (contractionLevels - 1) * 100) / 100
    contractionPercentRate = 5e-4
    gravityFactor = 9.8 * 20
    gravity = 1
    defaultMinLength = 1.2
    defaultMaxLength = defaultMinLength / (1 - maxMaxContraction)
    frictionFactor = 0.8
    numStepsAction = 2 / h
    defaultNumActions = 1
    defaultNumChannels = 4

    def __init__(self, modelName='tet'):
        
        # input: modelName: the file name of the geometry model, under ./data/model

        # ======= variable =======
        self.v = None       # vertices locations    [nv x 3]
        self.e = None       # edge                  [ne x 2] int
        self.v0 = None
        self.fixedVs = None
        self.lMax = None
        self.edgeActive = None
        self.edgeChannel = None
        self.script = None
        
        self.maxContraction = None
        self.vel = None     # velocity              [nv x 3]
        # self.f = None
        # self.l = None

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        self.scripting = True
        
        self.inflateChannel = None
        self.contractionPercent = None
        self.numChannels = None
        self.numActions = None
        
    def loadJson(self, name):
        with open(name) as ifile:
            content = ifile.read()
        data = json.loads(content)
        self.loadDict(data)
        return data
        
    def loadDict(self, data):
        self.v = np.array(data['v'])
        self.v -= self.v.mean(0)    # center the model
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
        self.v = np.array(self.v0)
        self.vel = np.zeros_like(self.v)

        self.iAction = 0
        self.numSteps = 0
        self.gravity = True
        self.simulate = True
        
        self.numChannels = numChannels if numChannels \
            else max(self.numChannels, int(np.max(self.edgeChannel) + 1)) \
            if self.numChannels else int(np.max(self.edgeChannel) + 1)
        
        self.numActions = numActions if numActions \
            else max(self.numActions, len(self.script[0])) \
            if self.numActions else len(self.script[0])
        self.inflateChannel = np.zeros(self.numChannels)
        self.contractionPercent = np.ones(self.numChannels)
        if resetScript:
            self.script = np.zeros([self.numChannels, self.numActions])
        
        
    def centroid(self):
        return np.sum(self.v, 0) / self.v.shape[0]
    
    def step(self, n=1):
        if not self.simulate:
            return
        
        for i in range(n):
            self.numSteps += 1

            # script
            if self.scripting:
                if self.numSteps > ((self.iAction + 1) % self.numActions) * Model.numStepsAction:
                    self.iAction = int(np.floor(self.numSteps / Model.numStepsAction) % self.numActions)
        
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
            model.l = l = np.sqrt(np.sum( vec ** 2, 1))
            
            l0 = np.copy(self.lMax)
            lMax = np.copy(self.lMax)
            lMin = lMax * (1 - self.maxContraction)
            l0[self.edgeActive] = (lMax - self.contractionPercent[self.edgeChannel] * (lMax - lMin))[self.edgeActive]
            
            fMagnitude = (l - l0) * Model.k
            fEdge = vec / l.reshape(-1, 1) * fMagnitude.reshape(-1, 1)
            np.add.at(f, iv0, fEdge)
            np.add.at(f, iv1, -fEdge)
            
            f[:, 2] -= Model.gravityFactor * Model.gravity
            model.f = f
            
            self.vel += Model.h * f
            
            self.vel[:, :2] *= 1 - Model.frictionFactor
            
            self.vel *= Model.dampingRatio

            velMag = np.sqrt(np.sum(self.vel ** 2, 1))
            while (velMag > 5).any():
                self.vel[(velMag > 5)] *= 0.9
                velMag = np.sqrt(np.sum(self.vel ** 2, 1))
            
            self.v += Model.h * self.vel
            
            boolUnderground = self.v[:, 2] < 0
            self.v[boolUnderground, 2] = 0
            self.vel[boolUnderground, 2] *= -1
    
    
    def iter(self, gene, visualize=False):
        # edgeChannel : (ne, ) int, [0, self.numChannels)
        # contractionPercentLevel : (ne, ) int, [0, Model.contractionLevels)
        # script : (na, nc) int, {0, 1}

        self.loadGene(gene)
        self.reset()
        
        if not visualize:
            if testing:
                self.step(10)
            else:
                self.step(int(Model.numStepsAction * self.numActions * 4))
            
        else:
            viewer = o3.visualization.VisualizerWithKeyCallback()
            viewer.create_window()

            render_opt = viewer.get_render_option()
            render_opt.mesh_show_back_face = True
            render_opt.mesh_show_wireframe = True
            render_opt.point_size = 8
            render_opt.line_width = 10
            render_opt.light_on = True

            ls = LineSet(model.v, model.e)
            viewer.add_geometry(ls)
    
            def timerCallback(vis):
                if self.numSteps > int(Model.numStepsAction * self.numActions * 2):
                    return
                self.step(50)
                ls.points = vector3d(model.v)
                viewer.update_geometry(ls)
    
            viewer.register_animation_callback(timerCallback)
    
            # def key_step(vis):
            #     pass

            # viewer.register_key_callback(65, key_step)
    
            drawGround(viewer)
            viewer.run()
            viewer.destroy_window()

        return self.v
    
    def gene(self):
        edgeChannel = self.edgeChannel
        contractionPercentLevel = self.maxContraction * Model.contractionLevels
        script = self.script.reshape(-1)
        g = np.concatenate([edgeChannel, contractionPercentLevel, script])
        
        ubEdgeChannel = np.ones_like(edgeChannel, dtype=np.int64) * (self.numChannels - 1)
        ubContractionPercentLevel = np.ones_like(contractionPercentLevel, dtype=np.int64) * (Model.contractionLevels - 1)
        ubScript = np.ones_like(script, dtype=np.int64)
        ub = np.concatenate([ubEdgeChannel, ubContractionPercentLevel, ubScript])
        
        lb = np.zeros_like(ub, dtype=np.int64)
        
        return g, ub, lb
    
    def lb(self):
        return self.gene()[2]
    
    def ub(self):
        # caution: the upper bound here is inclusive
        return self.gene()[1]
    
    def loadGene(self, gene):
        lEdgeChannel = len(self.edgeChannel)
        lContractionPercentLevel = len(self.maxContraction)
        lScript = len(self.script.reshape(-1))
        
        self.edgeChannel = np.array(gene[:lEdgeChannel], dtype=np.int64)
        a = lEdgeChannel
        b = a + lContractionPercentLevel
        self.maxContraction = np.array(gene[a:b] * Model.contractionInterval)
        a = b
        b = a + lScript
        self.script = np.array(gene[a:b], dtype=bool).reshape(self.numChannels, self.numActions)

        self.edgeActive = np.ones_like(self.edgeActive, dtype=bool)
    
    def exportJSON(self, gene, name):
        with open(name) as ifile:
            content = ifile.read()
        data = json.loads(content)
        self.loadGene(gene)
        data['edgeChannel'] = self.edgeChannel.tolist()
        data['edgeActive'] = self.edgeActive.tolist()
        data['maxContraction'] = self.maxContraction.tolist()
        data['script'] = self.script.tolist()
        data['numChannels'] = self.numChannels
        data['numActions'] = self.numActions
        with open('{}/output/out.json'.format(rootPath), 'w') as ofile:
            js = json.dumps(data)
            ofile.write(js)
        
# class Optimizer




if __name__ == "__main__":
    
    inFileDir = '{}/data/{}.json'.format(rootPath, inFileName)
    
    model = Model()
    model.loadJson(inFileDir)
    model.scripting = scripting
    model.reset(resetScript=True, numChannels=3, numActions=3)

    def locomotion(gene, direction='x'):
        v = model.iter(gene)
        centroid = v.mean(0)
        if direction == 'x':
            return centroid[0] - abs(centroid[1]) * 10
        elif direction == 'y':
            return centroid[1] - abs(centroid[0]) * 10
    
    locomotion_x = lambda g : locomotion(g, 'x')
    locomotion_y = lambda g: locomotion(g, 'y')
    criterion = locomotion_x if direction == "x" else locomotion_x
    
    ea = EvolutionAlgorithm(model=model, criterion=criterion,
                            nPop=numPopulation)
    gene = ea.maximize(2 if testing else numGeneration)
    
    model.loadGene(gene)
    model.exportJSON(gene, inFileDir)
    
    if visualize:
        model.iter(gene, True)
    
    
    
    
        
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    

