import igl
import numpy as np
import json
import copy
from utils.geometry import boundingBox, bboxDiagonal, center, translationMatrix, scaleMatrix, transform3d
from utils.truss import Truss


class Mesh(object):

    def __init__(self, meshDir):
        # ======= variable =======
        """
               load parameters from a json file into the mesh
               :param meshDir: dir of the json file
               :return: data as a dictionary
        """
        self.meshDir = meshDir
        with open(meshDir) as ifile:
            content = ifile.read()
        data = json.loads(content)

        self.surface = np.array(data['f'])
        self.v = np.array(data['v'])
        self.bv = boundingBox(self.v)
        self.keyPoints = np.array(data['keyPoints'])
        # may not need to define keyPoints I guess ?

    def affine(self, truss: Truss):
        mesh_c = center(self.bv)
        truss_c = center(truss.bv)

        dx, dy, dz = truss_c - mesh_c

        scale = bboxDiagonal(truss.bv) / bboxDiagonal(self.bv)

        self.v = transform3d(self.v, [translationMatrix(dx, dy, dz), scaleMatrix(scale, scale, scale)])
