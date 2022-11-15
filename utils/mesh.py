import igl
import numpy as np
import json
import copy
import math
from utils.geometry import boundingBox, bboxDiagonal, center, translationMatrix, scaleMatrix, transform3d, rigid_align, \
    best_fit_transform
from utils.truss import Truss


class Mesh(object):

    def __init__(self, meshDir, truss_bv):
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
        self.affine(truss_bv=truss_bv, init=True)

    # Longest side of mesh should align with the smallest side of truss in bounding box
    # hardcoded based on the bv structure
    def calc_scale(self, truss_bv, mesh_bv, alpha=1.0):
        truss_x = abs(truss_bv[0, 0] - truss_bv[4, 0])
        truss_y = abs(truss_bv[0, 1] - truss_bv[2, 1])
        truss_z = abs(truss_bv[0, 2] - truss_bv[1, 2])
        truss_min = np.min([truss_x, truss_y, truss_z])

        mesh_x = abs(mesh_bv[0, 0] - mesh_bv[4, 0])
        mesh_y = abs(mesh_bv[0, 1] - mesh_bv[2, 1])
        mesh_z = abs(mesh_bv[0, 2] - mesh_bv[1, 2])
        mesh_max = np.max([mesh_x, mesh_y, mesh_z])

        return (alpha * truss_min) / mesh_max

    def affine(self, truss_bv, init=False):
        scale = 1
        if init:
            scale = self.calc_scale(truss_bv=truss_bv, mesh_bv=self.bv, alpha=0.9)
        self.v = transform3d(self.v, [scaleMatrix(scale, scale, scale)])
        self.keyPoints = transform3d(self.keyPoints, [scaleMatrix(scale, scale, scale)])
        new_bv_mesh = boundingBox(self.v)

        # test applying only rigid for rotation

        mesh_c = center(new_bv_mesh)
        truss_c = center(truss_bv)
        dx, dy, dz = truss_c - mesh_c
        self.v = transform3d(self.v, [translationMatrix(dx, dy, dz)])
        self.keyPoints = transform3d(self.keyPoints, [scaleMatrix(scale, scale, scale)])

    def rigid_affine(self, v_prev, v):
        T, R, t = best_fit_transform(v_prev, v)

        self.v = transform3d(self.v, [T])
        self.keyPoints = transform3d(self.keyPoints, [T])
