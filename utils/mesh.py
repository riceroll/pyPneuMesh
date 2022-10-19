import numpy as np
import json
import copy

from utils.model import Model


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
        self.keyPoints = np.array(data['keyPoints'])
        self.v = np.array(data['v'])

    def mapScale(self, model: Model, indices: np.ndarray):
        mesh_v = self.v
        truss_v = model.v[indices]
        # assuming scale at the center
        scale = np.mean(mesh_v / truss_v)
        return scale
