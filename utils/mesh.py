import numpy as np
import json
import copy


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
