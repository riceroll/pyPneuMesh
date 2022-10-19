import igl
import numpy as np
import json
from utils.geometry import boundingBox


class Truss(object):

    def __init__(self, vs, indices):
        # ======= variable =======
        """
               load parameters from a json file into the mesh
               :param meshDir: dir of the json file
               :return: data as a dictionary
        """
        self.indices = indices
        self.vs = vs
        # print(vs.shape)
        self.bv = boundingBox(self.vs[-1])
        # last iteration boundingBox

    # def __init__(self, trussDir):
    #     with open(trussDir) as ifile:
    #         content = ifile.read()
    #     data = json.loads(content)
    #
    #     self.v = np.array(data['v'])
    #     self.bv = boundingBox(self.v)
