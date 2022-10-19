import numpy as np
import json

from utils.model import Model


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
