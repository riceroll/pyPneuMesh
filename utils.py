import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
from optimizer import EvolutionAlgorithm
rootPath = os.path.split(os.path.realpath(__file__))[0]
tPrev = time.time()

def test():
    return 2