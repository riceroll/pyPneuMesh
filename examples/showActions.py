import sys
import numpy as np
import json
from utils.model import Model
from utils.visualizer import visualizeActions

if len(sys.argv) > 1:
    modelDir = sys.argv[1]
else:
    modelDir = './output/_GA_72-4-14-36_pillbugnodir/g2495_0.76,1.00/3.63,0.98'

model = Model()
model.load(modelDir)
with open(modelDir) as iFile:
    data = json.load(iFile)
    actionSeq = np.array(data['script'])

actionSeq = np.hstack([actionSeq, actionSeq])

vs = visualizeActions(model, actionSeq, nLoop=1)