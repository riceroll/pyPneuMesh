# used to convert previous json file to npy file

from pyPneuMesh.utils import json2Data, data2Npy
import numpy as np


trussParamPillbug = np.load('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/pillbug.trussparam.npy', allow_pickle=True).all()

data = json2Data('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/'+ 'tentacle' + '.json')

# scale v0
data['v0'] = data['v0'] / 1.2 * (trussParamPillbug['MAX_ACTIVE_BEAM_LENGTH'] - trussParamPillbug['CONTRACTION_PER_LEVEL'] * (trussParamPillbug['NUM_CONTRACTION_LEVEL'] - 1))

data2Npy(data,npyFolderDir='/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTentacle/data/', name='tentacle',
            CONTRACTION_SPEED=trussParamPillbug['CONTRACTION_SPEED'],
            NUM_CONTRACTION_LEVEL=trussParamPillbug['NUM_CONTRACTION_LEVEL'],
            CONTRACTION_PER_LEVEL=trussParamPillbug['CONTRACTION_PER_LEVEL'],
            MAX_ACTIVE_BEAM_LENGTH=trussParamPillbug['MAX_ACTIVE_BEAM_LENGTH'],
            ACTION_TIME=trussParamPillbug['ACTION_TIME']
         )




