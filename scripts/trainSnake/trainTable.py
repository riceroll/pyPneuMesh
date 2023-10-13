# used to convert previous json file to npy file

from pyPneuMesh.utils import json2Data, data2Npy, readNpy
from pyPneuMesh.Model import Model

folderDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainSnake/data/"

names = [
    'snake'
]

for name in names:
    data = json2Data(folderDir+name + '.json')
    data2Npy(data, name=name, npyFolderDir='/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainSnake/data')


trussparam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainSnake/data/snake.trussparam.npy')
# simparam = readNpy('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainSnake/data/snake.simparam.npy')

# model = Model(trussparam, simparam)
#
# model.saveRendering('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/tripleTet0/data', 'tripleTet0')
#
# import numpy as np
#
#
# lengthMean = np.sqrt(((model.v0[model.e[:, 0]] - model.v0[model.e[:, 1]]) ** 2).sum(1)).mean()
#
# print('average length: {}'.format(lengthMean))
#
