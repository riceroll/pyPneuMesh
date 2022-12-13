# used to convert previous json file to npy file

from pyPneuMesh.utils import json2Data, data2Npy

folderDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/data/"

names = [
    'Lobster_0',
    'Lobster_1',
    'lobster_forward',
    'lobster_grab',
    'lobster_grab2',
    'lobster_grabgo',
    'lobster_walk',
    'table1',
    'table2',
    'table3',
    'table4',
    'table5',
    'table6'
]

for name in names:
    data = json2Data(folderDir+name + '.json')
    data2Npy(data, name=name)


