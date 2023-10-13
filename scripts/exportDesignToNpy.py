import numpy as np

from pyPneuMesh.utils import readMooDict
from pyPneuMesh.MOO import MOO

mooDict = readMooDict('scripts/trainTable_2022-12-09_5-51/data/')
moo = MOO(mooDict=mooDict)


data = {
    'contractionLevel': moo.model.contractionLevel,
    'edgeChannel': moo.model.edgeChannel
}

np.save('/Users/Roll/Desktop/dataPlotting/table.design.npy', data)


    
