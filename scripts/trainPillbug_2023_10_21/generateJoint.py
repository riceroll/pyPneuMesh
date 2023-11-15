from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.JointGenerator import JointGenerator
from pyPneuMesh.MOO import MOO


dataFolderDir = '/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/generateJoint_pillbug_2023_08_16/data/'
mooDict = readMooDict(dataFolderDir)
jointGeneratorParam = readNpy(dataFolderDir + 'pillbug.jointgeneratorparam.npy')

moo = MOO(mooDict=mooDict)
jg = JointGenerator(model=moo.model, jointGeneratorParam=jointGeneratorParam)

# Ps, E = jg.generateJoint(30)
# jg.animate(Ps, E, 0.01)

vtP, vE, vEIntersection, V = jg.generateJoints()
# jg.animate(vtP[0], vE[0], vEIntersection[0], 0.004)

jg.exportJoints(folderDir=dataFolderDir, name='pillbug', vtP=vtP, vE=vE, vEIntersection=vEIntersection, V=V)




