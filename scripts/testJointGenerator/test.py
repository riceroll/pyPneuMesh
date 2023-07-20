from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.JointGenerator import JointGenerator
from pyPneuMesh.MOO import MOO

mooDict = readMooDict('scripts/testJointGenerator/data/')
jointGeneratorParam = readNpy('scripts/testJointGenerator/data/table.jointgeneratorparam.npy')

moo = MOO(mooDict=mooDict)
jg = JointGenerator(model=moo.model, jointGeneratorParam=jointGeneratorParam)

# Ps, E = jg.generateJoint(30)
# jg.animate(Ps, E, 0.01)

vtP, vE, vEIntersection, V = jg.generateJoints()
# jg.animate(vtP[0], vE[0], vEIntersection[0], 0.004)

jg.exportJoints(folderDir='scripts/testJointGenerator/data', name='table', vtP=vtP, vE=vE, vEIntersection=vEIntersection, V=V)




