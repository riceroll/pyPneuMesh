import pickle5

from utils.mesh import Mesh
from utils.truss import Truss
from utils.geometry import boundingBox, rigid_align

result = pickle5.load(open('./output/1109_bean/iPool_91', 'rb'))

print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f}'.format(score[0]))

moo = result['elitePool'][0]['moo']
# moo.model.show()  # visualize the truss static shape and channels

actionSeq0 = moo.actionSeqs[0]  # control sequence of the second objective
# actionSeq1 = moo.actionSeqs[1]
# assert (actionSeq0.all() == actionSeq1.all())

# test it out ...
# mesh = Mesh('./data/eclipse_mesh.json', boundingBox(moo.model.v))
mesh2 = Mesh('./data/bean_mesh.json', boundingBox(moo.model.v))

moo.simulate(actionSeq0, nLoops=2, visualize=True, mesh=None)  # visualize the trajectory of the control
# moo.simulate(actionSeq1, nLoops=1, visualize=True, mesh=mesh)  # visualize the trajectory of the control
# moo.simulate(actionSeq0, nLoops=1, visualize=True, mesh=mesh2)
