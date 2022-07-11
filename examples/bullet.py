import pybullet as p
import time
import math

import pickle5
from utils.visualizer import showFrames



result = pickle5.load(open('./output/GA_531-8-36-53/iPool_580', 'rb'))

print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f} {:20f} {:20f} {:20f}'.format(score[0], score[1], score[2], score[3]))

moo = result['elitePool'][5]['moo']
model = moo.model




p.connect(p.GUI)

p.loadURDF("./data/plane.urdf")

scale = 2
model.v *= scale
cubes = [p.loadURDF("./data/cube_small.urdf", v[0], v[1], v[2]) for v in model.v]


constrs = []

for e in model.e:
    vec = model.v[e[1]] - model.v[e[0]]
    constr = p.createConstraint(cubes[e[0]], -1, cubes[e[1]], -1, p.JOINT_POINT2POINT, [0, 0, 0], [vec[0], vec[1], vec[2]], [0,0,0])
    constrs.append(constr)



# cubeId = p.loadURDF("cube_small.urdf", 0, 0, 1)
# cubeId2 = p.loadURDF("cube_small.urdf", 0, 0, 5)
# cubeId3 = p.loadURDF("cube_small.urdf", 0, 0, 4)


p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)
# cid = p.createConstraint(cubeId, -1, cubeId2, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 1, 1], [0, 0, 0])
# cid2 = p.createConstraint(cubeId, -1, cubeId3, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, -1, 1], [0, 0, 0])
# cid3 = p.createConstraint(cubeId2, -1, cubeId3, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, -2, 0], [0, 0, 0])


# print(cid)

print(p.getConstraintUniqueId(0))

prev = [0, 0, 1]
a = -math.pi
# while 1:
#   a = a + 0.01
#   if (a > math.pi):
#     a = -math.pi
#   time.sleep(.01)
#   p.setGravity(0, 0, -10)
#   pivot = [a, 0, 1]
#   orn = p.getQuaternionFromEuler([a, 0, 0])
  # p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)

# p.removeConstraint(cid)

#
# def showFrames(frames, es):
#     try:
#         o3, vector3d, vector3i, vector2i, LineSet, PointCloud, drawGround = _importVisualizer()
#         viewer = _create_window(o3)
#     except Exception as e:
#         print(e)
#         return
#
#     ls = LineSet(frames[0], es)
#     viewer.add_geometry(ls)
#
#     global iFrame
#     iFrame = 0
#
#     def callback(vis):
#         global iFrame
#
#         if iFrame >= len(frames):
#             vis.close()
#             vis.destroy_window()
#             return
#
#         ls.points = vector3d(frames[iFrame])
#         viewer.update_geometry(ls)
#
#         iFrame += 10
#
#     viewer.register_animation_callback(callback)
#
#     drawGround(viewer)
#     viewer.run()