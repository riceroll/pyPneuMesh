from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
import pickle5


result = pickle5.load(open('./output/GA_531-8-36-53/iPool_580', 'rb'))

print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f} {:20f} {:20f} {:20f}'.format(score[0], score[1], score[2], score[3]))

moo = result['elitePool'][5]['moo']
moo.model.show()    # visualize the truss static shape and channels

actionSeq = moo.actionSeqs[1]   # control sequence of the second objective
moo.simulate(actionSeq, nLoops=2, visualize=True)   # visualize the trajectory of the control
