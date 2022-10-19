import pickle5

result = pickle5.load(open('./output/GA_1012-18:58:5/iPool_39', 'rb'))

print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f}'.format(score[0]))

moo = result['elitePool'][0]['moo']
# moo.model.show()  # visualize the truss static shape and channels

actionSeq = moo.actionSeqs[0]  # control sequence of the second objective
moo.simulate(actionSeq, nLoops=2, visualize=True)  # visualize the trajectory of the control
