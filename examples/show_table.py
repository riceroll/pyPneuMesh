from utils.moo import MOO
import pickle5

result = pickle5.load(open('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/output/GA_1117-12:12:23/iPool_3', 'rb'))

for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    # print(i, ' {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(score[0], score[1], score[2], score[3], score[4]))

import numpy as np
scores = np.array([gene['score'] for gene in result['elitePool']])

for i in range(len(scores)):
    score = scores[i]
    if score[0] > 0.1:
        if score[1] > 0.5:
            if score[2] > 0.5:
                if score[3] > -0.7:
                    if score[4] > -0.2:
                        print(i, ' {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(score[0], score[1], score[2], score[3], score[4]))

moo = result['elitePool'][17]['moo']

















