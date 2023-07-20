# convert npy to
# {
#     'objectives': original
#     'subobjectves': 1d list of string
#     'unit': 1d list of string
#     'scores': iteration x genes x numSubObjectives
# }

import numpy as np
import os
import tqdm
def updateNpy(npyDir):
    data = np.load(npyDir, allow_pickle=True).all()
    objectives = data['objectives']
    subObjectives = []
    while True:
        if len(objectives) == 0:
            break
        
        subs = objectives.pop()
        while True:
            if len(subs) == 0:
                break
            subObjectives.append(subs.pop())
    
    scores = data['scores']     # mean and max scores, iIter: (mean_score0 / max_score0 / mean_score1 / max_score1 ...)
    if isinstance(scores, dict):    # iIter:
        print('is dict')
        indices = sorted(scores.keys())
        values = np.array([scores[i] for i in indices])
        iMax = max(indices)
        
        fullValues = np.zeros([iMax + 1, values.shape[1]])
        i0 = 0
        i1 = indices.pop(0)
        
        averaging = False
        if values.ndim == 3:
            averaging = True
        
        if not averaging:
            while True:
                # if averaging:
                #     score = scores[i1].mean()
                fullValues[i0] = scores[i1]
                i0 += 1
                
                if i0 > i1:
                    if len(indices) == 0:
                        break
                    else:
                        i1 = indices.pop(0)
        
        scores = fullValues
    
    data['scores'] = scores
    data['subObjectives'] = subObjectives
    return data
    
folderDir = '/Users/Roll/Desktop/MetaTruss-Submission/Plots/dataPlotting/'

data = {}
for name in tqdm.tqdm(list(os.listdir(folderDir))):
    if name[-3:] == 'npy':
        print(name)
        data[name] = updateNpy(folderDir + name)
    

