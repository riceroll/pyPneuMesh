import numpy as np
import json

def exportAnimation(frames, es, name=None):
    # use with showActions.py
    frames = np.array(frames).tolist()
    es = np.array(es, dtype=int).tolist()
    data = {
        'frames': frames,
        'es': es
    }
    js = json.dumps(data)
    
    if name:
        name = name
    else:
        name = 'output'
        
    outputDir = './output/{}.anime'.format(name)
    with open(outputDir, 'w') as oFile:
        oFile.write(js)


