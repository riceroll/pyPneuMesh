import sys
import json
import numpy as np

def readActionSeq(iDir):
    with open(iDir) as iFile:
        js = iFile.read()
        data = json.loads(js)
        actionSeq = np.array(data['script'], dtype=int)
    return actionSeq

def readData(iDir):
    with open(iDir) as iFile:
        js = iFile.read()
        data = json.loads(js)
    return data


outDir = sys.argv[1]
fDir = outDir+"_0"
lDir = outDir+"_1"
rDir = outDir+"_2"

data = readData(fDir)

actionSeqF = readActionSeq(fDir)
actionSeqL = readActionSeq(lDir)
actionSeqR = readActionSeq(rDir)

actionSeq = np.hstack([actionSeqF, actionSeqF, actionSeqL, actionSeqL, actionSeqF, actionSeqF, actionSeqR, actionSeqR, actionSeqF, actionSeqF])

data['script'] = actionSeq.tolist()

with open(outDir, 'w') as oFile:
    js = json.dumps(data)
    oFile.write(js)

