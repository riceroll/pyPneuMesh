import numpy as np

def readNpy(npyDir):
    return np.load(npyDir, allow_pickle=True).item()

def readMooDict(folderDir):
    import os
    root = os.curdir
    mooDict = dict()
    for filename in os.listdir(folderDir):
        fileDir = os.path.join(folderDir, filename)
        if 'trussparam' in fileDir:
            mooDict['trussParam'] = readNpy(fileDir)

        if 'simparam' in fileDir:
            mooDict['simParam'] = readNpy(fileDir)

        if 'actionseqs' in fileDir:
            mooDict['actionSeqs'] = readNpy(fileDir)

        if 'objectives' in fileDir:
            mooDict['objectives'] = readNpy(fileDir)

        if 'graphsetting' in fileDir:
            mooDict['graphSetting'] = readNpy(fileDir)
    
    assert(len(mooDict) == 5)
    return mooDict


def getDefaultValue(dictionary, key, defaultValue):
    if key in dictionary:
        return dictionary[key]
    else:
        return defaultValue

def getLength(v, e):
    assert(v.ndim == 2 and v.shape[1] == 3)
    assert(e.ndim == 2 and e.shape[1] == 2)
    
    return np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)


def json2Data(jsonDir):
    import json
    with open(jsonDir) as iFile:
        content = iFile.read()
        js = json.loads(content)
        data = dict()
        data['v0'] = np.array(js['v'], dtype=np.float64)
        data['e'] = np.array(js['e'], dtype=int)

        if 'edgeChannel' in js:
            data['edgeChannel'] = np.array(js['edgeChannel'], dtype=int)
        else:
            data['edgeChannel'] = np.zeros(len(data['e']), dtype=int)

        if 'edgeActive' in js:
            data['edgeActive'] = np.array(js['edgeActive'], dtype=bool)
        else:
            data['edgeActive'] = np.ones(len(data['e']), dtype=bool)
        
        if 'fixedVs' in js:
            data['vertexFixed'] = np.array(js['fixedVs'], dtype=bool)
        else:
            data['vertexFixed'] = np.zeros(len(data['v0']), dtype=bool)
        
        if 'maxContraction' in js:
            data['maxContraction'] = np.array(js['maxContraction'], dtype=np.float64)
        data['contractionLevel'] = np.zeros(len('e'), dtype=int)
        
        if 'script' in js:
            data['actionSeqs'] = {
                0: np.array(js['script'], dtype=int).T
            }
        else:
            data['actionSeqs'] = {}
        
        data['l'] = getLength(data['v0'], data['e'])
    
    return data

def data2Npy(data, name, npyFolderDir='/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/data/',
             CONTRACTION_SPEED=0.21,
             NUM_CONTRACTION_LEVEL=8,
             CONTRACTION_PER_LEVEL=0.088,
             MAX_ACTIVE_BEAM_LENGTH=1.75864,
             ACTION_TIME=3.0
             ):
    import pathlib
    
    trussParam = dict()
    trussParam['v0'] = data['v0']
    trussParam['e'] = data['e']
    trussParam['edgeChannel'] = data['edgeChannel']
    trussParam['edgeActive'] = data['edgeActive']
    trussParam['vertexFixed'] = data['vertexFixed']
    trussParam['CONTRACTION_SPEED'] = CONTRACTION_SPEED
    trussParam['NUM_CONTRACTION_LEVEL'] = NUM_CONTRACTION_LEVEL
    trussParam['CONTRACTION_PER_LEVEL'] = CONTRACTION_PER_LEVEL
    trussParam['MAX_ACTIVE_BEAM_LENGTH'] = MAX_ACTIVE_BEAM_LENGTH
    trussParam['ACTION_TIME'] = ACTION_TIME

    trussParam['contractionLevel'] = np.array((data['maxContraction'] / 0.05).round(), dtype=int)
    
    actionSeqs = data['actionSeqs']
    
    folderPath = pathlib.Path(npyFolderDir)
    trussParamPath = folderPath.joinpath('{}.trussparam'.format(name))
    actionSeqsPath = folderPath.joinpath('{}.actionseqs'.format(name))
    
    np.save(str(trussParamPath), trussParam)
    np.save(str(actionSeqsPath), actionSeqs)
    
    
    
    
    
    
    
        
        
    

    
    
    
    
    
    
    
    
    
    
    
        
        

