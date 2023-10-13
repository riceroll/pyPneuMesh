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
    

def getDistanceSingleSource(E, iSource):
    distance = np.ones(np.max(E) + 1) * -1
    distance[iSource] = 0
    
    adjacencyList = {}
    for e in E:
        iv0 = e[0]
        iv1 = e[1]
        if iv0 not in adjacencyList:
            adjacencyList[iv0] = []
        
        if iv1 not in adjacencyList:
            adjacencyList[iv1] = []
        
        adjacencyList[iv0].append(iv1)
        adjacencyList[iv1].append(iv0)
    
    queue = [iSource]
    while True:
        if len(queue) == 0:
            break
        
        iv = queue.pop(0)
        depth = distance[iv]
        
        neighbors = adjacencyList[iv]
        neighbors = [n for n in neighbors if distance[n] == -1]
        
        for n in neighbors:
            distance[n] = depth +1
            queue.append(n)
    
    return distance

def getDistance(E, iSources):
    distance = np.vstack([getDistanceSingleSource(E, iSource) for iSource in iSources])
    distance = distance.min(0)
    return distance

def getWeightFromDistance(distance, xMiddle=2.3, scale=2):
    # xMiddle, weight = 0.5 when it's xMiddle
    # scale, the larger the value the steeper the curve
    return np.arctan((distance - xMiddle) * scale) / np.pi + 0.5

def plotDistance(distance, V, E):
    # plot distance for helmet weight of joints
    
    import polyscope as ps
    try:
        ps.init()
    except:
        pass
    
    colors = np.array([[d/max(distance), 0, 0] for d in distance])
    
    
    ps.register_curve_network('curve', V, E)
    pc = ps.register_point_cloud('pc', V)
    pc.add_color_quantity('color', colors)
    ps.show()
        
    
    

    
    
    
        
        
    

    
    
    
    
    
    
    
    
    
    
    
        
        

