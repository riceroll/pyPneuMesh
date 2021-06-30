import numpy as np
from utils.geometry import getFrontDirection, getClosestVector

def cstrIntersection(vs: np.ndarray, es: np.ndarray):
    # TODO: not tested
    TInterval = 1
    
    vecs = [None] * len(es) ** 2
    iv = 0
    while iv < len(vs):
        v = vs[iv]
        ivec = 0
        for ie0, e0 in enumerate(es):
            for ie1, e1 in enumerate(es):
                a = v[e0[0]]
                b = v[e0[1]]
                c = v[e1[0]]
                d = v[e1[1]]
                ab = b - a
                cd = d - c
                lab = np.linalg.norm(ab)
                lcd = np.linalg.norm(cd)
                
                if ie0 == ie1:
                    vecs[ivec] = None
                elif np.linalg.norm((a + b) / 2 - (c + d) / 2) > (lab + lcd) / 2:
                    # distance too far, don't consider
                    vecs[ivec] = None
                else:
                    if np.linalg.norm(a - c) < 5e-2 and e0[0] != e1[0]:
                        print('node overlap')
                        return False
                    
                    if np.linalg.norm(a - d) < 5e-2 and e0[0] != e1[1]:
                        print('node overlap')
                        return False
                    
                    if np.linalg.norm(b - c) < 5e-2 and e0[1] != e1[0]:
                        print('node overlap')
                        return False

                    if np.linalg.norm(b - d) < 5e-2 and e0[1] != e1[1]:
                        print('node overlap')
                        return False

                    if e0[0] in e1 or e0[1] in e1 or e1[0] in e0 or e1[1] in e0:    # connected edges
                        if abs(abs((ab / lab ).dot(cd / lcd)) - 1) < 1e-3:
                            return False
                    else:
                        ret = getClosestVector(a, b, c, d)
                        prev = vecs[ivec]
                        if ret is None or prev is None:
                            pass
                        else:
                            if ret.dot(prev) < 0:
                                print('intersect')
                                assert False
                                return False
                        vecs[ivec] = ret
                ivec += 1
        iv += TInterval
        print(iv)
    return True

def testCstrIntersection(argv):
    from utils.mmo import MMO
    from model import Model
    from utils.objectives import objMoveForward, objFaceForward, objTurnLeft
    from utils.mmoCriterion import getCriterion

    setting = {
        'modelDir': './test/data/pillBugIn_intersection.json',
        'numChannels': 4,
        'numActions': 4,
        'numObjectives': 2,
        "channelMirrorMap": {
            0: 1,
            1: 0,
            2: -1,
            3: -1,
        },
        'objectives': [[objMoveForward, objFaceForward], [objTurnLeft]]
    }
    mmo = MMO(setting=setting)
    assert(mmo.actionSeqs.shape == (1, mmo.numChannels, mmo.numActions))
    vs, es = mmo.simulate(actionSeq=mmo.actionSeqs[0])
    print('start')
    ret = cstrIntersection(vs, es)
    print(ret)

    

tests = {
    # 'testTurn': testTurn,
}

if __name__ == "__main__":
    import sys
    # testTurn(sys.argv)
    testCstrIntersection(sys.argv)
