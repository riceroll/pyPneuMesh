# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

from utils.objectives.locomotion import MoveForward, FaceForward, TurnLeft, LowerBodyMax

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

MOOsetting = {
    'modelDir': './data/half_helmet.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: -1,
        1: -1,
        2: -1,
        3: -1
    },
    # 'env': ['flat', 'cube', 'cave'],
    'objectives': [[MoveForward, FaceForward], [TurnLeft], [LowerBodyMax]],
    'nLoopSimulate': 4
}

nWorkers = multiprocessing.cpu_count()
nGensPerPool = int(nWorkers / 8 * 5)
settingGA = {
    'nGenesPerPool': nWorkers,
    'nGensPerPool': int(nWorkers / 8 * 5),
    'nSurvivedMax': int(nGensPerPool * 0.5),

    'nWorkers': nWorkers,
    'plot': True,
    'mute': False,
    'saveHistory': True,
}

ga = GeneticAlgorithm(MOOSetting=MOOsetting, GASetting=settingGA)

if args.checkpoint:
    checkpointDir = args.checkpoint
    ga.loadCheckpoint(checkpointDir)

ga.run()
