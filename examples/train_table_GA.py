from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

MOOmoo = {
    'modelDir': './data/table.json',
    'numChannels': 6,
    'numActions': 4,
    'numObjectives': 3,
    "channelMirrorMap": {
        0: 1,
        2: -1,
        3: -1,
        4: 5,
    },
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]],
    'nLoopSimulate': 4
}

nWorkers = multiprocessing.cpu_count()
nGensPerPool = int(nWorkers / 8 * 5)
mooGA = {
    'nGenesPerPool': nWorkers,
    'nGensPerPool': int(nWorkers / 8 * 5),
    'nSurvivedMax': int(nGensPerPool * 0.5),
    
    'nWorkers': nWorkers,
    'plot': True,
    'mute': False,
    'saveHistory': True,
}


ga = GeneticAlgorithm(MOOSetting=MOOSetting, GASetting=settingGA)

if args.checkpoint:
    checkpointDir = args.checkpoint
    ga.loadCheckpoint(checkpointDir)


ga.run()


