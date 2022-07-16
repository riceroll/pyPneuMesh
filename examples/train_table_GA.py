from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

settingMOO = {
    'modelDir': './data/table.json',
    'modelSettingDir': './data/model_setting_large.json',
    'nActions': 4,
    'nLoopPerSim': 4,
    'objectives': [[objMoveForward, objFaceForward], [objTurnLeft], [objLowerBodyMax]],
}

nWorkers = multiprocessing.cpu_count()
nGensPerPool = int(nWorkers / 8 * 5)
settingGA = {
    'nGenesPerPool': nWorkers,
    'nGensPerPool': int(nWorkers / 8 * 5),
    'nSurvivedMax': int(nGensPerPool * 0.5),
    
    'nWorkers': nWorkers,
    # 'plot': True,
    'mute': False,
    'saveHistory': True,
}

ga = GeneticAlgorithm(MOOSetting=settingMOO, GASetting=settingGA)

if args.checkpoint:
    checkpointDir = args.checkpoint
    ga.loadCheckpoint(checkpointDir)


ga.run()


