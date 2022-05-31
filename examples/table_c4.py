from utils.mooCriterion import getCriterion
from utils.moo import MOO
from utils.objectives import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.GA import GeneticAlgorithm

setting = {
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

# mmo = MOO(setting)

ga = GeneticAlgorithm(MOOSetting=setting)

import multiprocessing
nWorkers = multiprocessing.cpu_count()

settingGA = ga.getDefaultSetting()
settingGA['nGenesPerPool'] = nWorkers
settingGA['nGensPerPool'] = int(nWorkers / 8 * 5)
settingGA['nSurvivedMax'] = int(settingGA['nGenesPerPool'] * 0.5)
settingGA['nWorkers'] = nWorkers
ga.loadSetting(settingGA)
ga.run()


