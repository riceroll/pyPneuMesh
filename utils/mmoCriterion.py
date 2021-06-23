# multi-objective optimization
from typing import Callable
import numpy as np
from utils.modelInterface import getModel, encodeGeneSpace, decodeGene, simulate
from utils.mmoSetting import MMOSetting

def getCriterion(mmoSetting: MMOSetting) -> Callable[[np.ndarray], np.ndarray]:
    def criterion(gene: np.ndarray) -> np.ndarray:
        ms = mmoSetting
        model, actionSeqs = decodeGene(mmoSetting=mmoSetting, gene=gene)
        assert(len(ms.objectives) == len(actionSeqs))
        
        rating = []
        for i in range(len(actionSeqs)):
            actionSeq = actionSeqs[i]
            model, actionSeqs = decodeGene(mmoSetting=mmoSetting, gene=gene)
            vs = simulate(model, actionSeq)
            
            objective = ms.objectives[i]
            for subObjective in objective:
                score = subObjective(vs)
                rating.append(score)
                
        rating = np.array(rating)
        return rating
    
    return criterion




