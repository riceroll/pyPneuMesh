# multi-objective optimization
from typing import Callable
import numpy as np
from utils.moo import MOO

def getCriterion(mmo: MOO) -> Callable[[np.ndarray], np.ndarray]:
    def criterion(gene) -> np.ndarray:
        moo = gene
        rating = []
        for i in range(len(moo.actionSeqs)):
            vs, es = moo.simulateOpenLoop(i)
            
            objective = moo.setting.objectives[i]
            for subObjective in objective:
                score = subObjective(vs, es)
                rating.append(score)
                
        rating = np.array(rating)
        return rating
    
    return criterion




