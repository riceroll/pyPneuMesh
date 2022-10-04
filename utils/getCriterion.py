# multi-objective optimization
from typing import Callable
import numpy as np
from utils.moo import MOO


def getCriterion(mmo: MOO) -> Callable[[np.ndarray], np.ndarray]:
    def criterion(gene) -> np.ndarray:

        mmo = gene
        rating = []
        for i in range(len(mmo.actionSeqs)):
            actionSeq = mmo.actionSeqs[i]

            vs, es = mmo.simulate(actionSeq)

            objective = mmo.objectives[i]
            indices = mmo.keyPointsIndices  # indices specifying the key points in list of vertices
            targetMesh = mmo.targetMeshes[i]  # face definition of target Mesh
            for subObjective in objective:
                score = subObjective(vs, es, indices, targetMesh)
                print(score)
                rating.append(score)

        rating = np.array(rating)
        return rating

    return criterion
