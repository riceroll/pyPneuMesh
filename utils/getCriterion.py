# multi-objective optimization
from typing import Callable
import numpy as np
from utils.moo import MOO
from utils.objectives.locomotion import Locomotion
from utils.truss import Truss


def getCriterion(mmo: MOO) -> Callable[[np.ndarray], np.ndarray]:
    def criterion(gene) -> np.ndarray:

        mmo = gene
        rating = []
        for i in range(len(mmo.actionSeqs)):
            actionSeq = mmo.actionSeqs[i]

            # TODO FIX THIS LATER REMOVE TARGETMESH
            vs, es = mmo.simulate(actionSeq)
            objective = mmo.objectives[i]
            indices = mmo.keyPointsIndices  # indices specifying the key points in list of vertices
            truss = Truss(vs, indices)
            for subObjective in objective:
                if issubclass(subObjective, Locomotion):
                    obj = subObjective(truss)
                else:
                    targetMesh = mmo.targetMeshes[i]  # face definition of target Mesh
                    obj = subObjective(truss, targetMesh)
                score = obj.execute()
                rating.append(score)

        rating = np.array(rating)
        return rating

    return criterion
