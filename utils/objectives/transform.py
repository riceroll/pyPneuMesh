import numpy as np
import igl as igl
from utils.mesh import Mesh

from utils.objectives.objective import Objective
from utils.truss import Truss
from utils.geometry import boundingBox


class Transform(Objective):

    def __init__(self, truss: Truss, mesh: Mesh):
        self.truss = truss
        self.mesh = mesh

    def execute(self):
        pass


class KeyPointsAlign(Transform):

    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        mesh_keypoints: np.ndarray = self.mesh.keyPoints

        v_keypoints = self.truss.vs[-1, self.truss.indices]

        assert (v_keypoints.shape == mesh_keypoints.shape)
        return -np.sqrt(((v_keypoints - mesh_keypoints) ** 2).sum(1)).mean()


class SurfaceAlign(Transform):
    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        self.mesh.affine(boundingBox(self.truss.vs[-1]))
        # apply affine transformation to the mesh to have equal ratio with the truss
        surface: np.ndarray = self.mesh.surface
        v_mesh: np.ndarray = self.mesh.v
        v_points = self.truss.vs[-1]
        dis, _, _ = igl.point_mesh_squared_distance(v_points, v_mesh, surface)
        return - (sum(dis) / len(dis))
