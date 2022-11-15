import argparse
from utils.mesh import Mesh
from utils.model import Model
from utils.truss import Truss
from utils.geometry import rigid_align, boundingBox, transform3d, translationMatrix, best_fit_transform, rotationMatrix
from utils.visualizer import showFrames
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

# mesh2 = Mesh('./data/bean_mesh.json')
# model = Model.load('./data/fullsphere.json')
model = Model()
model.load('./data/helmet.json')

mesh = Mesh('./data/eclipse_mesh.json', boundingBox(model.v))

print(model.v[9])
print(model.v[22])
print(model.v[23])
print(model.v[66])
print(model.v[67])
print(model.v[68])
# showFrames(frames, model.e, mesh=mesh, mesh_frames=m_frames)
