import argparse
from utils.mesh import Mesh
from utils.truss import Truss

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

mesh = Mesh('./data/sphere_mesh.json')

truss = Truss('./data/fullsphere.json')

mesh.affine(truss)
