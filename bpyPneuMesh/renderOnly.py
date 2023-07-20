import numpy as np
import bpy
import sys
sys.path.append('/Users/Roll/opt/anaconda3/lib/python3.7/site-packages/')
from skimage import exposure

import matplotlib.cm as cm
import matplotlib.colors as colors
scalar_mappable = cm.ScalarMappable(cmap='inferno')
number_to_color = scalar_mappable.get_cmap()


#
# frames = np.zeros([200, 50, 3])
# e = np.ones([30, 2])
# edgeChannel = np.ones(30)


# animation = {
#     'Vs': frames,
#     'E': e,
#     'edgeChannel': edgeChannel,
#     'h': 0.01,
# }
showForce = False
animation = np.load("/Users/Roll/Desktop/rendered/testing.animation.npy", allow_pickle=True).item()
frameInterval = 500

Vs = animation['Vs']
Fs = animation['Fs']
E = animation['E']
edgeChannel = animation['edgeChannel']

FsOriginal = Fs.copy()
Fs = np.array(exposure.equalize_hist(Fs.reshape(-1))).reshape(Fs.shape)



colors = [
    (0.1, 0.5, 0.8),
    (0.5, 0.1, 0.8),
    (0.9, 0.1, 0.4),
    (0.1, 0.8, 0.5),
    (0.5, 0.8, 0.2),
    (0.8, 0.5, 0.1)
]

radius = 0.015
bevel_resolution = 12
fps = 12
h = animation['h']
numFrames = len(Vs)
T = h * (numFrames - 1)
interval = 1.0 / 12
numFramesSampled = int(T / interval + 1)
# numFrames = 50



def newMaterial(id):
    mat = bpy.data.materials.new(name=id)
    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat


def newShader(id, color):
    r, g, b = color
    mat = newMaterial(id)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    shader = nodes.new(type='ShaderNodeBsdfDiffuse')
    nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)
    links.new(shader.outputs[0], output.inputs[0])
    return mat


def addCurveObject(color, props):
    curve = bpy.data.curves.new('edge', type='CURVE')
    obj = bpy.data.objects.new('edge', curve)
    bpy.context.collection.objects.link(obj)
    
    for key in props:
        obj.data[key] = props[key]
    
    line = curve.splines.new('POLY')
    line.points.add(1)
    curve.bevel_depth = radius
    curve.bevel_resolution = bevel_resolution
    curve.dimensions = '3D'
    
    mat = newShader('color', color)
    obj.data.materials.append(mat)
    
    return obj

def addSphereObject(color):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, enter_editmode=False, align='WORLD', location=(0, 0, 0),
                                          subdivisions=3)
    sphereObject = bpy.context.active_object
    mat = newShader('color', color)
    sphereObject.data.materials.append(mat)
    
    return sphereObject


# print(numFrames)
V0 = Vs[0]
curveObjects = []

ivsAdded = set()

for ie, e in enumerate(E):
    print(ie / len(E))
    
    color = colors[edgeChannel[ie]]
    
    curveObject = addCurveObject(color, {'id': ie})
    
    addSphere0 = e[0] not in ivsAdded
    addSphere1 = e[1] not in ivsAdded
    
    if addSphere0:
        sphereObject0 = addSphereObject(color)
    
    if addSphere1:
        sphereObject1 = addSphereObject(color)
    
    iV_prev = -1
    for iFrame in range(numFramesSampled):
        t = iFrame * interval
        iV = int(t // h)
        if iV == iV_prev:
            continue
        
        iV_prev = iV
        
        V = Vs[iV]
        v0 = V[e[0]]
        v1 = V[e[1]]
        curveObject.data.splines[0].points[0].co = (v0[0], v0[1], v0[2], 1.0)
        curveObject.data.splines[0].points[1].co = (v1[0], v1[1], v1[2], 1.0)
        curveObject.data.splines[0].points[0].keyframe_insert('co', frame=iFrame)
        curveObject.data.splines[0].points[1].keyframe_insert('co', frame=iFrame)
        
        if showForce:
            color = number_to_color(Fs[iFrame, ie])
            curveObject.data.materials[0].node_tree.nodes[1].inputs[0].default_value = color
            curveObject.data.materials[0].node_tree.nodes[1].inputs[0].keyframe_insert('default_value', frame=iFrame)

            curveObject.data.materials[0].node_tree.nodes[1].inputs[1].default_value = FsOriginal[iFrame, ie]
            curveObject.data.materials[0].node_tree.nodes[1].inputs[1].keyframe_insert('default_value', frame=iFrame)
            
        
        if addSphere0:
            sphereObject0.location = v0
            sphereObject0.keyframe_insert('location', frame=iFrame)
        if addSphere1:
            sphereObject1.location = v1
            sphereObject1.keyframe_insert('location', frame=iFrame)
    
    ivsAdded.add(e[0])
    ivsAdded.add(e[1])

bpy.ops.mesh.primitive_cube_add(size=1)
cube = bpy.context.active_object
cube.scale = (2, 2, 0.2)

iV_prev = -1
for iFrame in range(numFramesSampled):
    t = iFrame * interval
    iV = int(t // h)
    if iV == iV_prev:
        continue
    
    iV_prev = iV
    
    V = Vs[iV]
    vec0 = V[45] - V[46]
    vec1 = V[47] - V[48]
    
    normal = np.cross(vec0, vec1)
    normal /= np.linalg.norm(normal)
    up = np.array([0, 0, 1])
    
    axis = np.cross(up, normal)
    angle = np.arccos(np.sum(up * normal))
    
    cube.location = ((V[45] + V[46]) / 2).tolist()
    cube.rotation_euler = (axis * angle).tolist()
    
    cube.keyframe_insert('location', iFrame)
    cube.keyframe_insert('rotation_euler', iFrame)

bpy.context.scene.render.fps = fps

bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.1


bpy.ops.object.light_add(type='POINT', align='WORLD', location=(5, 5, 5))
bpy.context.active_object.data.energy=2000

bpy.ops.object.light_add(type='POINT', align='WORLD', location=(-5, 5, 5))
bpy.context.active_object.data.energy=2000

bpy.ops.object.light_add(type='POINT', align='WORLD', location=(5, -5, 5))
bpy.context.active_object.data.energy=2000

bpy.ops.object.light_add(type='POINT', align='WORLD', location=(-5, -5, 5))
bpy.context.active_object.data.energy=2000

bpy.ops.mesh.primitive_plane_add(size=1000, enter_editmode=False, align='WORLD', location=(0, 0, -5))

bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(3.67944574, -3.46289539,  2.47915459), rotation=(1.1093189716339111, 0.0, 0.8149281740188599))





















