import sys

sys.path.append('/Users/Roll/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append('/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh')

bl_info = {
    # required
    'name': 'PneuMesh',
    'blender': (2, 93, 0),
    'category': 'Object',
    # optional
    'version': (1, 0, 0),
    'author': 'Jianzhe Gu',
    'description': 'PneuMesh Visualizer',
}

import bpy
import re

# == GLOBAL VARIABLES
PROPS = [
    ('animation_dir',
     bpy.props.StringProperty(name='animation', default='/Users/Roll/Desktop/rendered/testing.animation.npy')),
    # ('suffix', bpy.props.StringProperty(name='Suffix', default='Suff')),
    # ('add_version', bpy.props.BoolProperty(name='Add Version', default=False)),
    # ('version', bpy.props.IntProperty(name='Version', default=1)),
]


class ObjectCreateTruss(bpy.types.Operator):
    bl_idname = 'opr.object_create_truss'
    bl_label = 'Object create truss'
    
    def execute(self, context):
        import numpy as np
        import bpy
        import sys
        sys.path.append('/Users/Roll/opt/anaconda3/lib/python3.7/site-packages/')
        # from skimage import exposure
        
        # import matplotlib.cm as cm
        # import matplotlib.colors as colors
        # scalar_mappable = cm.ScalarMappable(cmap='inferno')
        # number_to_color = scalar_mappable.get_cmap()
        
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
        print(bpy.types.Scene.animation_dir)
        animation = np.load(context.scene.animation_dir, allow_pickle=True).item()
        frameInterval = 500
        
        Vs = animation['Vs']
        Fs = animation['Fs']
        E = animation['E']
        edgeChannel = animation['edgeChannel']
        
        FsOriginal = Fs.copy()
        # Fs = np.array(exposure.equalize_hist(Fs.reshape(-1))).reshape(Fs.shape)
        
        
        colors = np.array([
            [250, 144, 146],
            [244, 138, 194],
            [179, 121, 211],
            [168, 159, 241],
            [121, 153, 219],
            [163, 231, 250]
        ], dtype=np.float) / 256
        
        
        # colorful
        colors = np.array([
            [242, 34, 51],
            [249, 152, 89],
            [251, 220, 88],
            [113, 191, 253],
            [103, 132, 232],
            [238, 255, 255]
        ], dtype=np.float) / 256
        
        def srgb2Linear(color_component):
            # sRGB to linear approximation
            if color_component < 0.04045:
                return color_component / 12.92
            else:
                return ((color_component + 0.055) / 1.055) ** 2.4
        
        def getRGB(hexColor):
            r = srgb2Linear(int(hexColor[0:2], 16) / 255.0)
            g = srgb2Linear(int(hexColor[2:4], 16) / 255.0)
            b = srgb2Linear(int(hexColor[4:6], 16) / 255.0)
            return [r, g, b]
        
        colors = ['F2B807', 'F5AB55', 'E3125F', '63A5BF', 'F27A5E', '66C8D2']
        
        colors = np.array([getRGB(c) for c in colors], dtype=np.float)
        
        
        # colors = [
        #     (0.007, 0.083, 0.500),
        #     (0.0088, 0.280, 0.628),
        #     (0.638, 0.030, 0.023),
        #     (0.5, 0.7, 0.86),
        #     (0.638, 0.131, 0.000),
        #     (0.7, 0.432, 0.008),
        #     (1.0, 1.0, 1.0)x
        # ]
    
        radius = 0.012
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
            transparent = color[0] == color[1] and color[0] == color[2] and color[0] == 1.0
            
            r, g, b = color
            mat = newMaterial(id)
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            
            shader = nodes.new(type='ShaderNodeBsdfPrincipled')
            nodes["Principled BSDF"].inputs[0].default_value = (r, g, b, 1)
            links.new(shader.outputs[0], output.inputs[0])
            
            if transparent:
                mat.blend_method = 'BLEND'
                nodes["Principled BSDF"].inputs[21].default_value = 0.0
                
            return mat

        materials = [newShader('color', c) for c in colors]
        
        def addCurveObject(iChannel, props):
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
            
            mat = materials[iChannel]
            obj.data.materials.append(mat)
            
            return obj
        
        def addSphereObject(iChannel):
            bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, enter_editmode=False, align='WORLD',
                                                  location=(0, 0, 0),
                                                  subdivisions=3)
            sphereObject = bpy.context.active_object
            mat = materials[iChannel]
            sphereObject.data.materials.append(mat)
            
            return sphereObject

        
        # print(numFrames)
        V0 = Vs[0]
        curveObjects = []
        
        ivsAdded = set()
        
        for ie, e in enumerate(E):
            print(ie / len(E))
            
            ic = edgeChannel[ie]
            color = colors[edgeChannel[ie]]
            
            curveObject = addCurveObject(ic, {'id': ie})
            
            addSphere0 = e[0] not in ivsAdded
            addSphere1 = e[1] not in ivsAdded
            
            # if addSphere0:
            #     sphereObject0 = addSphereObject(color)

            # if addSphere1:
            #     sphereObject1 = addSphereObject(color)
            
            if color[0] != 1.0:
                sphereObject0 = addSphereObject(ic)
                sphereObject1 = addSphereObject(ic)
            
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
                
                # if showForce:
                #     color = number_to_color(Fs[iFrame, ie])
                #     curveObject.data.materials[0].node_tree.nodes[1].inputs[0].default_value = color
                #     curveObject.data.materials[0].node_tree.nodes[1].inputs[0].keyframe_insert('default_value',
                #                                                                                frame=iFrame)
                #
                #     curveObject.data.materials[0].node_tree.nodes[1].inputs[1].default_value = FsOriginal[iFrame, ie]
                #     curveObject.data.materials[0].node_tree.nodes[1].inputs[1].keyframe_insert('default_value',
                #                                                                                frame=iFrame)
                #
                
                # if addSphere0:
                #     sphereObject0.location = v0
                #     sphereObject0.keyframe_insert('location', frame=iFrame)
                # if addSphere1:
                #     sphereObject1.location = v1
                #     sphereObject1.keyframe_insert('location', frame=iFrame)
                #
                if color[0] != 1.0:

                    sphereObject0.location = v0
                    sphereObject0.keyframe_insert('location', frame=iFrame)

                    sphereObject1.location = v1
                    sphereObject1.keyframe_insert('location', frame=iFrame)
            
            ivsAdded.add(e[0])
            ivsAdded.add(e[1])
        
        
        # # adding cube
        # bpy.ops.mesh.primitive_cube_add(size=1)
        # cube = bpy.context.active_object
        # cube.scale = (2, 2, 0.2)
        #
        # iV_prev = -1
        # for iFrame in range(numFramesSampled):
        #     t = iFrame * interval
        #     iV = int(t // h)
        #     if iV == iV_prev:
        #         continue
        #
        #     iV_prev = iV
        #
        #     V = Vs[iV]
        #     vec0 = V[45] - V[46]
        #     vec1 = V[47] - V[48]
        #
        #     normal = np.cross(vec0, vec1)
        #     normal /= np.linalg.norm(normal)
        #     up = np.array([0, 0, 1])
        #
        #     axis = np.cross(up, normal)
        #     angle = np.arccos(np.sum(up * normal))
        #
        #     cube.location = ((V[45] + V[46]) / 2).tolist()
        #     cube.rotation_euler = (axis * angle).tolist()
        #
        #     cube.keyframe_insert('location', iFrame)
        #     cube.keyframe_insert('rotation_euler', iFrame)
        
        bpy.context.scene.render.fps = fps
        
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.gamma = 0.8
        
        bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 0))
        bpy.context.object.data.energy = 1.0

        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.eevee.bloom_threshold = 1.1
        bpy.context.scene.eevee.use_gtao = True

        bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        
        bpy.context.scene.render.use_freestyle = True
        bpy.context.scene.render.line_thickness = 2.0
        bpy.context.scene.render.resolution_x = 3840
        bpy.context.scene.render.resolution_y = 2160
        
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(3.9237427711486816, -1.7984970808029175, 2.009699821472168),
                                  rotation=(1.1931158304214478, -7.194596491899574e-06, 1.1325969696044922))


        # bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(3.9237427711486816, 1.7984970808029175, 2.009699821472168),rotation=(1.1931158304214478, -7.194596491899574e-06, 3.141592653589 -1.1325969696044922))
        
        return {'FINISHED'}


# == PANELS
class PneuMeshPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_pneumesh'
    bl_label = 'PneuMesh'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    
    def draw(self, context):
        col = self.layout.column()
        for (prop_name, _) in PROPS:
            row = col.row()
            if prop_name == 'version':
                row = row.row()
                row.enabled = context.scene.add_version
            row.prop(context.scene, prop_name)
        
        col.operator('opr.object_create_truss', text='Create')


# == MAIN ROUTINE
CLASSES = [
    ObjectCreateTruss,
    PneuMeshPanel,
]


def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
    
    for klass in CLASSES:
        bpy.utils.register_class(klass)


def unregister():
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)
    
    for klass in CLASSES:
        bpy.utils.unregister_class(klass)


if __name__ == '__main__':
    register()
    
    
    