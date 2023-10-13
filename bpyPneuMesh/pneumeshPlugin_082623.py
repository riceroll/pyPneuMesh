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
     bpy.props.StringProperty(name='animation', default='/Users/Roll/Desktop/staticRendering/2channels.static.npy')),
    ('min_length',
     bpy.props.FloatProperty(name='min_length', default=0.12))
]


class ObjectCreateTruss(bpy.types.Operator):
    bl_idname = 'opr.object_create_truss'
    bl_label = 'Object create truss'
    
    def execute(self, context):
        import numpy as np
        import bpy
        import sys
        sys.path.append('/Users/Roll/opt/anaconda3/lib/python3.7/site-packages/')
        
        showForce = False
        print(bpy.types.Scene.animation_dir)
        animation = np.load(context.scene.animation_dir, allow_pickle=True).item()
        frameInterval = 500
        
        Vs = animation['Vs']
        E = animation['E']
        edgeChannel = animation['edgeChannel']
        
        
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
        
        colors = ['E3C53F', 'F2623A', 'E33252','63A5BF', 'F59B43',  '66C8D2']
        colorsa = np.array([getRGB(c) for c in colors], dtype=np.float)
        
        # import seaborn as sns
        #
        # cm = sns.color_palette("Spectral", n_colors=64)
        import colorsys
        
        rgbs = np.array([(0.6449826989619377, 0.03206459054209919, 0.2648212226066898),
         (0.6788158400615149, 0.06958861976163014, 0.2728181468665898),
         (0.7126489811610919, 0.10711264898116109, 0.28081507112648985),
         (0.746482122260669, 0.14463667820069204, 0.28881199538638985),
         (0.7803152633602461, 0.182160707420223, 0.2968089196462899),
         (0.8141484044598232, 0.2196847366397539, 0.3048058439061899),
         (0.8424452133794694, 0.25397923875432526, 0.3070357554786621),
         (0.8615148019992311, 0.28289119569396387, 0.2996539792387543),
         (0.8805843906189927, 0.3118031526336025, 0.2922722029988466),
         (0.8996539792387543, 0.340715109573241, 0.2848904267589389),
         (0.9187235678585159, 0.3696270665128797, 0.2775086505190312),
         (0.9377931564782777, 0.39853902345251824, 0.2701268742791234),
         (0.9568627450980393, 0.42745098039215684, 0.2627450980392157),
         (0.96239907727797, 0.46743560169165704, 0.281199538638985),
         (0.9679354094579008, 0.5074202229911572, 0.2996539792387543),
         (0.9734717416378317, 0.5474048442906574, 0.3181084198385236),
         (0.9776239907727797, 0.5773933102652824, 0.33194925028835054),
         (0.9831603229527105, 0.6173779315647827, 0.3504036908881199),
         (0.9886966551326413, 0.6573625528642827, 0.36885813148788915),
         (0.9923875432525952, 0.6938869665513263, 0.3900807381776239),
         (0.9930026912725874, 0.7246443675509417, 0.41591695501730086),
         (0.9936178392925799, 0.7554017685505574, 0.441753171856978),
         (0.9942329873125721, 0.7861591695501728, 0.46758938869665495),
         (0.9948481353325644, 0.8169165705497885, 0.4934256055363321),
         (0.9954632833525567, 0.8476739715494039, 0.519261822376009),
         (0.996078431372549, 0.8784313725490196, 0.5450980392156862),
         (0.9966935793925413, 0.8975009611687812, 0.5770857362552863),
         (0.9973087274125336, 0.9165705497885429, 0.6090734332948865),
         (0.9979238754325259, 0.9356401384083044, 0.6410611303344866),
         (0.9985390234525182, 0.9547097270280661, 0.6730488273740869),
         (0.9991541714725106, 0.9737793156478277, 0.7050365244136869),
         (0.9997693194925029, 0.9928489042675894, 0.7370242214532872),
         (0.9942329873125721, 0.9976931949250288, 0.7400230680507497),
         (0.9788542868127643, 0.9915417147251058, 0.7160322952710496),
         (0.9634755863129566, 0.9853902345251826, 0.6920415224913495),
         (0.9480968858131489, 0.9792387543252595, 0.6680507497116495),
         (0.932718185313341, 0.9730872741253365, 0.6440599769319494),
         (0.9173394848135333, 0.9669357939254134, 0.6200692041522493),
         (0.9019607843137256, 0.9607843137254902, 0.5960784313725491),
         (0.8656670511341794, 0.9460207612456749, 0.6034602076124567),
         (0.8293733179546331, 0.9312572087658594, 0.6108419838523644),
         (0.7930795847750868, 0.916493656286044, 0.6182237600922722),
         (0.7567858515955403, 0.9017301038062285, 0.6256055363321799),
         (0.720492118415994, 0.8869665513264131, 0.6329873125720876),
         (0.6841983852364477, 0.8722029988465976, 0.6403690888119954),
         (0.6440599769319495, 0.8562860438292965, 0.643521722414456),
         (0.60161476355248, 0.8396770472895041, 0.6441368704344483),
         (0.5591695501730105, 0.8230680507497117, 0.6447520184544406),
         (0.5273356401384084, 0.8106113033448674, 0.6452133794694349),
         (0.48489042675893923, 0.7940023068050751, 0.6458285274894272),
         (0.4424452133794695, 0.7773933102652826, 0.6464436755094195),
         (0.4, 0.7607843137254902, 0.6470588235294118),
         (0.3680123029603999, 0.7251057285659361, 0.6618223760092272),
         (0.33602460592079997, 0.6894271434063823, 0.6765859284890426),
         (0.30403690888119955, 0.6537485582468281, 0.6913494809688582),
         (0.27204921184159936, 0.6180699730872741, 0.7061130334486736),
         (0.24006151480199925, 0.58239138792772, 0.7208765859284891),
         (0.2080738177623993, 0.5467128027681664, 0.7356401384083044),
         (0.21299500192233756, 0.5114186851211072, 0.730795847750865),
         (0.24006151480199922, 0.47635524798154555, 0.7141868512110727),
         (0.2671280276816609, 0.4412918108419839, 0.6975778546712803),
         (0.2941945405613224, 0.4062283737024224, 0.680968858131488),
         (0.3212610534409842, 0.3711649365628604, 0.6643598615916955),
         (0.34832756632064593, 0.33610149942329876, 0.647750865051903)])
        
        rgbsNew = []
        for rgb in rgbs:
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1] * 1.4, hsv[2] * 0.72)
            rgbsNew.append(rgb)
        
        rgbs = np.array(rgbsNew)
        
        colorsb = np.array([rgbs[round((1.0 * i/32) * 64)] for i in range(32)], dtype=float)
        np.random.shuffle(colorsb)

        colors = np.vstack([colorsa, colorsb])
        
        
        radius = 0.012
        bevel_resolution = 12
        fps = 12
        h = 0.001
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
        transparentMaterial = newShader('transparent', (1.0, 1.0, 1.0))
        
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
            obj.data.materials.append(transparentMaterial)
            
            return obj
        
        def addSphereObject(iChannel, index):
            bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, enter_editmode=False, align='WORLD',
                                                  location=(0, 0, 0),
                                                  subdivisions=3)
            sphereObject = bpy.context.active_object
            sphereObject['id'] = index
            mat = materials[iChannel]
            sphereObject.data.materials.append(mat)
            sphereObject.data.materials.append(transparentMaterial)
            
            return sphereObject
        
        ivsAdded = set()
        
        for ie, e in enumerate(E):
            print(ie / len(E))
            
            ic = edgeChannel[ie]
            color = colors[edgeChannel[ie]]
            
            curveObject = addCurveObject(ic, {'id': ie})
            
            if color[0] != 1.0:
                sphereObject0 = addSphereObject(ic, int(e[0]))
                sphereObject1 = addSphereObject(ic, int(e[1]))
            
            iV_prev = -1
            for iFrame in range(numFramesSampled):
                t = iFrame * interval
                iV = int(t // h)
                if iV == iV_prev:
                    continue
                
                iV_prev = iV
                
                V = Vs[iV] / context.scene.min_length * 0.17
                v0 = V[e[0]]
                v1 = V[e[1]]
                curveObject.data.splines[0].points[0].co = (v0[0], v0[1], v0[2], 1.0)
                curveObject.data.splines[0].points[1].co = (v1[0], v1[1], v1[2], 1.0)
                curveObject.data.splines[0].points[0].keyframe_insert('co', frame=iFrame)
                curveObject.data.splines[0].points[1].keyframe_insert('co', frame=iFrame)
                
                if color[0] != 1.0:
                    sphereObject0.location = v0
                    sphereObject0.keyframe_insert('location', frame=iFrame)

                    sphereObject1.location = v1
                    sphereObject1.keyframe_insert('location', frame=iFrame)
            
            ivsAdded.add(e[0])
            ivsAdded.add(e[1])
        
        
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

        return {'FINISHED'}

class ObjectMoveMaterialSlotsUp(bpy.types.Operator):
    bl_idname = 'opr.object_move_material_slots_up'
    bl_label = 'Object move material slots up'
    
    def execute(self, context):
        """
        Moves material slots up or down for all selected objects in Blender.

        Parameters:
            direction (str): The direction to move the material slots. Either 'UP' or 'DOWN'.
        """
        import bpy
        
        # Loop through all selected objects
        for obj in bpy.context.selected_objects:
            # Set the object to be the active object
            bpy.context.view_layer.objects.active = obj
            
            # Get the length of the material slot list
            len_material_list = len(obj.material_slots)
            
            # Loop through material slots and move them up
            for i in range(1, len_material_list):
                bpy.context.object.active_material_index = i
                bpy.ops.object.material_slot_move(direction='UP')
        

            if obj.type == 'MESH':
                mesh = obj.data
                
                # Assuming you want to use the material in the first slot
                material_index = 0
                
                # Set all polygons to use this material index
                for polygon in mesh.polygons:
                    polygon.material_index = material_index
                    
                # Update & Free BMesh
                mesh.update()

        
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
        col.operator('opr.object_move_material_slots_up', text='Switch Material')


# == MAIN ROUTINE
CLASSES = [
    ObjectCreateTruss,
    ObjectMoveMaterialSlotsUp,
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
    
    
    