import os
import re
import bpy
import blf
import enum
import math
import locale
import operator
import mathutils
import bpy.utils.previews
from bpy_extras.io_utils import ExportHelper
from . import there


class Object:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Gender(enum.Enum):
    FEMALE = (0, 'Female', 0.24694)
    MALE = (1, 'Male', 0.24601)

    def __init__(self, index, title, length):
        self.index = index
        self.title = title
        self.skeleton = title[0].lower() + 't0'
        self.length = length


class Bone(enum.Enum):
    PELVIS = (1, 'Pelvis')
    LEFT_HIP = (2, 'LeftHip')
    LEFT_KNEE = (3, 'LeftKnee')
    LEFT_ANKLE = (4, 'LeftAnkle')
    LEFT_TOE_BASE = (5, 'LeftToeBase')
    RIGHT_HIP = (6, 'RightHip')
    RIGHT_KNEE = (7, 'RightKnee')
    RIGHT_ANKLE = (8, 'RightAnkle')
    RIGHT_TOE_BASE = (9, 'RightToeBase')
    SPINE_1 = (10, 'Spine1')
    SPINE_2 = (11, 'Spine2')
    SPINE_3 = (12, 'Spine3')
    BREASTS = (13, 'Breasts')
    SPINE_4 = (14, 'Spine4')
    LEFT_CLAVICLE = (15, 'LeftClavicle')
    LEFT_SHOULDER = (16, 'LeftShoulder')
    LEFT_ELBOW = (17, 'LeftElbow')
    LEFT_WRIST = (18, 'LeftWrist')
    LEFT_FINGERS_1 = (19, 'LeftFingers1')
    LEFT_FINGERS_2 = (20, 'LeftFingers2')
    LEFT_THUMB_1 = (21, 'LeftThumb1')
    LEFT_THUMB_2 = (22, 'LeftThumb2')
    NECK = (23, 'Neck')
    HEAD = (24, 'Head')
    RIGHT_CLAVICLE = (25, 'RightClavicle')
    RIGHT_SHOULDER = (26, 'RightShoulder')
    RIGHT_ELBOW = (27, 'RightElbow')
    RIGHT_WRIST = (28, 'RightWrist')
    RIGHT_FINGERS_1 = (29, 'RightFingers1')
    RIGHT_FINGERS_2 = (30, 'RightFingers2')
    RIGHT_THUMB_1 = (31, 'RightThumb1')
    RIGHT_THUMB_2 = (32, 'RightThumb2')

    def __init__(self, index, title):
        self.index = index
        self.title = title


# The below specs must match what the submission system is expecting
# They are designed to avoid exceeding the avatar memory usage and crashing the client
class Accoutrement(enum.Enum):
    HAIR = (
        'Hair', (
            Object(name='hair', title='Hair'),
        ), (
            Object(name='hair_round', title='Round'),
            Object(name='hair_square', title='Square'),
            Object(name='hair_tall', title='Tall'),
            Object(name='hair_wedge', title='Wedge'),
            Object(name='hair_pyramid', title='Pyramid'),
            Object(name='hair_eyesapart', title='EyesApart'),
        ),
        Object(
            lod_counts=(3, 3),
            distances=(15, 100, 1000),
            vertex_counts=(600, 450, 250),
            face_counts=(1100, 800, 450),
            bounds=(
                ((-0.35, 0.25, -0.35), (0.35, 1.2, 0.35)),
                ((-0.35, 0.25, -0.35), (0.35, 1.45, 0.35)),
            ),
            bone=None,
        ), (
            Object(skute_index=0, items=(
                Object(kit_id=2007, pieces=(
                    Object(piece_id=2007, texture_id=22007),
                )),
                Object(kit_id=900, pieces=(
                    Object(piece_id=900, texture_id=30900),
                )),
                Object(kit_id=1201, pieces=(
                    Object(piece_id=1201, texture_id=11201),
                )),
            )),
            Object(skute_index=0, items=(
                Object(kit_id=2507, pieces=(
                    Object(piece_id=2507, texture_id=12507),
                )),
                Object(kit_id=901, pieces=(
                    Object(piece_id=901, texture_id=30901),
                )),
                Object(kit_id=1708, pieces=(
                    Object(piece_id=1708, texture_id=11708),
                )),
            )),
        ), True,
    )
    EARRINGS = (
        'Earrings', (
            Object(name='earrings', title='Earrings'),
        ), (
            Object(name='round', title='Round'),
            Object(name='square', title='Square'),
            Object(name='tall', title='Tall'),
            Object(name='wedge', title='Wedge'),
            Object(name='pyramid', title='Pyramid'),
            Object(name='earswider', title='EarsWider'),
            Object(name='eyesapart', title='EyesApart'),
        ),
        Object(
            lod_counts=(2, 3),
            distances=(15, 100, 1000),
            vertex_counts=(450, 250, 200),
            face_counts=(800, 500, 300),
            bounds=(
                ((-0.35, 0.25, -0.35), (0.35, 1.2, 0.35)),
                ((-0.35, 0.25, -0.35), (0.35, 1.45, 0.35)),
            ),
            bone=Bone.HEAD,
        ), (
            Object(skute_index=1, items=(
                Object(kit_id=2001, pieces=(
                    Object(piece_id=2001, texture_id=12001),
                )),
                Object(kit_id=2151, pieces=(
                    Object(piece_id=2151, texture_id=22151),
                )),
                Object(kit_id=900, pieces=(
                    Object(piece_id=900, texture_id=30900),
                )),
                Object(kit_id=1201, pieces=(
                    Object(piece_id=1201, texture_id=11201),
                )),
            )),
            Object(skute_index=1, items=(
                Object(kit_id=2501, pieces=(
                    Object(piece_id=2501, texture_id=12501),
                )),
                Object(kit_id=2651, pieces=(
                    Object(piece_id=2651, texture_id=12651),
                )),
                Object(kit_id=901, pieces=(
                    Object(piece_id=901, texture_id=30901),
                )),
                Object(kit_id=1708, pieces=(
                    Object(piece_id=1708, texture_id=11708),
                )),
            )),
        ), True,
    )
    GLASSES = (
        'Glasses', (
            Object(name='glasses', title='Glasses'),
        ), (
            Object(name='round', title='Round'),
            Object(name='square', title='Square'),
            Object(name='tall', title='Tall'),
            Object(name='wedge', title='Wedge'),
            Object(name='pyramid', title='Pyramid'),
            Object(name='eyesapart', title='EyesApart'),
            Object(name='nosebridgelarger', title='NoseBridgeLarger'),
        ),
        Object(
            lod_counts=(2, 3),
            distances=(15, 100, 1000),
            vertex_counts=(450, 250, 200),
            face_counts=(800, 500, 300),
            bounds=(
                ((-0.35, 0.25, -0.35), (0.35, 1.2, 0.35)),
                ((-0.35, 0.25, -0.35), (0.35, 1.45, 0.35)),
            ),
            bone=Bone.HEAD,
        ), (
            Object(skute_index=1, items=(
                Object(kit_id=2001, pieces=(
                    Object(piece_id=2001, texture_id=12001),
                )),
                Object(kit_id=2111, pieces=(
                    Object(piece_id=2111, texture_id=22111),
                )),
                Object(kit_id=900, pieces=(
                    Object(piece_id=900, texture_id=30900),
                )),
                Object(kit_id=1201, pieces=(
                    Object(piece_id=1201, texture_id=11201),
                )),
            )),
            Object(skute_index=1, items=(
                Object(kit_id=2501, pieces=(
                    Object(piece_id=2501, texture_id=12501),
                )),
                Object(kit_id=2611, pieces=(
                    Object(piece_id=2611, texture_id=22611),
                )),
                Object(kit_id=901, pieces=(
                    Object(piece_id=901, texture_id=30901),
                )),
                Object(kit_id=1708, pieces=(
                    Object(piece_id=1708, texture_id=11708),
                )),
            )),
        ), True,
    )
    HEAD = ('Head', (), (), None, None, False)
    UPPERBODY = ('UpperBody', (), (), None, None, False)
    COMMON = (
        None, (
            Object(name='fc_hairscrunchy', title='HairScrunchy'),
            Object(name='fc_eyebrow', title='Eyebrow'),
            Object(name='fc_eyeshadow', title='EyeShadow'),
            Object(name='fc_mascara', title='Mascara'),
            Object(name='fc_upperlip', title='UpperLip'),
            Object(name='fc_lowerlip', title='LowerLip'),
            Object(name='fc_teeth', title='Teeth'),
            Object(name='fc_tongue', title='Tongue'),
            Object(name='fc_innermouth', title='InnerMouth'),
            Object(name='fc_fleshblushing', title='FleshBlushing'),
            Object(name='fc_fleshshadedblushing', title='FleshShadedBlushing'),
            Object(name='fc_flesh', title='Flesh'),
            Object(name='fc_fleshshaded', title='FleshShaded'),
        ), (
            Object(name='round', title='Round'),
            Object(name='square', title='Square'),
            Object(name='tall', title='Tall'),
            Object(name='wedge', title='Wedge'),
            Object(name='pyramid', title='Pyramid'),
            Object(name='cheeks', title='Cheeks'),
            Object(name='chindown', title='ChinLower'),
            Object(name='chinforward', title='ChinForward'),
            Object(name='chinthinner', title='ChinThinner'),
            Object(name='chinwide', title='ChinWider'),
            Object(name='earswider', title='EarsWider'),
            Object(name='eyebrowscloser', title='EyebrowsCloser'),
            Object(name='eyebrowthicker', title='EyebrowThicker'),
            Object(name='eyebrowthinner', title='EyebrowThinner'),
            Object(name='eyesapart', title='EyesApart'),
            Object(name='eyesbigger', title='EyesBigger'),
            Object(name='eyescloser', title='EyesCloser'),
            Object(name='eyessmaller', title='EyesSmaller'),
            Object(name='eyetilt', title='EyeTilt'),
            Object(name='jawsquare', title='JawSquare'),
            Object(name='lipsthicker', title='LipsThicker'),
            Object(name='lipsthinner', title='LipsThinner'),
            Object(name='mouthsmaller', title='MouthSmaller'),
            Object(name='muzzle', title='Muzzle'),
            Object(name='nosebridgelarger', title='NoseBridgeLarger'),
            Object(name='nosebridgesmaller', title='NoseBridgeSmaller'),
            Object(name='nosebulbsmall', title='NoseBulbSmaller'),
            Object(name='nosehook', title='NoseHook'),
            Object(name='noselonger', title='NoseLonger'),
            Object(name='nosewider', title='NoseWider'),
        ), None, None, False,
    )

    def __init__(self, title, materials, phenomorphs, specs, kits, is_valid):
        self.title = title
        self.materials = materials
        self.phenomorphs = phenomorphs
        self.specs = specs
        self.kits = kits
        self.is_valid = is_valid

    def get_material_name(self, title, is_optional=False):
        title = title.lower()
        for material in self.materials:
            if material.title.lower() == title:
                return material.name
        if is_optional and self != self.COMMON:
            return self.COMMON.get_material_name(title)
        return None

    def get_material_title(self, name, is_optional=False):
        for material in self.materials:
            if material.name == name:
                return material.title
        if is_optional and self != self.COMMON:
            return self.COMMON.get_material_title(name)
        return None

    def get_phenomorph_name(self, title, is_optional=False):
        title = title.lower()
        for phenomorph in self.phenomorphs:
            if phenomorph.title.lower() == title:
                return phenomorph.name
        if is_optional and self != self.COMMON:
            return self.COMMON.get_phenomorph_name(title)
        return None

    def get_phenomorph_title(self, name, is_optional=False):
        for phenomorph in self.phenomorphs:
            if phenomorph.name == name:
                return phenomorph.title
        if is_optional and self != self.COMMON:
            return self.COMMON.get_phenomorph_title(name)
        return None


class ExportSkuteBase:
    save_style: bpy.props.BoolProperty(
        name='StyleMaker Settings',
        description='Also save a .style file',
        default=False,
    )
    accoutrement: bpy.props.EnumProperty(
        name='',
        description='Category',
        default=Accoutrement.HAIR.name,
        items=[(a.name, a.title, f'Export the {a.title.lower()}') for a in Accoutrement if a.is_valid],
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text='Include')
        box.prop(self, 'save_style')
        box = layout.box()
        box.label(text='Category')
        box.prop(self, 'accoutrement')

    def check(self, context):
        old_filepath = self.filepath
        filename = os.path.basename(self.filepath)
        if filename != '':
            stem, ext = os.path.splitext(filename)
            if stem.startswith('.') and not ext:
                stem, ext = '', stem
            ext_lower = ext.lower()
            if ext_lower != '.skute':
                if ext == '':
                    filepath = self.filepath
                else:
                    filepath = self.filepath[:-len(ext)]
                if filepath != '':
                    self.filepath = filepath + '.skute'
        return self.filepath != old_filepath

    def invoke(self, context, event):
        try:
            del context.scene['ThereExportSettings']
        except KeyError:
            pass
        preferences = context.preferences.addons[__package__].preferences
        self.save_style = preferences.save_style
        self.accoutrement = bpy.context.window_manager.there_skute_accoutrements.accoutrement
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            self.check(context)
            bpy.ops.object.mode_set(mode='OBJECT')
            assert bpy.context.mode == 'OBJECT', 'Exporting must be done in Object Mode.'
            context.window_manager.progress_begin(0, 100)
            context.window_manager.progress_update(0)
            self.data = Object()
            self.data.skute = there.Skute(path=self.filepath)
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_armature = [o for o in bpy_scene.objects if o.parent is None and o.type == 'ARMATURE'][0]
            except IndexError:
                raise RuntimeError('The armature object was not found.')
            for gender in Gender:
                if gender.title.lower() == self.get_basename(bpy_armature.name).lower() and gender.length == round(bpy_armature.data.bones['Head'].length, 5):
                    self.data.gender = gender
                    break
            else:
                raise RuntimeError('The gender could not be determined.')
            try:
                self.data.accoutrement = Accoutrement[self.accoutrement]
            except KeyError:
                raise RuntimeError('The item could not be determined.')
            if not self.data.accoutrement.is_valid:
                raise RuntimeError('The item is not valid.')
            try:
                bpy_object = [o for o in bpy_armature.children if o.type == 'EMPTY' and self.get_basename(o.name).lower() == self.data.accoutrement.title.lower()][0]
            except IndexError:
                raise RuntimeError('The "%s" object was not found.' % self.data.accoutrement.title)
            self.data.skute.skeleton = self.data.gender.skeleton
            self.gather_lods(bpy_armature, bpy_object)
            context.window_manager.progress_update(25)
            self.sort_lods()
            context.window_manager.progress_update(35)
            self.gather_materials()
            context.window_manager.progress_update(45)
            self.normalize_weights()
            context.window_manager.progress_update(45)
            self.scale_skute()
            context.window_manager.progress_update(55)
            self.data.skute.save()
            context.window_manager.progress_update(65)
            if self.save_style:
                style = there.Style(path=os.path.splitext(self.filepath)[0] + '.style', items=self.create_style_items())
                style.save()
            context.window_manager.progress_update(100)
        except (RuntimeError, AssertionError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        context.window_manager.progress_end()
        return {'FINISHED'}

    def gather_lods(self, bpy_armature, bpy_object):
        matrix_root_inverted = bpy_armature.matrix_world.inverted()
        bone_lookup = {bpy_armature.data.bones[i].name: i + 1 for i in range(len(bpy_armature.data.bones))}
        for bpy_lod in bpy_object.children:
            if bpy_lod.type != 'MESH':
                continue
            for bpy_vertex_group in bpy_lod.vertex_groups:
                if bpy_vertex_group.name not in bone_lookup and not bpy_vertex_group.name.startswith('_'):
                    raise RuntimeError('Vertex group "%s" does not refer to a bone.' % bpy_vertex_group.name)
            lod = there.LOD()
            matrix_skute = matrix_root_inverted @ bpy_lod.matrix_world
            matrix_rotation = matrix_skute.to_quaternion().to_matrix()
            if bpy.app.version < (4, 1, 0):
                bpy_lod.data.calc_normals_split()
            components = Object(
                positions=[[-v[0], v[2], v[1]] for v in [matrix_skute @ v.co for v in bpy_lod.data.vertices]],
                indices=[v.vertex_index for v in bpy_lod.data.loops],
                normals=[[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.normal).normalized() for v in bpy_lod.data.loops]],
                uvs=[[[d.uv[0], 1.0 - d.uv[1]] for d in e.data] for e in bpy_lod.data.uv_layers][:1],
                bones=[bone_lookup.get(g.name, -1) for g in bpy_lod.vertex_groups],
                weights=[{g.group: g.weight for g in v.groups} for v in bpy_lod.data.vertices],
                shapes=[],
            )
            optimized = Object(
                vertices=[],
                map={},
                shapes=[],
            )
            if bpy_lod.data.shape_keys is not None:
                for bpy_shape in bpy_lod.data.shape_keys.key_blocks[1:]:
                    name = self.data.accoutrement.get_phenomorph_name(self.get_basename(bpy_shape.name), is_optional=True)
                    if name is None:
                        continue
                    shape_normals = bpy_shape.normals_split_get()
                    shape_normals = [mathutils.Vector(shape_normals[i:i + 3]) for i in range(0, len(shape_normals), 3)]
                    shape = Object(
                        index=len(components.shapes),
                        positions=[[-v[0], v[2], v[1]] for v in [matrix_skute @ v.co for v in bpy_shape.data]],
                        normals=[[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v).normalized() for v in shape_normals]],
                    )
                    components.shapes.append(shape)
                    optimized.shapes.append(there.Target(name=name))
            if bpy.app.version < (4, 1, 0):
                bpy_lod.data.free_normals_split()
            for index, name in enumerate(bpy_lod.material_slots.keys()):
                if name not in self.data.skute.materials:
                    self.data.skute.materials[name] = there.Material(name=name)
                mesh = there.Mesh()
                mesh.material = self.data.skute.materials[name]
                bpy_polygons = [p for p in bpy_lod.data.polygons if p.material_index == index]
                if len(bpy_polygons) == 0:
                    continue
                mesh.indices = self.optimize_mesh(bpy_polygons=bpy_polygons, components=components, optimized=optimized)
                if len(mesh.indices) == 0:
                    continue
                lod.meshes.append(mesh)
            lod.vertices = optimized.vertices
            lod.phenomorphs = optimized.shapes
            lod.vertex_count += len(lod.vertices)
            for mesh in lod.meshes:
                lod.face_count += len(mesh.indices) // 3
            self.data.skute.lods.append(lod)

    def optimize_mesh(self, bpy_polygons, components, optimized):
        positions = components.positions
        indices = components.indices
        normals = components.normals
        uvs = components.uvs
        bones = components.bones
        weights = components.weights
        shapes = components.shapes
        if len(uvs) == 0:
            uvs = [(0.0, 0.0)] * len(normals)
        else:
            uvs = uvs[0]
        optimized_indices = []
        for bpy_polygon in bpy_polygons:
            triangles = []
            for index in range(2, len(bpy_polygon.loop_indices)):
                triangles.append([bpy_polygon.loop_indices[0], bpy_polygon.loop_indices[index - 1], bpy_polygon.loop_indices[index]])
            for triangle in triangles:
                for index in triangle:
                    key = '%s:%s:%s' % (
                        indices[index],
                        '%.03f:%.03f:%.03f' % (normals[index][0], normals[index][1], normals[index][2]),
                        '%.03f:%.03f' % (uvs[index][0], uvs[index][1]),
                    )
                    optimized_index = optimized.map.get(key)
                    if optimized_index is None:
                        optimized_index = len(optimized.vertices)
                        optimized.map[key] = optimized_index
                        optimized.vertices.append(there.LOD.Vertex(
                            position=positions[indices[index]],
                            normal=normals[index],
                            uv=uvs[index],
                            bone_indices=[bones[v] for v in weights[indices[index]].keys()],
                            bone_weights=[v for v in weights[indices[index]].values()],
                        ))
                        for shape in shapes:
                            delta_position = [v[0] - v[1] for v in zip(shape.positions[indices[index]], positions[indices[index]])]
                            if self.is_close(delta_position, (0.0, 0.0, 0.0)):
                                continue
                            delta_normal = [v[0] - v[1] for v in zip(shape.normals[index], normals[index])]
                            optimized.shapes[shape.index].deltas.append(there.Target.Delta(
                                index=optimized_index,
                                position=delta_position,
                                normal=delta_normal,
                            ))
                    optimized_indices.append(optimized_index)
        return optimized_indices

    def sort_lods(self):
        self.data.skute.lods.sort(key=operator.attrgetter('vertex_count'), reverse=True)
        if self.data.accoutrement.specs is None:
            raise RuntimeError('"%s" is not configured.' % self.data.accoutrement.title)
        # The below specs must match what the submission system is expecting
        # They are designed to avoid exceeding the avatar memory usage and crashing the client
        spec_lod_counts = self.data.accoutrement.specs.lod_counts
        spec_distances = self.data.accoutrement.specs.distances
        spec_vertex_counts = self.data.accoutrement.specs.vertex_counts
        spec_face_counts = self.data.accoutrement.specs.face_counts
        spec_bounds = self.data.accoutrement.specs.bounds[self.data.gender.index]
        spec_bone = self.data.accoutrement.specs.bone
        lod_count = len(self.data.skute.lods)
        if lod_count < spec_lod_counts[0] or lod_count > spec_lod_counts[1]:
            if spec_lod_counts[0] == spec_lod_counts[1]:
                raise RuntimeError('"%s" should contain %s LODs.' % (
                    self.data.accoutrement.title,
                    spec_lod_counts[0],
                ))
            else:
                raise RuntimeError('"%s" should contain between %s and %s LODs.' % (
                    self.data.accoutrement.title,
                    spec_lod_counts[0],
                    spec_lod_counts[1],
                ))
        for index, lod in enumerate(self.data.skute.lods):
            lod.index = index
            lod.distance = spec_distances[index]
            if lod.vertex_count > spec_vertex_counts[index]:
                raise RuntimeError('LOD%s contains too many vertices.' % index)
            if lod.face_count > spec_face_counts[index]:
                raise RuntimeError('LOD%s contains too many faces.' % index)
            if index < 2:
                lod_phenomorphs = [p.name for p in lod.phenomorphs]
                for phenomorph in self.data.accoutrement.phenomorphs:
                    if phenomorph.name not in lod_phenomorphs:
                        raise RuntimeError('LOD%s should contain a "%s" shape key.' % (index, phenomorph.title))
            else:
                lod.phenomorphs = []
            lod.bounds = [
                [min([v.position[i] for v in lod.vertices]) for i in range(3)],
                [max([v.position[i] for v in lod.vertices]) for i in range(3)],
            ]
            if False in [lod.bounds[0][i] >= spec_bounds[0][i] and lod.bounds[1][i] <= spec_bounds[1][i] for i in range(3)]:
                raise RuntimeError('LOD%s is outside the bounding box.' % index)
            if spec_bone is not None:
                for vertex in lod.vertices:
                    if len(vertex.bone_indices) != 1 or vertex.bone_indices[0] != spec_bone.index or vertex.bone_weights[0] != 1.0:
                        raise RuntimeError('LOD%s must be entirely weighted to the "%s" bone' % (index, spec_bone.title))
            for phenomorph in lod.phenomorphs:
                if len(phenomorph.deltas) == 0:
                    continue
                phenomorph.bounds = [
                    [min([z[0].position[i] + z[1].position[i] for z in zip(lod.vertices, phenomorph.deltas)]) for i in range(3)],
                    [max([z[0].position[i] + z[1].position[i] for z in zip(lod.vertices, phenomorph.deltas)]) for i in range(3)],
                ]
                if False in [phenomorph.bounds[0][i] >= spec_bounds[0][i] and phenomorph.bounds[1][i] <= spec_bounds[1][i] for i in range(3)]:
                    name = self.data.accoutrement.get_phenomorph_title(phenomorph.name, is_optional=True)
                    raise RuntimeError('LOD%s with "%s" shape is outside the bounding box.' % (index, name))

    def gather_materials(self):
        self.data.skute.materials = list(self.data.skute.materials.values())
        for index, material in enumerate(self.data.skute.materials):
            bpy_material = bpy.data.materials[material.name]
            material.name = self.data.accoutrement.get_material_name(self.get_basename(bpy_material.name), is_optional=True)
            if material.name is None:
                raise RuntimeError('Material "%s" is not supported.' % bpy_material.name)
            material.index = index
            if not bpy_material.use_backface_culling:
                raise RuntimeError('Material "%s" does not support two sided rendering.' % bpy_material.name)
            if bpy_material.blend_method in ['BLEND', 'CLIP']:
                raise RuntimeError('Material "%s" does not support alpha.' % bpy_material.name)
            if material.name.startswith('fc_'):
                continue
            if not bpy_material.use_nodes:
                raise RuntimeError('Material "%s" is not configured to use nodes.' % bpy_material.name)
            bpy_material_output = bpy_material.node_tree.nodes.get('Material Output')
            if bpy_material_output is None:
                raise RuntimeError('Material "%s" does not have a Material Output node.' % bpy_material.name)
            bpy_input = bpy_material_output.inputs.get('Surface')
            if not bpy_input.is_linked:
                raise RuntimeError('Material "%s" does not have a linked Surface.' % bpy_material.name)
            bpy_link_node = bpy_input.links[0].from_node
            if bpy_link_node.type == 'GROUP':
                bpy_node_tree = bpy_link_node.node_tree
                if bpy_node_tree.name == 'There BSDF':
                    self.gather_there_bsdf(bpy_material, bpy_link_node, material)
                else:
                    raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (bpy_material.name, bpy_node_tree.name))
            elif bpy_link_node.type == 'BSDF_PRINCIPLED':
                self.gather_principled_bsdf(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'BSDF_DIFFUSE':
                self.gather_diffuse_bsdf(bpy_material, bpy_link_node, material)
            else:
                raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (bpy_material.name, bpy_link_node.name))
        for material in self.data.accoutrement.materials:
            if material.name.startswith('fc_'):
                continue
            if len([m for m in self.data.skute.materials if m.name == material.name and m.texture is not None]) == 0:
                raise RuntimeError('Material "%s" is missing.' % (material.title))

    def gather_there_bsdf(self, bpy_material, bpy_there_node, material):
        material.texture = self.gather_texture(bpy_there_node, 'Color')
        if material.texture is None:
            raise RuntimeError('Material "%s" needs a Color image.' % bpy_material.name)

    def gather_principled_bsdf(self, bpy_material, bpy_principled_node, material):
        material.texture = self.gather_texture(bpy_principled_node, 'Base Color')
        if material.texture is None:
            raise RuntimeError('Material "%s" needs a Base Color image.' % bpy_material.name)

    def gather_diffuse_bsdf(self, bpy_material, bpy_diffuse_node, material):
        material.texture = self.gather_texture(bpy_diffuse_node, 'Color')
        if material.texture is None:
            raise RuntimeError('Material "%s" needs a Color texture.' % bpy_material.name)

    def gather_texture(self, bpy_material_node, name):
        bpy_input = bpy_material_node.inputs.get(name)
        if bpy_input is None:
            return None
        if not bpy_input.is_linked:
            return None
        if bpy_input.is_multi_input:
            return None
        bpy_link_node = bpy_input.links[0].from_node
        if bpy_link_node.type == 'TEX_IMAGE':
            bpy_image = bpy_link_node.image
            if bpy_image is None:
                return None
            path = bpy_image.filepath_from_user()
            if path == '':
                return None
            return path
        return None

    def create_style_items(self):
        kits = self.data.accoutrement.kits
        if kits is None:
            raise RuntimeError('"%s" is not configured.' % self.data.accoutrement.title)
        kit = kits[self.data.gender.index]
        items = []
        for index, kit_item in enumerate(kit.items):
            item = there.Style.Item(kit_id=kit_item.kit_id)
            if index == kit.skute_index:
                item.skute = self.data.skute
            for kit_piece in kit_item.pieces:
                piece = there.Style.Piece(piece_id=kit_piece.piece_id, texture_id=kit_piece.texture_id)
                item.pieces.append(piece)
            items.append(item)
        return items

    def normalize_weights(self):
        for lod in self.data.skute.lods:
            for vertex in lod.vertices:
                values = sorted([v for v in zip(vertex.bone_indices, vertex.bone_weights) if v[0] >= 0 and v[1] > 0.0], key=lambda v: v[1], reverse=True)
                total = sum([v[1] for v in values] + [0.0])
                count = len(values)
                if count == 0 or total < 0.1:
                    raise RuntimeError('LOD%s is not rigged to the armature bones.' % lod.index)
                elif count == 1:
                    values.append([values[0][0], 0.0])
                elif count > 2:
                    raise RuntimeError('LOD%s is rigged to more than 2 bones per vertex.' % lod.index)
                vertex.bone_indices = [v[0] for v in values]
                vertex.bone_weights = [v[1] / total for v in values]

    def scale_skute(self):
        scales = [(s, pow(2.0, s - 32)) for s in range(32, 34)]
        try:
            value = max([abs(v) for o in self.data.skute.lods for b in o.bounds for v in b])
        except ValueError:
            value = 0.0
        try:
            scale = [s for s in scales if value <= s[1]][0]
        except IndexError:
            raise RuntimeError('The skute is too big to export.')
        self.data.skute.scale = scale[0]
        for lod in self.data.skute.lods:
            for vertex in lod.vertices:
                vertex.position = [v / scale[1] for v in vertex.position]
            for phenomorph in lod.phenomorphs:
                for delta in phenomorph.deltas:
                    delta.position = [v / scale[1] for v in delta.position]

    def get_stats(self, is_quick=False):
        try:
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_armature = [o for o in bpy_scene.objects if o.parent is None and o.type == 'ARMATURE'][0]
            except IndexError:
                return None
            for gender in Gender:
                if gender.title.lower() == self.get_basename(bpy_armature.name).lower() and gender.length == round(bpy_armature.data.bones['Head'].length, 5):
                    break
            else:
                return None
            stats = Object(
                accoutrements=[],
            )
            if is_quick:
                return stats
            if bpy.context.mode != 'OBJECT':
                return None
            bone_lookup = {bpy_armature.data.bones[i].name: i + 1 for i in range(len(bpy_armature.data.bones))}
            for bpy_object in bpy_armature.children:
                if bpy_object.type != 'EMPTY':
                    continue
                accoutrement = getattr(Accoutrement, self.get_basename(bpy_object.name).upper(), None)
                if accoutrement is None or not accoutrement.is_valid:
                    continue
                stats_accoutrement = Object(
                    name=accoutrement.title,
                    lods=[],
                )
                for bpy_lod in bpy_object.children:
                    if bpy_lod.type != 'MESH':
                        continue
                    stats_lod = Object(
                        name=bpy_lod.name,
                        vertex_count=0,
                        triangle_count=0,
                        shape_count=0,
                    )
                    if bpy.app.version < (4, 1, 0):
                        bpy_lod.data.calc_normals_split()
                    positions = [list(v) for v in [v.co for v in bpy_lod.data.vertices]]
                    indices = [v.vertex_index for v in bpy_lod.data.loops]
                    normals = [list(v) for v in [v.normal for v in bpy_lod.data.loops]]
                    uvs = [[list(d.uv) for d in e.data] for e in bpy_lod.data.uv_layers][:1]
                    bones = [bone_lookup.get(g.name, -1) for g in bpy_lod.vertex_groups]
                    weights = [{g.group: g.weight for g in v.groups} for v in bpy_lod.data.vertices]
                    if len(uvs) == 0:
                        uvs = [(0.0, 0.0)] * len(normals)
                    else:
                        uvs = uvs[0]
                    if bpy.app.version < (4, 1, 0):
                        bpy_lod.data.free_normals_split()
                    if bpy_lod.data.shape_keys is not None:
                        for bpy_shape in bpy_lod.data.shape_keys.key_blocks[1:]:
                            name = accoutrement.get_phenomorph_name(self.get_basename(bpy_shape.name), is_optional=True)
                            if name is None:
                                continue
                            stats_lod.shape_count += 1
                    for index, name in enumerate(bpy_lod.material_slots.keys()):
                        bpy_polygons = [p for p in bpy_lod.data.polygons if p.material_index == index]
                        if len(bpy_polygons) == 0:
                            continue
                        optimized_keys = set()
                        for bpy_polygon in bpy_polygons:
                            for index in range(2, len(bpy_polygon.loop_indices)):
                                triangle = [bpy_polygon.loop_indices[0], bpy_polygon.loop_indices[index - 1], bpy_polygon.loop_indices[index]]
                                stats_lod.triangle_count += 1
                                for index in triangle:
                                    key = '%s:%s:%s' % (
                                        indices[index],
                                        '%.03f:%.03f:%.03f' % (normals[index][0], normals[index][1], normals[index][2]),
                                        '%.03f:%.03f' % (uvs[index][0], uvs[index][1]),
                                    )
                                    if key not in optimized_keys:
                                        optimized_keys.add(key)
                                        stats_lod.vertex_count += 1
                    stats_accoutrement.lods.append(stats_lod)
                stats.accoutrements.append(stats_accoutrement)
        except (RuntimeError, AssertionError) as error:
            return None
        return stats

    @staticmethod
    def get_basename(name):
        return re.sub(r'\.\d+$', '', name)

    @staticmethod
    def is_close(a, b):
        return False not in [math.isclose(a[i], b[i], abs_tol=0.00001) for i in range(len(a))]


class ExportSkute(bpy.types.Operator, ExportSkuteBase, ExportHelper):
    """Export scene as There Skute file"""
    bl_idname = 'export_scene.skute'
    bl_label = 'Export There Skute'
    filename_ext = '.skute'
    filter_glob: bpy.props.StringProperty(
        default='*.skute',
        options={'HIDDEN'},
    )

    @staticmethod
    def handle_menu_export(self, context):
        self.layout.operator(ExportSkute.bl_idname, text='There Skute (.skute)')


class ExportSkutePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    save_style: bpy.props.BoolProperty(
        name='Export StyleMaker Settings',
        description='Save a .style file when exporting a skute',
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'save_style')


class ThereAccoutrementPropertyGroup(bpy.types.PropertyGroup):
    accoutrement: bpy.props.EnumProperty(
        name='',
        description='Category',
        default=Accoutrement.HAIR.name,
        items=[(a.name, a.title, f'Focus on the {a.title.lower()}') for a in Accoutrement if a.is_valid],
    )


class ThereLODOperator(bpy.types.Operator):
    bl_idname = __package__ + '.set_lod'
    bl_label = 'LOD'
    bl_description = 'Show LOD in viewport'
    id: bpy.props.StringProperty()

    def execute(self, context):
        id = self.id.lower()
        bpy_lods = ThereOutlinerPanel.get_lods(context)
        for bpy_lod in bpy_lods:
            name = ThereOutlinerPanel.get_basename(bpy_lod.name).lower()
            bpy_lod.hide_set(name != id)
        ThereOutlinerPanel.set_active(context, bpy_lods)
        return {'FINISHED'}

    @classmethod
    def is_depressed(cls, context, id):
        id = id.lower()
        bpy_lods = ThereOutlinerPanel.get_lods(context)
        if len(bpy_lods) == 0:
            return False
        for bpy_lod in bpy_lods:
            name = ThereOutlinerPanel.get_basename(bpy_lod.name).lower()
            if bpy_lod.hide_get() != (name != id):
                return False
        return True


class ThereShapeKeyOperator(bpy.types.Operator):
    bl_idname = __package__ + '.set_shapekey'
    bl_label = 'SHAPEKEY'
    bl_description = 'Show shape key in viewport (hold Ctrl for multiple)'
    id: bpy.props.StringProperty()

    def execute(self, context):
        id = self.id.lower()
        states = {}
        bpy_lods = [b for b in ThereOutlinerPanel.get_lods(context) if b.data.shape_keys is not None]
        for bpy_lod in bpy_lods:
            for shape_key in bpy_lod.data.shape_keys.key_blocks:
                name = shape_key.name.lower()
                if name == 'basis':
                    continue
                if name not in states:
                    states[name] = shape_key.mute
                else:
                    states[name] = states[name] or shape_key.mute
        unmute_count = len([v for v in states.values() if not v])
        for bpy_lod in bpy_lods:
            for shape_key in bpy_lod.data.shape_keys.key_blocks:
                name = shape_key.name.lower()
                if name == 'basis':
                    continue
                if id == 'basis':
                    shape_key.mute = True
                elif name == id:
                    if not self.ctrl and not states.get(name, True) and unmute_count > 1:
                        shape_key.mute = False
                    else:
                        shape_key.mute = not states[name] if name in states else False
                elif not self.ctrl:
                    shape_key.mute = True
                if not shape_key.mute:
                    shape_key.value = 1.0
        ThereOutlinerPanel.set_active(context, bpy_lods, id)
        return {'FINISHED'}

    def invoke(self, context, event):
        self.ctrl = event.ctrl
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        bpy_lods = [b for b in ThereOutlinerPanel.get_lods(context) if b.data.shape_keys is not None and not b.hide_get()]
        if len(bpy_lods) == 0:
            return False
        for bpy_lod in bpy_lods:
            for shape_key in bpy_lod.data.shape_keys.key_blocks:
                name = shape_key.name.lower()
                if name == 'basis':
                    continue
                return True
        return False

    @classmethod
    def is_depressed(cls, context, id):
        id = id.lower()
        bpy_accoutrements = ThereOutlinerPanel.get_accoutrements(context)
        if len(bpy_accoutrements) == 0:
            return False
        shape_key = None
        bpy_lods = [b for b in ThereOutlinerPanel.get_lods(context) if b.data.shape_keys is not None and not b.hide_get()]
        for bpy_lod in bpy_lods:
            for shape_key in bpy_lod.data.shape_keys.key_blocks:
                name = shape_key.name.lower()
                if name == 'basis':
                    continue
                if id == 'basis':
                    if not shape_key.mute:
                        return False
                elif name == id:
                    if shape_key.mute:
                        return False
        return shape_key is not None or id == 'basis'


class ThereOutlinerPanel(bpy.types.Panel):
    bl_idname = 'OUTLINER_PT_there_skute'
    bl_space_type = 'OUTLINER'
    bl_region_type = 'HEADER'
    bl_label = 'There Helper'
    previews = None

    def draw(self, context):
        shapes = set()
        bpy_lods = [b for b in self.get_lods(context) if b.data.shape_keys is not None and not b.hide_get()]
        for bpy_lod in bpy_lods:
            for shape_key in bpy_lod.data.shape_keys.key_blocks:
                name = shape_key.name.lower()
                shapes.add(name)
        layout = self.layout
        layout.label(text='Category')
        layout.prop(context.window_manager.there_skute_accoutrements, 'accoutrement')
        layout.label(text='LODs')
        row = layout.row()
        for text in ['LOD0', 'LOD1', 'LOD2']:
            row.operator(ThereLODOperator.bl_idname, text=text, depress=ThereLODOperator.is_depressed(context, text)).id = text
        row = layout.row()
        row.label(text='Shape Keys')
        row.prop(context.window_manager, 'there_skute_optional_shapes')
        text = 'Basis'
        layout.operator(ThereShapeKeyOperator.bl_idname, text=text, depress=ThereShapeKeyOperator.is_depressed(context, text)).id = text
        phenomorphs = Accoutrement.COMMON.phenomorphs
        if not context.window_manager.there_skute_optional_shapes:
            try:
                active_accoutrement = Accoutrement[context.window_manager.there_skute_accoutrements.accoutrement]
            except KeyError:
                active_accoutrement = None
            if active_accoutrement is not None:
                phenomorphs = active_accoutrement.phenomorphs
        for phenomorph in phenomorphs:
            if phenomorph.title.lower() not in shapes:
                continue
            text = phenomorph.title
            layout.operator(ThereShapeKeyOperator.bl_idname, text=text, depress=ThereShapeKeyOperator.is_depressed(context, text)).id = text

    @classmethod
    def poll(cls, context):
        bpy_armature = ThereOutlinerPanel.get_armature(context)
        if bpy_armature is None:
            return False
        return True

    @staticmethod
    def get_basename(name):
        return re.sub(r'\.\d+$', '', name)

    @staticmethod
    def get_armature(context):
        bpy_scene = context.scene
        bpy_armatures = [o for o in bpy_scene.objects if o.parent is None and o.type == 'ARMATURE']
        if len(bpy_armatures) == 0:
            return None
        bpy_armature = bpy_armatures[0]
        if ThereOutlinerPanel.get_basename(bpy_armature.name).lower() not in [e.title.lower() for e in Gender]:
            return None
        return bpy_armature

    @staticmethod
    def get_accoutrements(context):
        bpy_armature = ThereOutlinerPanel.get_armature(context)
        if bpy_armature is None:
            return []
        accoutrement_titles = [e.title.lower() for e in Accoutrement if e.title is not None]
        return [o for o in bpy_armature.children if o.type == 'EMPTY' and ThereOutlinerPanel.get_basename(o.name).lower() in accoutrement_titles]

    @staticmethod
    def get_lods(context):
        bpy_lods = []
        bpy_accoutrements = ThereOutlinerPanel.get_accoutrements(context)
        for bpy_accoutrement in bpy_accoutrements:
            for bpy_lod in bpy_accoutrement.children:
                name = ThereOutlinerPanel.get_basename(bpy_lod.name).lower()
                if not name.startswith('lod') or bpy_lod.type != 'MESH':
                    continue
                bpy_lods.append(bpy_lod)
        return bpy_lods

    @staticmethod
    def set_active(context, bpy_lods, shape_id=None):
        for bpy_selected in context.selected_objects:
            bpy_selected.select_set(False)
        try:
            active_accoutrement = Accoutrement[bpy.context.window_manager.there_skute_accoutrements.accoutrement]
        except KeyError:
            return
        for bpy_lod in bpy_lods:
            if bpy_lod.hide_get():
                continue
            if ThereOutlinerPanel.get_basename(bpy_lod.parent.name).lower() != active_accoutrement.title.lower():
                continue
            context.view_layer.objects.active = bpy_lod
            if shape_id is not None:
                for index, shape_key in enumerate(bpy_lod.data.shape_keys.key_blocks):
                    if shape_key.name.lower() == shape_id:
                        bpy_lod.active_shape_key_index = index
                        break
                else:
                    bpy_lod.active_shape_key_index = 0
            break

    @staticmethod
    def handle_outliner_header(self, context):
        layout = self.layout
        if context.space_data.display_mode == 'VIEW_LAYER' and ThereOutlinerPanel.poll(context):
            layout.popover(ThereOutlinerPanel.bl_idname, text='', icon_value=ThereOutlinerPanel.previews['THERE_SKUTE_HELPER'].icon_id)


class SkuteStatistics:
    handler = None
    rows = None
    active = None

    @classmethod
    def draw(cls):
        if cls.rows is None:
            return
        scale = bpy.context.preferences.system.ui_scale
        if bpy.app.version < (4, 0, 0):
            blf.size(0, 11.0 * scale, 72)
        else:
            blf.size(0, 11.0 * scale)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        if bpy.app.version >= (4, 2, 0):
            blf.shadow(0, 6, 0.0, 0.0, 0.0, 1.0)
            blf.enable(0, blf.SHADOW)
        for y, row in enumerate(reversed(cls.rows)):
            for column in row:
                blf.position(0, (10.0 + column[0]) * scale, (17.0 + y * 17.0) * scale, 0.0)
                blf.draw(0, column[1])
        if bpy.app.version >= (4, 2, 0):
            blf.disable(0, blf.SHADOW)

    @staticmethod
    @bpy.app.handlers.persistent
    def init(*args, **kwargs):
        cls = SkuteStatistics
        cls.rows = None
        cls.active = None

    @staticmethod
    @bpy.app.handlers.persistent
    def update(*args, **kwargs):
        cls = SkuteStatistics
        if cls.active != bpy.context.view_layer.objects.active:
            cls.active = bpy.context.view_layer.objects.active
            active = cls.active
            if active is not None and active.type == 'MESH':
                active = active.parent
            if active is not None and active.type == 'EMPTY' and active.parent is not None and active.parent.type == 'ARMATURE':
                title = active.name.lower()
                for accoutrement in Accoutrement:
                    if accoutrement.is_valid and accoutrement.title.lower() == title:
                        bpy.context.window_manager.there_skute_accoutrements.accoutrement = accoutrement.name
                        break
        if not bpy.context.window_manager.there_skute_stats:
            cls.rows = None
            return
        stats = ExportSkuteBase().get_stats()
        if stats is not None:
            locale.setlocale(locale.LC_ALL, '')
            cls.rows = []
            try:
                active_accoutrement = Accoutrement[bpy.context.window_manager.there_skute_accoutrements.accoutrement]
            except KeyError:
                return
            for accoutrement in stats.accoutrements:
                if accoutrement.name != active_accoutrement.title:
                    continue
                cls.rows.append([[0, accoutrement.name]])
                for lod in accoutrement.lods:
                    cls.rows.append([[10, lod.name]])
                    cls.rows.append([[20, 'Vertices'], [80, '{:n}'.format(lod.vertex_count)]])
                    cls.rows.append([[20, 'Triangles'], [80, '{:n}'.format(lod.triangle_count)]])
                    if lod.shape_count > 0:
                        cls.rows.append([[20, 'Shapes'], [80, '{:n}'.format(lod.shape_count)]])
        else:
            cls.rows = None

    @classmethod
    def poll(cls, context):
        stats = ExportSkuteBase().get_stats(is_quick=True)
        if stats is None:
            return False
        return True

    @staticmethod
    def overlay_options(self, context):
        cls = SkuteStatistics
        if not cls.poll(context):
            return
        layout = self.layout
        layout.label(text='There Skute')
        row = layout.row()
        row.prop(context.window_manager, 'there_skute_stats')
        row.prop(context.window_manager.there_skute_accoutrements, 'accoutrement')


def register_exporter():
    bpy.utils.register_class(ExportSkute)
    bpy.utils.register_class(ExportSkutePreferences)
    bpy.utils.register_class(ThereAccoutrementPropertyGroup)
    bpy.utils.register_class(ThereLODOperator)
    bpy.utils.register_class(ThereShapeKeyOperator)
    bpy.utils.register_class(ThereOutlinerPanel)
    ThereOutlinerPanel.previews = bpy.utils.previews.new()
    ThereOutlinerPanel.previews.load('THERE_SKUTE_HELPER', os.path.abspath(os.path.join(os.path.dirname(__file__), 'helper.png')), 'IMAGE')
    bpy.types.TOPBAR_MT_file_export.append(ExportSkute.handle_menu_export)
    bpy.types.OUTLINER_HT_header.append(ThereOutlinerPanel.handle_outliner_header)
    bpy.app.handlers.load_pre.append(SkuteStatistics.init)
    bpy.app.handlers.load_post.append(SkuteStatistics.update)
    bpy.app.handlers.depsgraph_update_post.append(SkuteStatistics.update)
    SkuteStatistics.handler = bpy.types.SpaceView3D.draw_handler_add(SkuteStatistics.draw, tuple(), 'WINDOW', 'POST_PIXEL')
    bpy.types.VIEW3D_PT_overlay.append(SkuteStatistics.overlay_options)
    bpy.types.WindowManager.there_skute_stats = bpy.props.BoolProperty(name='Statistics', default=True)
    bpy.types.WindowManager.there_skute_optional_shapes = bpy.props.BoolProperty(name='Optional', default=True)
    bpy.types.WindowManager.there_skute_accoutrements = bpy.props.PointerProperty(type=ThereAccoutrementPropertyGroup)


def unregister_exporter():
    bpy.utils.unregister_class(ExportSkute)
    bpy.utils.unregister_class(ExportSkutePreferences)
    bpy.utils.unregister_class(ThereAccoutrementPropertyGroup)
    bpy.utils.unregister_class(ThereLODOperator)
    bpy.utils.unregister_class(ThereShapeKeyOperator)
    bpy.utils.unregister_class(ThereOutlinerPanel)
    bpy.utils.previews.remove(ThereOutlinerPanel.previews)
    bpy.types.TOPBAR_MT_file_export.remove(ExportSkute.handle_menu_export)
    bpy.types.OUTLINER_HT_header.remove(ThereOutlinerPanel.handle_outliner_header)
    bpy.app.handlers.load_pre.remove(SkuteStatistics.init)
    bpy.app.handlers.load_post.remove(SkuteStatistics.update)
    bpy.app.handlers.depsgraph_update_post.remove(SkuteStatistics.update)
    bpy.types.SpaceView3D.draw_handler_remove(SkuteStatistics.handler, 'WINDOW')
    bpy.types.VIEW3D_PT_overlay.remove(SkuteStatistics.overlay_options)
    del bpy.types.WindowManager.there_skute_stats
    del bpy.types.WindowManager.there_skute_optional_shapes
    del bpy.types.WindowManager.there_skute_accoutrements