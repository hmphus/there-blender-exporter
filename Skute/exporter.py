import os
import re
import bpy
import enum
import math
import bmesh
import operator
from bpy_extras.io_utils import ExportHelper
from . import there


class Gender(enum.Enum):
    FEMALE = ('Female', 0.24694)
    MALE = ('Male', 0.24601)

    def __init__(self, title, length):
        self.title = title
        self.skeleton = title[0].lower() + 't0'
        self.length = length


class Accoutrement(enum.Enum):
    HAIR = (
        'Hair', (
            ('fc_hairscrunchy', 'HairScrunchy'),
            ('hair', 'Hair'),
        ), (
            ('hair_round', 'Round'),
            ('hair_square', 'Square'),
            ('hair_tall', 'Tall'),
            ('hair_wedge', 'Wedge'),
            ('hair_pyramid', 'Pyramid'),
            ('hair_eyesapart', 'EyesApart'),
        )
    )

    def __init__(self, title, materials, phenomorphs):
        self.title = title
        self.materials = materials
        self.phenomorphs = phenomorphs

    def get_material_name(self, title):
        title = title.lower()
        for material in self.materials:
            if material[1].lower() == title:
                return material[0]
        return None

    def get_phenomorph_name(self, title):
        title = title.lower()
        for phenomorph in self.phenomorphs:
            if phenomorph[1].lower() == title:
                return phenomorph[0]
        return None


class ExportSkuteBase:
    save_style: bpy.props.BoolProperty(
        name='StyleMaker Settings',
        description='Also save a .style file',
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text='Include')
        box.prop(self, 'save_style')

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
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            self.check(context)
            bpy.ops.object.mode_set(mode='OBJECT')
            assert bpy.context.mode == 'OBJECT', 'Exporting must be done in Object Mode.'
            context.window_manager.progress_begin(0, 100)
            context.window_manager.progress_update(0)
            self.skute = there.Skute(path=self.filepath)
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_armature = [o for o in bpy_scene.objects if o.parent is None and o.type == 'ARMATURE'][0]
            except IndexError:
                raise RuntimeError('The armature object was not found.')
            for gender in Gender:
                if gender.title.lower() == self.get_basename(bpy_armature.name).lower() and gender.length == round(bpy_armature.data.bones['Head'].length, 5):
                    self.gender = gender
                    break
            else:
                raise RuntimeError('The gender could not be determined.')
            try:
                bpy_object = [o for o in bpy_armature.children if o.type == 'EMPTY' and self.get_basename(o.name).lower() == self.accoutrement.title.lower()][0]
            except IndexError:
                raise RuntimeError('The "%s" object was not found.' % self.accoutrement.title)
            self.skute.skeleton = self.gender.skeleton
            self.gather_lods(bpy_armature, bpy_object)
            context.window_manager.progress_update(25)
            self.sort_lods()
            context.window_manager.progress_update(35)
            self.gather_materials()
            context.window_manager.progress_update(45)
            self.skute.save()
            context.window_manager.progress_update(55)
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
            lod = there.LOD()
            matrix_skute = matrix_root_inverted @ bpy_lod.matrix_world
            matrix_rotation = matrix_skute.to_quaternion().to_matrix()
            bpy_lod.data.calc_normals_split()
            bpy_lod.data.calc_tangents()
            components = {
                'positions': [[-v[0], v[2], v[1]] for v in [matrix_skute @ v.co for v in bpy_lod.data.vertices]],
                'indices': [v.vertex_index for v in bpy_lod.data.loops],
                'normals': [[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.normal).normalized() for v in bpy_lod.data.loops]],
                'uvs': [[[d.uv[0], 1.0 - d.uv[1]] for d in e.data] for e in bpy_lod.data.uv_layers][:1],
                'bones': [bone_lookup[g.name] for g in bpy_lod.vertex_groups],
                'weights': [{g.group: g.weight for g in v.groups} for v in bpy_lod.data.vertices],
                'shapes': [],
            }
            optimized = {
                'vertices': [],
                'map': {},
                'shapes': [],
            }
            if bpy_lod.data.shape_keys is not None:
                for bpy_shape in bpy_lod.data.shape_keys.key_blocks[1:]:
                    name = self.accoutrement.get_phenomorph_name(self.get_basename(bpy_shape.name))
                    if name is None:
                        continue
                    shape = {
                        'index': len(components['shapes']),
                        'positions': [[-v[0], v[2], v[1]] for v in [matrix_skute @ v.co for v in bpy_shape.data]],
                    }
                    components['shapes'].append(shape)
                    optimized['shapes'].append(there.Target(name=name))
            bpy_lod.data.free_tangents()
            bpy_lod.data.free_normals_split()
            for index, name in enumerate(bpy_lod.material_slots.keys()):
                if name not in self.skute.materials:
                    self.skute.materials[name] = there.Material(name=name)
                mesh = there.Mesh()
                mesh.material = self.skute.materials[name]
                bpy_polygons = [p for p in bpy_lod.data.polygons if p.material_index == index]
                if len(bpy_polygons) == 0:
                    continue
                mesh.indices = self.optimize_mesh(bpy_polygons=bpy_polygons, components=components, optimized=optimized)
                if len(mesh.indices) == 0:
                    continue
                lod.meshes.append(mesh)
            lod.vertices = optimized['vertices']
            lod.phenomorphs = optimized['shapes']
            lod.vertex_count += len(lod.vertices)
            for mesh in lod.meshes:
                lod.face_count += len(mesh.indices) // 3
            self.skute.lods.append(lod)

    def optimize_mesh(self, bpy_polygons, components, optimized):
        positions = components['positions']
        indices = components['indices']
        normals = components['normals']
        uvs = components['uvs']
        bones = components['bones']
        weights = components['weights']
        shapes = components['shapes']
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
                    optimized_index = optimized['map'].get(key)
                    if optimized_index is None:
                        optimized_index = len(optimized['vertices'])
                        optimized['map'][key] = optimized_index
                        optimized['vertices'].append(there.LOD.Vertex(
                            position=positions[indices[index]],
                            normal=normals[index],
                            uv=uvs[index],
                            bone_indices=[bones[v] for v in weights[indices[index]].keys()],
                            bone_weights=[v for v in weights[indices[index]].values()],
                        ))
                        for shape in shapes:
                            delta_position = [v[0] - v[1] for v in zip(positions[indices[index]], shape['positions'][indices[index]])]
                            if self.is_close(delta_position, (0.0, 0.0, 0.0)):
                                continue
                            optimized['shapes'][shape['index']].deltas.append(there.Target.Delta(
                                index=optimized_index,
                                position=delta_position,
                                normal=None,  # TODO: Add normal
                            ))
                    optimized_indices.append(optimized_index)
        return optimized_indices

    def sort_lods(self):
        self.skute.lods.sort(key=operator.attrgetter('vertex_count'), reverse=True)
        if len(self.skute.lods) != 3:
            raise RuntimeError('%s should contain 3 LODs.' % self.accoutrement.title)
        distances = [15, 100, 1000]
        for index, lod in enumerate(self.skute.lods):
            lod.distance = distances[index]
        # TODO: Check for all phenomorphs except for LOD2

    def gather_materials(self):
        self.skute.materials = list(self.skute.materials.values())
        for index, material in enumerate(self.skute.materials):
            bpy_material = bpy.data.materials[material.name]
            material.name = self.accoutrement.get_material_name(self.get_basename(bpy_material.name))
            if material.name is None:
                raise RuntimeError('Material "%s" is not supported.' % bpy_material.name)
            material.index = index
            if not bpy_material.use_backface_culling:
                raise RuntimeError('Material "%s" does not support two sided rendering.' % bpy_material.name)
            if bpy_material.blend_method != 'OPAQUE':
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
        items = []
        if self.accoutrement == Accoutrement.HAIR:
            if self.gender == Gender.FEMALE:
                item = there.Style.Item(kit_id=2007, skute=self.skute)
                piece = there.Style.Piece(piece_id=2007, texture_id=22007)
                item.pieces.append(piece)
                items.append(item)
                item = there.Style.Item(kit_id=900)
                piece = there.Style.Piece(piece_id=900, texture_id=30900)
                item.pieces.append(piece)
                items.append(item)
                item = there.Style.Item(kit_id=1201)
                piece = there.Style.Piece(piece_id=1201, texture_id=11201)
                item.pieces.append(piece)
                items.append(item)
            if self.gender == Gender.MALE:
                item = there.Style.Item(kit_id=2507, skute=self.skute)
                piece = there.Style.Piece(piece_id=2507, texture_id=12507)
                item.pieces.append(piece)
                items.append(item)
                item = there.Style.Item(kit_id=901)
                piece = there.Style.Piece(piece_id=901, texture_id=30901)
                item.pieces.append(piece)
                items.append(item)
                item = there.Style.Item(kit_id=1708)
                piece = there.Style.Piece(piece_id=1708, texture_id=11708)
                item.pieces.append(piece)
                items.append(item)
        return items

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
    accoutrement = Accoutrement.HAIR


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


def menu_export_handler(self, context):
    self.layout.operator(ExportSkute.bl_idname, text='There Skute (.skute)')


def register_exporter():
    bpy.utils.register_class(ExportSkute)
    bpy.utils.register_class(ExportSkutePreferences)
    bpy.types.TOPBAR_MT_file_export.append(menu_export_handler)


def unregister_exporter():
    bpy.utils.unregister_class(ExportSkute)
    bpy.utils.unregister_class(ExportSkutePreferences)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export_handler)