import os
import re
import bpy
import enum
import math
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
    FEMALE = ('Female', 0.24694)
    MALE = ('Male', 0.24601)

    def __init__(self, title, length):
        self.title = title
        self.skeleton = title[0].lower() + 't0'
        self.length = length


class Accoutrement(enum.Enum):
    HAIR = (
        'Hair', (
            Object(name='hair', title='Hair'),
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
            Object(name='hair_round', title='Round'),
            Object(name='hair_square', title='Square'),
            Object(name='hair_tall', title='Tall'),
            Object(name='hair_wedge', title='Wedge'),
            Object(name='hair_pyramid', title='Pyramid'),
            Object(name='hair_eyesapart', title='EyesApart'),
        ), True
    )
    HEAD = ('Head', (), (), False)
    UPPERBODY = ('UpperBody', (), (), False)

    def __init__(self, title, materials, phenomorphs, is_valid):
        self.title = title
        self.materials = materials
        self.phenomorphs = phenomorphs
        self.is_valid = is_valid

    def get_material_name(self, title):
        title = title.lower()
        for material in self.materials:
            if material.title.lower() == title:
                return material.name
        return None

    def get_phenomorph_name(self, title):
        title = title.lower()
        for phenomorph in self.phenomorphs:
            if phenomorph.title.lower() == title:
                return phenomorph.name
        return None

    def get_phenomorph_title(self, name):
        for phenomorph in self.phenomorphs:
            if phenomorph.name == name:
                return phenomorph.title
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
            if not self.accoutrement.is_valid:
                raise RuntimeError('The accoutrement is not valid.')
            self.skute.skeleton = self.gender.skeleton
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
            self.skute.save()
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
                    name = self.accoutrement.get_phenomorph_name(self.get_basename(bpy_shape.name))
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
            lod.vertices = optimized.vertices
            lod.phenomorphs = optimized.shapes
            lod.vertex_count += len(lod.vertices)
            for mesh in lod.meshes:
                lod.face_count += len(mesh.indices) // 3
            self.skute.lods.append(lod)

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
        self.skute.lods.sort(key=operator.attrgetter('vertex_count'), reverse=True)
        if len(self.skute.lods) != 3:
            raise RuntimeError('"%s" should contain 3 LODs.' % self.accoutrement.title)
        if self.accoutrement == Accoutrement.HAIR:
            spec_distances = (15, 100, 1000)
            spec_vertex_counts = (600, 450, 250)
            spec_face_counts = (1100, 800, 450)
            if self.gender == Gender.FEMALE:
                spec_bounds = ((-0.35, 0.25, -0.35), (0.35, 1.2, 0.35))
            else:
                spec_bounds = ((-0.35, 0.25, -0.35), (0.35, 1.45, 0.35))
        else:
            raise RuntimeError('"%s" is not configured.' % self.accoutrement.title)
        for index, lod in enumerate(self.skute.lods):
            lod.index = index
            lod.distance = spec_distances[index]
            if lod.vertex_count > spec_vertex_counts[index]:
                raise RuntimeError('LOD%s contains too many vertices.' % index)
            if lod.face_count > spec_face_counts[index]:
                raise RuntimeError('LOD%s contains too many faces.' % index)
            if index < 2:
                lod_phenomorphs = [p.name for p in lod.phenomorphs]
                for phenomorph in self.accoutrement.phenomorphs:
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
            for phenomorph in lod.phenomorphs:
                if len(phenomorph.deltas) == 0:
                    continue
                phenomorph.bounds = [
                    [min([z[0].position[i] + z[1].position[i] for z in zip(lod.vertices, phenomorph.deltas)]) for i in range(3)],
                    [max([z[0].position[i] + z[1].position[i] for z in zip(lod.vertices, phenomorph.deltas)]) for i in range(3)],
                ]
                if False in [phenomorph.bounds[0][i] >= spec_bounds[0][i] and phenomorph.bounds[1][i] <= spec_bounds[1][i] for i in range(3)]:
                    name = self.accoutrement.get_phenomorph_title(phenomorph.name)
                    raise RuntimeError('LOD%s with "%s" shape is outside the bounding box.' % (index, name))

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
        for material in self.accoutrement.materials:
            if material.name.startswith('fc_'):
                continue
            if len([m for m in self.skute.materials if m.name == material.name and m.texture is not None]) == 0:
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

    def normalize_weights(self):
        for lod in self.skute.lods:
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
            value = max([abs(v) for o in self.skute.lods for b in o.bounds for v in b])
        except ValueError:
            value = 0.0
        try:
            scale = [s for s in scales if value <= s[1]][0]
        except IndexError:
            raise RuntimeError('The skute is too big to export.')
        self.skute.scale = scale[0]
        for lod in self.skute.lods:
            for vertex in lod.vertices:
                vertex.position = [v / scale[1] for v in vertex.position]
            for phenomorph in lod.phenomorphs:
                for delta in phenomorph.deltas:
                    delta.position = [v / scale[1] for v in delta.position]

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
        layout = self.layout
        layout.label(text='LODs')
        row = layout.row()
        for text in ['LOD0', 'LOD1', 'LOD2']:
            row.operator(ThereLODOperator.bl_idname, text=text, depress=ThereLODOperator.is_depressed(context, text)).id = text
        layout.label(text='Shape Keys')
        text = 'Basis'
        layout.operator(ThereShapeKeyOperator.bl_idname, text=text, depress=ThereShapeKeyOperator.is_depressed(context, text)).id = text
        for phenomorph in Accoutrement.HAIR.phenomorphs:
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
        return [o for o in bpy_armature.children if o.type == 'EMPTY' and ThereOutlinerPanel.get_basename(o.name).lower() in [e.title.lower() for e in Accoutrement]]

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
        for bpy_lod in bpy_lods:
            if bpy_lod.hide_get():
                continue
            if ThereOutlinerPanel.get_basename(bpy_lod.parent.name).lower() != Accoutrement.HAIR.title.lower():
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


def register_exporter():
    bpy.utils.register_class(ExportSkute)
    bpy.utils.register_class(ExportSkutePreferences)
    bpy.utils.register_class(ThereLODOperator)
    bpy.utils.register_class(ThereShapeKeyOperator)
    bpy.utils.register_class(ThereOutlinerPanel)
    ThereOutlinerPanel.previews = bpy.utils.previews.new()
    ThereOutlinerPanel.previews.load('THERE_SKUTE_HELPER', os.path.abspath(os.path.join(os.path.dirname(__file__), 'helper.png')), 'IMAGE')
    bpy.types.TOPBAR_MT_file_export.append(ExportSkute.handle_menu_export)
    bpy.types.OUTLINER_HT_header.append(ThereOutlinerPanel.handle_outliner_header)


def unregister_exporter():
    bpy.utils.unregister_class(ExportSkute)
    bpy.utils.unregister_class(ExportSkutePreferences)
    bpy.utils.unregister_class(ThereLODOperator)
    bpy.utils.unregister_class(ThereShapeKeyOperator)
    bpy.utils.unregister_class(ThereOutlinerPanel)
    bpy.utils.previews.remove(ThereOutlinerPanel.previews)
    bpy.types.TOPBAR_MT_file_export.remove(ExportSkute.handle_menu_export)
    bpy.types.OUTLINER_HT_header.remove(ThereOutlinerPanel.handle_outliner_header)