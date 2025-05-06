import os
import re
import bpy
import blf
import math
import bmesh
import locale
import operator
from bpy_extras.io_utils import ExportHelper
from . import there


class Object:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExportModelBase:
    save_preview: bpy.props.BoolProperty(
        name='Previewer Settings',
        description='Also save a .preview file',
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text='Include')
        box.prop(self, 'save_preview')

    def check(self, context):
        old_filepath = self.filepath
        filename = os.path.basename(self.filepath)
        if filename != '':
            stem, ext = os.path.splitext(filename)
            if stem.startswith('.') and not ext:
                stem, ext = '', stem
            ext_lower = ext.lower()
            if ext_lower != '.model':
                if ext == '':
                    filepath = self.filepath
                else:
                    filepath = self.filepath[:-len(ext)]
                if filepath != '':
                    self.filepath = filepath + '.model'
        return self.filepath != old_filepath

    def invoke(self, context, event):
        try:
            del context.scene['ThereExportSettings']
        except KeyError:
            pass
        preferences = context.preferences.addons[__package__].preferences
        self.save_preview = preferences.save_preview
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            self.check(context)
            bpy.ops.object.mode_set(mode='OBJECT')
            assert bpy.context.mode == 'OBJECT', 'Exporting must be done in Object Mode.'
            context.window_manager.progress_begin(0, 100)
            context.window_manager.progress_update(0)
            self.model = there.Model(path=self.filepath)
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_node = [o for o in bpy_scene.objects if o.parent is None and o.type == 'EMPTY'][0]
            except IndexError:
                raise RuntimeError('The root object was not found.')
            self.gather_node_properties(bpy_node)
            self.model.nodes.append(self.gather_nodes(bpy_node))
            assert self.model.nodes[0] is not None, 'The root object was not found.'
            context.window_manager.progress_update(25)
            self.sort_nodes()
            context.window_manager.progress_update(35)
            self.flatten_nodes()
            context.window_manager.progress_update(45)
            self.gather_materials()
            context.window_manager.progress_update(55)
            self.flatten_meshes()
            context.window_manager.progress_update(65)
            self.scale_meshes()
            context.window_manager.progress_update(75)
            self.model.save()
            context.window_manager.progress_update(85)
            if self.save_preview:
                there.Preview(path=os.path.splitext(self.filepath)[0] + '.preview', model=self.model).save()
            context.window_manager.progress_update(100)
        except (RuntimeError, AssertionError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        context.window_manager.progress_end()
        return {'FINISHED'}

    def gather_node_properties(self, bpy_node):
        props = {k.lower(): v for k, v in bpy_node.items() if type(v) in [str, int, float]}
        distances = [int(props.get('lod%s' % i, [8, 20, 50, 400][i])) for i in range(4)]
        if distances[0] < 1:
            distances[0] = 1
        for i in range(1, 4):
            if distances[i - 1] >= distances[i]:
                distances[i] = round(distances[i - 1] * 1.5)
        self.model.lods = [there.LOD(index=i, distance=d) for i, d in enumerate(distances)]

    def gather_nodes(self, bpy_node, level=0, matrix_root_inverted=None):
        if bpy_node.type not in ['EMPTY', 'MESH']:
            return None
        node = there.Node(name=self.get_basename(bpy_node.name))
        if level == 0:
            matrix_root_inverted = bpy_node.matrix_local.inverted()
            node.position = [0.0, 0.0, 0.0]
            node.orientation = [1.0, 0.0, 0.0, 0.0]
            assert self.is_close(bpy_node.scale, [1.0, 1.0, 1.0]), 'Apply scale to root objects in scene before exporting.'
            assert self.is_close(bpy_node.delta_scale, [1.0, 1.0, 1.0]), 'Apply delta scale to root objects in scene before exporting.'
        else:
            is_collision = (level == 1 and node.name.lower() == 'col')
            matrix_model = matrix_root_inverted @ bpy_node.matrix_world
            matrix_rotation = matrix_model.to_quaternion().to_matrix()
            position = matrix_model.to_translation()
            node.position = [-position[0], position[2], position[1]]
            node.orientation = list(matrix_model.to_quaternion().inverted().normalized())
            if bpy_node.type == 'MESH':
                if is_collision:
                    positions, polygons = self.optimize_collision(bpy_collision_node=bpy_node)
                    positions = [matrix_model @ v for v in positions]
                    collision = there.Collision()
                    collision.vertices = [[-v[0], v[2], v[1]] for v in positions]
                    collision.polygons = polygons
                    collision.center = [
                        (min([v[0] for v in collision.vertices]) + max([v[0] for v in collision.vertices])) / 2.0,
                        (min([v[1] for v in collision.vertices]) + max([v[1] for v in collision.vertices])) / 2.0,
                        (min([v[2] for v in collision.vertices]) + max([v[2] for v in collision.vertices])) / 2.0,
                    ]
                    self.model.collision = collision
                    return None
                if len(bpy_node.material_slots) == 0:
                    self.report({'WARNING'}, 'Object "%s" is missing a material and will not be exported.' % bpy_node.name)
                else:
                    if bpy.app.version < (4, 1, 0):
                        bpy_node.data.calc_normals_split()
                    bpy_node.data.calc_tangents()
                    components = Object(
                        positions=[[-v[0], v[2], v[1]] for v in [matrix_model @ v.co for v in bpy_node.data.vertices]],
                        indices=[v.vertex_index for v in bpy_node.data.loops],
                        normals=[[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.normal).normalized() for v in bpy_node.data.loops]],
                        tangents=[[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.tangent).normalized() for v in bpy_node.data.loops]],
                        bitangents=[[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.bitangent).normalized() for v in bpy_node.data.loops]],
                        uvs=[[[d.uv[0], 1.0 - d.uv[1]] for d in e.data] for e in bpy_node.data.uv_layers][:2],
                        colors=[],
                    )
                    if bpy.app.version < (3, 2, 0):
                        components.colors = [[self.color_as_uint(d.color) for d in e.data] for e in bpy_node.data.vertex_colors][:1]
                    else:
                        for color_attrs in bpy_node.data.color_attributes[:1]:
                            if color_attrs.domain == 'POINT':
                                components.colors.append([self.color_as_uint(color_attrs.data[i].color) for i in components.indices])
                            elif color_attrs.domain == 'CORNER':
                                components.colors.append([self.color_as_uint(d.color) for d in color_attrs.data])
                    if len(components.colors) == 1:
                        for color in components.colors[0]:
                            if color != 0xFFFFFFFF:
                                break
                        else:
                            del components.colors[0]
                    bpy_node.data.free_tangents()
                    if bpy.app.version < (4, 1, 0):
                        bpy_node.data.free_normals_split()
                for index, name in enumerate(bpy_node.material_slots.keys()):
                    if name not in self.model.materials:
                        self.model.materials[name] = there.Material(name=name)
                    mesh = there.Mesh()
                    mesh.material = self.model.materials[name]
                    bpy_polygons = [p for p in bpy_node.data.polygons if p.material_index == index]
                    if len(bpy_polygons) == 0:
                        continue
                    mesh.vertices, mesh.indices = self.optimize_mesh(bpy_polygons=bpy_polygons, components=components)
                    if len(mesh.vertices) == 0 or len(mesh.indices) == 0 or len(mesh.vertices[0].uvs) == 0:
                        continue
                    node.meshes.append(mesh)
            if is_collision:
                return None
        for bpy_child in bpy_node.children:
            child = self.gather_nodes(bpy_child, level=level + 1, matrix_root_inverted=matrix_root_inverted)
            if child is not None:
                node.children.append(child)
                for mesh in child.meshes:
                    node.vertex_count += len(mesh.vertices)
                    node.face_count += len(mesh.indices) // 3
        return node

    def sort_nodes(self):
        self.model.nodes[0].children.sort(key=operator.attrgetter('vertex_count'), reverse=True)

    def flatten_nodes(self, node=None, parent_index=0):
        if node is None:
            node = self.model.nodes[0].children[0]
        node.index = len(self.model.nodes)
        node.parent_index = parent_index
        self.model.nodes.append(node)
        for child in sorted(node.children, key=lambda n: self.get_basename(n.name)):
            self.flatten_nodes(child, node.index)

    def gather_materials(self):
        self.model.materials = list(self.model.materials.values())
        for index, material in enumerate(self.model.materials):
            bpy_material = bpy.data.materials[material.name]
            material.index = index
            material.is_two_sided = not bpy_material.use_backface_culling
            material.is_lit = True
            if not bpy_material.use_nodes:
                raise RuntimeError('Material "%s" is not configured to use nodes.' % material.name)
            bpy_material_output = bpy_material.node_tree.nodes.get('Material Output')
            if bpy_material_output is None:
                raise RuntimeError('Material "%s" does not have a Material Output node.' % material.name)
            bpy_input = bpy_material_output.inputs.get('Surface')
            if not bpy_input.is_linked:
                raise RuntimeError('Material "%s" does not have a linked Surface.' % material.name)
            bpy_link_node = bpy_input.links[0].from_node
            if bpy_link_node.type == 'GROUP':
                bpy_node_tree = bpy_link_node.node_tree
                if bpy_node_tree.name == 'There BSDF':
                    self.gather_there_bsdf(bpy_material, bpy_link_node, material)
                else:
                    raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (material.name, bpy_node_tree.name))
            elif bpy_link_node.type == 'BSDF_PRINCIPLED':
                self.gather_base_principled_bsdf(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'BSDF_DIFFUSE':
                self.gather_base_diffuse_bsdf(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'MIX_SHADER':
                self.gather_mix(bpy_material, bpy_link_node, material)
            else:
                raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (material.name, bpy_link_node.name))
            self.gather_material_properties(bpy_material, material)

    def gather_material_properties(self, bpy_material, material):
        props = {k.lower(): v for k, v in bpy_material.items() if type(v) in [str, int, float]}
        if material.draw_mode != there.Material.DrawMode.DEFAULT:
            try:
                material.draw_mode = there.Material.DrawMode(int(props.get('draw mode')))
            except (TypeError, ValueError):
                pass
        try:
            material.animation_type = there.Material.AnimationType(int(props.get('animation type')))
        except (TypeError, ValueError):
            pass
        if material.animation_type != there.Material.AnimationType.STATIC:
            try:
                material.animation_loop = there.Material.AnimationLoop(int(props.get('animation loop')))
            except (TypeError, ValueError):
                pass
            try:
                material.animation_strips = max(0, min(int(props.get('animation strip count')), 32))
            except (TypeError, ValueError):
                pass
            try:
                material.animation_frames = max(0, min(int(props.get('animation frame count')), 32))
            except (TypeError, ValueError):
                pass
            try:
                material.animation_speed = max(0.0, min(float(props.get('animation speed')), 100.0))
            except (TypeError, ValueError):
                pass
            animation_arg_name_1 = None
            animation_arg_name_2 = None
            if material.animation_type == there.Material.AnimationType.SCROLLING:
                animation_arg_name_1 = 'angle'
                animation_arg_name_2 = 'inset'
            elif material.animation_type == there.Material.AnimationType.ONE_STRIP:
                animation_arg_name_1 = 'strip'
            elif material.animation_type == there.Material.AnimationType.BROWNIAN_MOTION:
                animation_arg_name_1 = 'multiplier'
            elif material.animation_type == there.Material.AnimationType.RANDOM_FRAME:
                animation_arg_name_2 = 'inset'
            elif material.animation_type == there.Material.AnimationType.SWAYING:
                animation_arg_name_1 = 'angle'
                animation_arg_name_2 = 'amplitude'
            elif material.animation_type == there.Material.AnimationType.FRAME_LOOP:
                animation_arg_name_1 = 'start frame'
                animation_arg_name_2 = 'end frame'
            if animation_arg_name_1 is not None:
                animation_arg_names = [animation_arg_name_1, 'argument 1', 'arg 1', 'argument', 'arg']
                try:
                    material.animation_arg_1 = max(-512.0, min(float([props.get('animation %s' % k) for k in animation_arg_names if k in props][0]), 512.0))
                except (TypeError, ValueError, IndexError):
                    pass
            if animation_arg_name_2 is not None:
                animation_arg_names = [animation_arg_name_2, 'argument 2', 'arg 2']
                try:
                    material.animation_arg_2 = max(-512.0, min(float([props.get('animation %s' % k) for k in animation_arg_names if k in props][0]), 512.0))
                except (TypeError, ValueError, IndexError):
                    pass

    def gather_there_bsdf(self, bpy_material, bpy_there_node, material):
        color_texture = self.gather_texture(bpy_there_node, 'Color')
        emission_texture = self.gather_texture(bpy_there_node, 'Emission')
        alpha_texture = self.gather_texture(bpy_there_node, 'Alpha')
        detail_color_texture = self.gather_texture(bpy_there_node, 'Detail Color')
        detail_alpha_texture = self.gather_texture(bpy_there_node, 'Detail Alpha')
        lighting_texture = self.gather_texture(bpy_there_node, 'Lighting')
        normal_texture = self.gather_texture(bpy_there_node, 'Normal')
        gloss_color_texture = self.gather_texture(bpy_there_node, 'Gloss Color')
        gloss_alpha_texture = self.gather_texture(bpy_there_node, 'Gloss Alpha')
        specular_power = max(0.0, min(self.gather_float(bpy_there_node, 'Specular Power', 40.0), 511.0))
        specular_color = self.gather_color(bpy_there_node, 'Specular', 0x202020) & 0xFFFFFF
        environment_color = self.gather_color(bpy_there_node, 'Environment', 0x000000) & 0xFFFFFF
        if lighting_texture is not None:
            if color_texture is not None:
                emission_texture = color_texture
                color_texture = None
                self.report({'WARNING'}, 'Material "%s" is using a Color image for baked lighting.' % material.name)
            if emission_texture is None:
                raise RuntimeError('Material "%s" needs an Emission image.' % material.name)
            if detail_color_texture is not None:
                raise RuntimeError('Material "%s" has both Detail Color and Lighting images.' % material.name)
        if color_texture is not None:
            material.textures[there.Material.Slot.COLOR] = color_texture
            if emission_texture is not None:
                material.textures[there.Material.Slot.EMISSION] = emission_texture
        else:
            color_texture = emission_texture
            emission_texture = None
            if color_texture is not None:
                material.is_lit = False
                material.textures[there.Material.Slot.COLOR] = color_texture
                if lighting_texture is not None:
                    material.textures[there.Material.Slot.LIGHTING] = lighting_texture
            else:
                raise RuntimeError('Material "%s" needs a Color or Emission image.' % material.name)
        if bpy_material.blend_method == 'CLIP':
            if alpha_texture is not None:
                if alpha_texture == color_texture:
                    material.draw_mode = there.Material.DrawMode.CHROMAKEY
                else:
                    material.textures[there.Material.Slot.CUTOUT] = alpha_texture
            else:
                raise RuntimeError('Material "%s" is set to Alpha Clip but is missing an Alpha image.' % material.name)
        elif bpy_material.blend_method == 'BLEND':
            if alpha_texture is not None:
                alpha_ext = os.path.splitext(alpha_texture)[1].lower()
                if bpy.app.version < (4, 2, 0) or alpha_ext == '.jpg':
                    if alpha_texture == color_texture:
                        material.draw_mode = there.Material.DrawMode.BLENDED
                    else:
                        material.textures[there.Material.Slot.OPACITY] = alpha_texture
                elif alpha_ext == '.png':
                    if alpha_texture == color_texture:
                        material.draw_mode = there.Material.DrawMode.CHROMAKEY
                    else:
                        material.textures[there.Material.Slot.CUTOUT] = alpha_texture
            else:
                raise RuntimeError('Material "%s" is set to Alpha Blend but is missing an Alpha image.' % material.name)
        if detail_color_texture is not None:
            if detail_color_texture != detail_alpha_texture:
                self.report({'WARNING'}, 'Material "%s" does not have the same Detail Color and Detail Alpha images.' % material.name)
            material.textures[there.Material.Slot.DETAIL] = detail_color_texture
        if normal_texture is not None:
            material.textures[there.Material.Slot.NORMAL] = normal_texture
        if gloss_color_texture is not None:
            if gloss_color_texture != gloss_alpha_texture:
                self.report({'WARNING'}, 'Material "%s" does not have the same Gloss Color and Gloss Alpha images.' % material.name)
            material.textures[there.Material.Slot.GLOSS] = gloss_color_texture
            material.specular_power = specular_power
            material.specular_color = specular_color
            material.environment_color = environment_color

    def gather_base_principled_bsdf(self, bpy_material, bpy_principled_node, material):
        color_texture = self.gather_texture(bpy_principled_node, 'Base Color')
        emission_texture = self.gather_texture(bpy_principled_node, 'Emission')
        alpha_texture = self.gather_texture(bpy_principled_node, 'Alpha')
        normal_texture = self.gather_texture(bpy_principled_node, 'Normal')
        if color_texture is None and emission_texture is None:
            emission_texture, lighting_texture = self.gather_multiply_textures(bpy_principled_node, 'Emission')
            if emission_texture is None:
                emission_texture, lighting_texture = self.gather_multiply_textures(bpy_principled_node, 'Base Color')
                if emission_texture is not None:
                    self.report({'WARNING'}, 'Material "%s" is using a Base Color image for baked lighting.' % material.name)
        if color_texture is not None:
            material.textures[there.Material.Slot.COLOR] = color_texture
            if emission_texture is not None:
                material.textures[there.Material.Slot.EMISSION] = emission_texture
        else:
            color_texture = emission_texture
            emission_texture = None
            if color_texture is not None:
                material.is_lit = False
                material.textures[there.Material.Slot.COLOR] = color_texture
                if lighting_texture is not None:
                    material.textures[there.Material.Slot.LIGHTING] = lighting_texture
            else:
                raise RuntimeError('Material "%s" needs a Base Color or Emission image.' % material.name)
        if bpy_material.blend_method == 'CLIP':
            if alpha_texture is not None:
                if alpha_texture == color_texture:
                    material.draw_mode = there.Material.DrawMode.CHROMAKEY
                else:
                    material.textures[there.Material.Slot.CUTOUT] = alpha_texture
            else:
                raise RuntimeError('Material "%s" is set to Alpha Clip but is missing an Alpha image.' % material.name)
        elif bpy_material.blend_method == 'BLEND':
            if alpha_texture is not None:
                alpha_ext = os.path.splitext(alpha_texture)[1].lower()
                if bpy.app.version < (4, 2, 0) or alpha_ext == '.jpg':
                    if alpha_texture == color_texture:
                        material.draw_mode = there.Material.DrawMode.BLENDED
                    else:
                        material.textures[there.Material.Slot.OPACITY] = alpha_texture
                elif alpha_ext == '.png':
                    if alpha_texture == color_texture:
                        material.draw_mode = there.Material.DrawMode.CHROMAKEY
                    else:
                        material.textures[there.Material.Slot.CUTOUT] = alpha_texture
            else:
                raise RuntimeError('Material "%s" is set to Alpha Blend but is missing an Alpha image.' % material.name)
        if normal_texture is not None:
            material.textures[there.Material.Slot.NORMAL] = normal_texture

    def gather_base_diffuse_bsdf(self, bpy_material, bpy_diffuse_node, material):
        color_texture = self.gather_texture(bpy_diffuse_node, 'Color')
        normal_texture = self.gather_texture(bpy_diffuse_node, 'Normal')
        if color_texture is not None:
            material.textures[there.Material.Slot.COLOR] = color_texture
        else:
            raise RuntimeError('Material "%s" needs a Color image.' % material.name)
        if normal_texture is not None:
            material.textures[there.Material.Slot.NORMAL] = normal_texture

    def gather_detail_principled_bsdf(self, bpy_material, bpy_principled_node, material):
        detail_texture = self.gather_texture(bpy_principled_node, 'Base Color')
        if detail_texture is not None:
            material.textures[there.Material.Slot.DETAIL] = detail_texture

    def gather_detail_diffuse_bsdf(self, bpy_material, bpy_diffuse_node, material):
        detail_texture = self.gather_texture(bpy_diffuse_node, 'Color')
        if detail_texture is not None:
            material.textures[there.Material.Slot.DETAIL] = detail_texture

    def gather_mix(self, bpy_material, bpy_mix_node, material):
        for index, bpy_input in enumerate(bpy_mix_node.inputs[1:3]):
            assert bpy_input.type == 'SHADER'
            bpy_link_node = bpy_input.links[0].from_node
            if bpy_link_node.type == 'BSDF_PRINCIPLED':
                if index == 0:
                    self.gather_base_principled_bsdf(bpy_material, bpy_link_node, material)
                elif index == 1:
                    self.gather_detail_principled_bsdf(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'BSDF_DIFFUSE':
                if index == 0:
                    self.gather_base_diffuse_bsdf(bpy_material, bpy_link_node, material)
                elif index == 1:
                    self.gather_detail_diffuse_bsdf(bpy_material, bpy_link_node, material)
            else:
                raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (material.name, bpy_link_node.type))

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
        if bpy_link_node.type == 'NORMAL_MAP':
            return self.gather_texture(bpy_link_node, 'Color')
        return None

    def gather_multiply_textures(self, bpy_material_node, name):
        bpy_input = bpy_material_node.inputs.get(name)
        if bpy_input is None:
            return None, None
        if not bpy_input.is_linked:
            return None, None
        if bpy_input.is_multi_input:
            return None, None
        bpy_link_node = bpy_input.links[0].from_node
        if bpy_link_node.type != 'MIX_RGB':
            return None, None
        if bpy_link_node.blend_type != 'MULTIPLY':
            return None, None
        color_texture_1 = self.gather_texture(bpy_link_node, 'Color1')
        if color_texture_1 is None:
            return None, None
        color_texture_2 = self.gather_texture(bpy_link_node, 'Color2')
        if color_texture_2 is None:
            return None, None
        return color_texture_1, color_texture_2

    def gather_float(self, bpy_material_node, name, value):
        bpy_input = bpy_material_node.inputs.get(name)
        if bpy_input is None:
            return value
        if bpy_input.type == 'VALUE':
            value = float(bpy_input.default_value)
        if not bpy_input.is_linked:
            return value
        bpy_link_node = bpy_input.links[0].from_node
        if bpy_link_node.type == 'VALUE':
            value = float(bpy_link_node.outputs[0].default_value)
        return value

    def gather_color(self, bpy_material_node, name, value):
        bpy_input = bpy_material_node.inputs.get(name)
        if bpy_input is None:
            return value
        if bpy_input.type == 'RGBA':
            value = self.color_as_uint(bpy_input.default_value)
        if not bpy_input.is_linked:
            return value
        bpy_link_node = bpy_input.links[0].from_node
        if bpy_link_node.type == 'RGB':
            value = self.color_as_uint(bpy_link_node.outputs[0].default_value)
        return value

    def optimize_collision(self, bpy_collision_node):
        bmesh_collision = bmesh.new()
        bmesh_collision.from_mesh(bpy_collision_node.data)
        bmesh.ops.triangulate(bmesh_collision, faces=bmesh_collision.faces)
        bmesh.ops.dissolve_degenerate(bmesh_collision, edges=bmesh_collision.edges, dist=0.0001)
        bmesh.ops.dissolve_limit(bmesh_collision, verts=bmesh_collision.verts, edges=bmesh_collision.edges, angle_limit=0.0175, delimit={'NORMAL'})
        bmesh.ops.connect_verts_concave(bmesh_collision, faces=bmesh_collision.faces)
        optimized_positions = [v.co for v in bmesh_collision.verts]
        optimized_polygons = [[v.index for v in f.verts] for f in bmesh_collision.faces]
        bmesh_collision.free()
        return (optimized_positions, optimized_polygons)

    def optimize_mesh(self, bpy_polygons, components):
        positions = components.positions
        indices = components.indices
        normals = components.normals
        tangents = components.tangents
        bitangents = components.bitangents
        colors = components.colors
        uvs = components.uvs
        optimized_vertices = []
        optimized_indices = []
        optimized_map = {}
        for bpy_polygon in bpy_polygons:
            triangles = []
            for i in range(2, len(bpy_polygon.loop_indices)):
                triangles.append([bpy_polygon.loop_indices[0], bpy_polygon.loop_indices[i - 1], bpy_polygon.loop_indices[i]])
            for triangle in triangles:
                for index in triangle:
                    key = '%s:%s:%s:%s' % (
                        indices[index],
                        '%.03f:%.03f:%.03f' % (normals[index][0], normals[index][1], normals[index][2]),
                        ':'.join(['%x' % c[index] for c in colors]),
                        ':'.join(['%.03f:%.03f' % (u[index][0], u[index][1]) for u in uvs]),
                    )
                    optimized_index = optimized_map.get(key)
                    if optimized_index is None:
                        optimized_index = len(optimized_vertices)
                        optimized_map[key] = optimized_index
                        optimized_vertices.append(there.Mesh.Vertex(
                            position=positions[indices[index]],
                            normal=normals[index],
                            tangent=tangents[index],
                            bitangent=bitangents[index],
                            colors=[c[index] for c in colors],
                            uvs=[u[index] for u in uvs],
                        ))
                    optimized_indices.append(optimized_index)
        return (optimized_vertices, optimized_indices)

    def flatten_meshes(self, lod=None, node=None, node_index=None):
        if lod is None or node is None or node_index is None:
            for lod in self.model.lods:
                if lod.index >= len(self.model.nodes[0].children):
                    self.model.lods = self.model.lods[:lod.index]
                    break
                self.flatten_meshes(lod=lod, node=self.model.nodes[0].children[lod.index], node_index=1)
            return
        assert node_index < len(self.model.nodes), 'LOD%s has too many nodes.' % lod.index
        if node_index > 1:
            if node.name != self.model.nodes[node_index].name:
                self.report({'WARNING'}, 'LOD%s has a different name than LOD0 and may not animate correctly.' % lod.index)
        for mesh in node.meshes:
            mesh.node = self.model.nodes[node_index]
            lod.meshes.append(mesh)
        del node.meshes
        node_index += 1
        for child in sorted(node.children, key=lambda n: self.get_basename(n.name)):
            node_index = self.flatten_meshes(lod=lod, node=child, node_index=node_index)
        return node_index

    def scale_meshes(self):
        scales = [(s, pow(2.0, s - 32)) for s in range(33, 64)]
        for lod in self.model.lods:
            try:
                value = max([max([max([abs(p) for p in v.position]) for v in m.vertices]) for m in lod.meshes])
            except ValueError:
                value = 0.0
            try:
                scale = [s for s in scales if value <= s[1]][0]
            except IndexError:
                raise RuntimeError('The model is too big to export.')
            lod.scale = scale[0]
            for mesh in lod.meshes:
                for vertex in mesh.vertices:
                    vertex.position = [v / scale[1] for v in vertex.position]

    def get_stats(self, is_quick=False):
        try:
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_node = [o for o in bpy_scene.objects if o.parent is None and o.type == 'EMPTY'][0]
            except IndexError:
                return None
            props = {k.lower(): v for k, v in bpy_node.items() if type(v) in [str, int, float]}
            if 'lod0' not in props:
                return None
            stats = Object(
                collision=None,
                lods=[],
            )
            if is_quick:
                return stats
            if bpy.context.mode != 'OBJECT':
                return None
            for bpy_object in bpy_node.children:
                if bpy_object.type not in ['EMPTY', 'MESH']:
                    continue
                if bpy_object.name.lower() == 'col':
                    positions, polygons = self.optimize_collision(bpy_object)
                    stats.collision = Object(
                        vertex_count=len(positions),
                        polygon_count=len(polygons),
                    )
                else:
                    stats_lod = Object(
                        name=bpy_object.name,
                        vertex_count=0,
                        triangle_count=0,
                    )
                    self.get_node_stats(stats_lod, bpy_object)
                    stats.lods.append(stats_lod)
        except (RuntimeError, AssertionError) as error:
            return None
        return stats

    def get_node_stats(self, stats, bpy_node):
        if bpy_node.type == 'MESH':
            if bpy.app.version < (4, 1, 0):
                bpy_node.data.calc_normals_split()
            bpy_node.data.calc_tangents()
            positions = [list(v) for v in [v.co for v in bpy_node.data.vertices]]
            indices = [v.vertex_index for v in bpy_node.data.loops]
            normals = [list(v) for v in [v.normal for v in bpy_node.data.loops]]
            tangents = [list(v) for v in [v.tangent for v in bpy_node.data.loops]]
            bitangents = [list(v) for v in [v.bitangent for v in bpy_node.data.loops]]
            uvs = [[list(d.uv) for d in e.data] for e in bpy_node.data.uv_layers][:2]
            colors = []
            if bpy.app.version < (3, 2, 0):
                colors = [[self.color_as_uint(d.color) for d in e.data] for e in bpy_node.data.vertex_colors][:1]
            else:
                for color_attrs in bpy_node.data.color_attributes[:1]:
                    if color_attrs.domain == 'POINT':
                        components.colors.append([self.color_as_uint(color_attrs.data[i].color) for i in components.indices])
                    elif color_attrs.domain == 'CORNER':
                        colors.append([self.color_as_uint(d.color) for d in color_attrs.data])
            if len(colors) == 1:
                for color in colors[0]:
                    if color != 0xFFFFFFFF:
                        break
                else:
                    del colors[0]
            bpy_node.data.free_tangents()
            if bpy.app.version < (4, 1, 0):
                bpy_node.data.free_normals_split()
            if len(positions) > 0 and len(indices) > 0 and len(uvs) > 0:
                for index, name in enumerate(bpy_node.material_slots.keys()):
                    bpy_polygons = [p for p in bpy_node.data.polygons if p.material_index == index]
                    if len(bpy_polygons) == 0:
                        continue
                    optimized_keys = set()
                    for bpy_polygon in bpy_polygons:
                        for index in range(2, len(bpy_polygon.loop_indices)):
                            triangle = [bpy_polygon.loop_indices[0], bpy_polygon.loop_indices[index - 1], bpy_polygon.loop_indices[index]]
                            stats.triangle_count += 1
                            for index in triangle:
                                key = '%s:%s:%s:%s' % (
                                    indices[index],
                                    '%.03f:%.03f:%.03f' % (normals[index][0], normals[index][1], normals[index][2]),
                                    ':'.join(['%x' % c[index] for c in colors]),
                                    ':'.join(['%.03f:%.03f' % (u[index][0], u[index][1]) for u in uvs]),
                                )
                                if key not in optimized_keys:
                                    optimized_keys.add(key)
                                    stats.vertex_count += 1
        for bpy_object in bpy_node.children:
            if bpy_object.type not in ['EMPTY', 'MESH']:
                continue
            self.get_node_stats(stats, bpy_object)

    @staticmethod
    def get_basename(name):
        return re.sub(r'\.\d+$', '', name)

    @staticmethod
    def is_close(a, b):
        return False not in [math.isclose(a[i], b[i], abs_tol=0.00001) for i in range(len(a))]

    @staticmethod
    def color_as_uint(color):
        return (int(color[3] * 255.0) << 24) | (int(color[0] * 255.0) << 16) | (int(color[1] * 255.0) << 8) | int(color[2] * 255.0)


class ExportModel(bpy.types.Operator, ExportModelBase, ExportHelper):
    """Export scene as There Model file"""
    bl_idname = 'export_scene.model'
    bl_label = 'Export There Model'
    filename_ext = '.model'
    filter_glob: bpy.props.StringProperty(
        default='*.model',
        options={'HIDDEN'},
    )

    @staticmethod
    def handle_menu_export(self, context):
        self.layout.operator(ExportModel.bl_idname, text='There Model (.model)')


class ExportModelPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    save_preview: bpy.props.BoolProperty(
        name='Export Previewer Settings',
        description='Save a .preview file when exporting a model',
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'save_preview')


class ModelStatistics:
    handler = None
    rows = None

    @classmethod
    def draw(cls):
        if cls.rows is None:
            return
        blf.size(0, 11.0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.shadow(0, 6, 0.0, 0.0, 0.0, 1.0)
        blf.enable(0, blf.SHADOW)
        for y, row in enumerate(reversed(cls.rows)):
            for column in row:
                blf.position(0, 10.0 + column[0], 17.0 + y * 17.0, 0.0)
                blf.draw(0, column[1])
        blf.disable(0, blf.SHADOW)

    @classmethod
    @bpy.app.handlers.persistent
    def init(cls, *args, **kwargs):
        cls.rows = None

    @classmethod
    @bpy.app.handlers.persistent
    def update(cls, *args, **kwargs):
        if not bpy.context.window_manager.show_there_model_stats:
            cls.rows = None
            return
        stats = ExportModelBase().get_stats()
        if stats is not None:
            locale.setlocale(locale.LC_ALL, '')
            cls.rows = []
            collision = stats.collision
            if collision is not None:
                cls.rows.append([[10, 'Collision']])
                cls.rows.append([[20, 'Vertices'], [80, '{:n}'.format(collision.vertex_count)]])
                cls.rows.append([[20, 'Polygons'], [80, '{:n}'.format(collision.polygon_count)]])
            for lod in stats.lods:
                cls.rows.append([[10, lod.name]])
                cls.rows.append([[20, 'Vertices'], [80, '{:n}'.format(lod.vertex_count)]])
                cls.rows.append([[20, 'Triangles'], [80, '{:n}'.format(lod.triangle_count)]])
        else:
            cls.rows = None

    @classmethod
    def poll(cls, context):
        stats = ExportModelBase().get_stats(is_quick=True)
        if stats is None:
            return False
        return True

    @staticmethod
    def overlay_options(self, context):
        cls = ModelStatistics
        if not cls.poll(context):
            return
        layout = self.layout
        layout.label(text='There Model')
        layout.prop(context.window_manager, 'show_there_model_stats')


def register_exporter():
    bpy.utils.register_class(ExportModel)
    bpy.utils.register_class(ExportModelPreferences)
    bpy.types.TOPBAR_MT_file_export.append(ExportModel.handle_menu_export)
    bpy.app.handlers.load_pre.append(ModelStatistics.init)
    bpy.app.handlers.load_post.append(ModelStatistics.update)
    bpy.app.handlers.depsgraph_update_post.append(ModelStatistics.update)
    ModelStatistics.handler = bpy.types.SpaceView3D.draw_handler_add(ModelStatistics.draw, tuple(), 'WINDOW', 'POST_PIXEL')
    bpy.types.VIEW3D_PT_overlay.append(ModelStatistics.overlay_options)
    bpy.types.WindowManager.show_there_model_stats = bpy.props.BoolProperty(name='Statistics', default=True)


def unregister_exporter():
    bpy.utils.unregister_class(ExportModel)
    bpy.utils.unregister_class(ExportModelPreferences)
    bpy.types.TOPBAR_MT_file_export.remove(ExportModel.handle_menu_export)
    bpy.app.handlers.load_pre.remove(ModelStatistics.init)
    bpy.app.handlers.load_post.remove(ModelStatistics.update)
    bpy.app.handlers.depsgraph_update_post.remove(ModelStatistics.update)
    bpy.types.SpaceView3D.draw_handler_remove(ModelStatistics.handler, 'WINDOW')
    bpy.types.VIEW3D_PT_overlay.remove(ModelStatistics.overlay_options)
    del bpy.types.WindowManager.show_there_model_stats