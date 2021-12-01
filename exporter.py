import os
import re
import bpy
import math
import operator
from bpy_extras.io_utils import ExportHelper
from . import there


class ExportModelBase:
    scene_key = "ThereExportSettings"

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
        settings = context.scene.get(self.scene_key)
        if settings:
            try:
                for key, value in settings.items():
                    setattr(self, key, value)
            except (AttributeError, TypeError):
                del context.scene[self.scene_key]
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            self.check(context)
            assert bpy.context.mode != 'EDIT_MESH', 'Exporting while in Edit Mode is not supported.'
            context.scene[self.scene_key] = {
                'save_preview': self.save_preview,
            }
            context.window_manager.progress_begin(0, 100)
            context.window_manager.progress_update(0)
            self.model = there.Model(path=self.filepath)
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_node = [o for o in bpy_scene.objects if o.proxy is None and o.parent is None and o.type == 'EMPTY'][0]
            except IndexError:
                raise RuntimeError('The root object was not found.')
            self.gather_properties(bpy_node)
            self.model.nodes.append(self.gather_nodes(bpy_node))
            assert self.model.nodes[0] is not None, 'The root object was not found.'
            context.window_manager.progress_update(25)
            self.sort_nodes()
            context.window_manager.progress_update(35)
            self.flatten_nodes()
            context.window_manager.progress_update(45)
            self.gather_materials()
            context.window_manager.progress_update(50)
            self.flatten_meshes()
            context.window_manager.progress_update(60)
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

    def gather_properties(self, bpy_node):
        props = dict([[k.lower(), v] for k, v in bpy_node.items() if type(v) in [str, int, float]])
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
        node = there.Node(name=re.sub(r'\.\d+$', '', bpy_node.name))
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
                    positions = [matrix_model @ v.co for v in bpy_node.data.vertices]
                    collision = there.Collision()
                    collision.vertices = [[-v[0], v[2], v[1]] for v in positions]
                    collision.polygons = self.optimize_collision(bpy_polygons=bpy_node.data.polygons)
                    collision.center = [
                        (min([v[0] for v in collision.vertices]) + max([v[0] for v in collision.vertices])) / 2.0,
                        (min([v[1] for v in collision.vertices]) + max([v[1] for v in collision.vertices])) / 2.0,
                        (min([v[2] for v in collision.vertices]) + max([v[2] for v in collision.vertices])) / 2.0,
                    ]
                    self.model.collision = collision
                    return None
                bpy_node.data.calc_normals_split()
                bpy_node.data.calc_tangents()
                components = {
                    'positions': [[-v[0], v[2], v[1]] for v in [matrix_model @ v.co for v in bpy_node.data.vertices]],
                    'indices': [v.vertex_index for v in bpy_node.data.loops],
                    'normals': [[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.normal).normalized() for v in bpy_node.data.loops]],
                    'tangents': [[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.tangent).normalized() for v in bpy_node.data.loops]],
                    'bitangents': [[-v[0], v[2], v[1]] for v in [(matrix_rotation @ v.bitangent).normalized() for v in bpy_node.data.loops]],
                    'colors': [[self.color_as_uint(d.color) for d in e.data] for e in bpy_node.data.vertex_colors][:1],
                    'uvs': [[[d.uv[0], 1.0 - d.uv[1]] for d in e.data] for e in bpy_node.data.uv_layers][:2],
                }
                bpy_node.data.free_tangents()
                bpy_node.data.free_normals_split()
                for index, name in enumerate(bpy_node.material_slots.keys()):
                    if name not in self.model.materials:
                        self.model.materials[name] = there.Material(name=name)
                    mesh = there.Mesh()
                    mesh.material = self.model.materials[name]
                    bpy_polygons = [p for p in bpy_node.data.polygons if p.material_index == index]
                    if len(bpy_polygons) == 0:
                        continue
                    mesh.vertices, mesh.indices = self.optimize_mesh(bpy_polygons=bpy_polygons, name=name, components=components)
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
        for child in node.children:
            self.flatten_nodes(child, node.index)

    def gather_materials(self):
        # TODO: Add gloss texture
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
            if bpy_link_node.type == 'BSDF_PRINCIPLED':
                self.gather_base_principled(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'BSDF_DIFFUSE':
                self.gather_base_diffuse(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'MIX_SHADER':
                self.gather_mix(bpy_material, bpy_link_node, material)
            else:
                raise RuntimeError('Material "%s" configured with an unsupported %s node.' % (material.name, bpy_link_node.type))

    def gather_base_principled(self, bpy_material, bpy_principled_node, material):
        color_texture = self.gather_texture(bpy_principled_node, 'Base Color')
        emission_texture = self.gather_texture(bpy_principled_node, 'Emission')
        alpha_texture = self.gather_texture(bpy_principled_node, 'Alpha')
        normal_texture = self.gather_texture(bpy_principled_node, 'Normal')
        if color_texture is None:
            color_texture, lighting_texture = self.gather_multiply_textures(bpy_principled_node, 'Base Color')
        else:
            lighting_texture = None
        if color_texture is not None:
            material.textures[there.Material.Slot.COLOR] = color_texture
            if lighting_texture is not None:
                material.textures[there.Material.Slot.LIGHTING] = lighting_texture
                material.is_lit = False
            if emission_texture is not None:
                material.textures[there.Material.Slot.EMISSION] = emission_texture
        else:
            color_texture = emission_texture
            emission_texture = None
            if color_texture is not None:
                material.textures[there.Material.Slot.COLOR] = color_texture
                material.is_lit = False
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
                if alpha_texture == color_texture:
                    material.draw_mode = there.Material.DrawMode.BLENDED
                else:
                    material.textures[there.Material.Slot.OPACITY] = alpha_texture
            else:
                raise RuntimeError('Material "%s" is set to Alpha Blend but is missing an Alpha image.' % material.name)
        if normal_texture is not None:
            material.textures[there.Material.Slot.NORMAL] = normal_texture

    def gather_base_diffuse(self, bpy_material, bpy_diffuse_node, material):
        color_texture = self.gather_texture(bpy_diffuse_node, 'Color')
        normal_texture = self.gather_texture(bpy_diffuse_node, 'Normal')
        if color_texture is None:
            color_texture, lighting_texture = self.gather_multiply_textures(bpy_diffuse_node, 'Color')
        else:
            lighting_texture = None
        if color_texture is not None:
            material.textures[there.Material.Slot.COLOR] = color_texture
            if lighting_texture is not None:
                material.textures[there.Material.Slot.LIGHTING] = lighting_texture
                material.is_lit = False
        else:
            raise RuntimeError('Material "%s" needs a Color image.' % material.name)
        if normal_texture is not None:
            material.textures[there.Material.Slot.NORMAL] = normal_texture

    def gather_detail_principled(self, bpy_material, bpy_principled_node, material):
        detail_texture = self.gather_texture(bpy_principled_node, 'Base Color')
        if detail_texture is not None:
            material.textures[there.Material.Slot.DETAIL] = detail_texture

    def gather_detail_diffuse(self, bpy_material, bpy_diffuse_node, material):
        detail_texture = self.gather_texture(bpy_diffuse_node, 'Color')
        if detail_texture is not None:
            material.textures[there.Material.Slot.DETAIL] = detail_texture

    def gather_mix(self, bpy_material, bpy_mix_node, material):
        for index, bpy_input in enumerate(bpy_mix_node.inputs[1:3]):
            assert bpy_input.type == 'SHADER'
            bpy_link_node = bpy_input.links[0].from_node
            if bpy_link_node.type == 'BSDF_PRINCIPLED':
                if index == 0:
                    self.gather_base_principled(bpy_material, bpy_link_node, material)
                elif index == 1:
                    self.gather_detail_principled(bpy_material, bpy_link_node, material)
            elif bpy_link_node.type == 'BSDF_DIFFUSE':
                if index == 0:
                    self.gather_base_diffuse(bpy_material, bpy_link_node, material)
                elif index == 1:
                    self.gather_detail_diffuse(bpy_material, bpy_link_node, material)
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

    def optimize_collision(self, bpy_polygons):
        normal_groups = []
        normal_grouped = [False] * len(bpy_polygons)
        for i1 in range(len(bpy_polygons)):
            if normal_grouped[i1]:
                continue
            normal_group = [i1]
            normal_grouped[i1] = True
            for i2 in range(i1 + 1, len(bpy_polygons)):
                if math.isclose(bpy_polygons[i1].normal.dot(bpy_polygons[i2].normal), 1.0, abs_tol=0.001):
                    normal_group.append(i2)
                    normal_grouped[i2] = True
            normal_groups.append(normal_group)
        polygon_groups = []
        for normal_group in normal_groups:
            polygon_grouped = [False] * len(normal_group)
            for i1 in range(len(normal_group)):
                if polygon_grouped[i1]:
                    continue
                vertices1 = list(bpy_polygons[normal_group[i1]].vertices)
                polygon_grouped[i1] = True
                for i2 in range(i1 + 1, len(normal_group)):
                    vertices2 = list(bpy_polygons[normal_group[i2]].vertices)
                    if len(vertices2) > 3:
                        continue
                    vertices3 = [v for v in vertices2 if v not in vertices1]
                    if len(vertices3) == 1:
                        vertices2.remove(vertices3[0])
                        for i3 in range(len(vertices1)):
                            if vertices1[0] in vertices2 and vertices1[-1] in vertices2:
                                vertices1 = vertices1 + vertices3
                                polygon_grouped[i2] = True
                                break
                            if len(vertices1) > 3:
                                break
                            vertices1.append(vertices1.pop(0))
                polygon_groups.append(vertices1)
        return polygon_groups

    def optimize_mesh(self, bpy_polygons, name, components):
        positions = components['positions']
        indices = components['indices']
        normals = components['normals']
        tangents = components['tangents']
        bitangents = components['bitangents']
        colors = components['colors']
        uvs = components['uvs']
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
        for child in node.children:
            node_index = self.flatten_meshes(lod=lod, node=child, node_index=node_index)
        return node_index

    def scale_meshes(self):
        scales = [(s, pow(2.0, s - 32)) for s in range(33, 64)]
        for lod in self.model.lods:
            try:
                value = max([max([max([abs(p) for p in v.position]) for v in m.vertices]) for m in lod.meshes])
            except ValueError:
                value = 0
            try:
                scale = [s for s in scales if value <= s[1]][0]
            except IndexError:
                raise RuntimeError('The model is too big to export.')
            lod.scale = scale[0]
            for mesh in lod.meshes:
                for vertex in mesh.vertices:
                    vertex.position = [n / scale[1] for n in vertex.position]

    @staticmethod
    def is_close(a, b):
        return False not in [math.isclose(a[i], b[i], abs_tol=0.00001) for i in range(len(a))]

    @staticmethod
    def color_as_uint(color):
        return (int(color[3] * 255.0) << 24) | (int(color[2] * 255.0) << 16) | (int(color[1] * 255.0) << 8) | int(color[0] * 255.0)


class ExportModel(bpy.types.Operator, ExportModelBase, ExportHelper):
    """Export scene as There Model file"""
    bl_idname = 'export_scene.model'
    bl_label = 'Export There Model'
    filename_ext = '.model'
    filter_glob: bpy.props.StringProperty(default='*.model', options={'HIDDEN'})


def menu_func_export(self, context):
    self.layout.operator(ExportModel.bl_idname, text='There Model (.model)')


def register_exporter():
    bpy.utils.register_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister_exporter():
    bpy.utils.unregister_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)