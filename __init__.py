import os
import re
import math
import copy
import socket
import operator
import bpy
from bpy_extras.io_utils import ExportHelper


bl_info = {
    'name': 'There Model format',
    'author': 'Brian Gontowski',
    'version': (1, 0, 0),
    'blender': (2, 93, 0),
    'location': 'File > Import-Export',
    'description': 'Import-Export as There Model',
    'support': 'COMMUNITY',
    'category': 'Import-Export',
    'doc_url': 'https://github.com/hmphus/there-blender-exporter/wiki',
    'tracker_url': 'https://github.com/hmphus/there-blender-exporter/issues',
}


def get_version_string():
    return str(bl_info['version'][0]) + '.' + str(bl_info['version'][1]) + '.' + str(bl_info['version'][2])


def reload_package(module_dict_main):
    import importlib
    from pathlib import Path

    def reload_package_recursive(current_dir, module_dict):
        for path in current_dir.iterdir():
            if '__init__' in str(path) or path.stem not in module_dict:
                continue
            if path.is_file() and path.suffix == '.py':
                importlib.reload(module_dict[path.stem])
            elif path.is_dir():
                reload_package_recursive(path, module_dict[path.stem].__dict__)

    reload_package_recursive(Path(__file__).parent, module_dict_main)


if 'bpy' in locals():
    reload_package(locals())


class ExportModelBase:
    class ThereModel:
        class Marker:
            def __init__(self, position=-1, mask=0):
                self.position = position
                self.mask = mask

            def clone(self):
                return copy.copy(self)

        def __init__(self):
            self.lods = None
            self.nodes = []
            self.materials = {}
            self.collision = None

        def save(self, path):
            assert type(self.materials) == list, 'The materials were not flattened.'
            self.data = bytearray()
            self.marker = ExportModelBase.ThereModel.Marker()
            self.store_header()
            with open(path, 'wb') as file:
                file.write(self.data)

        def align(self):
            self.marker.mask = 0

        def seek(self, marker=None, offset=None):
            origin_marker = self.marker.clone()
            if marker is not None:
                self.marker.position = marker.position
                self.marker.mask = marker.mask
            if offset is not None:
                self.marker.position += offset
            return origin_marker

        def store(self, value, width):
            mask = 1 << (width - 1)
            while mask > 0:
                if self.marker.mask == 0:
                    self.marker.position += 1
                    self.marker.mask = 128
                    while self.marker.position >= len(self.data):
                        self.data += b'\0'
                if value & mask != 0:
                    self.data[self.marker.position] |= self.marker.mask
                    value ^= mask
                self.marker.mask >>= 1
                mask >>= 1
            if value != 0:
                raise OverflowError('An overflow has occured while storing the data.')

        def store_text(self, value, width, length=None, end=None):
            if length is None:
                length = len(value)
                self.store_int(length, end=end)
            for i in range(length):
                if i < len(value):
                    self.store(ord(value[i]), width=width)
                else:
                    self.store(0, width=width)
            self.align()

        def store_bool(self, value):
            value = self.store_uint(1 if value else 0, width=1)

        def store_int(self, value, width=0, start=0, end=0, step=1):
            assert value >= 0, 'Only positive integers are supported.'
            self.store_uint(value, width=width, start=start, end=end, step=step)

        def store_uint(self, value, width=0, start=0, end=0, step=1):
            if width == 0:
                width = int(math.ceil(math.log((float(end) - float(start) + 1.0) / float(step), 2.0)))
            elif end > start:
                step = int((float(end) - float(start)) / (math.pow(2.0, float(width)) - 1.0))
            assert start == 0, 'The start argument is invalid.'
            assert step == 1, 'The step argument is invalid.'
            self.store(value, width=width)

        def store_float(self, value, width=0, start=0.0, end=0.0, step=1.0):
            if width == 0:
                width = int(math.ceil(math.log((end - start) / step, 2.0)))
            elif end > start:
                step = (end - start) / (math.pow(2.0, float(width)) - 1.0)
            self.store(round((value - start) / step), width=width)

        def store_header(self):
            self.store_text('SOM ', width=8, length=4)
            self.store_int(10, width=32)
            self.align()
            self.store_bool(False)
            self.store_bool(False)
            self.store_int(2, end=8)
            self.align()
            self.store_float(7.0, width=32, start=-2000000.0, end=2000000.0)
            self.store_float(10.0, width=32, start=-2000000.0, end=2000000.0)
            self.store_float(-3.0, width=32, start=-2000000.0, end=2000000.0)
            self.store_float(-10.0, width=32, start=-2000000.0, end=2000000.0)
            self.store_float(1.0, width=32, start=-2000000.0, end=2000000.0)
            self.store_float(0.0, width=32, start=-2000000.0, end=2000000.0)
            self.align()
            self.store_uint(0xff606060, width=32)
            self.store_uint(0xff606060, width=32)
            self.align()
            self.store_int(1, width=32)
            self.store_int(1, width=32)
            self.align()
            self.store_uint(0xff808080, width=32)
            self.store_int(0, width=32)
            self.store_float(135.0, width=32, start=0.0, end=360.0)
            self.store_int(1, width=32)
            self.store_uint(0xff000000, width=32)
            self.store_int(255, width=8)
            self.store_float(0.0, width=32, start=0.0, end=360.0)
            self.store_float(270.0, width=32, start=0.0, end=360.0)
            self.store_int(1, width=32)
            self.store_uint(0xff000000, width=32)
            self.store_int(255, width=8)
            self.store_float(0.0, width=32, start=0.0, end=360.0)
            self.store_int(1, end=32)
            self.align()
            self.store_uint(0xffff0000, width=32)
            self.align()
            self.store_uint(0xffff0080, width=32)
            self.align()
            self.store_int(len(self.materials), end=256)
            self.store_int(len(self.lods), width=32)
            self.store_int(len(self.lods), width=32)
            self.store_int(1, width=32)
            self.store_int(len(self.nodes) - 1, width=32)
            self.align()
            self.store_materials()
            self.store_nodes()
            self.store_collisions()
            self.store_components()
            self.store_families()

        def store_materials(self):
            marker1 = self.seek(offset=2)
            for material in self.materials:
                marker3 = self.seek(offset=2)
                self.store_text(material.name, width=7, end=32)
                self.store_int(2, width=8)
                self.store_int(1, width=8)
                bool_mask = 1
                self.store_uint(bool_mask, width=16)
                float_mask = 0
                self.store_uint(float_mask, width=16)
                color_mask = 0
                self.store_uint(color_mask, width=16)
                map_mask = 0
                if 'color' in material.textures:
                    map_mask |= 1 << 0
                if 'opacity' in material.textures:
                    map_mask |= 1 << 1
                elif 'cutout' in material.textures:
                    map_mask |= 1 << 2
                if 'lighting' in material.textures or 'detail' in material.textures:
                    map_mask |= 1 << 3
                if 'gloss' in material.textures:
                    map_mask |= 1 << 4
                if 'emission' in material.textures:
                    map_mask |= 1 << 5
                if 'normal' in material.textures:
                    map_mask |= 1 << 6
                self.store_uint(map_mask, width=16)
                bool_values = 0
                if material.is_lit:
                    bool_values |= 1 << 0
                if material.is_two_sided:
                    bool_values |= 1 << 1
                if 'lighting' in material.textures:
                    bool_values |= 1 << 6
                self.store_uint(bool_values, width=16)
                if 'color' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'opacity' in material.textures:
                    self.store_text('cm/t555cmxx_devdefault.jpg', width=7, end=48)
                elif 'cutout' in material.textures:
                    self.store_text('cm/t555cmyy_devdefault.png', width=7, end=48)
                if 'lighting' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                elif 'detail' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'gloss' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'emission' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'normal' in material.textures:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                int_mask = 0
                self.store_uint(int_mask, width=16)
                self.align()
                marker4 = self.seek(marker=marker3)
                self.store_uint(socket.htons(marker4.position - marker3.position), width=16)
                self.seek(marker=marker4)
            marker2 = self.seek(marker=marker1)
            self.store_uint(socket.htons(marker2.position - marker1.position), width=16)
            self.seek(marker=marker2)

        def store_nodes(self):
            for node in self.nodes[1:]:
                self.store_float(node.position[0], width=32, start=-2000000.0, end=2000000.0)
                self.store_float(node.position[1], width=32, start=-2000000.0, end=2000000.0)
                self.store_float(node.position[2], width=32, start=-2000000.0, end=2000000.0)
                self.align()
            for node in self.nodes:
                self.store_int(3001 if node.parent_index is None else node.parent_index, width=16)
                self.align()
            for node in self.nodes:
                values = [v * v for v in node.orientation[1:]]
                index = values.index(max(values))
                sign = -1.0 if node.orientation[index + 1] < 0.0 else 1.0
                values = [node.orientation[i] * sign for i in range(4) if i != index + 1]
                self.store_int(index, width=2)
                self.store_float(values[0], width=24, start=-1.0, end=1.0)
                self.store_float(values[1], width=23, start=-1.0, end=1.0)
                self.store_float(values[2], width=23, start=-1.0, end=1.0)
                self.align()
            for node in self.nodes:
                self.store_text(node.name, width=7, end=40)
                self.align()

        def store_collisions(self):
            self.store_uint(1, width=16)
            if self.collision is None:
                self.store_uint(0, width=32)
            else:
                self.store_uint(1, width=32)
                self.store_uint(len(self.collision.vertices), width=16)
                self.store_uint(len(self.collision.polygons), width=16)
                self.store_uint(sum([len(p) for p in self.collision.polygons]), width=32)
                self.store_uint(0, width=32)
                self.store_float(self.collision.center[0], width=32, start=-2000000.0, end=2000000.0)
                self.store_float(self.collision.center[1], width=32, start=-2000000.0, end=2000000.0)
                self.store_float(self.collision.center[2], width=32, start=-2000000.0, end=2000000.0)
                for vertex in self.collision.vertices:
                    self.store_float(vertex[0], width=32, start=-2000000.0, end=2000000.0)
                    self.store_float(vertex[1], width=32, start=-2000000.0, end=2000000.0)
                    self.store_float(vertex[2], width=32, start=-2000000.0, end=2000000.0)
                for polygon in self.collision.polygons:
                    self.store_uint(len(polygon), width=8)
                    for index in polygon:
                        self.store_uint(index, width=16)
            self.align()

        def store_components(self):
            for lod in self.lods:
                self.store_uint(len(lod.meshes), width=32)
                self.store_uint(lod.scale, width=6)
                self.align()
                for mesh in lod.meshes:
                    vertex_format = 0
                    vertex_functions = []
                    vertex_format |= 1 << 0
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=14, start=-1.024, end=1.023) for v in vertex.position])
                    vertex_format |= 1 << 3
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=6, start=-1.0, end=1.0) for v in vertex.normal])
                    if len(mesh.vertices[0].colors) >= 1:
                        vertex_format |= 1 << 4
                        vertex_functions.append(lambda vertex: self.store_uint(vertex.colors[0], width=32))
                    if len(mesh.vertices[0].uvs) >= 1:
                        vertex_format |= 1 << 5
                        vertex_functions.append(lambda vertex: [self.store_float(v, width=18, start=-256.0, end=255.998046875) for v in vertex.uvs[0]])
                    if len(mesh.vertices[0].uvs) >= 2:
                        vertex_format |= 1 << 6
                        vertex_functions.append(lambda vertex: [self.store_float(v, width=18, start=-256.0, end=255.998046875) for v in vertex.uvs[1]])
                    index_width = max(4, int(math.ceil(math.log(len(mesh.indices), 2))))
                    assert index_width < 16, 'The mesh is too complicated to export.'
                    self.store_uint(1, end=8)
                    self.store_uint(8, end=8)
                    self.store_uint(mesh.node.index - 1, width=8)
                    for i in range(1, 8):
                        self.store_uint(0, width=8)
                    self.store_uint(mesh.material.index, end=256)
                    self.store_uint(vertex_format, width=32)
                    self.store_uint(vertex_format, width=32)
                    self.store_uint(7, width=32)
                    self.store_uint(len(mesh.vertices), width=32)
                    self.store_uint(len(mesh.indices), width=32)
                    self.align()
                    for vertex in mesh.vertices:
                        for vertex_function in vertex_functions:
                            vertex_function(vertex)
                        self.align()
                    self.store_uint(len(mesh.indices), width=index_width)
                    self.align()
                    for index in mesh.indices:
                        self.store_uint(index, width=index_width)
                    self.align()

        def store_families(self):
            for lod in self.lods:
                self.store_float(float(lod.distance), width=32, start=0.0, end=100000.0)
                self.store_uint(lod.index, width=32)
                self.align()

    class ThereNode:
        def __init__(self, name):
            self.name = name
            self.index = None
            self.meshes = []
            self.children = []
            self.vertex_count = 0
            self.face_count = 0
            self.parent_index = None

    class ThereMaterial:
        def __init__(self, name):
            self.name = name
            self.index = None
            self.is_lit = True
            self.is_two_sided = False
            self.textures = {}

    class ThereCollision:
        pass

    class ThereLOD:
        def __init__(self, index, distance):
            self.index = index
            self.distance = distance
            self.meshes = []
            self.scale = None

    class ThereMesh:
        class Vertex:
            def __init__(self, position, normal, colors, uvs):
                self.position = position
                self.normal = normal
                self.colors = colors
                self.uvs = uvs

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
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        self.check(context)
        context.window_manager.progress_begin(0, 100)
        try:
            context.window_manager.progress_update(0)
            self.model = ExportModelBase.ThereModel()
            try:
                bpy_scene = bpy.data.scenes[bpy.context.scene.name]
                bpy_node = [o for o in bpy_scene.objects if o.proxy is None and o.parent is None and o.type == 'EMPTY'][0]
            except IndexError:
                raise RuntimeError('The root note was not found.')
            self.gather_properties(bpy_node)
            self.model.nodes.append(self.gather_nodes(bpy_node))
            assert self.model.nodes[0] is not None, 'The root note was not found.'
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
            self.model.save(self.filepath)
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
        self.model.lods = [ExportModelBase.ThereLOD(index=i, distance=d) for i, d in enumerate(distances)]

    def gather_nodes(self, bpy_node, level=0):
        if bpy_node.type not in ['EMPTY', 'MESH']:
            return None
        node = ExportModelBase.ThereNode(name=re.sub(r'\.\d+$', '', bpy_node.name))
        is_collision = (level == 1 and node.name.lower() == 'col')
        node.position = [-bpy_node.location[0], bpy_node.location[2], bpy_node.location[1]]
        if bpy_node.rotation_mode == 'QUATERNION':
            node.orientation = list(bpy_node.rotation_quaternion.inverted().normalized())
        else:
            node.orientation = list(bpy_node.rotation_euler.to_quaternion().inverted().normalized())
        assert self.is_close(bpy_node.scale, [1.0, 1.0, 1.0]), 'Apply scale to all objects in scene before exporting.'
        if level == 0:
            assert self.is_close(node.position, [0.0, 0.0, 0.0]), 'Apply position to root node in scene before exporting.'
            assert self.is_close(node.orientation, [1.0, 0.0, 0.0, 0.0]), 'Apply rotation to root node in scene before exporting.'
        if bpy_node.type == 'MESH':
            if is_collision:
                positions = [bpy_node.matrix_world @ v.co for v in bpy_node.data.vertices]
                collision = ExportModelBase.ThereCollision()
                collision.vertices = [[-v[0], v[2], v[1]] for v in positions]
                collision.polygons = self.optimize_collision(bpy_polygons=bpy_node.data.polygons)
                collision.center = [
                    (min([v[0] for v in collision.vertices]) + max([v[0] for v in collision.vertices])) / 2.0,
                    (min([v[1] for v in collision.vertices]) + max([v[1] for v in collision.vertices])) / 2.0,
                    (min([v[2] for v in collision.vertices]) + max([v[2] for v in collision.vertices])) / 2.0,
                ]
                self.model.collision = collision
                return None
            positions = [bpy_node.matrix_world @ v.co for v in bpy_node.data.vertices]
            positions = [[-v[0], v[2], v[1]] for v in positions]
            normals = [list(v.normal) for v in bpy_node.data.vertices]
            indices = [i.vertex_index for i in bpy_node.data.loops]
            colors = [[self.color_as_uint(d.color) for d in e.data] for e in bpy_node.data.vertex_colors][:1]
            uvs = [[[d.uv[0], 1.0 - d.uv[1]] for d in e.data] for e in bpy_node.data.uv_layers][:2]
            for index, name in enumerate(bpy_node.material_slots.keys()):
                if name not in self.model.materials:
                    self.model.materials[name] = ExportModelBase.ThereMaterial(name=name)
                mesh = ExportModelBase.ThereMesh()
                mesh.material = self.model.materials[name]
                bpy_polygons = [p for p in bpy_node.data.polygons if p.material_index == index]
                if len(bpy_polygons) == 0:
                    continue
                mesh.vertices, mesh.indices = self.optimize_mesh(bpy_polygons=bpy_polygons, positions=positions, normals=normals, indices=indices, colors=colors, uvs=uvs, name=name)
                if len(mesh.vertices) == 0 or len(mesh.indices) == 0 or len(mesh.vertices[0].uvs) == 0:
                    continue
                node.meshes.append(mesh)
        if is_collision:
            return None
        for bpy_child in bpy_node.children:
            child = self.gather_nodes(bpy_child, level=level + 1)
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
        self.model.materials = list(self.model.materials.values())
        for index, material in enumerate(self.model.materials):
            bpy_material = bpy.data.materials[material.name]
            material.index = index
            material.is_two_sided = not bpy_material.use_backface_culling
            material.is_lit = True
            color_texture = self.gather_texture(bpy_material, 'Base Color')
            emission_texture = self.gather_texture(bpy_material, 'Emission')
            alpha_texture = self.gather_texture(bpy_material, 'Alpha')
            if color_texture is not None:
                material.textures['color'] = color_texture
                if emission_texture is not None:
                    material.textures['emission'] = emission_texture
            else:
                if emission_texture is not None:
                    material.textures['color'] = emission_texture
                    material.is_lit = False
                else:
                    raise RuntimeError('Add a base color or emission image to the "%s" material.' % material.name)
            if bpy_material.blend_method == 'CLIP':
                if alpha_texture is not None:
                    material.textures['cutout'] = alpha_texture
                else:
                    raise RuntimeError('Add an alpha image to the "%s" material or set its blend mode to opaque.' % material.name)
            elif bpy_material.blend_method == 'BLEND':
                if alpha_texture is not None:
                    material.textures['opacity'] = alpha_texture
                else:
                    raise RuntimeError('Add an alpha image to the "%s" material or set its blend mode to opaque.' % material.name)
            # TODO: Add lighting, detail, gloss, and normal textures

    def gather_texture(self, bpy_material, name):
        if not bpy_material.use_nodes:
            return None
        bpy_node = bpy_material.node_tree.nodes.get('Principled BSDF')
        if bpy_node is None:
            return None
        bpy_input = bpy_node.inputs.get(name)
        if bpy_input is None:
            return None
        if not bpy_input.is_linked:
            return None
        if bpy_input.is_multi_input:
            return None
        bpy_link = bpy_input.links[0]
        if bpy_link.from_node.type != 'TEX_IMAGE':
            return None
        if bpy_link.from_node.image is None:
            return None
        return bpy_link.from_node.image.name

    def optimize_collision(self, bpy_polygons):
        assert True not in [len(p.vertices) > 4 for p in bpy_polygons], 'The collision mesh must be triangulated.'
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

    def optimize_mesh(self, bpy_polygons, positions, normals, indices, colors, uvs, name):
        assert True not in [len(p.loop_indices) > 4 for p in bpy_polygons], 'The mesh for %s must be triangulated.' % name
        optimized_vertices = []
        optimized_indices = []
        optimized_map = {}
        for bpy_polygon in bpy_polygons:
            triangles = [[bpy_polygon.loop_indices[0], bpy_polygon.loop_indices[1], bpy_polygon.loop_indices[2]]]
            if len(bpy_polygon.loop_indices) == 4:
                triangles.append([bpy_polygon.loop_indices[2], bpy_polygon.loop_indices[1], bpy_polygon.loop_indices[3]])
            for triangle in triangles:
                for index in triangle:
                    key = '%s:%s:%s' % (
                        indices[index],
                        ':'.join(['%x' % c[index] for c in colors]),
                        ':'.join(['%.03f:%.03f' % (u[index][0], u[index][1]) for u in uvs]),
                    )
                    optimized_index = optimized_map.get(key)
                    if optimized_index is None:
                        optimized_index = len(optimized_vertices)
                        optimized_map[key] = optimized_index
                        optimized_vertices.append(ExportModelBase.ThereMesh.Vertex(
                            position=positions[indices[index]],
                            normal=normals[indices[index]],
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
            value = max([max([max([abs(p) for p in v.position]) for v in m.vertices]) for m in lod.meshes])
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


classes = (
    ExportModel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)