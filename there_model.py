import os
import re
import math
import copy
import enum
import socket
import operator
import bpy
from bpy_extras.io_utils import ExportHelper


bl_info = {
    'name': 'There Model format',
    'author': 'Brian Gontowski',
    'version': (0, 9, 0),
    'blender': (2, 91, 0),
    'location': 'File > Import-Export',
    'description': 'Import-Export as There Model',
    'support': 'TESTING',
    'category': 'Import-Export',
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
    class Model:
        class Marker:
            def __init__(self, position=-1, mask=0):
                self.position = position
                self.mask = mask

            def clone(self):
                return copy.copy(self)

        def __init__(self):
            self.nodes = []
            self.materials = {}

        def save(self, path):
            self.data = bytearray()
            self.marker = ExportModelBase.Model.Marker()
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
            mask = pow(2, width - 1)
            while mask > 0:
                if self.marker.mask == 0:
                    self.marker.position += 1
                    self.marker.mask = 128
                    while self.marker.position >= len(self.data):
                        self.data += b'\0'
                if value & mask != 0:
                    self.data[self.marker.position] |= self.marker.mask
                self.marker.mask >>= 1
                mask >>= 1

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
            assert value >= 0
            self.store_uint(value, width=width, start=start, end=end, step=step)

        def store_uint(self, value, width=0, start=0, end=0, step=1):
            if width == 0:
                width = int(math.ceil(math.log((float(end) - float(start) + 1.0) / float(step), 2.0)))
            elif end > start:
                step = int((float(end) - float(start)) / (math.pow(2.0, float(width)) - 1.0))
            assert start == 0
            assert step == 1
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
            self.store_int(1, width=32)
            self.store_int(1, width=32)
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
            for name, material in self.materials.items():
                marker3 = self.seek(offset=2)
                self.store_text(name, width=7, end=32)
                self.store_int(2, width=8)
                self.store_int(1, width=8)
                bool_mask = 1
                self.store_uint(bool_mask, width=16)
                float_mask = 0
                self.store_uint(float_mask, width=16)
                color_mask = 0
                self.store_uint(color_mask, width=16)
                map_mask = 0
                if 'color' in material.maps:
                    map_mask |= 1 << 0
                if 'opacity' in material.maps:
                    map_mask |= 1 << 1
                elif 'cutout' in material.maps:
                    map_mask |= 1 << 2
                if 'lighting' in material.maps or 'detail' in material.maps:
                    map_mask |= 1 << 3
                if 'gloss' in material.maps:
                    map_mask |= 1 << 4
                if 'emission' in material.maps:
                    map_mask |= 1 << 5
                if 'normal' in material.maps:
                    map_mask |= 1 << 6
                self.store_uint(map_mask, width=16)
                bool_values = 0
                if material.is_lit:
                    bool_values |= 1 << 0
                if material.is_two_sided:
                    bool_values |= 1 << 1
                if 'lighting' in material.maps:
                    bool_values |= 1 << 6
                self.store_uint(bool_values, width=16)
                if 'color' in material.maps:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'opacity' in material.maps:
                    self.store_text('cm/t555cmxx_devdefault.jpg', width=7, end=48)
                elif 'cutout' in material.maps:
                    self.store_text('cm/t555cmyy_devdefault.png', width=7, end=48)
                if 'lighting' in material.maps:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                elif 'detail' in material.maps:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'gloss' in material.maps:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'emission' in material.maps:
                    self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
                if 'normal' in material.maps:
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

        def store_collisions(self, count=0):
            self.store_uint(1, width=16)
            self.store_uint(count, width=32)
            self.align()

        def store_components(self, count=1):
            for i in range(count):
                self.store_uint(0, width=32)
                self.store_uint(0, width=6)
            self.align()

        def store_families(self, count1=1, count2=1):
            for i1 in range(count1):
                self.store_float(float(self.distances[i1]), width=32, start=0.0, end=100000.0)
                for i2 in range(count2):
                    self.store_uint(0, width=32)
            self.align()

    class Node:
        def __init__(self):
            self.index = None
            self.mesh = None
            self.children = []
            self.vertex_count = 0
            self.parent_index = None

    class Material:
        def __init__(self):
            self.is_lit = True
            self.is_two_sided = False
            self.maps = {
                'color': 'color.png',
                #'opacity': 'opacity.jpg',
                #'cutout': 'cutout.png',
                #'lighting': 'lighting.png',
                #'detail': 'detail.png',
                #'gloss': 'gloss.png',
                #'emission': 'emission.png',
                #'normal': 'normal.png',
            }

    class Mesh:
        pass

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
        context.window_manager.progress_update(0)
        self.model = ExportModelBase.Model()
        bpy_scene = bpy.data.scenes[bpy.context.scene.name]
        bpy_node = [o for o in bpy_scene.objects if o.proxy is None and o.parent is None][0]
        props = dict([[k.lower(), v] for k, v in bpy_node.items() if type(v) == str])
        self.model.distances = [int(props.get('lod%s' % i, [8, 20, 50, 400][i])) for i in range(4)]
        if self.model.distances[0] < 1:
            self.model.distances[0] = 1
        for i in range(1, 4):
            if self.model.distances[i - 1] >= self.model.distances[i]:
                self.model.distances[i] = round(self.model.distances[i - 1] * 1.5)
        self.model.nodes.append(self.gather_nodes(bpy_node))
        assert self.model.nodes[0] is not None
        self.flatten_nodes(self.sort_nodes(self.model.nodes[0].children), parent_index=0)
        self.model.save(self.filepath)
        context.window_manager.progress_end()
        return {'FINISHED'}

    def gather_nodes(self, bpy_node):
        if bpy_node.type not in ['EMPTY', 'MESH']:
            return None
        if False in [s == 1.0 for s in bpy_node.scale]:
            raise RuntimeError('Apply scale to all objects in scene before exporting')
        node = ExportModelBase.Node()
        node.name = re.sub(r'\.\d+$', '', bpy_node.name)
        node.position = [-bpy_node.location[0], bpy_node.location[2], bpy_node.location[1]]
        node.orientation = list(bpy_node.rotation_quaternion.inverted().normalized())
        if bpy_node.type == 'MESH':
            node.mesh = ExportModelBase.Mesh()
            node.mesh.vertices = bpy_node.data.vertices
            node.mesh.uv_layers = bpy_node.data.uv_layers
            node.mesh.vertex_colors = bpy_node.data.vertex_colors
            node.mesh.polygons = bpy_node.data.polygons
            for slot in bpy_node.material_slots.keys():
                self.model.materials[slot] = ExportModelBase.Material()
        for bpy_child in bpy_node.children:
            child = self.gather_nodes(bpy_child)
            if child is not None:
                node.children.append(child)
                if child.mesh is not None:
                    node.vertex_count += len(child.mesh.vertices)
        return node

    def sort_nodes(self, nodes):
        sorted_nodes = [n for n in nodes if n.name.lower() != 'col']
        sorted_nodes.sort(key=operator.attrgetter('vertex_count'), reverse=True)
        for i, node in enumerate(sorted_nodes):
            node.lod_index = i
        return sorted_nodes[0]

    def flatten_nodes(self, node, parent_index):
        node.index = len(self.model.nodes)
        node.parent_index = parent_index
        self.model.nodes.append(node)
        for child in node.children:
            self.flatten_nodes(child, node.index)


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