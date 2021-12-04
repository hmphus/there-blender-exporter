import os
import math
import copy
import enum
import socket


class Node:
    def __init__(self, name):
        self.name = name
        self.index = None
        self.meshes = []
        self.children = []
        self.vertex_count = 0
        self.face_count = 0
        self.parent_index = None
        self.position = None
        self.orientation = None


class Material:
    class Slot(enum.Enum):
        COLOR = (0, 'colormap')
        OPACITY = (1, 'opacitymap')
        CUTOUT = (2, 'cutoutmap')
        LIGHTING = (3, 'detailmap')
        DETAIL = (4, 'detailmap')
        GLOSS = (5, 'reflectivitymap')
        EMISSION = (6, 'emissionmap')
        NORMAL = (7, 'normalmap')

        def __init__(self, index, map_name):
            self.index = index
            self.slot_name = self.name.lower()
            self.map_name = map_name

    class DrawMode(enum.IntEnum):
        DEFAULT = 0
        OPAQUE = 1
        BLENDED = 2
        FILTER = 3
        CHROMAKEY = 4
        ADDITIVE = 5

    def __init__(self, name):
        self.name = name
        self.index = None
        self.is_lit = True
        self.is_two_sided = False
        self.draw_mode = Material.DrawMode.DEFAULT
        self.specular_power = None
        self.specular_color = None
        self.environment_color = None
        self.textures = {}


class Collision:
    def __init__(self):
        self.vertices = None
        self.polygons = None
        self.center = None


class LOD:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance
        self.meshes = []
        self.scale = None


class Mesh:
    class Vertex:
        def __init__(self, position, normal, tangent, bitangent, colors, uvs):
            self.position = position
            self.normal = normal
            self.tangent = tangent
            self.bitangent = bitangent
            self.colors = colors
            self.uvs = uvs

    def __init__(self):
        self.vertices = None
        self.indices = None
        self.material = None
        self.node = None


class Model:
    class Marker:
        def __init__(self, position=-1, mask=0):
            self.position = position
            self.mask = mask

        def clone(self):
            return copy.copy(self)

    def __init__(self, path):
        self.path = path
        self.lods = None
        self.nodes = []
        self.materials = {}
        self.collision = None

    def save(self):
        assert isinstance(self.materials, list), 'The materials were not flattened.'
        self.data = bytearray()
        self.marker = Model.Marker()
        self.store_header()
        with open(self.path, 'wb') as file:
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
            if material.specular_power is not None:
                float_mask |= 1 << 5
            self.store_uint(float_mask, width=16)
            color_mask = 0
            if material.specular_color is not None:
                color_mask |= 1 << 2
            if material.environment_color is not None:
                color_mask |= 1 << 4
            self.store_uint(color_mask, width=16)
            map_mask = 0
            if Material.Slot.COLOR in material.textures:
                map_mask |= 1 << 0
            if Material.Slot.OPACITY in material.textures:
                map_mask |= 1 << 1
            elif Material.Slot.CUTOUT in material.textures:
                map_mask |= 1 << 2
            if Material.Slot.LIGHTING in material.textures or Material.Slot.DETAIL in material.textures:
                map_mask |= 1 << 3
            if Material.Slot.GLOSS in material.textures:
                map_mask |= 1 << 4
            if Material.Slot.EMISSION in material.textures:
                map_mask |= 1 << 5
            if Material.Slot.NORMAL in material.textures:
                map_mask |= 1 << 6
            self.store_uint(map_mask, width=16)
            bool_values = 0
            if material.is_lit:
                bool_values |= 1 << 0
            if material.is_two_sided:
                bool_values |= 1 << 1
            if Material.Slot.LIGHTING in material.textures:
                bool_values |= 1 << 6
            self.store_uint(bool_values, width=16)
            if material.specular_power is not None:
                self.store_uint(15488, width=16)  # TODO: Implement this
            if material.specular_color is not None:
                self.store_uint(material.specular_color, width=24)
            if material.environment_color is not None:
                self.store_uint(material.environment_color, width=24)
            if Material.Slot.COLOR in material.textures:
                self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
            if Material.Slot.OPACITY in material.textures:
                self.store_text('cm/t555cmxx_devdefault.jpg', width=7, end=48)
            elif Material.Slot.CUTOUT in material.textures:
                self.store_text('cm/t555cmyy_devdefault.png', width=7, end=48)
            if Material.Slot.LIGHTING in material.textures:
                self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
            elif Material.Slot.DETAIL in material.textures:
                self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
            if Material.Slot.GLOSS in material.textures:
                self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
            if Material.Slot.EMISSION in material.textures:
                self.store_text('cm/t555cm_devdefault.jpg', width=7, end=48)
            if Material.Slot.NORMAL in material.textures:
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
                store_tangents = Material.Slot.NORMAL in mesh.material.textures
                vertex_format = 0
                vertex_functions = []
                vertex_format |= 1 << 0
                vertex_functions.append(lambda vertex: [self.store_float(v, width=14, start=-1.024, end=1.023) for v in vertex.position])
                vertex_format |= 1 << 3
                vertex_functions.append(lambda vertex: [self.store_float(v, width=6, start=-1.0, end=1.0) for v in vertex.normal])
                if len(mesh.vertices[0].colors) >= 1 and not store_tangents:
                    vertex_format |= 1 << 4
                    vertex_functions.append(lambda vertex: self.store_uint(vertex.colors[0], width=32))
                if len(mesh.vertices[0].uvs) >= 1:
                    vertex_format |= 1 << 5
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=18, start=-256.0, end=255.998046875) for v in vertex.uvs[0]])
                if len(mesh.vertices[0].uvs) >= 2 and not store_tangents:
                    vertex_format |= 1 << 6
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=18, start=-256.0, end=255.998046875) for v in vertex.uvs[1]])
                if store_tangents:
                    vertex_format |= 1 << 7
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=6, start=-1.0, end=1.0) for v in vertex.tangent])
                    vertex_format |= 1 << 8
                    vertex_functions.append(lambda vertex: [self.store_float(v, width=6, start=-1.0, end=1.0) for v in vertex.bitangent])
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


class Preview:
    class Xml:
        def __init__(self, tag, value=None):
            self.tag = tag
            self.value = value
            self.children = []

        def append(self, xml):
            self.children.append(xml)

        def to_text(self, level=0):
            space = (' ' * level)
            text = '%s<%s>' % (space, self.tag)
            if self.value is not None:
                if isinstance(self.value, str):
                    text += self.value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elif isinstance(self.value, int) or isinstance(self.value, bool):
                    text += '%d' % self.value
                elif isinstance(self.value, float):
                    text += '%.03f' % self.value
            else:
                text += '\n'
                for child in self.children:
                    text += child.to_text(level + 2)
                text += space
            text += '</%s>\n' % self.tag
            return text

    def __init__(self, path, model):
        self.path = path
        self.model = model

    def save(self):
        assert isinstance(self.model.materials, list), 'The materials were not flattened.'
        xml = Preview.Xml('Preview')
        xml.append(Preview.Xml('Model', os.path.normpath(self.model.path)))
        for material in self.model.materials:
            xml_material = Preview.Xml('Material')
            xml_material.append(Preview.Xml('Name', material.name))
            for texture_slot in Material.Slot:
                texture_path = material.textures.get(texture_slot)
                if texture_path is not None:
                    texture_path = os.path.splitdrive(os.path.normpath(os.path.join(os.path.dirname(self.model.path), texture_path)))[1]
                    xml_material.append(Preview.Xml(texture_slot.map_name, texture_path))
            xml_material.append(Preview.Xml('lit', material.is_lit))
            xml_material.append(Preview.Xml('twosided', material.is_two_sided))
            xml_material.append(Preview.Xml('lightmap', Material.Slot.LIGHTING in material.textures))
            xml_material.append(Preview.Xml('drawmode', material.draw_mode.value))
            if material.specular_power is not None:
                xml_material.append(Preview.Xml('specularpower', material.specular_power))
            if material.specular_color is not None:
                xml_material.append(Preview.Xml('specular', '#%06x' % material.specular_color))
            if material.environment_color is not None:
                xml_material.append(Preview.Xml('environment', '#%06x' % material.environment_color))
            xml.append(xml_material)
        with open(self.path, 'w', encoding='utf-8') as file:
            file.write(xml.to_text())