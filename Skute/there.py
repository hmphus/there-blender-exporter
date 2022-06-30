import os
import math
import copy


class Material:
    def __init__(self, name):
        self.index = None
        self.name = name
        self.texture = None


class LOD:
    class Vertex:
        def __init__(self, position, normal, uv, bone_indices, bone_weights):
            self.position = position
            self.normal = normal
            self.uv = uv
            self.bone_indices = bone_indices
            self.bone_weights = bone_weights

    def __init__(self):
        self.distance = None
        self.vertices = []
        self.welds = []
        self.meshes = []
        self.phenomorphs = []
        self.siblings = []
        self.morphs = []
        self.vertex_count = 0
        self.face_count = 0
        self.bounds = None


class Mesh:
    def __init__(self):
        self.indices = []
        self.material = None


class Target:
    class Delta:
        def __init__(self, index, position, normal):
            self.index = index
            self.position = position
            self.normal = normal

    def __init__(self, name):
        self.name = name
        self.deltas = []
        self.phenomorphs = []
        self.bounds = None


class Skute:
    class Marker:
        def __init__(self, position=-1, mask=0):
            self.position = position
            self.mask = mask

        def clone(self):
            return copy.copy(self)

    class Block:
        def __init__(self, key):
            self.key = key

        def __call__(self, func):
            key = self.key

            def wrapper(self, *args):
                self.store_text(key, width=8, length=2)
                marker1 = self.seek(offset=3)
                func(self, *args)
                self.align()
                marker2 = self.seek(marker=marker1)
                self.store_uint(0, width=4)
                self.store_uint(marker2.position - marker1.position - 3, width=20)
                self.seek(marker=marker2)
            return wrapper

    def __init__(self, path):
        self.path = path
        self.skeleton = None
        self.lods = []
        self.scale = None
        self.materials = {}

    def save(self):
        assert isinstance(self.materials, list), 'The materials were not flattened.'
        self.data = bytearray()
        self.marker = Skute.Marker()
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

    def store_half_float(self, value):
        value = self.float_to_uint32(value)
        sign = (value & 0x80000000) >> 31
        exponent = (value & 0x7F800000) >> 23
        mantissa = (value & 0x7FFFFF) >> 12
        exponent -= 125
        if exponent < 0:
            mantissa = (mantissa | 0x800) >> (1 - exponent)
            exponent = 0
        self.store_uint(sign, width=1)
        self.store_uint(exponent, width=4)
        self.store_uint(mantissa, width=11)

    def store_header(self):
        self.store_text('SKUT', width=8, length=4)
        self.store_uint(4, width=8)
        self.align()
        self.store_file_info()
        self.store_bone_block()
        self.store_material_block()
        self.store_lod_block()

    @Block('FI')
    def store_file_info(self):
        self.store_uint(self.scale, width=6)
        self.store_uint(len(self.lods), end=15)
        self.store_uint(0, width=3)

    @Block('BL')
    def store_bone_block(self):
        self.store_bone_info()

    @Block('BI')
    def store_bone_info(self):
        self.store_text(self.skeleton, width=7, end=40)
        self.store_uint(33, end=255)
        self.store_uint(0, end=255)

    @Block('MT')
    def store_material_block(self):
        self.store_material_info()
        if len(self.materials) > 0:
            self.store_materials()

    @Block('MI')
    def store_material_info(self):
        self.store_uint(len(self.materials), end=1023)

    @Block('MD')
    def store_materials(self):
        for material in self.materials:
            self.store_text(material.name, width=7, end=32)

    @Block('LD')
    def store_lod_block(self):
        for lod in self.lods:
            self.store_lod_info(lod)
            self.store_archetypes(lod)

    @Block('LI')
    def store_lod_info(self, lod):
        self.store_uint(1, end=255)
        self.store_float(float(lod.distance), width=16, start=0.0, end=2047.0)

    @Block('AR')
    def store_archetypes(self, lod):
        vertex_buffers = [lod.vertices[i:i + 4095] for i in range(0, len(lod.vertices), 4095)]
        assert len(vertex_buffers) == 1, 'The mesh is too complicated to export.'
        index_width = max(4, int(math.ceil(math.log(len(lod.vertices) - 1, 2))))
        assert index_width < 16, 'The mesh is too complicated to export.'
        self.store_archetype_info(lod, len(vertex_buffers), index_width)
        if len(vertex_buffers) > 0:
            self.store_vertex_buffer_block(lod, vertex_buffers, index_width)
        if len(lod.phenomorphs) > 0:
            self.store_phenomorphs(lod.phenomorphs, index_width)
        if len(lod.siblings) > 0:
            self.store_siblings(lod.siblings)
        if len(lod.morphs) > 0:
            self.store_morphs(lod.morphs)

    @Block('AI')
    def store_archetype_info(self, lod, vertex_buffer_count, index_width):
        self.store_uint(vertex_buffer_count, end=15)
        self.store_uint(len(lod.phenomorphs), end=4095)
        self.store_uint(len(lod.siblings), end=4095)
        self.store_uint(len(lod.morphs), end=4095)
        self.store_uint(index_width, end=15)

    @Block('VB')
    def store_vertex_buffer_block(self, lod, vertex_buffers, index_width):
        for vertices in vertex_buffers:
            self.store_vertex_buffer_info(lod, vertices)
            self.store_vertex_positions(vertices)
            self.store_vertex_normals(vertices)
            self.store_vertex_uvs(vertices)
            self.store_vertex_weights(vertices, 0)
            self.store_vertex_weights(vertices, 1)
            self.store_index_buffers(lod.meshes, index_width)
            self.store_weldable_list(lod.welds, index_width)

    @Block('VI')
    def store_vertex_buffer_info(self, lod, vertices):
        self.store_uint(len(vertices), end=65535)
        self.store_uint(len(lod.meshes), end=4095)

    @Block('PS')
    def store_vertex_positions(self, vertices):
        for vertex in vertices:
            for value in vertex.position:
                self.store_float(value, width=16, start=-1.024, end=1.023)

    @Block('NR')
    def store_vertex_normals(self, vertices):
        for vertex in vertices:
            for value in vertex.normal:
                self.store_float(value, width=8, start=-1.024, end=1.023)

    @Block('TX')
    def store_vertex_uvs(self, vertices):
        for vertex in vertices:
            for value in vertex.uv:
                self.store_float(value, width=16, start=-64.0, end=63.9921875)

    @Block('WI')
    def store_vertex_weights(self, vertices, index):
        for vertex in vertices:
            self.store_float(vertex.bone_weights[index], width=8, start=0.0, end=1.0)
        self.align()
        self.store_uint(len(vertices), end=16383)
        self.align()
        for vertex in vertices:
            self.store_uint(vertex.bone_indices[index] - 1, width=6)

    @Block('IB')
    def store_index_buffers(self, meshes, index_width):
        for mesh in meshes:
            self.store_index_buffer_info(mesh)
            self.store_index_values(mesh, index_width)

    @Block('II')
    def store_index_buffer_info(self, mesh):
        self.store_uint(len(mesh.indices), end=16383)
        self.store_int(mesh.material.index, end=1023)

    @Block('IX')
    def store_index_values(self, mesh, index_width):
        self.store_int(len(mesh.indices), end=16383)
        self.align()
        for value in mesh.indices:
            self.store_int(value, width=index_width)

    @Block('WL')
    def store_weldable_list(self, welds, index_width):
        self.store_weldable_list_info(welds)
        if len(welds) > 0:
            self.store_weldable_vertices(welds, index_width)
            self.store_weldable_matches(welds, index_width)
            self.store_weldable_references(welds)

    @Block('WE')
    def store_weldable_list_info(self, welds):
        self.store_int(len(welds), end=65535)

    @Block('WV')
    def store_weldable_vertices(self, welds, index_width):
        raise NotImplementedError('Welds are not implemented.')

    @Block('WM')
    def store_weldable_matches(self, welds, index_width):
        raise NotImplementedError('Welds are not implemented.')

    @Block('WN')
    def store_weldable_references(self, welds):
        raise NotImplementedError('Welds are not implemented.')

    @Block('PM')
    def store_phenomorphs(self, phenomorphs, index_width):
        for target in phenomorphs:
            self.store_delta_vertex_buffer_info(target)
            if len(target.deltas) > 0:
                self.store_delta_positions(target.deltas)
                self.store_delta_normals(target.deltas)
            self.store_delta_index_buffer_info(target)
            if len(target.deltas) > 0:
                self.store_delta_indexes(target.deltas, index_width)

    @Block('SB')
    def store_siblings(self):
        raise NotImplementedError('Siblings are not implemented.')

    @Block('SI')
    def store_sibling_info(self):
        raise NotImplementedError('Siblings are not implemented.')

    @Block('MP')
    def store_morphs(self):
        raise NotImplementedError('Morphs are not implemented.')

    @Block('DV')
    def store_delta_vertex_buffer_info(self, target):
        self.store_text(target.name, width=7, end=32)
        self.store_int(len(target.deltas), end=65535)

    @Block('PD')
    def store_delta_positions(self, deltas):
        for delta in deltas:
            for value in delta.position:
                self.store_float(value, width=16, start=-1.024, end=1.023)

    @Block('ND')
    def store_delta_normals(self, deltas):
        for delta in deltas:
            for value in delta.normal:
                self.store_float(value, width=8, start=-2.048, end=2.047)

    @Block('DI')
    def store_delta_index_buffer_info(self, target):
        self.store_int(len(target.deltas), end=16383)

    @Block('DX')
    def store_delta_indexes(self, deltas, index_width):
        self.store_int(len(deltas), end=16383)
        self.align()
        for delta in deltas:
            self.store_int(delta.index, width=index_width)


class Style:
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

    class Item:
        def __init__(self, kit_id, skute=None):
            self.kit_id = kit_id
            self.pieces = []
            self.skute = skute

    class Piece:
        def __init__(self, piece_id, texture_id=None, colors=None):
            self.piece_id = piece_id
            self.texture_id = texture_id
            self.colors = colors if colors is not None else ['202020', '404040', '808080']

    def __init__(self, path, items):
        self.path = path
        self.items = items

    def save(self):
        xml = Style.Xml('Stylemaker')
        for item in self.items:
            xml_item = Style.Xml('Item')
            xml_item.append(Style.Xml('KitId', item.kit_id))
            xml_item.append(Style.Xml('NumPieces', len(item.pieces)))
            for piece in item.pieces:
                xml_piece = Style.Xml('Piece')
                xml_piece.append(Style.Xml('PieceIndex', piece.piece_id))
                xml_piece.append(Style.Xml('TextureIndex', piece.texture_id))
                for index, color in enumerate(piece.colors):
                    xml_piece.append(Style.Xml('Color%s' % (index + 1), color))
                xml_item.append(xml_piece)
            if item.skute is not None:
                assert isinstance(item.skute.materials, list), 'The materials were not flattened.'
                index = 0
                for material in item.skute.materials:
                    if material.texture is not None:
                        xml_texture = Style.Xml('Texture')
                        xml_texture.append(Style.Xml('Number', (index + 1)))
                        xml_texture.append(Style.Xml('File', os.path.normpath(os.path.join(os.path.dirname(item.skute.path), material.texture))))
                        xml_item.append(xml_texture)
                        index += 1
                xml_item.append(Style.Xml('CustomSkute', item.skute.path))
            xml.append(xml_item)
        with open(self.path, 'w', encoding='utf-8') as file:
            file.write(xml.to_text())