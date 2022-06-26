import os
import math
import copy
import enum
import struct


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


class Skute:
    def __init__(self, path):
        self.path = path
        self.skeleton = None
        self.lods = []
        self.scale = None
        self.materials = {}

    def save(self):
        pass


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