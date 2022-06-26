import os
import math
import copy
import enum
import struct


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

    def get_material_name(self, blender_name):
        blender_name = blender_name.lower()
        for material in self.materials:
            if material[1].lower() == blender_name:
                return material[0]
        return None

    def get_phenomorph_name(self, blender_name):
        blender_name = blender_name.lower()
        for phenomorph in self.phenomorphs:
            if phenomorph[1].lower() == blender_name:
                return phenomorph[0]
        return None


class Skute:
    def __init__(self, path):
        self.path = path
        self.skeleton = None
        self.lods = []
        self.scale = None
        self.materials = {}


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