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


import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper


class ExportModelBase:
    class Model:
        def __init__(self):
            self.node = ExportModelBase.Node()
            self.materials = {}
            self.mesh = None

    class Node:
        def __init__(self):
            self.children = []

    class Material:
        pass

    class Mesh:
        pass

    def check(self, context):
        import os
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
        self.model.distances = [int(props.get('lod%s' % i, 0)) for i in range(4)]
        self.gather_node(bpy_node, self.model.node)
        print(self.model.distances)
        print(list(self.model.materials.keys()))
        self.debug_node(0, self.model.node)
        context.window_manager.progress_end()
        return {'FINISHED'}

    def gather_node(self, bpy_node, node):
        import re
        name = re.sub(r'\.\d+$', '', bpy_node.name)
        if bpy_node.type in ['EMPTY', 'MESH']:
            if False in [s == 1.0 for s in bpy_node.scale]:
                raise RuntimeError('Apply scale to all objects in scene before exporting')
            node.name = name
            node.position = list(bpy_node.location)
            node.orientation = list(bpy_node.rotation_quaternion)
            if bpy_node.type == 'MESH':
                node.mesh = ExportModelBase.Mesh()
                for slot in bpy_node.material_slots.keys():
                    self.model.materials[slot] = ExportModelBase.Material()
        for bpy_child in bpy_node.children:
            child = ExportModelBase.Node()
            self.gather_node(bpy_child, child)
            node.children.append(child)

    def debug_node(self, level, node):
        print('%s%s' % (level * 2 * ' ', node.name))
        for child in node.children:
            self.debug_node(level + 1, child)


class ExportModel(bpy.types.Operator, ExportModelBase, ExportHelper):
    """Export scene as There Model file"""
    bl_idname = 'export_scene.model'
    bl_label = 'Export There Model'
    filename_ext = '.model'
    filter_glob: StringProperty(default='*.model', options={'HIDDEN'})


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