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
        scene = bpy.data.scenes[bpy.context.scene.name]
        node = [o for o in bpy.data.scenes[0].objects if o.proxy is None and o.parent is None][0]
        print(node.keys())
        self.process_node(node)
        context.window_manager.progress_end()
        return {'FINISHED'}

    def process_node(self, node):
        import re
        name = re.sub(r'\.\d+$', '', node.name)
        if node.type == 'EMPTY':
            print('%s %s' % (node.type, name))
        if node.type == 'MESH':
            print('%s %s %s' % (node.type, name, node.material_slots.keys()))
        for child in node.children:
            self.process_node(child)


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