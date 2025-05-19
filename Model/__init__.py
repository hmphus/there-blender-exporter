bl_info = {
    'name': 'There Model format',
    'author': 'Brian Gontowski',
    'version': (1, 3, 0),
    'blender': (3, 1, 0),
    'location': 'File > Import-Export',
    'description': 'Export as Model for There.com',
    'support': 'COMMUNITY',
    'category': 'Import-Export',
    'doc_url': 'https://github.com/hmphus/there-blender-exporter/wiki',
    'tracker_url': 'https://www.facebook.com/groups/1614251348595174',
}


def register():
    from .exporter import register_exporter
    register_exporter()
    from .shader import register_shader
    register_shader()


def unregister():
    from .exporter import unregister_exporter
    unregister_exporter()
    from .shader import unregister_shader
    unregister_shader()