bl_info = {
    'name': 'There Skute format',
    'author': 'Brian Gontowski',
    'version': (1, 2, 8),
    'blender': (3, 1, 0),
    'location': 'File > Import-Export',
    'description': 'Export as Skute for There.com',
    'support': 'COMMUNITY',
    'category': 'Import-Export',
    'doc_url': 'https://github.com/hmphus/there-blender-exporter/wiki',
    'tracker_url': 'https://www.facebook.com/groups/1614251348595174',
}


def register():
    from .exporter import register_exporter
    register_exporter()


def unregister():
    from .exporter import unregister_exporter
    unregister_exporter()