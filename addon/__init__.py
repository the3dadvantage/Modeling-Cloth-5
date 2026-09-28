"""Modeling Cloth 5 -- addon entry point.

The modules are written to work two ways: as loose text datablocks inside a
.blend while developing (each one finds the others through bpy.data.texts),
and as this package once installed.  Every module tries the text datablocks
first and falls back to a relative import, so nothing here has to change
between the two.

Only MC_ui is imported: it pulls in MC5, which pulls in everything else, and
it owns the class list.  register() / unregister() just hand over to it.
"""

bl_info = {
    "name": "Modeling Cloth 5",
    "author": "Rich Colburn",
    "version": (0, 9, 0),          # beta; keep in step with blender_manifest.toml
    "blender": (5, 0, 0),          # minimum Blender version, not the addon's
    "location": "View3D > Sidebar > MC5",
    "description": "Cloth simulation you can keep modelling with",
    "category": "Physics",
}

from . import MC_ui


def register():
    MC_ui.register()


def unregister():
    MC_ui.unregister()
