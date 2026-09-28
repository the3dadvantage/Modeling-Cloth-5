"""Import the collision modules the way an installed addon would.

A fresh Blender with NO text datablocks, and the modules copied into a real
package folder.  Every bpy.data.texts[...] lookup fails, so each module has to
reach its `from . import` fallback -- which the collision modules did not have.
"""
import bpy, os, sys, shutil, tempfile, importlib, traceback, glob

# the addon's modules, found relative to this file so the tests run
# wherever the repository is checked out (MC_SRC overrides)
SRC = os.environ.get("MC_SRC") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Python")
# the modules are loaded as text datablocks with no path of their own, so say
# where the per-platform libraries are (cpp/build_msvc.bat, cpp/build_unix.sh)
os.environ.setdefault("MC_LIB_DIR",
                      os.path.join(os.path.dirname(SRC), "addon", "lib"))
root = tempfile.mkdtemp(prefix="mc_pkgtest_")
pkg = os.path.join(root, "mc_pkg")
os.makedirs(pkg)
for f in glob.glob(os.path.join(SRC, "*.py")):
    shutil.copy(f, pkg)
open(os.path.join(pkg, "__init__.py"), "w").close()
sys.path.insert(0, root)

print("PKG texts in this Blender: %d (must be 0)" % len(bpy.data.texts), flush=True)
for mod in ("utils", "object_collide", "self_collide_2", "split_tuner", "MC5",
            "MC_ui"):
    try:
        m = importlib.import_module("mc_pkg." + mod)
        extra = ""
        if mod in ("object_collide", "self_collide_2"):
            extra = "  (U is %s)" % m.U.__name__
        print("PKG %-16s imported%s" % (mod, extra), flush=True)
    except Exception as e:
        print("PKG %-16s FAILED %s: %s" % (mod, type(e).__name__, e), flush=True)
        traceback.print_exc()
shutil.rmtree(root, ignore_errors=True)
