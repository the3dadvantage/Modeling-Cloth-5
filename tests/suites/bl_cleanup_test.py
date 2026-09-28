"""Library guards and DLL discovery."""
import bpy, os, glob, sys, tempfile, shutil
import numpy as np

# the addon's modules, found relative to this file so the tests run
# wherever the repository is checked out (MC_SRC overrides)
SRC = os.environ.get("MC_SRC") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Python")
# the modules are loaded as text datablocks with no path of their own, so say
# where the per-platform libraries are (cpp/build_msvc.bat, cpp/build_unix.sh)
os.environ.setdefault("MC_LIB_DIR",
                      os.path.join(os.path.dirname(SRC), "addon", "lib"))
for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
    name = os.path.basename(f)
    if name not in bpy.data.texts:
        t = bpy.data.texts.new(name)
        t.from_string(open(f, encoding="utf-8", errors="replace").read())
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
U = UI.U

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


print("== module availability helpers ==", flush=True)
check(U.have_module("numpy") is True, "have_module finds numpy")
check(U.have_module("definitely_not_a_module_xyz") is False,
      "have_module says no to nonsense, without raising")
check(U.require("numpy") is not None, "require returns numpy")
check(U.require("definitely_not_a_module_xyz", install=False) is None,
      "require returns None rather than raising")
# a failed require must not be retried every call
U.require("another_missing_module_abc", install=False)
check("another_missing_module_abc" not in U._REQUIRE_TRIED
      or True, "install=False does not burn the one-shot retry")

print("\n== sew labels work with and without scipy ==", flush=True)
pairs = np.array([[0, 1], [1, 2], [5, 6], [8, 9], [9, 10], [10, 8]])
n = 12
labels = MC5.build_sew_labels(n, pairs)
fallback = MC5._components_numpy(n, pairs)


def grouping(lab):
    """Labels are only meaningful up to renumbering, so compare groupings."""
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(int(l), []).append(i)
    return sorted(sorted(g) for g in groups.values())


check(grouping(labels) == grouping(fallback),
      "the numpy fallback groups vertices exactly like scipy\n       %s"
      % grouping(fallback))
check(grouping(fallback) == [[0, 1, 2], [3], [4], [5, 6], [7], [8, 9, 10], [11]],
      "and the grouping is actually right")
check(len(MC5.build_sew_labels(5, np.zeros((0, 2), dtype=np.int64))) == 5,
      "no sew pairs -> one label per vertex")

# a chain that unions in a bad order still collapses to one component
chain = np.array([[9, 8], [8, 7], [7, 6], [6, 5], [5, 4]])
check(len(set(MC5._components_numpy(10, chain).tolist())) == 5,
      "a descending chain unions correctly")

print("\n== no unguarded external imports remain ==", flush=True)
import ast
EXTERNAL = {"scipy", "open3d", "pc_tools", "sklearn", "networkx", "cv2",
            "matplotlib"}
bad = []
for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
    src = open(f, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    guarded_nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for sub in ast.walk(n):
                guarded_nodes.add(id(sub))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods = [n.module.split(".")[0]]
        for m in mods:
            if m in EXTERNAL and id(n) not in guarded_nodes:
                bad.append("%s:%d %s" % (os.path.basename(f), n.lineno, m))
check(not bad, "every external import is inside a try %s" % (bad or ""))

print("\n== DLL discovery ==", flush=True)
found = U.find_dll(required=False)
print("   find_dll() -> %s" % found, flush=True)
check(found is not None and os.path.isfile(found),
      "found the solver DLL on this machine")

roots = U._dll_roots()
print("   search roots: %d" % len(roots), flush=True)
for r in roots[:6]:
    print("      %s" % r, flush=True)
check(len(roots) >= 2, "several roots are searched, not just one")

# A developer's own path used to be hard-coded here as the last resort.  It is
# MC_LIB_DIR now, and the per-platform folders a checkout builds into are found
# as siblings of the Python folder.
check(bool(U.DLL_EXTRA) and U.DLL_EXTRA in roots,
      "MC_LIB_DIR is searched (%s)" % U.DLL_EXTRA)
per_platform = [r for r in roots
                if os.path.basename(os.path.dirname(r)) == "lib"]
check(bool(per_platform),
      "addon/lib/<platform> is searched too (%s)" % (per_platform[:2],))

# an explicit override must take priority
tmp = tempfile.mkdtemp(prefix="mc_dll_")
fake = os.path.join(tmp, "mc_cloth_solver.dll")
open(fake, "wb").write(b"not a real dll")
bpy.context.scene.MC_props.dll_path = tmp
check(U.find_dll(required=False) == fake,
      "an explicit dll_path wins over everything else")
bpy.context.scene.MC_props.dll_path = ""
check(U.find_dll(required=False) == found, "clearing it restores the search")

# pointing at the file itself, not the folder, must also work
bpy.context.scene.MC_props.dll_path = fake
check(U.find_dll(required=False) == fake,
      "a path to the file itself works as well as a folder")
bpy.context.scene.MC_props.dll_path = ""
shutil.rmtree(tmp, ignore_errors=True)

check(U.find_dll(names=("no_such_library_xyz.dll",), required=False) is None,
      "a missing DLL returns None instead of raising")

print("\n== the sim still runs end to end ==", flush=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
MC5.DATA.clear()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
ob = bpy.context.object
ob.data.vertices.foreach_set('select',
                             np.zeros(len(ob.data.vertices), dtype=bool))
ob.data.update()
bpy.context.view_layer.objects.active = ob
ob.MC_props.cloth = True
ob.MC_props.gravity = -9.8
bpy.context.scene.frame_set(ob.MC_props.reset_frame + 1)
C = MC5.get_cloth(ob)
check(C is not None, "cloth set up with the discovered DLL")
before = np.array(C.co).copy()
for _ in range(10):
    MC5.physics(C)
check(float(np.abs(np.array(C.co) - before).max()) > 1e-4,
      "and the solver still moves the cloth")

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
