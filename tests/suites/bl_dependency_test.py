"""The optional scipy dependency.

Only magnetic targets need scipy.  Everything about that has to survive a
machine with no network:

  * the addon imports and registers with scipy missing
  * the install is attempted once per session, not once per frame
  * a failed install warns once and leaves the sim running
  * the Settings panel offers the install, and the operator reports failure
  * pip is told --no-deps for scipy, so it cannot shadow Blender's numpy

The real pip is not run here: subprocess.check_call is stood in for, which is
also how "no network" is simulated.
"""
import bpy, os, sys, glob, types
import numpy as np

sys.excepthook = lambda *e: (sys.__excepthook__(*e), sys.stdout.flush(), os._exit(1))

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
    n = os.path.basename(f)
    if n not in bpy.data.texts:
        bpy.data.texts.new(n).from_string(open(f, encoding="utf-8", errors="replace").read())
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
MC5 = UI.MC5
U = MC5.U

popups = []
U.popup_error = lambda *a, **k: popups.append(a[0] if a else "")

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


print("== the addon works with scipy missing ==", flush=True)
check(hasattr(bpy.types.Object, "MC_props"), "registered without scipy at import")
check("scipy" not in sys.modules or U.have_module("scipy"),
      "nothing imported scipy behind our backs")

# pretend scipy is unavailable (it may well be installed on this machine) and
# that pip cannot reach anything
calls = []
real_call = U.subprocess.check_call


def fake_call(cmd, **kw):
    calls.append(list(cmd))
    if "install" in cmd:
        raise OSError("no network")
    return 0


class NoScipy:
    """Blocks scipy at the import machinery, the way a machine without it
    behaves -- patching find_spec alone would not stop import_module."""
    def find_spec(self, name, path=None, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("blocked by the test", name=name)
        return None


blocker = NoScipy()
stashed = {k: sys.modules.pop(k) for k in list(sys.modules)
           if k == "scipy" or k.startswith("scipy.")}
sys.meta_path.insert(0, blocker)
U.subprocess.check_call = fake_call
U._REQUIRE_TRIED.discard("scipy")
MC5._SCIPY_WARNED = False
check(not U.have_module("scipy"), "scipy looks absent for the rest of the test")

print("\n== a failed install is tried once, not every frame ==", flush=True)
first = U.require("scipy")
n_after_first = len([c for c in calls if "install" in c])
for _ in range(5):
    U.require("scipy")
n_after_more = len([c for c in calls if "install" in c])
check(first is None, "require() returns None instead of raising")
check(n_after_first == 1 and n_after_more == 1,
      "pip ran once for five calls (%d)" % n_after_more)

print("\n== pip is asked for the right thing ==", flush=True)
U._REQUIRE_TRIED.discard("scipy")
calls.clear()
U.external_lib(module="scipy")
inst = [c for c in calls if "install" in c][0]
check("--no-deps" in inst, "scipy is installed --no-deps, so Blender's numpy "
                           "is not shadowed (%s)" % " ".join(inst[-4:]))
check("--target" in inst and "scripts" in " ".join(inst).lower(),
      "into the user modules folder")
check(inst[0] == sys.executable and inst[0].lower().endswith("python.exe"),
      "using Blender's own python (%s)" % inst[0])
check(U.external_lib(module="scipy") is False,
      "returns False when pip fails")

print("\n== a sim with magnetic targets keeps running ==", flush=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
cloth = bpy.context.object
cloth.data.vertices.foreach_set('select', np.zeros(len(cloth.data.vertices), dtype=bool))
cloth.data.update()
cloth.MC_props.cloth = True
cloth.MC_props.animated = True
cloth.MC_props.magnetic_force = 1.0
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, -1.5))
target = bpy.context.object
target.MC_props.magnetic_target = True
bpy.context.view_layer.objects.active = cloth

popups.clear()
U._REQUIRE_TRIED.discard("scipy")
MC5._SCIPY_WARNED = False
for f in range(1, 8):
    bpy.context.scene.frame_set(f)
C = MC5.get_cloth(cloth)
check(C is not None and np.isfinite(np.array(C.co)).all(),
      "the cloth still simulates with no scipy")
check(len(popups) == 1, "warned exactly once over 7 frames (%d)" % len(popups))
check("scipy" in popups[0].lower(), "and the warning says what to do (%s)" % popups[0][:60])

print("\n== the settings panel offers the install ==", flush=True)


class Rec:
    def __init__(self, ops):
        self.ops = ops
        self.scale_y = self.scale_x = 1.0
        self.alert = False
        self.enabled = self.active = True
        self.use_property_split = False

    def _chain(self, *a, **k):
        return Rec(self.ops)
    box = row = column = split = column_flow = grid_flow = _chain

    def prop(self, *a, **k):
        return self

    def operator(self, idname, **k):
        self.ops.append(idname)
        return types.SimpleNamespace(module="")

    def label(self, *a, **k):
        return None
    separator = template_list = menu = prop_search = label


class Self:
    def __init__(self, panel):
        self.layout = Rec(getattr(self, "ops", []))
        self.ops = self.layout.ops
        for k in dir(panel):
            v = panel.__dict__.get(k)
            if callable(v) and not k.startswith("__") and k not in ("draw", "poll"):
                setattr(self, k, v.__get__(self))


s = Self(UI.MC_PT_panel_settings)
UI.MC_PT_panel_settings.draw(s, bpy.context)
check("mc.install_dependency" in s.ops, "the Install button is there (%s)"
      % [o for o in s.ops if "install" in o])

# an {'ERROR'} report comes back to python as a RuntimeError; in the UI it is
# just a red message in the status bar
try:
    res = bpy.ops.mc.install_dependency(module="scipy")
    msg = ""
except RuntimeError as e:
    res, msg = {'CANCELLED'}, str(e)
check(res == {'CANCELLED'} and "scipy" in msg,
      "the operator reports failure instead of hanging (%s)" % msg[:60])

print("\n== with scipy present ==", flush=True)
sys.meta_path.remove(blocker)
sys.modules.update(stashed)
U.subprocess.check_call = real_call
s = Self(UI.MC_PT_panel_settings)
UI.MC_PT_panel_settings.draw(s, bpy.context)
if U.have_module("scipy"):
    check("mc.install_dependency" not in s.ops, "no Install button once it is there")
else:
    check(True, "scipy genuinely absent here, skipped")
check(callable(U.KDTree), "utils.KDTree exists (it used to be a NameError)")

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
