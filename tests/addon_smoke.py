"""Enable the packaged addon (no text datablocks anywhere) and simulate."""
import bpy, sys, os, addon_utils
import numpy as np

MOD = "bl_ext.user_default.modeling_cloth"
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


names = [m.__name__ for m in addon_utils.modules()]
check(MOD in names, "addon is installed (%s)" % [n for n in names if "cloth" in n])
addon_utils.enable(MOD, default_set=True, persistent=True)
check(addon_utils.check(MOD)[1], "addon enabled")

mod = sys.modules[MOD]
MC5 = mod.MC_ui.MC5
check(len(bpy.data.texts) == 0, "no text datablocks involved (%d)" % len(bpy.data.texts))
check(hasattr(bpy.types.Object, "MC_props"), "object properties registered")

print("== libraries ==", flush=True)
dll = MC5.U.find_dll()
check(dll is not None and os.path.dirname(dll) == os.path.dirname(mod.__file__),
      "solver DLL found inside the addon folder (%s)" % dll)
nat = MC5.NATIVE.get()
check(nat is not None, "mc_collide.dll loaded (%s)" % MC5.NATIVE.LAST_ERROR)

print("== it simulates ==", flush=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2.0)
ob = bpy.context.object
ob.data.vertices.foreach_set('select', np.zeros(len(ob.data.vertices), dtype=bool))
ob.data.update()
ob.MC_props.cloth = True
ob.MC_props.animated = True
ob.MC_props.gravity = -9.8
C = MC5.get_cloth(ob)
check(C is not None and getattr(C, "solver", None) is not None,
      "cloth built with the C++ solver")
start = np.array(C.co).copy()
for f in range(1, 11):
    bpy.context.scene.frame_set(f)
drop = float(start[:, 2].mean() - np.array(C.co)[:, 2].mean())
check(drop > 0.05, "cloth fell under gravity (%.3f)" % drop)
check(np.isfinite(np.array(C.co)).all(), "no NaNs")

print("== collision runs ==", flush=True)
ob.MC_props.self_collision = True
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, -1.2))
sph = bpy.context.object
sph.MC_props.ob_collision = True
bpy.context.view_layer.objects.active = ob
for f in range(11, 26):
    bpy.context.scene.frame_set(f)
co = np.array(C.co)
check(np.isfinite(co).all(), "still finite with object + self collision on")
check(MC5.collision_backend() is not None, "collision ran on the C++ backend")

print("== bundled scipy ==", flush=True)
# BLENDER_USER_RESOURCES points somewhere throwaway for this run, so the user's
# own scripts/modules scipy is not on the path: anything importable here came
# from the wheel Blender installed with the extension.
import importlib.util
addon_dir = os.path.dirname(mod.__file__)
root = os.path.dirname(os.path.dirname(addon_dir))      # the extensions folder
wheels = os.path.join(addon_dir, "wheels")
bundled = os.path.isdir(wheels) and any(f.endswith(".whl") for f in os.listdir(wheels))

if not bundled:
    # a --no-wheels build: scipy is pip-installed on demand instead, which this
    # isolated run deliberately cannot do
    print("  SKIP  no wheel in this build, so scipy is not expected", flush=True)
else:
    spec = importlib.util.find_spec("scipy")
    check(spec is not None, "scipy is importable with nothing pip-installed")
    if spec is not None:
        where = os.path.abspath(spec.origin or "")
        check(os.path.commonpath([where, root]) == root,
              "and it came from the extension, not the system (%s)" % where)

        from scipy.spatial import cKDTree
        t = cKDTree(np.random.rand(50, 3))
        d, i = t.query(np.random.rand(5, 3), k=1)
        check(d.shape == (5,),
              "it works against Blender's own numpy (%s)" % np.__version__)

print("== magnetic targets ==", flush=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=1.0,
                                location=(5, 0, 1))
mag = bpy.context.object
mag.data.vertices.foreach_set('select', np.zeros(len(mag.data.vertices), dtype=bool))
mag.data.update()
mag.MC_props.cloth = True
mag.MC_props.animated = True
mag.MC_props.magnetic_force = 1.0
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(5, 0, 0))
bpy.context.object.MC_props.magnetic_target = True
bpy.context.view_layer.objects.active = mag
for f in range(1, 6):
    bpy.context.scene.frame_set(f)
MC = MC5.get_cloth(mag)
check(MC is not None and np.isfinite(np.array(MC.co)).all(),
      "a magnetic-target sim runs out of the box")
if bundled:
    check(getattr(MC, "tree", None) is not None, "and it built its kd-tree")
else:
    # without scipy the feature warns once and the sim carries on, which is the
    # behaviour that matters here
    check(np.isfinite(np.array(MC.co)).all(),
          "and it carried on without scipy rather than failing")

print("== cache ==", flush=True)
import tempfile
bpy.context.scene.MC_props.cache_dir = tempfile.mkdtemp(prefix="mc_addon_") + os.sep
MC5.CACHE.save_frame(ob, 1, C.co)
check(MC5.CACHE.cached_frames(ob) == [1], "cache writes and reads back")

print("== panels draw ==", flush=True)
import types


class Rec:
    def __init__(self):
        self.scale_y = self.scale_x = 1.0
        self.alert = False
        self.enabled = self.active = True
        self.use_property_split = False

    def _chain(self, *a, **k):
        return Rec()
    box = row = column = split = column_flow = grid_flow = _chain

    def prop(self, *a, **k):
        return self

    def operator(self, *a, **k):
        return types.SimpleNamespace()

    def label(self, *a, **k):
        return None
    separator = template_list = menu = prop_search = label


class Self:
    """Panels call their own helpers from draw(), so bind them on."""
    def __init__(self, panel):
        self.layout = Rec()
        for k in dir(panel):
            v = panel.__dict__.get(k)
            if callable(v) and not k.startswith("__") and k not in ("draw", "poll"):
                setattr(self, k, v.__get__(self))


n = 0
for cls in mod.MC_ui.CLASSES:
    if isinstance(cls, type) and issubclass(cls, bpy.types.Panel):
        cls.draw(Self(cls), bpy.context)
        n += 1
check(n >= 10, "%d panels drew with no errors" % n)

print("== disable / re-enable ==", flush=True)
addon_utils.disable(MOD)
check(not hasattr(bpy.types.Object, "MC_props"), "properties removed on disable")
check("mc_handler" not in [h.__name__ for h in bpy.app.handlers.frame_change_post],
      "frame handler removed on disable")
addon_utils.enable(MOD)
check(hasattr(bpy.types.Object, "MC_props"), "re-enables cleanly")

print("================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
