"""After the cleanup: does everything still register, draw, and simulate?

Headless Blender never draws a panel, so a panel still referencing a removed
property would sail through registration and only break in the UI.  Each
panel's draw() is driven here with a layout that records every property it
asks for, and each one is checked against what is actually registered.
"""
import bpy, os, glob, sys
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

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


REMOVED = {"grid_angle_limit", "grid_size", "grid_triangles", "grid_merge_dist",
           "grid_smoothing", "grid_debug_idx", "offset_grid", "rotate_grid",
           "scale_grid", "use_uv_shape", "sc_dynamic_search", "sc_search_radius",
           "bisect_precision", "ob_collision_search_radius", "dev_mode",
           "sc_box_count", "sc_box_count_auto", "ob_box_count",
           "ob_box_count_auto",
           # second round, after asking
           "debug", "debug_2", "ob_tri_bounds"}

print("== registration ==", flush=True)
obp = {p.identifier for p in bpy.types.Object.bl_rna.properties["MC_props"]
       .fixed_type.properties}
scp = {p.identifier for p in bpy.types.Scene.bl_rna.properties["MC_props"]
       .fixed_type.properties}
check(len(obp) > 50, "MC_props registered (%d props)" % len(obp))
check("debug_mode" in scp, "debug_mode exists on the scene")
still = sorted((obp | scp) & REMOVED)
check(not still, "no removed property is still registered %s" % (still or ""))

print("\n== preset keys all point at real properties ==", flush=True)
missing = [k for k in UI.PRESET_KEYS if k not in obp]
check(not missing, "every PRESET_KEYS entry exists %s" % (missing or ""))
dead_keys = [k for k in UI.PRESET_KEYS if k in REMOVED]
check(not dead_keys, "no removed property left in PRESET_KEYS %s"
      % (dead_keys or ""))


class Rec:
    """A layout that accepts anything and remembers the props asked for."""

    def __init__(self, log):
        self.log = log
        self.scale_y = 1.0
        self.scale_x = 1.0
        self.alert = False
        self.enabled = True
        self.active = True
        self.use_property_split = False

    def prop(self, data, name, *a, **k):
        self.log.append((data, name))
        return self

    def _chain(self, *a, **k):
        return Rec(self.log)

    box = row = column = split = column_flow = grid_flow = _chain

    def operator(self, *a, **k):
        class O:
            pass
        return O()

    def label(self, *a, **k):
        return None

    separator = template_list = menu = prop_search = label


class Self:
    def __init__(self, log, panel=None):
        self.layout = Rec(log)
        # panels can call their own helpers from draw() -- the wind panel does,
        # through self._settings() -- so bind the panel's methods onto us
        if panel is not None:
            for k in dir(panel):
                if k.startswith("__") or k in ("draw", "poll"):
                    continue
                v = panel.__dict__.get(k)
                if callable(v):
                    setattr(self, k, v.__get__(self))


def draw_all(debug):
    bpy.context.scene.MC_props.debug_mode = debug
    panels = [c for c in UI.CLASSES
              if isinstance(c, type) and issubclass(c, bpy.types.Panel)]
    problems, asked = [], 0
    for P in panels:
        log = []
        # respect poll: a panel that hides itself in this state is never drawn
        # in it, and may rely on that (no "no object selected" placeholder)
        try:
            if hasattr(P, "poll") and not P.poll(bpy.context):
                continue
        except Exception as e:
            problems.append("%s.poll raised %s: %s" % (P.__name__, type(e).__name__, e))
            continue
        try:
            P.draw(Self(log, P), bpy.context)
        except Exception as e:
            problems.append("%s raised %s: %s" % (P.__name__, type(e).__name__, e))
            continue
        for data, name in log:
            asked += 1
            if data is None:
                continue
            try:
                known = name in data.bl_rna.properties
            except Exception:
                known = hasattr(data, name)
            if not known:
                problems.append("%s draws unknown '%s'" % (P.__name__, name))
    return panels, problems, asked


# A cloth, a collider and a magnetic target, so that every panel's poll passes
# and the draw of each one is actually exercised -- the scene-list panels hide
# themselves until their data exists.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
MC5.DATA.clear()
MC5.OC_DATA["obs"] = []
MC5.COLLISION_DATA.clear()
MC5.MAGNETIC_DATA.clear()

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, -2))
collider = bpy.context.object
collider.MC_props.ob_collision = True

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 3, 0))
bpy.context.object.MC_props.magnetic_target = True

bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
ob = bpy.context.object
ob.data.vertices.foreach_set('select',
                             np.zeros(len(ob.data.vertices), dtype=bool))
ob.data.update()
bpy.context.view_layer.objects.active = ob
ob.MC_props.cloth = True
ob.MC_props.self_collision = True
ob.MC_props.ob_recollide = True

for debug in (False, True):
    print("\n== every panel draws, debug_mode=%s ==" % debug, flush=True)
    panels, problems, asked = draw_all(debug)
    check(not problems, "%d panels drew %d props with no errors %s"
          % (len(panels), asked, problems or ""))

print("\n== debug_mode hides exactly what was agreed ==", flush=True)


def names(debug):
    """Every property any panel draws, at this debug setting."""
    bpy.context.scene.MC_props.debug_mode = debug
    out = set()
    for P in UI.CLASSES:
        if isinstance(P, type) and issubclass(P, bpy.types.Panel):
            log = []
            try:
                P.draw(Self(log, P), bpy.context)
            except Exception:
                pass
            out |= {n for _, n in log}
    return out


off, on = names(False), names(True)
hidden = on - off
print("   shown only in debug mode: %s" % sorted(hidden), flush=True)
AGREED = {"text_export_path", "dll_path",                       # dev tools
          "e_bend_force",                  # a development dial, not a setting
          "ob_collision_tri_damping", "ob_collision_edge_damping",  # asked
          "sc_box_depth", "sc_box_depth_auto",                  # asked
          "ob_box_depth", "ob_box_depth_auto"}
check(hidden == AGREED, "the hidden set is exactly the agreed one%s"
      % ("" if hidden == AGREED else
         " (extra %s, missing %s)" % (sorted(hidden - AGREED),
                                     sorted(AGREED - hidden))))
check("debug_mode" in off, "the toggle itself is always visible")
KEPT = {"cl_point_tris", "ob_point_tris", "ob_edges", "ob_recollide"}
check(KEPT <= off, "contact toggles and recollide stay visible, as asked %s"
      % (sorted(KEPT - off) or ""))
check(not (off | on) & REMOVED, "no panel draws a removed property")
bpy.context.scene.MC_props.debug_mode = False

print("\n== nothing loads a retired module ==", flush=True)
import ast
RETIRED = {"MC_grid", "MC_pierce", "Text", "c_plus_bridge_v2", "panels",
           "properties", "self_collide", "sew_bend"}
refs = []
for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
    tree = ast.parse(open(f, encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value.endswith(".py") and n.value[:-3] in RETIRED:
                refs.append("%s:%d" % (os.path.basename(f), n.lineno))
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            for a in n.names:
                if a.name in RETIRED:
                    refs.append("%s:%d" % (os.path.basename(f), n.lineno))
check(not refs, "no live module loads a retired one %s" % (refs or ""))
check(not any(os.path.exists(os.path.join(SRC, m + ".py")) for m in RETIRED),
      "and none of them are left in the addon folder")

print("\n== the sim still runs ==", flush=True)
bpy.context.scene.frame_set(ob.MC_props.reset_frame + 1)
ob.MC_props.gravity = -9.8
C = MC5.get_cloth(ob)
before = np.array(C.co).copy()
for _ in range(10):
    MC5.physics(C)
check(float(np.abs(np.array(C.co) - before).max()) > 1e-4,
      "a cloth with self collision still simulates")

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
