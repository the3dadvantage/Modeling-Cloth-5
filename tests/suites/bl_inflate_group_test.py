"""The MC_inflate vertex group weights the inflate force per vertex.

Default 1.0 everywhere, so an unpainted group has to leave inflate exactly as
it was.
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
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def build(**settings):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=2.0, location=(0, 0, 0))
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = 0.0
    p.velocity = 1.0
    p.stretch = 1.25
    p.bend_force = 0.5
    for k, v in settings.items():
        setattr(p, k, v)
    scene.frame_set(p.reset_frame + 1)
    return ob


def sim(ob, steps=25):
    C = MC5.get_cloth(ob)
    for _ in range(steps):
        MC5.physics(C)
    return np.array(MC5.get_cloth(ob).co).copy()


print("== the group exists and defaults to 1.0 ==", flush=True)
ob = build()
C = MC5.get_cloth(ob)
check("MC_inflate" in C.group_data, "MC_inflate is in group_data")
check("MC_inflate" in [g.name for g in ob.vertex_groups],
      "and is a real vertex group on the object")
check(float(np.abs(C.group_data["MC_inflate"] - 1.0).max()) == 0.0,
      "defaults to 1.0 everywhere")
check(C.group_data["MC_inflate"].shape == (len(ob.data.vertices), 1),
      "shaped (vc, 1) like the other groups %s"
      % (C.group_data["MC_inflate"].shape,))

print("\n== unpainted, inflate is unchanged ==", flush=True)
# a flat grid's normals are +Z, so inflate drives it along +Z
base = sim(build(inflate=0.0))
blown = sim(build(inflate=0.02))
check(float(np.abs(blown - base).max()) > 1e-4, "inflate moves the cloth")
check(blown[:, 2].mean() > base[:, 2].mean(), "along the normals")

print("\n== the weight scales it ==", flush=True)
ob = build(inflate=0.02)
C = MC5.get_cloth(ob)
C.group_data["MC_inflate"][:] = 0.0
none = sim(ob)
check(float(np.abs(none - base).max()) < 1e-9,
      "a zero weight removes inflate entirely (%.3e)"
      % float(np.abs(none - base).max()))

ob = build(inflate=0.02)
C = MC5.get_cloth(ob)
C.group_data["MC_inflate"][:] = 0.5
half = sim(ob)
full = blown
d_half = float(np.abs(half[:, 2] - base[:, 2]).mean())
d_full = float(np.abs(full[:, 2] - base[:, 2]).mean())
print("   mean z displacement: full %.6f, half %.6f" % (d_full, d_half),
      flush=True)
check(d_half < d_full, "a 0.5 weight inflates less than 1.0")
check(abs(d_half / max(d_full, 1e-12) - 0.5) < 0.05,
      "and does so by about half (ratio %.4f)" % (d_half / max(d_full, 1e-12)))

print("\n== a partial weight only moves the painted verts ==", flush=True)
ob = build(inflate=0.02)
C = MC5.get_cloth(ob)
n = len(ob.data.vertices)
C.group_data["MC_inflate"][:] = 0.0
C.group_data["MC_inflate"][: n // 2] = 1.0
part = sim(ob)
moved = np.abs(part[:, 2] - base[:, 2])
full_moved = np.abs(full[:, 2] - base[:, 2])
lo, hi = moved[n // 2:].mean(), moved[: n // 2].mean()
print("   painted half %.4f, unpainted half %.4f, fully painted %.4f"
      % (hi, lo, full_moved[n // 2:].mean()), flush=True)
check(hi > 1e-5, "the painted half moved")
check(lo < hi, "the unpainted half moved less (%.4f vs %.4f)" % (lo, hi))
# it cannot be zero -- the sheet is connected, so stretch and bend drag the
# unpainted verts along.  What matters is that they move less than they would
# if they were painted too.
check(lo < full_moved[n // 2:].mean() * 0.75,
      "and much less than when it is painted as well (%.4f vs %.4f)"
      % (lo, full_moved[n // 2:].mean()))

print("\n== group_check still handles it ==", flush=True)
ob = build(inflate=0.02)
C = MC5.get_cloth(ob)
idx = ob.vertex_groups["MC_inflate"].index
ob.vertex_groups.active_index = idx
try:
    MC5.G.group_check(C, active_name="MC_inflate")
    check(True, "group_check accepts MC_inflate")
except Exception as e:
    check(False, "group_check raised %s: %s" % (type(e).__name__, e))
check(C.group_data["MC_inflate"].shape == (n, 1),
      "and left the array shape alone")

print("\n== the other groups are untouched ==", flush=True)
ob = build()
C = MC5.get_cloth(ob)
for g in ("MC_bend", "MC_stretch", "MC_pin", "MC_velocity", "MC_friction",
          "MC_wind"):
    check(g in C.group_data, "%s still present" % g)

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
