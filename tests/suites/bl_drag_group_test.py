"""MC_air_drag and MC_inverted_air_drag weight the two drag forces per vertex.

Both default to 1.0, so an unpainted group has to leave the existing behaviour
bit-identical.  Inverted drag is the fiddly one: it damps the whole velocity by
(1 - amount) and adds the normal component back, so the weight has to scale the
damping too or a zero weight would still bleed velocity away.
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


def build(prim="grid", **settings):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    if prim == "sphere":
        # Inverted drag damps velocity then restores the normal component, so
        # on a flat sheet falling along its own normal the two cancel and it is
        # a no-op.  A sphere has normals pointing every way, so gravity has a
        # tangential component over most of it and the force actually bites.
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12,
                                             radius=1.0)
    else:
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
    p.gravity = -9.8          # something for drag to act on
    p.velocity = 0.99
    p.stretch = 1.25
    p.bend_force = 0.5
    for k, v in settings.items():
        setattr(p, k, v)
    scene.frame_set(p.reset_frame + 1)
    return ob


def sim(ob, steps=30, weights=None):
    C = MC5.get_cloth(ob)
    if weights:
        for k, v in weights.items():
            C.group_data[k][:] = v
    for _ in range(steps):
        MC5.physics(C)
    return np.array(MC5.get_cloth(ob).co).copy()


print("== both groups exist at 1.0 ==", flush=True)
ob = build()
C = MC5.get_cloth(ob)
n = len(ob.data.vertices)
for g in ("MC_air_drag", "MC_inverted_air_drag"):
    check(g in C.group_data, "%s is in group_data" % g)
    check(g in [v.name for v in ob.vertex_groups],
          "%s is a real vertex group" % g)
    check(float(np.abs(C.group_data[g] - 1.0).max()) == 0.0,
          "%s defaults to 1.0" % g)
    check(C.group_data[g].shape == (n, 1), "%s is shaped (vc, 1)" % g)

print("\n== unpainted, both drags behave as before ==", flush=True)
# a full weight must reproduce the un-weighted maths exactly
plain = sim(build(air_drag=0.3))
weighted = sim(build(air_drag=0.3), weights={"MC_air_drag": 1.0})
check(float(np.abs(plain - weighted).max()) == 0.0,
      "air drag at weight 1.0 is bit-identical (%.3e)"
      % float(np.abs(plain - weighted).max()))

plain_i = sim(build(inverted_air_drag=0.3))
weighted_i = sim(build(inverted_air_drag=0.3),
                 weights={"MC_inverted_air_drag": 1.0})
check(float(np.abs(plain_i - weighted_i).max()) == 0.0,
      "inverted air drag at weight 1.0 is bit-identical (%.3e)"
      % float(np.abs(plain_i - weighted_i).max()))

print("\n== a zero weight switches each drag off ==", flush=True)
none = sim(build(air_drag=0.0))
off = sim(build(air_drag=0.3), weights={"MC_air_drag": 0.0})
check(float(np.abs(none - off).max()) == 0.0,
      "air drag weight 0 == air drag off (%.3e)"
      % float(np.abs(none - off).max()))

none_i = sim(build(inverted_air_drag=0.0))
off_i = sim(build(inverted_air_drag=0.3),
            weights={"MC_inverted_air_drag": 0.0})
check(float(np.abs(none_i - off_i).max()) == 0.0,
      "inverted weight 0 == inverted off, no velocity bled away (%.3e)"
      % float(np.abs(none_i - off_i).max()))

print("\n== the weight scales the effect ==", flush=True)
# drag resists falling, so more drag means the cloth has fallen less far
z_off = sim(build(air_drag=0.3), weights={"MC_air_drag": 0.0})[:, 2].mean()
z_half = sim(build(air_drag=0.3), weights={"MC_air_drag": 0.5})[:, 2].mean()
z_full = sim(build(air_drag=0.3), weights={"MC_air_drag": 1.0})[:, 2].mean()
print("   mean z: weight 0 %.4f, 0.5 %.4f, 1.0 %.4f" % (z_off, z_half, z_full),
      flush=True)
check(z_off < z_half < z_full,
      "more air drag weight means less fall (%.4f < %.4f < %.4f)"
      % (z_off, z_half, z_full))

def spread(w):
    """How far the sphere's verts have moved from where they started."""
    ob = build("sphere", inverted_air_drag=0.3)
    start = np.array(MC5.get_cloth(ob).co).copy()
    end = sim(ob, weights={"MC_inverted_air_drag": w})
    return float(np.linalg.norm(end - start, axis=1).mean())


i_off, i_half, i_full = spread(0.0), spread(0.5), spread(1.0)
print("   sphere mean displacement: weight 0 %.4f, 0.5 %.4f, 1.0 %.4f"
      % (i_off, i_half, i_full), flush=True)
check(abs(i_half - i_off) > 1e-6 and abs(i_full - i_half) > 1e-6,
      "inverted drag weight changes the result")
check((i_off < i_half < i_full) or (i_off > i_half > i_full),
      "and does so monotonically (%.4f, %.4f, %.4f)"
      % (i_off, i_half, i_full))

# and confirm the no-op on a flat sheet is real, not a wiring mistake
f_off = sim(build(inverted_air_drag=0.3),
            weights={"MC_inverted_air_drag": 0.0})[:, 2].mean()
f_full = sim(build(inverted_air_drag=0.3),
             weights={"MC_inverted_air_drag": 1.0})[:, 2].mean()
check(abs(f_off - f_full) < 1e-3,
      "on a flat sheet falling along its normal, inverted drag is a no-op "
      "at any weight (%.4f vs %.4f)" % (f_off, f_full))

print("\n== the two groups are independent ==", flush=True)
a_only = sim(build(air_drag=0.3, inverted_air_drag=0.3),
             weights={"MC_air_drag": 1.0, "MC_inverted_air_drag": 0.0})
i_only = sim(build(air_drag=0.3, inverted_air_drag=0.3),
             weights={"MC_air_drag": 0.0, "MC_inverted_air_drag": 1.0})
check(float(np.abs(a_only - i_only).max()) > 1e-6,
      "painting one does not paint the other")

print("\n== partial painting ==", flush=True)
ob = build(air_drag=0.4)
C = MC5.get_cloth(ob)
C.group_data["MC_air_drag"][:] = 0.0
C.group_data["MC_air_drag"][: n // 2] = 1.0
for _ in range(30):
    MC5.physics(C)
part = np.array(MC5.get_cloth(ob).co)
check(part[: n // 2, 2].mean() > part[n // 2:, 2].mean(),
      "the dragged half fell less than the free half (%.4f vs %.4f)"
      % (part[: n // 2, 2].mean(), part[n // 2:, 2].mean()))
check(np.all(np.isfinite(part)), "stayed finite")

print("\n== group_check handles both ==", flush=True)
ob = build(air_drag=0.2, inverted_air_drag=0.2)
C = MC5.get_cloth(ob)
for g in ("MC_air_drag", "MC_inverted_air_drag"):
    ob.vertex_groups.active_index = ob.vertex_groups[g].index
    try:
        MC5.G.group_check(C, active_name=g)
        check(True, "group_check accepts %s" % g)
    except Exception as e:
        check(False, "group_check on %s raised %s: %s" % (g, type(e).__name__, e))
    check(C.group_data[g].shape == (n, 1), "%s kept its shape" % g)

print("\n== every earlier group is still there ==", flush=True)
for g in ("MC_bend", "MC_stretch", "MC_pin", "MC_velocity", "MC_friction",
          "MC_wind", "MC_inflate"):
    check(g in C.group_data, "%s still present" % g)

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
