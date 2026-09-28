"""Gravity is a world direction, and can be aimed with an object.

It used to be added straight onto local +Z, so a rotated cloth fell along its
own axis rather than down.  An unrotated cloth must be completely unaffected by
the change.
"""
import bpy, os, glob, sys
import numpy as np
from mathutils import Euler

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


def close(a, b, tol=1e-6):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= tol


def build(rot=None, **settings):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6,
                                    size=2.0, location=(0, 0, 0))
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    if rot is not None:
        ob.rotation_euler = Euler(rot)
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = -9.8
    p.velocity = 0.98
    p.stretch = 1.25
    p.bend_force = 0.5
    for k, v in settings.items():
        setattr(p, k, v)
    scene.frame_set(p.reset_frame + 1)
    return ob


def world_of(ob, co):
    mw = np.array(ob.matrix_world, dtype=np.float64)
    return np.asarray(co, dtype=np.float64) @ mw[:3, :3].T + mw[:3, 3]


def sim(ob, steps=30):
    C = MC5.get_cloth(ob)
    start = world_of(ob, np.array(C.co).copy())
    for _ in range(steps):
        MC5.physics(C)
    return start, world_of(ob, np.array(MC5.get_cloth(ob).co))


print("== an unrotated cloth is unchanged ==", flush=True)
ob = build()
C = MC5.get_cloth(ob)
g = MC5.gravity_local(C)
check(close(g, [0.0, 0.0, -9.8 * 0.001], 1e-9),
      "gravity resolves to local -Z on an unrotated cloth %s" % np.round(g, 6))
start, end = sim(build())
drop = (end - start)[:, 2].mean()
check(drop < -0.1, "it falls (world z moved %.4f)" % drop)
check(abs((end - start)[:, 0].mean()) < 1e-6
      and abs((end - start)[:, 1].mean()) < 1e-6,
      "straight down, with no sideways drift")

print("\n== a rotated cloth still falls DOWN, not along its own axis ==",
      flush=True)
# pitched 90 degrees about X.  Local +Z now points along world -Y, so the old
# local-space gravity would have pushed it sideways in world terms.
ob = build(rot=(np.pi / 2.0, 0.0, 0.0))
C = MC5.get_cloth(ob)
g = MC5.gravity_local(C)
# gravity points world -Z, and pitching +90 about X carries local -Y round to
# world -Z, so downward reads as local -Y here
check(close(g, [0.0, -9.8 * 0.001, 0.0], 1e-9),
      "gravity becomes local -Y on a cloth pitched 90deg %s" % np.round(g, 6))

start, end = sim(build(rot=(np.pi / 2.0, 0.0, 0.0)))
delta = (end - start).mean(axis=0)
print("   world displacement %s" % np.round(delta, 4), flush=True)
check(delta[2] < -0.1, "world z went down (%.4f)" % delta[2])
check(abs(delta[0]) < 1e-5 and abs(delta[1]) < 1e-5,
      "and not sideways (%.5f, %.5f)" % (delta[0], delta[1]))

print("\n== tilted cloths all fall the same way ==", flush=True)
drops = []
for r in ((0, 0, 0), (0.4, 0, 0), (0, 0.7, 0), (0.3, -0.5, 1.1)):
    s, e = sim(build(rot=r), steps=25)
    d = (e - s).mean(axis=0)
    drops.append(d)
    print("   rot %-18s -> %s" % (np.round(r, 2), np.round(d, 4)), flush=True)
zs = [d[2] for d in drops]
check(max(zs) - min(zs) < 1e-4,
      "every orientation falls the same distance (%.6f spread)"
      % (max(zs) - min(zs)))
check(all(abs(d[0]) < 1e-4 and abs(d[1]) < 1e-4 for d in drops),
      "and none of them drift sideways")

print("\n== a gravity object aims it ==", flush=True)
ob = build()
bpy.ops.object.empty_add()
e = bpy.context.object
e.rotation_euler = Euler((0.0, np.pi / 2.0, 0.0))     # +Z now points world +X
bpy.context.view_layer.update()
ob.MC_props.gravity_object = e
C = MC5.get_cloth(ob)
g = MC5.gravity_local(C)
# the arrow's +Z is where things fall, and only the size of the gravity value
# counts, so a negative gravity does not flip it back
check(close(g, [9.8 * 0.001, 0.0, 0.0], 1e-8),
      "gravity now pulls along the object's +Z %s" % np.round(g, 6))
ob.MC_props.gravity = 9.8
check(close(MC5.gravity_local(MC5.get_cloth(ob)), g, 1e-9),
      "and the sign of the Gravity value makes no difference when aimed")
ob.MC_props.gravity = -9.8

start, end = sim(ob, steps=25)
delta = (end - start).mean(axis=0)
print("   aimed displacement %s" % np.round(delta, 4), flush=True)
check(abs(delta[0]) > 0.1, "the cloth moved along world X (%.4f)" % delta[0])
check(abs(delta[2]) < 1e-4, "and not down (%.5f)" % delta[2])

print("\n== the operator builds a usable empty ==", flush=True)
ob = build()
bpy.context.view_layer.objects.active = ob
res = bpy.ops.mc.gravity_add_object()
check(res == {'FINISHED'}, "mc.gravity_add_object ran (%s)" % (res,))
gob = ob.MC_props.gravity_object
check(gob is not None and gob.type == 'EMPTY', "it made an Empty and linked it")
bpy.context.view_layer.update()
C = MC5.get_cloth(ob)
g = MC5.gravity_local(C)
check(g[2] < 0.0, "which points gravity downward by default %s" % np.round(g, 6))

print("\n== edge cases ==", flush=True)
ob = build(gravity=0.0)
check(MC5.gravity_local(MC5.get_cloth(ob)) is None, "zero gravity -> None")

ob = build()
bpy.ops.object.empty_add()
e = bpy.context.object
ob.MC_props.gravity_object = e
e.scale = (0.0, 0.0, 0.0)
bpy.context.view_layer.update()
check(MC5.gravity_local(MC5.get_cloth(ob)) is None,
      "a zero-scaled aim object -> None, not a divide by zero")

ob = build()
bpy.ops.object.empty_add()
e = bpy.context.object
ob.MC_props.gravity_object = e
bpy.data.objects.remove(e)
check(ob.MC_props.gravity_object is None, "Blender cleared the deleted pointer")
g = MC5.gravity_local(MC5.get_cloth(ob))
check(g is not None and np.all(np.isfinite(g)),
      "and gravity falls back to world Z %s" % np.round(g, 6))

ob = build(rot=(0.0, 0.0, 0.0))
ob.scale = (2.0, 2.0, 2.0)
bpy.context.view_layer.update()
C = MC5.get_cloth(ob)
g = MC5.gravity_local(C)
check(close(g, [0.0, 0.0, -9.8 * 0.001 / 2.0], 1e-9),
      "a scaled cloth gets gravity in its own units %s" % np.round(g, 6))

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
