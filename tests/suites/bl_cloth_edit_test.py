"""Editing the cloth mesh itself while a target is attached.

The other direction of the target workflow.  When the user edits the cloth
object rather than the pattern, the target has to be brought onto the new
topology -- keeping its own shape, because the target is what the rest lengths
and bend frames are read from -- and the sim has to carry on.

  * the target follows the cloth's topology
  * the target keeps its shape: old verts stay where the target had them
  * the sim carries on -- positions and velocity survive the edit
  * new verts arrive moving with the cloth around them
  * it works whether or not the cloth is the active object
  * the target does not overwrite the edit on the next frame
"""
import bpy, bmesh, os, sys, glob
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
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
U = MC5.U
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def build(sub=5):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=sub, y_subdivisions=sub, size=2.0)
    cloth = bpy.context.object
    cloth.name = "Cloth"
    cloth.data.vertices.foreach_set('select', np.zeros(len(cloth.data.vertices), dtype=bool))
    cloth.data.update()

    # a target with a shape of its own, so "the target kept its shape" means
    # something: pull it into a dome
    tmesh = cloth.data.copy()
    tco = np.empty((len(tmesh.vertices), 3), dtype=np.float32)
    tmesh.vertices.foreach_get('co', tco.ravel())
    tco[:, 2] += 0.3 * (1.0 - (tco[:, 0] ** 2 + tco[:, 1] ** 2) / 2.0)
    tmesh.vertices.foreach_set('co', tco.ravel())
    tmesh.update()
    target = bpy.data.objects.new("Target", tmesh)
    scene.collection.objects.link(target)

    bpy.context.view_layer.objects.active = cloth
    p = cloth.MC_props
    p.cloth = True
    p.gravity = -3.0
    p.animated = True
    p.target_object = target
    return cloth, target


def step(n=1):
    for _ in range(n):
        scene.frame_set(scene.frame_current + 1)
    return MC5.get_cloth(cloth)


def drifting(cloth, C, vel=(0.01, 0.005, -0.02)):
    """No gravity, no damping, one velocity for every vert: a frame becomes a
    rigid translation, so 'did the edit disturb the sim' is exact."""
    p = cloth.MC_props
    p.gravity = 0.0
    p.velocity = 1.0
    C.velocity[:] = np.array(vel, dtype=np.float32)
    return np.array(vel, dtype=np.float32)


def key_co(ob, key):
    k = ob.data.shape_keys.key_blocks[key]
    co = np.empty((len(k.data), 3), dtype=np.float32)
    k.data.foreach_get('co', co.ravel())
    return co


def mesh_co(ob):
    co = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())
    return co


def deselect(ob):
    """A selected vert is a grabbed vert and is held in place (see
    pin_selection), and an edit leaves its geometry selected.  That is by
    design, so these tests deselect first and measure the sim proper."""
    m = ob.data
    m.vertices.foreach_set('select', np.zeros(len(m.vertices), dtype=bool))
    m.edges.foreach_set('select', np.zeros(len(m.edges), dtype=bool))
    m.polygons.foreach_set('select', np.zeros(len(m.polygons), dtype=bool))
    m.update()


def subdivide_cloth(cloth, edge=0):
    """Cut an edge of the cloth, the way a knife cut does.  Blender carries
    the shape keys through, so MC_current and MC_target get interpolated
    values for the new vert."""
    bm = bmesh.new()
    bm.from_mesh(cloth.data)
    bm.edges.ensure_lookup_table()
    ends = [v.index for v in bm.edges[edge].verts]
    bmesh.ops.subdivide_edges(bm, edges=[bm.edges[edge]], cuts=1, use_grid_fill=True)
    bm.to_mesh(cloth.data)
    bm.free()
    deselect(cloth)
    return ends


print("== the target follows the cloth's topology ==", flush=True)
cloth, target = build()
C = step(3)
n0 = C.vc
tco_before = mesh_co(target).copy()
ends = subdivide_cloth(cloth)
C = step(1)
check(C.vc == n0 + 1, "the cloth gained a vert (%d -> %d)" % (n0, C.vc))
check(len(target.data.vertices) == C.vc,
      "the target followed (%d verts)" % len(target.data.vertices))
tco_after = mesh_co(target)
check(np.abs(tco_after[:n0] - tco_before).max() < 1e-5,
      "the target kept its own shape for the verts it already had (%.3g)"
      % float(np.abs(tco_after[:n0] - tco_before).max()))
mid = tco_before[ends].mean(axis=0)
check(float(np.linalg.norm(tco_after[n0] - mid)) < 0.05,
      "and the new target vert sits on the target's surface, not the cloth's")

print("\n== ids are re-issued on both ==", flush=True)
t_ids = MC5.read_mana(target.data)
c_ids = MC5.read_mana(cloth.data)
check(t_ids is not None and sorted(t_ids.tolist()) == list(range(C.vc)),
      "the target's ids are unique again")
check(c_ids is not None and sorted(c_ids.tolist()) == list(range(C.vc)),
      "the cloth's ids are unique again")

print("\n== the simulation carries on ==", flush=True)
cloth, target = build()
C = step(2)
v = drifting(cloth, C)
n0 = C.vc
co_before = np.array(C.co).copy()
vel_before = np.array(C.velocity).copy()
ends = subdivide_cloth(cloth)
C = step(1)
# Unlike an edit of the target, an edit of the cloth changes the springs
# themselves -- the new vert brings rest lengths of its own -- so the solver
# does respond.  What matters is that the response is a fraction of a step
# rather than the whole of it, which is what losing the velocity looked like.
step_size = float(np.abs(v).max())
drift = float(np.abs(np.array(C.co)[:n0] - (co_before + v)).max()) / step_size
check(drift < 0.2, "old verts advanced one normal step (off by %.1f%% of a step)"
      % (drift * 100))
carried = float(np.abs(np.array(C.velocity)[:n0] - vel_before).max()) / step_size
check(carried < 0.2, "old verts kept their velocity (off by %.1f%%)" % (carried * 100))
new_vel = np.array(C.velocity)[n0]
check(float(np.linalg.norm(new_vel - v)) / step_size < 0.2,
      "the new vert arrived moving with the cloth (%s vs %s)"
      % (np.round(new_vel, 4), v))
check(float(np.abs(new_vel).max()) > 0.5 * step_size,
      "(and not from rest, which is what the rebuild used to leave it at)")
new_pos = np.array(C.co)[n0]
want = np.array(C.co)[ends].mean(axis=0)
check(float(np.linalg.norm(new_pos - want)) < 0.05,
      "and it sits between the verts of the edge it was cut into")

print("\n== the cloth need not be the active object ==", flush=True)
cloth, target = build()
C = step(2)
n0 = C.vc
ends = subdivide_cloth(cloth)
bpy.context.view_layer.objects.active = target       # something else is active
C = step(1)
check(len(target.data.vertices) == C.vc,
      "the target still followed (%d vs %d)" % (len(target.data.vertices), C.vc))
bpy.context.view_layer.objects.active = cloth

print("\n== the edit is not undone on the next frame ==", flush=True)
co_after_edit = np.array(C.co).copy()
vc_after_edit = C.vc
C = step(3)
check(C.vc == vc_after_edit,
      "the vert count held after more frames (%d vs %d)" % (C.vc, vc_after_edit))
check(np.isfinite(np.array(C.co)).all(), "and the sim is still finite")

print("\n== deleting from the cloth ==", flush=True)
cloth, target = build()
C = step(2)
v = drifting(cloth, C)
n0 = C.vc
co_before = np.array(C.co).copy()
bm = bmesh.new()
bm.from_mesh(cloth.data)
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[3]], context='VERTS')
bm.to_mesh(cloth.data)
bm.free()
deselect(cloth)
C = step(1)
keep = np.array([i for i in range(n0) if i != 3])
check(C.vc == n0 - 1, "the cloth lost a vert (%d -> %d)" % (n0, C.vc))
check(len(target.data.vertices) == C.vc, "the target followed")
drift = float(np.abs(np.array(C.co) - (co_before[keep] + v)).max()) / float(np.abs(v).max())
check(drift < 0.2, "the survivors carried on (off by %.1f%% of a step)" % (drift * 100))
vel = np.array(C.velocity)
check(float(np.abs(vel).min()) > 0.5 * float(np.abs(v).min()),
      "and none of them were reset to rest")

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
