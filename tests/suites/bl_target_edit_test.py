"""Editing the target mesh while the simulation runs.

The point of the target workflow: the user keeps modelling the pattern while
the cloth is simulating.  New verts have to be placed in the *running* cloth
(mapped through the target's triangles), and everything already simulating has
to carry on untouched -- same positions, same velocity.

What is checked here:

  * verts are followed by id, not by what happens to be selected
  * a vert cut into an edge lands where that edge is in the sim, not where the
    target's rest shape puts it, and it follows the cloth's stretch
  * old verts keep their exact simulated positions and their velocity
  * deleting verts keeps the survivors' state
  * extruded verts, and geometry joined in from another mesh, do not crash
  * an edit with nothing old to hang the new verts on is survivable
  * vertex groups follow the verts
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

    target = bpy.data.objects.new("Target", cloth.data.copy())
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


def edit(fn, select=None):
    """Run a bmesh edit on the target, then set the selection the way a user's
    operator would leave it (or deliberately wrongly)."""
    bm = bmesh.new()
    bm.from_mesh(target.data)
    fn(bm)
    bm.to_mesh(target.data)
    bm.free()
    m = target.data
    if select is not None:
        sel = np.zeros(len(m.vertices), dtype=bool)
        sel[select] = True
        m.vertices.foreach_set('select', sel)
    m.update()


def subdivide_edge0(bm):
    bm.edges.ensure_lookup_table()
    bmesh.ops.subdivide_edges(bm, edges=[bm.edges[0]], cuts=1, use_grid_fill=True)


def drifting(cloth, C, vel=(0.01, 0.005, -0.02)):
    """Put the cloth in a state a frame advances exactly: no gravity, no
    damping, and the same velocity on every vert, so a step is a rigid
    translation the solver has nothing to correct.  That makes "did the edit
    disturb the sim" an exact comparison rather than a tolerance."""
    p = cloth.MC_props
    p.gravity = 0.0
    p.velocity = 1.0
    C.velocity[:] = np.array(vel, dtype=np.float32)
    return np.array(vel, dtype=np.float32)


print("== identity comes from ids, not the selection ==", flush=True)
cloth, target = build()
C = step(2)
v = drifting(cloth, C)
n0 = C.vc
co_before = np.array(C.co).copy()
vel_before = np.array(C.velocity).copy()

# the two ends of the edge that is about to be cut
bm = bmesh.new(); bm.from_mesh(target.data); bm.edges.ensure_lookup_table()
e0 = [v_.index for v_ in bm.edges[0].verts]
bm.free()

# every vert selected: the old code called everything selected "new"
edit(subdivide_edge0, select=slice(None))
C = step(1)
check(C.vc == n0 + 1, "one vert added (%d -> %d)" % (n0, C.vc))
co_after = np.array(C.co)
drift = np.abs(co_after[:n0] - (co_before + v)).max()
check(drift < 1e-5,
      "old verts advanced exactly one normal step, nothing else (%.3g)" % drift)

new_i = n0                      # the cut vert is appended
mid = co_after[e0].mean(axis=0)
d_mid = float(np.linalg.norm(co_after[new_i] - mid))
edge_len = float(np.linalg.norm(co_after[e0[1]] - co_after[e0[0]]))
check(d_mid < 0.02 * edge_len,
      "the cut vert landed on the simulated edge (%.4g of an edge off centre)"
      % (d_mid / edge_len))

tco = np.empty((len(target.data.vertices), 3), dtype=np.float32)
target.data.vertices.foreach_get('co', tco.ravel())
check(float(np.linalg.norm(co_after[new_i] - tco[new_i])) > 1e-3,
      "and not simply where the target's rest shape has it")

print("\n== the simulation carries on ==", flush=True)
vel_after = np.array(C.velocity)
carried = np.abs(vel_after[:n0] - vel_before).max()
check(carried < 1e-5, "old verts kept their velocity (max diff %.3g)" % carried)
check(float(np.linalg.norm(vel_after[new_i] - v)) < 1e-5,
      "the new vert starts out moving with the cloth, not from rest (%s)"
      % vel_after[new_i])
C = step(3)
check(np.isfinite(np.array(C.co)).all(), "and it keeps simulating")

print("\n== a new vert follows the cloth's stretch ==", flush=True)
# the plotting on its own: a grid, and the "simulated" copy of it stretched 2x
# in x.  A vert cut into the middle of a target edge belongs in the middle of
# that edge as the sim has it, not where the target's rest shape puts it.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=2.0)
g = bpy.context.object
n0 = len(g.data.vertices)
bm = bmesh.new(); bm.from_mesh(g.data); bm.edges.ensure_lookup_table()
e0 = [v_.index for v_ in bm.edges[0].verts]
bmesh.ops.subdivide_edges(bm, edges=[bm.edges[0]], cuts=1, use_grid_fill=True)
bm.to_mesh(g.data); bm.free()
g.data.update()

obm = U.get_bmesh(g)
tco = np.empty((len(g.data.vertices), 3), dtype=np.float32)
g.data.vertices.foreach_get('co', tco.ravel())
cco = tco * np.array([2.0, 1.0, 1.0], dtype=np.float32)
old = np.ones(tco.shape[0], dtype=bool)
old[n0:] = False

anchors = MC5.find_anchors(obm, np.arange(tco.shape[0])[~old], old)
plotted = MC5.plot_new_verts(anchors, tco, cco, obm)
got = plotted.get(n0)
want = cco[e0].mean(axis=0)
check(got is not None and np.allclose(got, want, atol=1e-4),
      "mid-edge in the target -> mid-edge in the stretched cloth (got %s, want %s)"
      % (None if got is None else np.round(got, 3), np.round(want, 3)))
check(not np.allclose(tco[n0] * [1, 1, 1], want, atol=1e-4),
      "(and the two are genuinely different places)")

print("\n== deleting verts ==", flush=True)
cloth, target = build()
C = step(2)
v = drifting(cloth, C)
n0 = C.vc
co_before = np.array(C.co).copy()
vel_before = np.array(C.velocity).copy()


def del_vert(bm):
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[3]], context='VERTS')


edit(del_vert)
C = step(1)
keep = np.array([i for i in range(n0) if i != 3])
check(C.vc == n0 - 1, "one vert removed (%d -> %d)" % (n0, C.vc))
drift = np.abs(np.array(C.co) - (co_before[keep] + v)).max()
check(drift < 1e-4, "survivors carried on undisturbed (%.3g)" % drift)
check(np.abs(np.array(C.velocity) - vel_before[keep]).max() < 1e-4,
      "survivors kept their velocity")

print("\n== extrude, and geometry joined in ==", flush=True)
cloth, target = build()
C = step(3)
n0 = C.vc


def extrude(bm):
    bm.faces.ensure_lookup_table()
    r = bmesh.ops.extrude_face_region(bm, geom=[bm.faces[0]])
    for g in r["geom"]:
        if isinstance(g, bmesh.types.BMVert):
            g.co.z += 0.4


edit(extrude)
C = step(1)
check(C.vc > n0 and np.isfinite(np.array(C.co)).all(),
      "extruded verts land and the sim survives (%d -> %d)" % (n0, C.vc))
C = step(2)
check(np.isfinite(np.array(C.co)).all(), "still finite a few frames later")

# a separate island: verts with no id at all, nothing old to anchor them to
cloth, target = build()
C = step(3)
n0 = C.vc


def add_island(bm):
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.3,
                          matrix=__import__("mathutils").Matrix.Translation((4, 4, 4)))


edit(add_island)
C = step(1)
check(C.vc == n0 + 4, "the island's verts were added (%d -> %d)" % (n0, C.vc))
check(np.isfinite(np.array(C.co)).all(),
      "an island with nothing to anchor to no longer crashes")

print("\n== vertex groups follow the verts ==", flush=True)
cloth, target = build()
n0 = len(cloth.data.vertices)
pin = np.zeros(n0, dtype=np.float32)
pin[[0, 1, 2]] = 1.0
U.write_vertex_group(cloth, "MC_pin", pin)
cloth.MC_props.cloth = True          # rebuild so the group is picked up
C = step(3)
check(float(np.array(C.group_data["MC_pin"]).ravel()[[0, 1, 2]].min()) > 0.99,
      "pin group read back before the edit")
edit(subdivide_edge0)
C = step(1)
after = np.array(C.group_data["MC_pin"]).ravel()
check(after.shape[0] == C.vc and float(after[[0, 1, 2]].min()) > 0.99,
      "pinned verts are still pinned after the edit")
check(float(after[n0]) == 0.0, "the new vert got the group's default (%.2f)" % after[n0])

print("\n== the id map itself ==", flush=True)
ids = np.array([0, 1, 2, 2, 3], dtype=np.int32)      # vert 2 was split
new_co = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.5, 0, 0], [3, 0, 0]], dtype=np.float32)
old_co = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
m = MC5.map_target_verts(ids, 4, new_co, old_co)
check(m.tolist() == [0, 1, 2, -1, 3],
      "a duplicated id keeps the copy nearest the old position (%s)" % m.tolist())
m2 = MC5.map_target_verts(np.array([0, 1, 9, 2], dtype=np.int32), 3)
check(m2.tolist() == [0, 1, -1, 2], "an id we never issued is new (%s)" % m2.tolist())

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
