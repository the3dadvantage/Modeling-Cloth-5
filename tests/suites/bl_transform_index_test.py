"""Regressions for bugs the benchmark harness turned up.

1. U.revert_transforms applied the inverse rotation twice, so a cloth OBJECT
   with any rotation saw its colliders turned the wrong way.
2. get_tri_edges / get_edge_normal_data build triangle->edge and edge->face
   maps from element indices after triangulating.  Guarded here in case a
   Blender version stops keeping those indices valid (5.2 keeps them).
3. In edit mode those helpers triangulated the LIVE edit bmesh -- the user's
   own mesh.  (Confirmed on the pre-fix code: 16 quads -> 32 triangles.)
"""
import bpy
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

UI = B.load()
U = UI.U
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("TEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


# ---- 1. revert_transforms is the inverse of apply_transforms --------------
B.clear()
ob = B.grid(3, 1.0, loc=(0.3, -1.2, 0.7), rot=(0.4, -0.9, 1.3))
ob.scale = (1.7, 0.4, 2.2)
bpy.context.view_layer.update()
rng = np.random.default_rng(1)
local = rng.uniform(-1, 1, (50, 3))
back = U.revert_transforms(ob, U.apply_transforms(ob, local))
err = float(np.abs(back - local).max())
check("revert_transforms inverts apply_transforms (rotated, non-uniform scale)",
      err < 1e-5, "(max error %.2g)" % err)

# ---- 2. triangle -> edge and edge -> face maps are consistent -------------
for name, ob in (("quad sphere", B.sphere(1.0, 12, 6)),
                 ("quad grid", B.grid(5, 2.0)),
                 ("ngon", B.mesh_object("Ngon", [(np.cos(a), np.sin(a), 0.0)
                                                 for a in np.linspace(0, 2 * np.pi, 7)[:-1]],
                                        [tuple(range(6))]))):
    for _ in range(3):                     # stale indices vary; try a few times
        tridex, tri_eidx, tri_eidxer, tri_edge_idx = U.get_tri_edges(ob)
        e_idxer, e_counts, e_add = U.get_edge_normal_data(ob)
        ok_edges = True
        for t, eids in zip(tridex, tri_edge_idx):
            for e in eids:
                if not (0 <= e < len(tri_eidx)) or not set(tri_eidx[e]) <= set(t):
                    ok_edges = False
        ok_faces = bool(np.all((0 <= e_idxer) & (e_idxer < len(tridex))))
        if ok_faces:
            for f, e in zip(e_idxer, e_add):
                if not set(tri_eidx[e]) <= set(tridex[f]):
                    ok_faces = False
                    break
        if not (ok_edges and ok_faces):
            break
    check("%s: every triangle's edges belong to it" % name, ok_edges)
    check("%s: every edge's faces contain it" % name, ok_faces)

# ---- 3. edit mode: the user's mesh is not triangulated --------------------
B.clear()
ob = B.grid(4, 2.0)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
n_polys = len(ob.data.polygons)
bpy.ops.object.mode_set(mode='EDIT')
U.get_tri_edges(ob)
U.get_edge_normal_data(ob)
bpy.ops.object.mode_set(mode='OBJECT')
check("edit mode: mesh keeps its quads", len(ob.data.polygons) == n_polys,
      "(%d -> %d polygons)" % (n_polys, len(ob.data.polygons)))

print("================ %s ================"
      % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
