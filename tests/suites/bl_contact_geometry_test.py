"""Contact geometry and move distances, on hand-built cases.

  * Each narrow-phase contact (the same functions recollide re-runs), in the
    shell, moves its contact point to exactly one thickness out.
  * A point just past a face's corner is pushed straight away from the corner.
    With the old barycentric slack it was pushed along the face's plane
    normal, which past a sharp tip points back across the tip.
  * Points either side of a cone tip are not pushed toward each other.
  * Edge contacts push along the real separation, not the collider edge's
    normal (whose plane runs past the end of the edge).
"""
import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

B.load()
OC = B.mc5().OC
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("TEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


TH = 0.05
CACHE = {"char_len": None}
UP = np.array([0.0, 0.0, 1.0])


def fake_cloth(co):
    C = types.SimpleNamespace()
    C.co = np.asarray(co, dtype=np.float32)
    C.start_co = C.co.copy()
    C.vc = len(co)
    C.ob = types.SimpleNamespace(MC_props=types.SimpleNamespace(
        ob_collision_tri_damping=1.0, ob_collision_edge_damping=1.0))
    return C


def moved(C, idx, mv):
    out = C.co.astype(np.float64).copy()
    np.add.at(out, idx, mv)
    return out


# ---- point_to_triangle: vertex 1 cm above a z=0 face ------------------------
C = fake_cloth([[0.25, 0.25, 0.01]])
tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
C.local_trico = tri
C.start_local_trico = tri.copy()
C.local_normals = UP[None]
C.start_local_normals = UP[None]
idx, mv = OC.point_to_triangle({}, C, np.array([0]), np.array([0]), TH, CACHE)
after = moved(C, idx, mv)
check("face: vertex lands exactly one thickness above the face",
      abs(after[0, 2] - TH) < 1e-6, "(z %.6f)" % after[0, 2])

# ---- triangle_to_point: a tip under a cloth triangle's interior -------------
for w_name, apex_xy in (("mid-triangle", (1 / 3, 1 / 3)), ("near a corner", (0.1, 0.1))):
    C = fake_cloth([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    C.tridex = np.array([[0, 1, 2]])
    C.local_vert_norms = UP[None]
    P = np.array([apex_xy[0], apex_xy[1], -0.01])        # 1 cm below the cloth
    data = {"joined_co": np.array([[P, P]])}
    idx, mv = OC.triangle_to_point(data, C, np.array([0]), np.array([0]), TH, CACHE)[:2]
    after = moved(C, idx, mv)
    w = np.array([1 - apex_xy[0] - apex_xy[1], apex_xy[0], apex_xy[1]])
    over = w @ after
    check("tip, %s: point over the tip lands exactly one thickness above it" % w_name,
          abs(over[2] - (P[2] + TH)) < 1e-6, "(z %.6f, want %.6f)" % (over[2], P[2] + TH))

# ---- edge_to_edge: cloth edge crossing over a collider edge ----------------
C = fake_cloth([[0.5, -0.5, 0.01], [0.5, 0.5, 0.01]])      # along y, 1 cm up
C.tri_eidx = np.array([[0, 1]])
ob = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])          # along x at z=0
data = {"joined_co": np.stack([ob, ob], axis=1), "tridex_eidx": np.array([[0, 1]]),
        "local_edge_normals": np.array([[0.0, 0.0, 1.0]]),
        "start_local_edge_normals": np.array([[0.0, 0.0, 1.0]])}
idx, mv = OC.edge_to_edge(data, C, np.array([0]), np.array([0]), TH, CACHE)[:2]
after = moved(C, idx, mv)
mid = 0.5 * (after[0] + after[1])
check("edge: closest point lands exactly one thickness out",
      abs(mid[2] - TH) < 1e-6, "(z %.6f)" % mid[2])

# the same, but the collider edge's normal is tilted 60 degrees off the real
# separation: the push must still follow the separation (straight up)
tilt = np.array([np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))])
data["local_edge_normals"] = tilt[None]
data["start_local_edge_normals"] = tilt[None]
C = fake_cloth([[0.5, -0.5, 0.01], [0.5, 0.5, 0.01]])
C.tri_eidx = np.array([[0, 1]])
idx, mv = OC.edge_to_edge(data, C, np.array([0]), np.array([0]), TH, CACHE)[:2]
after = moved(C, idx, mv)
mid = 0.5 * (after[0] + after[1])
check("edge, tilted edge normal: pushed along the real separation",
      abs(mid[0] - 0.5) < 1e-6 and abs(mid[2] - TH) < 1e-6,
      "(moved to x %.4f z %.4f; along the edge normal it would go sideways)" % (mid[0], mid[2]))

# ---- combining several contacts on one vertex (_project_contacts) ---------
def combined(parts, nv=1):
    x, _ = OC._project_contacts(nv, parts)
    return x


x = combined([(np.array([0, 0]), np.array([[0, 0, 0.03], [0, 0, 0.05]]))])
check("combine, pushes that agree: the strongest one, exactly",
      np.allclose(x[0], [0, 0, 0.05]), "(%s)" % np.round(x[0], 4))

# the cone-tip case: one push of 3 cm and thirty 1 mm pushes nearby in direction
tiny = np.tile([[0.0005, 0.0, 0.001]], (30, 1))
x = combined([(np.zeros(31, dtype=int), np.vstack([[[0, 0, 0.03]], tiny]))])
check("combine, one big push among 30 tiny ones: the big one is not diluted",
      x[0, 2] >= 0.03 - 1e-9 and np.linalg.norm(x[0]) < 0.0301,
      "(moved %s; averaged it would be %.4f)" % (np.round(x[0], 4), 0.03 / 31 + 0.001 * 30 / 31))

# a 90 degree crease: two faces at +-45 degrees each wanting 2 cm out
u1, u2 = np.array([-1, 0, 1]) / np.sqrt(2), np.array([1, 0, 1]) / np.sqrt(2)
x = combined([(np.array([0, 0]), np.vstack([u1 * 0.02, u2 * 0.02]))])
check("combine, crease: both faces met exactly (rises sqrt(2) x the push)",
      abs(x[0] @ u1 - 0.02) < 1e-9 and abs(x[0] @ u2 - 0.02) < 1e-9,
      "(moved %s)" % np.round(x[0], 4))

# opposing pushes (pinched): each met, nothing more
x = combined([(np.array([0, 0]), np.array([[0.01, 0, 0], [-0.01, 0, 0]]))])
check("combine, opposing pushes: no runaway", np.linalg.norm(x[0]) <= 0.0101,
      "(moved %s)" % np.round(x[0], 4))

# a triangle contact (tip under mid-triangle, wants its point up 3 cm) whose
# corner 0 is also lifted 3 cm by its own point contact: the triangle
# contact must ask only for what is still missing, not stack on top
w = np.array([1 / 3, 1 / 3, 1 / 3])
tri_moves = (w / (w @ w))[:, None] * np.array([0, 0, 0.03])
x = combined([(np.array([0]), np.array([[0, 0, 0.03]])),
              (np.array([0, 1, 2]), tri_moves, (np.zeros(3, dtype=int), w))], nv=3)
over = w @ x[:, 2]
check("combine, triangle contact with a corner already lifted: met, not overshot",
      abs(over - 0.03) < 1e-6 and x[0, 2] <= 0.03 + 1e-9,
      "(point over the tip rose %.4f, want 0.03; corners %s)" % (over, np.round(x[:, 2], 4)))


# ---- past a corner: pushed away from the corner, not along the plane ------
tri = np.array([[[0.0, 0.0, 0.0], [-0.5, -0.05, -0.8], [-0.5, 0.05, -0.8]]])
e1, e2 = tri[0, 1] - tri[0, 0], tri[0, 2] - tri[0, 0]
fn = np.cross(e1, e2)
fn /= np.linalg.norm(fn)
if fn[0] > 0:                   # outward normal of this face points to -x, up
    fn = -fn
# 3 cm past the corner along the face's own plane, 1 cm above it: inside the
# barycentric slack band (the face is 0.94 long), closest point is the corner
along = tri[0, 0] - 0.5 * (tri[0, 1] + tri[0, 2])
along /= np.linalg.norm(along)
p = (tri[0, 0] + 0.03 * along + 0.01 * fn)[None]
zone, dist, nrm, _ = OC._face_contact(tri, fn[None], p)
radial = (p[0] - tri[0, 0]) / np.linalg.norm(p[0] - tri[0, 0])
old_in, _, _ = OC.inside_triangles(tri, p, margin=-OC.OC_SWEPT_MARGIN)
check("past a corner: inside the slack band (the case being tested)",
      bool(old_in[0]) and bool(zone[0]))
want = np.linalg.norm(p[0] - tri[0, 0])
check("past a corner: distance measured to the corner",
      abs(dist[0] - want) < 1e-9, "(%.4f vs %.4f)" % (dist[0], want))
check("past a corner: pushed straight away from it, not along the face normal",
      np.dot(nrm[0], radial) > 0.999 and np.dot(nrm[0], fn) < 0.99,
      "(normal %s, the face normal would be %s)" % (np.round(nrm[0], 3), np.round(fn, 3)))

# ---- two points straddling a cone tip are not pushed together -------------
B.clear()
cone = B.collider(B.cone(0.5, 0.8, 32))
MC5 = B.mc5()
MC5.update_ob_colliders(MC5.OC_DATA)
co = np.array(B.world(cone, np.array([v.co[:] for v in cone.data.vertices])))
tris = co[MC5.OC_DATA["tridexes"]]
normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
apex = np.array([0.0, 0.0, 0.8])
worst_new, worst_old, n_new, n_old = 0.0, 0.0, 0, 0
for side in (+1, -1):
    pt = apex + np.array([side * 0.02, 0.0, 0.03])
    P = np.repeat(pt[None], len(tris), 0)
    # the rule this replaced: slack inside test, plane distance, face normal
    d = np.einsum("ij,ij->i", P - tris[:, 0], normals)
    old_in, _, _ = OC.inside_triangles(tris, P, margin=-OC.OC_SWEPT_MARGIN)
    old = old_in & (d < TH) & (d > -OC.OC_PUSH_CAP * TH)
    n_old += int(old.sum())
    if old.any():
        worst_old = min(worst_old, float((normals[old, 0] * side).min()))
    zone, dist, nrm, _ = OC._face_contact(tris, normals, P)
    act = zone & (dist < TH)
    n_new += int(act.sum())
    if act.any():
        # the x part of every push must point AWAY from the other point
        worst_new = min(worst_new, float((nrm[act, 0] * side).min()))
check("old plane rule: points beside the tip pushed back across it (the bug)",
      n_old > 0 and worst_old < -0.1,
      "(%d face contacts, most inward x push %.3f)" % (n_old, worst_old))
check("new rule: still in contact beside the tip", n_new > 0, "(%d face contacts)" % n_new)
check("new rule: points either side of the tip only pushed apart",
      worst_new >= -1e-9, "(most inward x push %.3f)" % worst_new)

print("================ %s ================"
      % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
