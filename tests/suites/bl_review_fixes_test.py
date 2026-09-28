"""Tests for the six code-review fixes."""
import bpy, os, glob, sys, types, ast
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
OC, SC = MC5.OC, MC5.SC
scene = bpy.context.scene
rng = np.random.default_rng(5)

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


# ------------------------------------------------------------------ fix 1a
print("== 1. inside_triangles works at every scale ==", flush=True)
for L in (1.0, 0.1, 0.02, 0.01, 0.005, 0.001, 1e-4):
    tri = np.array([[[0, 0, 0], [L, 0, 0], [0, L, 0]]], dtype=np.float32)
    pin = np.array([[L / 4, L / 4, 0]], dtype=np.float32)
    pout = np.array([[5 * L, 5 * L, 0]], dtype=np.float32)
    for mod, tag in ((OC, "object"), (SC, "self")):
        ci, wi, _ = mod.inside_triangles(tri, pin)
        co_, wo, _ = mod.inside_triangles(tri, pout)
        good = (bool(ci[0]) and not bool(co_[0])
                and np.allclose(wi[0], [0.5, 0.25, 0.25], atol=1e-5))
        if not good or L in (0.01, 1e-4):
            check(good, "%-6s L=%-7g inside %s, 5L away %s, weights %s"
                  % (tag, L, bool(ci[0]), bool(co_[0]),
                     np.round(wi[0], 4).tolist()))

# random triangles at many scales: the inside test must agree with an
# independent area-based test
agree = True
for scale in (10.0, 1.0, 0.01, 1e-3):
    tris = rng.normal(size=(400, 3, 3)) * scale
    pts = tris.mean(axis=1) + rng.normal(size=(400, 3)) * scale * 0.4
    # project the points onto each triangle's plane so "inside" is meaningful
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    pts -= np.einsum("ij,ij->i", pts - tris[:, 0], n)[:, None] * n
    c, w, _ = OC.inside_triangles(tris, pts)
    a = lambda p, q, r: np.linalg.norm(np.cross(q - p, r - p), axis=1)
    full = a(tris[:, 0], tris[:, 1], tris[:, 2])
    sub = (a(pts, tris[:, 1], tris[:, 2]) + a(tris[:, 0], pts, tris[:, 2])
           + a(tris[:, 0], tris[:, 1], pts))
    ref = np.abs(sub - full) <= 1e-6 * full
    agree &= bool(np.all(c == ref))
check(agree, "on 1600 random triangles from 10 m to 1 mm it agrees with an "
             "independent area test")

print("\n   degenerate triangles", flush=True)
bad = np.array([
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],        # collinear
    [[0, 0, 0], [0, 0, 0], [0, 1, 0]],        # a repeated vertex
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],        # a point
], dtype=np.float64)
c, w, _ = OC.inside_triangles(bad, np.zeros((3, 3)))
check(not c.any(), "no degenerate triangle claims a point")
check(np.all(np.isfinite(w)) and np.allclose(w, 1.0 / 3.0),
      "and they get neutral, finite weights %s" % np.round(w, 3).tolist())

# ------------------------------------------------------------------ fix 1b
print("\n== 1. closest_point_segments works at every scale ==", flush=True)
for L in (1.0, 0.02, 0.01, 0.001, 1e-4):
    p1 = np.array([[-L / 2, 0, 0]]); q1 = np.array([[L / 2, 0, 0]])
    p2 = np.array([[0, -L / 2, L * 0.1]]); q2 = np.array([[0, L / 2, L * 0.1]])
    for mod, tag in ((OC, "object"), (SC, "self")):
        _, _, s, t = mod.closest_point_segments(p1, q1, p2, q2)
        if L in (0.01, 1e-4) or not (abs(s[0] - 0.5) < 1e-9 and abs(t[0] - 0.5) < 1e-9):
            check(abs(s[0] - 0.5) < 1e-9 and abs(t[0] - 0.5) < 1e-9,
                  "%-6s L=%-7g crossing edges give s %.4f t %.4f (want 0.5)"
                  % (tag, L, s[0], t[0]))


def brute(p1, q1, p2, q2, k=400):
    """Densely sampled minimum distance, as ground truth."""
    s = np.linspace(0, 1, k)
    A = p1[:, None] + (q1 - p1)[:, None] * s[None, :, None]
    B = p2[:, None] + (q2 - p2)[:, None] * s[None, :, None]
    d = np.linalg.norm(A[:, :, None] - B[:, None, :], axis=3)
    return d.reshape(len(p1), -1).min(axis=1)


worst = 0.0
for scale in (10.0, 1.0, 0.01, 1e-3):
    P = rng.normal(size=(4, 150, 3)) * scale
    cpa, cpb, _, _ = OC.closest_point_segments(*P)
    got = np.linalg.norm(cpa - cpb, axis=1)
    ref = brute(*P)
    worst = max(worst, float(np.max((got - ref) / scale)))
check(worst < 1e-3,
      "on 600 random segment pairs from 10 m to 1 mm it never exceeds a dense "
      "brute-force minimum (worst excess %.2e of scale)" % worst)

print("\n   degenerate segments", flush=True)
p = np.array([[0.0, 0, 0]]); q = np.array([[1.0, 0, 0]])
pt = np.array([[0.3, 2.0, 0]])
_, _, s, t = OC.closest_point_segments(p, q, pt, pt)
check(abs(s[0] - 0.3) < 1e-12 and t[0] == 0.0,
      "second segment a point: s projects onto the first (s %.3f, want 0.3) "
      "-- the case the old code did not handle" % s[0])
_, _, s, t = OC.closest_point_segments(pt, pt, p, q)
check(abs(t[0] - 0.3) < 1e-12 and s[0] == 0.0,
      "first segment a point: t projects onto the second (t %.3f)" % t[0])
cpa, cpb, s, t = OC.closest_point_segments(pt, pt, pt + 1.0, pt + 1.0)
check(np.all(np.isfinite(cpa)) and s[0] == 0 and t[0] == 0,
      "both points: finite, s = t = 0")
par = OC.closest_point_segments(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]),
                                np.array([[0.0, 1, 0]]), np.array([[1.0, 1, 0]]))
check(np.all(np.isfinite(par[0])) and abs(np.linalg.norm(par[0] - par[1]) - 1.0) < 1e-12,
      "exactly parallel edges: finite, distance 1.0")

# both copies identical
same = True
for _ in range(20):
    P = rng.normal(size=(4, 50, 3)) * rng.choice([1.0, 0.01])
    a = OC.closest_point_segments(*P)
    b = SC.closest_point_segments(*P)
    same &= all(np.array_equal(x, y) for x, y in zip(a, b))
    T = rng.normal(size=(50, 3, 3)); Q = rng.normal(size=(50, 3))
    same &= all(np.array_equal(x, y) for x, y in
                zip(OC.inside_triangles(T, Q), SC.inside_triangles(T, Q)))
check(same, "the object and self collision copies are bit-identical")

# ------------------------------------------------------------------ fix 4
print("\n== 4. object collision catches a diagonal tunnel ==", flush=True)


def fake(p0, p1, tri):
    tri = np.array([tri], dtype=np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return types.SimpleNamespace(
        co=np.array([p1], dtype=np.float32),
        start_co=np.array([p0], dtype=np.float32),
        local_trico=tri, start_local_trico=tri.copy(),
        local_normals=n, start_local_normals=n.copy())


TRI = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
thick = 0.05
C = fake([0.05, 0.05, 0.5], [0.85, 0.85, -0.5], TRI)
vh, mv = OC.point_to_triangle({}, C, np.array([0]), np.array([0]), thick, {})
check(vh is not None and 0 in vh.tolist(),
      "a vertex crossing the triangle's interior and ending beside it is caught")
check(vh is not None and mv[0, 2] > 0,
      "and pushed back up the way it came (dz %+.4f)"
      % (mv[0, 2] if vh is not None else float("nan")))

C = fake([1.2, 1.2, 0.5], [0.9, 0.9, -0.5], TRI)     # crosses OUTSIDE it
vh, mv = OC.point_to_triangle({}, C, np.array([0]), np.array([0]), thick, {})
check(vh is None, "a vertex that passes beside the triangle is left alone")

C = fake([0.2, 0.2, 0.5], [0.2, 0.2, -0.5], TRI)     # straight through
vh, mv = OC.point_to_triangle({}, C, np.array([0]), np.array([0]), thick, {})
check(vh is not None, "a straight-down tunnel is still caught")

C = fake([0.2, 0.2, 0.02], [0.2, 0.2, -0.02], TRI)   # sub-thickness wobble
vh, mv = OC.point_to_triangle({}, C, np.array([0]), np.array([0]), thick, {})
check(vh is None or np.all(np.abs(mv) <= thick * OC.OC_PUSH_CAP + 1e-9),
      "a sub-thickness wobble is not treated as a tunnel")

# ------------------------------------------------------------------ fix 8
print("\n== 8. clipped barycentric weights sum to 1 ==", flush=True)
w = OC._clip_weights(np.array([[-0.05, 0.5, 0.55], [0.2, 0.3, 0.5],
                               [-1.0, -1.0, 3.0], [-0.1, -0.1, -0.1]]))
check(np.allclose(w.sum(axis=1), 1.0), "every row sums to 1 %s"
      % np.round(w.sum(axis=1), 6).tolist())
check(np.allclose(w[1], [0.2, 0.3, 0.5]), "weights already inside are untouched")
check(np.all(w >= 0), "none negative")

# ------------------------------------------------------------------ fix 2
print("\n== 2. velocity damping no longer depends on the substep count ==",
      flush=True)


def resting():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=2.0, location=(0, 0, 0.02))
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=6.0)
    bpy.context.object.MC_props.ob_collision = True
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    ob.MC_props.ob_collision_radius = 0.05
    MC5.update_ob_colliders(MC5.OC_DATA)
    scene.frame_set(ob.MC_props.reset_frame + 1)
    ob.MC_props.collision_substeps = 1
    C = MC5.get_cloth(ob)
    MC5.physics(C)
    return ob, C


ob, C = resting()
rest = np.array(C.co).copy()
rest[:, 2] = 0.0
factors = {}
for n in (1, 2, 4, 8, 16):
    ob.MC_props.collision_substeps = n
    C.co[:] = rest
    C.start_co[:] = rest
    C.velocity[:] = 0.0
    C.velocity[:, 0] = 0.01
    MC5.collision_stage(C)
    factors[n] = float(np.abs(C.velocity[:, 0]).mean()) / 0.01
print("   tangential velocity kept: %s"
      % {n: round(f, 4) for n, f in factors.items()}, flush=True)
check(abs(factors[1] - 0.9) < 1e-6,
      "one substep keeps 0.9, exactly as before the fix (%.4f)" % factors[1])
check(max(factors.values()) - min(factors.values()) < 1e-6,
      "and every substep count now keeps the same (spread %.2e)"
      % (max(factors.values()) - min(factors.values())))

print("\n   self collision: feedback applied once however many substeps",
      flush=True)
calls = {"n": 0}
_real = SC.velocity_feedback


def counting(C):
    calls["n"] += 1
    return _real(C)


SC.velocity_feedback = counting
ob.MC_props.self_collision = True
for n in (1, 4, 16):
    ob.MC_props.collision_substeps = n
    calls["n"] = 0
    C.co[:] = rest
    C.start_co[:] = rest
    MC5.collision_stage(C)
    check(calls["n"] == 1, "%2d substeps -> velocity_feedback ran %d time"
          % (n, calls["n"]))
SC.velocity_feedback = _real

# the accumulate/apply contract, directly
nv = C.co.shape[0]
SC.begin_velocity(C)
C.velocity[:] = 1.0
h = np.zeros(nv, dtype=bool); h[:5] = True
z = np.zeros((nv, 3))
for _ in range(8):                     # eight substeps touching the same verts
    SC._vel_accumulate(C, h, h.copy(), z)
SC.velocity_feedback(C)
damp = float(ob.MC_props.sc_vel_damping)
check(np.allclose(C.velocity[:5], damp) and np.allclose(C.velocity[5:], 1.0),
      "eight substeps of contact damp by sc_vel_damping once (%.3f), not %.3g"
      % (float(C.velocity[0, 0]), damp ** 8))

# ------------------------------------------------------------------ fixes 3, 7
print("\n== 3 & 7. imports and dead state ==", flush=True)
for f in ("object_collide.py", "self_collide_2.py"):
    tree = ast.parse(open(os.path.join(SRC, f), encoding="utf-8").read())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names}
    rel = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
           and n.level == 1 and any(a.name == "utils" for a in n.names)]
    check("pc_tools" not in mods and "importlib" not in mods,
          "%s no longer imports pc_tools / importlib" % f)
    check(len(rel) == 1, "%s has the `from . import utils` package fallback" % f)
live = [f for f in glob.glob(os.path.join(SRC, "*.py"))
        if "stored_forces" in open(f, encoding="utf-8").read()]
check(not live, "stored_forces is gone everywhere %s"
      % [os.path.basename(f) for f in live])
check(not hasattr(MC5.get_cloth(ob), "stored_forces"),
      "and a freshly set-up cloth no longer allocates it")

print("\n== the sim still runs end to end ==", flush=True)
ob, C = resting()
ob.MC_props.gravity = -9.8
ob.MC_props.self_collision = True
ob.MC_props.collision_auto_substeps = True
for _ in range(20):
    MC5.physics(C)
check(np.all(np.isfinite(C.co)) and np.all(np.isfinite(C.velocity)),
      "object + self collision with auto substeps stays finite")

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
