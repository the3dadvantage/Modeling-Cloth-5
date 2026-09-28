"""Collision benchmark harness -- the reference the C++ port is measured against.

Builds deterministic cloth/collider scenes, steps the real solver, and records
where every vertex ended up plus a few physical measurements.  A run can be
stored as a baseline and later runs compared against it.

Scenes are chosen so every collision code path actually runs:

  object collision   point->triangle, triangle->point, edge->edge, friction,
                     a moving collider, the substep loop, and small (1 cm)
                     geometry.  ob_point_tris, ob_edges and ob_friction are all
                     OFF by default, so scenes left at defaults would only ever
                     exercise point->triangle.
  self collision     vertex->triangle and edge->edge between two layers, with
                     and without substeps.

Lessons built in, each learned the hard way:

  * Engagement is CHECKED, not assumed.  A cloth that falls straight through
    its collider is perfectly deterministic and matches its own baseline
    forever.  Every scene counts real contacts during the run and has a
    physical test; capture refuses to store a baseline that fails either.
  * Frame counts live in exactly one place (Scene.frames).  A test once saved
    12-frame runs over 40-frame baselines and produced phantom failures.
  * Every physical parameter and toggle is set explicitly, so changing a
    default somewhere cannot silently move every baseline.
  * Selected vertices are pinned by the solver, so every mesh is built with
    nothing selected.
  * Meshes come from explicit vertex/face lists: Blender 5.2's primitive
    generators return faces in a different order on each call.
  * gravity is added straight onto velocity, so it must be negative to fall.

Baselines are .npz files in tests/baselines, each carrying a fingerprint of the
code that produced it, so a check can say whether a difference is expected.
"""

import bpy
import bmesh
import os
import glob
import time
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# MC_SRC points the harness at another copy of the addon (a backup, a branch)
SRC = os.path.normpath(os.environ.get("MC_SRC") or os.path.join(HERE, "..", "Python"))
BASE_DIR = os.path.join(HERE, "baselines")
# Which collision implementation runs: "PYTHON" (the reference the baselines
# are captured from) or "CPP" (checked against them).  run_bench --backend.
BACKEND = os.environ.get("MC_BACKEND", "PYTHON").upper()

_UI = None


# ------------------------------------------------------------------ loading
def load():
    """Load the addon from the source folder and register it, once."""
    global _UI
    if _UI is not None:
        return _UI
    for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        name = os.path.basename(f)
        if name in bpy.data.texts:
            continue
        t = bpy.data.texts.new(name)
        t.from_string(open(f, encoding="utf-8", errors="replace").read())
        # as when the addon is loaded from disk: DLLs beside the code are found
        t.filepath = f
    _UI = bpy.data.texts["MC_ui.py"].as_module()
    _UI.register()
    _UI.U.popup_error = lambda *a, **k: None     # needs a window manager
    return _UI


def mc5():
    return load().MC5


def fingerprint():
    """What produced a result: every module, the solver DLL, and Blender."""
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    dll = load().U.find_dll(required=False)
    if dll and os.path.isfile(dll):
        h.update(open(dll, "rb").read())
    return {"code": h.hexdigest()[:16],
            "blender": bpy.app.version_string,
            "numpy": np.__version__,
            "backend": BACKEND,
            "dll": os.path.basename(dll) if dll else "none"}


# ------------------------------------------------------------------ helpers
def clear():
    MC5 = mc5()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.objects):
        for b in list(block):
            if b.users == 0:
                block.remove(b)
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []


def deselect(ob):
    n = len(ob.data.vertices)
    ob.data.vertices.foreach_set('select', np.zeros(n, dtype=bool))
    ob.data.update()


# Meshes are built from explicit vertex / face lists, never Blender's
# primitive generators.  In 5.2 primitive_uv_sphere_add AND
# bmesh.ops.create_uvsphere both return the same faces in a different ORDER
# from one call to the next (thread timing).  Contacts accumulate in face
# order, so float round-off -- and from there the whole run -- changed between
# identical runs.  from_pydata keeps exactly the order given here.
def mesh_object(name, verts, faces, loc=(0, 0, 0), rot=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [],
                   [tuple(map(int, f)) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = rot
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()
    deselect(ob)
    return ob


def grid(subs, size, loc=(0, 0, 0), rot=(0, 0, 0)):
    """Same layout as primitive_grid_add: (subs+1)^2 verts, x fastest."""
    n = subs + 1
    lin = np.linspace(-size / 2.0, size / 2.0, n)
    xs, ys = np.meshgrid(lin, lin)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i)
             for j in range(subs) for i in range(subs)]
    return mesh_object("Grid", verts, faces, loc, rot)


def cube(size, loc=(0, 0, 0), rot=(0, 0, 0)):
    """Same layout as primitive_cube_add."""
    h = size / 2.0
    verts = [(sx * h, sy * h, sz * h) for sx in (-1, 1) for sy in (-1, 1)
             for sz in (-1, 1)]
    faces = [(0, 1, 3, 2), (2, 3, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0),
             (2, 6, 4, 0), (7, 3, 1, 5)]
    return mesh_object("Cube", verts, faces, loc, rot)


def sphere(radius, segments, rings, loc=(0, 0, 0)):
    """UV sphere: two poles, rings-1 rings of `segments` verts, faces outward."""
    verts = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        th = np.pi * i / rings
        for j in range(segments):
            ph = 2.0 * np.pi * j / segments
            verts.append((radius * np.sin(th) * np.cos(ph),
                          radius * np.sin(th) * np.sin(ph),
                          radius * np.cos(th)))
    verts.append((0.0, 0.0, -radius))
    bottom = len(verts) - 1

    def ring(i, j):                       # vert index, ring i in 1..rings-1
        return 1 + (i - 1) * segments + (j % segments)

    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(segments)]
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append((ring(i, j), ring(i + 1, j), ring(i + 1, j + 1),
                          ring(i, j + 1)))
    faces += [(ring(rings - 1, j + 1), ring(rings - 1, j), bottom)
              for j in range(segments)]
    return mesh_object("Sphere", verts, faces, loc)

def cone(radius, height, segments, loc=(0, 0, 0)):
    """Sharp cone, apex straight up at loc + (0, 0, height), base capped."""
    verts = [(0.0, 0.0, height)]
    for j in range(segments):
        a = 2.0 * np.pi * j / segments
        verts.append((radius * np.cos(a), radius * np.sin(a), 0.0))
    verts.append((0.0, 0.0, 0.0))
    base = len(verts) - 1
    faces = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    faces += [(base, 1 + (j + 1) % segments, 1 + j) for j in range(segments)]
    return mesh_object("Cone", verts, faces, loc)


def collider(ob, friction=1.0):
    ob.MC_props.ob_collision = True
    ob.MC_props.ob_collider_friction = friction
    deselect(ob)
    return ob


# Every property that affects a result, set explicitly.  Defaults are not
# trusted: if one changes, every baseline would move at once.
PIN = dict(
    gravity=-9.8, velocity=0.98, stretch=1.25, bend_force=0.5,
    bend_stabilize=1.0, pre_shrink=1.0, shrink_grow=1.0,
    preserve_boundary_edges=False,
    inflate=0.0, air_drag=0.0, inverted_air_drag=0.0, wind=(0.0, 0.0, 0.0),
    cl_point_tris=True, ob_point_tris=False, ob_edges=False,
    ob_collision_radius=0.05, ob_friction=0.0, ob_static_threshold=0.0,
    ob_collision_tri_damping=1.0, ob_collision_edge_damping=1.0,
    ob_recollide=False, ob_recollide_every=1,
    self_collision=False, sc_radius=0.05, sc_edges=True, sc_damping=1.0,
    sc_vel_damping=0.5,
    sc_box_depth=5, sc_box_depth_auto=False,
    ob_box_depth=5, ob_box_depth_auto=False,
    collision_substeps=1, collision_auto_substeps=False,
    collision_max_substeps=8, collision_substep_margin=0.5,
)

ALL_PHASES = dict(cl_point_tris=True, ob_point_tris=True, ob_edges=True)


def make_cloth(ob, **overrides):
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    for k, v in PIN.items():
        setattr(p, k, v)
    for k, v in overrides.items():
        setattr(p, k, v)
    return ob


def world(ob, co):
    mw = np.array(ob.matrix_world, dtype=np.float64)
    return np.asarray(co, dtype=np.float64) @ mw[:3, :3].T + mw[:3, 3]


# ------------------------------------------------------------------- scenes
class Scene:
    """A scene: how to build it, how long to run it, and how to tell that
    collision really took part."""

    def __init__(self, name, frames, build, engaged, doc):
        self.name = name
        self.frames = frames
        self.build = build          # () -> (cloth_ob, extra)
        self.engaged = engaged      # (result) -> (bool, message)
        self.doc = doc


def _need(res, key, n=1):
    got = int(res.get(key, 0))
    return got >= n, "%s %d" % (key, got)


def build_oc_floor():
    ob = grid(16, 2.0, (0, 0, 0.6))
    collider(grid(4, 6.0))
    make_cloth(ob)                                  # default phases only
    return ob, {"floor_z": 0.0, "margin": 0.05}


def engaged_oc_floor(r):
    ok_c, m_c = _need(r, "oc_contacts")
    low = r["final_min_z"]
    ok_p = 0.0 < low < 0.1
    return ok_c and ok_p, "%s; rests at world z %.4f" % (m_c, low)


def build_oc_sphere():
    ob = grid(16, 2.0, (0, 0, 0.68))
    collider(sphere(0.6, 24, 12))
    make_cloth(ob, gravity=-1.5, **ALL_PHASES)
    return ob, {"sphere": ((0, 0, 0), 0.6), "margin": 0.05}


def engaged_oc_sphere(r):
    ok_c, m_c = _need(r, "oc_contacts")
    top = r["final_max_z"]
    ok_p = 0.6 < top < 0.72 and r["sphere_pen_worst"] < 0.05
    return ok_c and ok_p, ("%s; top at %.4f, worst penetration %.4f"
                           % (m_c, top, r["sphere_pen_worst"]))


def build_oc_ridge():
    ob = grid(16, 2.0, (0, 0, 0.8))
    # a cube turned 45 degrees about Y puts a sharp ridge on top: cloth draped
    # over it contacts the ridge EDGE and the corner vertices, which is what
    # edge->edge and triangle->point exist for
    collider(cube(1.0, rot=(0, np.radians(45), 0)))
    make_cloth(ob, gravity=-1.5, **ALL_PHASES)
    return ob, {"ridge_z": 0.5 * np.sqrt(2.0), "margin": 0.05}


def engaged_oc_ridge(r):
    ok_c, m_c = _need(r, "oc_contacts")
    top = r["final_max_z"]
    ridge = 0.5 * np.sqrt(2.0)
    ok_p = ridge - 0.02 < top < ridge + 0.12
    return ok_c and ok_p, "%s; top %.4f over a ridge at %.4f" % (m_c, top, ridge)


def build_oc_substeps():
    # dropped from high enough that one collision test per frame tunnels
    # (measured: contact is lost once a step exceeds ~2x the margin); eight
    # substeps hold it.  Exercises the substep loop and the once-per-step
    # velocity feedback.
    ob = grid(12, 2.0, (0, 0, 1.2))
    collider(sphere(0.6, 24, 12))
    make_cloth(ob, collision_substeps=8, **ALL_PHASES)
    return ob, {"sphere": ((0, 0, 0), 0.6), "margin": 0.05}


def engaged_oc_substeps(r):
    ok_c, m_c = _need(r, "oc_contacts")
    top = r["final_max_z"]
    return ok_c and top > 0.3, "%s; top at %.4f (held, not tunnelled)" % (m_c, top)


def build_oc_moving():
    # The sphere moves 0.03 per frame, under the 0.05 margin.  Substeps only
    # subdivide the CLOTH's motion; the collider still jumps its whole frame
    # in the first substep, so a collider faster than the margin punches
    # through whatever the substep count (a known gap, not baselined).
    ob = grid(12, 2.0, (0, 0, 0))
    mover = collider(sphere(0.4, 16, 8, (0, 0, -0.45)))
    make_cloth(ob, gravity=-0.5, collision_auto_substeps=True,
               collision_max_substeps=16, **ALL_PHASES)
    return ob, {"mover": mover, "speed": 0.03, "margin": 0.05}


def engaged_oc_moving(r):
    # the sphere's top ends at -0.45 + 40*0.03 + 0.4 = 1.15; the cloth has to
    # be riding on it
    ok_c, m_c = _need(r, "oc_contacts")
    return ok_c and r["final_max_z"] > 1.1, ("%s; lifted to %.4f"
                                             % (m_c, r["final_max_z"]))


def build_oc_transformed():
    # The cloth OBJECT is rotated and non-uniformly scaled; the collider is
    # rotated too.  Colliders are brought into the cloth's local space every
    # step, and that inverse transform was once wrong for any rotation: an
    # unrotated cloth never showed it, a rotated one fell through the floor.
    tilt = np.radians(20.0)
    normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    ob = grid(12, 1.0, tuple(normal * 0.06), rot=(0, tilt, 0))
    ob.scale = (1.5, 0.75, 1.0)
    bpy.context.view_layer.update()
    collider(grid(4, 5.0, rot=(0, tilt, 0)))
    make_cloth(ob, gravity=-3.0)
    return ob, {"plane_normal": normal, "margin": 0.05}


def engaged_oc_transformed(r):
    ok_c, m_c = _need(r, "oc_contacts")
    return ok_c and 0.0 <= r["plane_gap_min"] and r["plane_gap_max"] < 0.15, (
        "%s; gap to the tilted plane %.4f..%.4f"
        % (m_c, r["plane_gap_min"], r["plane_gap_max"]))


def build_oc_friction():
    tilt = np.radians(20.0)
    # a plane tilted 20 degrees, cloth resting on it; with friction on it
    # should barely slide (tan 20 deg = 0.36 < mu)
    normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    ob = grid(12, 1.5, tuple(normal * 0.06), rot=(0, tilt, 0))
    collider(grid(4, 5.0, rot=(0, tilt, 0)), friction=1.0)
    make_cloth(ob, gravity=-3.0, ob_friction=0.6, **ALL_PHASES)
    return ob, {"plane_normal": normal, "margin": 0.05}


def engaged_oc_friction(r):
    ok_c, m_c = _need(r, "oc_contacts")
    ok_f, m_f = _need(r, "friction_calls")
    on_plane = r["plane_gap_min"] >= 0.0 and r["plane_gap_max"] < 0.15
    return ok_c and ok_f and on_plane, (
        "%s; %s; gap to the plane %.4f..%.4f, slid %.4f"
        % (m_c, m_f, r["plane_gap_min"], r["plane_gap_max"], r["slide"]))


def build_oc_small():
    # 1 cm geometry: the scale the old absolute epsilon broke.  Cloth edges
    # are 1 cm and so are the collider's triangles.
    ob = grid(20, 0.2, (0, 0, 0.068))
    collider(sphere(0.06, 36, 18))
    # gravity scaled with the geometry.  At -0.3 the hanging edges speed up
    # until one step moves more than the 5 mm margin and the cloth punches
    # through at frame 37: that is the substep problem (4 substeps hold it),
    # not a small-geometry one, and oc_substeps already covers it.
    make_cloth(ob, gravity=-0.15, ob_collision_radius=0.005, **ALL_PHASES)
    return ob, {"sphere": ((0, 0, 0), 0.06), "margin": 0.005}


def engaged_oc_small(r):
    ok_c, m_c = _need(r, "oc_contacts")
    top = r["final_max_z"]
    ok_p = 0.058 < top < 0.075 and r["sphere_pen_worst"] < 0.005
    return ok_c and ok_p, ("%s; top %.4f, worst penetration %.5f"
                           % (m_c, top, r["sphere_pen_worst"]))


CONE_APEX = 0.8
# grid spacing is 2/30 = 0.0667; this offset puts the apex inside a triangle
# rather than exactly under a cloth vertex
CONE_GRID_OFFSET = (0.013, 0.021)
# gravity -1: at -3 the springs cannot carry the sheet's own weight (edges
# stretch 2x and the whole sheet slides down the cone), and no contact holds
# that.  At -1 the sheet can rest on the tip.
CONE_SUBS, CONE_SIZE, CONE_GRAVITY, CONE_BEND = 30, 2.0, -1.0, 20.0


def build_oc_cone():
    # A stiff grid dropped onto a sharp tip.  With high bend stiffness the bend
    # iterations flatten the grid and drag the verts at the tip back down
    # through it; the single collision pass at the end then shoves only those
    # verts up again, leaving a spike.  Recollide inside the solver loop is
    # meant to fix exactly this.  The tip is a collider VERTEX against cloth
    # triangles, so ob_point_tris has to be on.
    ob = grid(CONE_SUBS, CONE_SIZE, CONE_GRID_OFFSET + (CONE_APEX + 0.08,))
    collider(cone(0.5, CONE_APEX, 32))
    # recollide ON: it is what this scene tests.  --set ob_recollide=False
    # shows the tip going through (clearance -0.18, edges stretched 2x).
    make_cloth(ob, gravity=CONE_GRAVITY, bend_force=CONE_BEND,
               stretch=CONE_BEND + 1.25, ob_recollide=True, ob_recollide_every=1,
               **ALL_PHASES)
    return ob, {"cone_apex": (0.0, 0.0, CONE_APEX), "margin": 0.05}


def engaged_oc_cone(r):
    ok_c, m_c = _need(r, "oc_contacts")
    # the cloth must have come to rest on the tip, not slid off
    # and the tip must never be through it, at any frame
    ok_p = CONE_APEX - 0.02 < r["final_max_z"] < CONE_APEX + 0.15
    ok_t = r["tip_clear_worst"] > 0.0
    return ok_c and ok_p and ok_t, (
        "%s; top %.4f, tip clearance worst %+.4f final %+.4f, spike %.4f, "
        "stretch %.2f" % (m_c, r["final_max_z"], r["tip_clear_worst"],
                          r["tip_clear"], r["spike"], r["stretch_max"]))


def _tip_clearance(wc, tridex, apex):
    """Height of the cloth surface above the apex, measured on the cloth
    triangle straight over it.  Negative means the tip is through the cloth
    even if every VERTEX is outside the cone."""
    ax, ay, az = apex
    t = wc[tridex]                                    # (T,3,3)
    a, b, c = t[:, 0, :2], t[:, 1, :2], t[:, 2, :2]
    v0, v1, p = b - a, c - a, np.array([ax, ay]) - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-14
    den = np.where(ok, den, 1.0)
    u = (p[:, 0] * v1[:, 1] - v1[:, 0] * p[:, 1]) / den
    v = (v0[:, 0] * p[:, 1] - p[:, 0] * v0[:, 1]) / den
    inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    if not inside.any():
        return float("nan")
    z = t[:, 0, 2] + u * (t[:, 1, 2] - t[:, 0, 2]) + v * (t[:, 2, 2] - t[:, 0, 2])
    return float(z[inside].max() - az)


def _mesh_topology(ob):
    """Edges, and which verts are on the boundary (an edge with one face)."""
    me = ob.data
    edges = np.array([e.vertices[:] for e in me.edges], dtype=np.int64)
    key = {tuple(sorted(e)): i for i, e in enumerate(edges.tolist())}
    faces_per_edge = np.zeros(len(edges), dtype=np.int64)
    for p in me.polygons:
        vs = list(p.vertices)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            faces_per_edge[key[tuple(sorted((a, b)))]] += 1
    boundary = np.zeros(len(me.vertices), dtype=bool)
    boundary[edges[faces_per_edge == 1].ravel()] = True
    return edges, boundary


def _stretch_max(wc, rest_wc, edges):
    """Longest edge as a multiple of its starting length."""
    now = np.linalg.norm(wc[edges[:, 0]] - wc[edges[:, 1]], axis=1)
    rest = np.linalg.norm(rest_wc[edges[:, 0]] - rest_wc[edges[:, 1]], axis=1)
    return float((now / np.maximum(rest, 1e-12)).max())


def _spike(wc, edges, boundary):
    """Largest distance of an interior vertex from the average of its
    neighbours: how sharply one vertex sticks out of the surface around it.
    Boundary verts are skipped -- their neighbours are all on one side, so a
    flat grid's corners would otherwise dominate."""
    n = wc.shape[0]
    acc = np.zeros_like(wc)
    cnt = np.zeros(n)
    np.add.at(acc, edges[:, 0], wc[edges[:, 1]])
    np.add.at(acc, edges[:, 1], wc[edges[:, 0]])
    np.add.at(cnt, edges[:, 0], 1.0)
    np.add.at(cnt, edges[:, 1], 1.0)
    lap = wc - acc / np.maximum(cnt, 1.0)[:, None]
    lap = np.linalg.norm(lap, axis=1)[~boundary]
    return float(lap.max()) if lap.size else 0.0


def _folded(subs=20):
    """A sheet folded in half into two layers 0.02 apart, over a floor."""
    ob = grid(subs, 2.0, (0, 0, 0.3))
    me = ob.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    top = co[:, 0] > 0.0
    co[top, 0] = -co[top, 0]
    co[top, 2] += 0.02
    me.vertices.foreach_set('co', co.ravel())
    me.update()
    collider(grid(4, 6.0))
    return ob, top


def build_sc_folded():
    ob, top = _folded()
    make_cloth(ob, gravity=-3.0, self_collision=True, sc_radius=0.03,
               ob_collision_radius=0.04)
    return ob, {"layer_top": top, "floor_z": 0.0, "margin": 0.04}


def build_sc_folded_substeps():
    ob, top = _folded()
    make_cloth(ob, gravity=-3.0, self_collision=True, sc_radius=0.03,
               ob_collision_radius=0.04, collision_substeps=4)
    return ob, {"layer_top": top, "floor_z": 0.0, "margin": 0.04}


def engaged_sc_folded(r):
    ok_s, m_s = _need(r, "sc_contacts")
    gap = r["layer_gap"]
    # held apart at roughly sc_radius; if self collision had not engaged the
    # two layers would have merged to ~0
    return ok_s and 0.015 < gap < 0.06, "%s; layers held %.4f apart" % (m_s, gap)


SCENES = {s.name: s for s in (
    Scene("oc_floor", 40, build_oc_floor, engaged_oc_floor,
          "drop onto a floor, default phases (point->triangle only)"),
    Scene("oc_sphere", 60, build_oc_sphere, engaged_oc_sphere,
          "slow drape over a sphere, all three phases"),
    Scene("oc_ridge", 60, build_oc_ridge, engaged_oc_ridge,
          "drape over a cube ridge: edge->edge and triangle->point"),
    Scene("oc_substeps", 22, build_oc_substeps, engaged_oc_substeps,
          "fast drop held by 8 substeps: substep loop, velocity feedback"),
    Scene("oc_moving", 40, build_oc_moving, engaged_oc_moving,
          "a collider rising through the cloth, auto substeps"),
    Scene("oc_friction", 40, build_oc_friction, engaged_oc_friction,
          "cloth on a 20 degree slope with friction"),
    Scene("oc_transformed", 40, build_oc_transformed, engaged_oc_transformed,
          "rotated, non-uniformly scaled cloth on a rotated collider"),
    Scene("oc_small", 60, build_oc_small, engaged_oc_small,
          "1 cm edges on a 1 cm-tessellated sphere"),
    Scene("oc_cone", 40, build_oc_cone, engaged_oc_cone,
          "stiff grid on a sharp cone tip: collider vertex -> cloth triangle"),
    Scene("sc_folded", 40, build_sc_folded, engaged_sc_folded,
          "two stacked layers: self vertex->triangle and edge->edge"),
    Scene("sc_folded_substeps", 40, build_sc_folded_substeps, engaged_sc_folded,
          "the same with 4 substeps: self collision in the substep loop"),
)}


# ---------------------------------------------------------------------- run
class _Counters:
    """Wraps the collision modules to count real contacts during a run."""

    def __init__(self, MC5):
        self.MC5 = MC5
        self.n = {"oc_contacts": 0, "sc_contacts": 0, "friction_calls": 0}

    def __enter__(self):
        OC, SC = self.MC5.OC, self.MC5.SC
        self._oc, self._sc = OC._vel_accumulate, SC._vel_accumulate
        self._fr = OC._friction_applied
        n = self.n

        def oc(C, any_hit, total_step):
            n["oc_contacts"] += int(np.count_nonzero(any_hit))
            return self._oc(C, any_hit, total_step)

        def sc(C, any_hit, vert_swept, total_step):
            n["sc_contacts"] += int(np.count_nonzero(any_hit))
            return self._sc(C, any_hit, vert_swept, total_step)

        def fr(count):
            n["friction_calls"] += int(count)
            return self._fr(count)

        OC._vel_accumulate, SC._vel_accumulate = oc, sc
        OC._friction_applied = fr
        return self

    def __exit__(self, *exc):
        OC, SC = self.MC5.OC, self.MC5.SC
        OC._vel_accumulate, SC._vel_accumulate = self._oc, self._sc
        OC._friction_applied = self._fr
        return False


def _sphere_pen(wc, centre, radius):
    d = np.linalg.norm(wc - np.asarray(centre, dtype=np.float64), axis=1)
    inside = d < radius
    return float((radius - d[inside]).max()) if inside.any() else 0.0


def run_scene(name, overrides=None):
    """Build, step and measure one scene.  `overrides` sets cloth properties
    after the build -- for comparing variants, never for baselines."""
    MC5 = mc5()
    scene_def = SCENES[name]
    clear()
    ob, extra = scene_def.build()
    for k, v in (overrides or {}).items():
        setattr(ob.MC_props, k, v)
    bpy.context.scene.MC_props.collision_backend = BACKEND
    if BACKEND == "CPP" and MC5.NATIVE.get() is None:
        raise RuntimeError("C++ collision requested but unavailable: %s" % MC5.NATIVE.LAST_ERROR)
    if MC5.OC_DATA["obs"] or any(o.MC_props.ob_collision for o in bpy.data.objects):
        MC5.update_ob_colliders(MC5.OC_DATA)

    scene = bpy.context.scene
    reset = ob.MC_props.reset_frame
    C = MC5.get_cloth(ob)
    start_w = world(ob, np.array(C.co))
    sphere = extra.get("sphere")
    normal = extra.get("plane_normal")
    mover, speed = extra.get("mover"), extra.get("speed", 0.0)
    pen_worst, floor_worst, gap_max, gap_min = 0.0, 0.0, 0.0, np.inf
    apex = extra.get("cone_apex")
    tip_worst = np.inf
    tridex = np.asarray(C.tridex)
    edges, boundary = _mesh_topology(ob)

    t0 = time.perf_counter()
    with _Counters(MC5) as counters:
        for f in range(scene_def.frames):
            scene.frame_set(reset + 1 + f)      # past reset_frame, or it resets
            if mover is not None:
                mover.location.z += speed
                bpy.context.view_layer.update()
            MC5.physics(C)
            wc = world(ob, np.array(C.co))
            if not np.all(np.isfinite(wc)):
                break
            if sphere is not None:
                pen_worst = max(pen_worst, _sphere_pen(wc, *sphere))
            if "floor_z" in extra:
                floor_worst = max(floor_worst, float(extra["floor_z"] - wc[:, 2].min()))
            if normal is not None:
                g = wc @ normal
                gap_max = max(gap_max, float(g.max()))
                gap_min = min(gap_min, float(g.min()))
            if apex is not None:
                tc = _tip_clearance(wc, tridex, apex)
                if np.isfinite(tc):
                    tip_worst = min(tip_worst, tc)
    elapsed = time.perf_counter() - t0

    co = np.array(C.co, dtype=np.float64)
    wc = world(ob, co)
    finite = bool(np.all(np.isfinite(co)))
    res = {
        "scene": name, "frames": scene_def.frames, "co": co, "finite": finite,
        "final_min_z": float(wc[:, 2].min()) if finite else float("nan"),
        "final_max_z": float(wc[:, 2].max()) if finite else float("nan"),
        "sphere_pen_worst": pen_worst,
        "floor_pen_worst": floor_worst,
        "plane_gap_max": gap_max,
        "plane_gap_min": gap_min if normal is not None else 0.0,
        "time_s": elapsed,
    }
    res.update(counters.n)
    if normal is not None:
        downhill = np.array([-normal[2], 0.0, normal[0]])
        res["slide"] = float(np.abs((wc - start_w) @ downhill).mean())
    if apex is not None:
        res["tip_clear"] = _tip_clearance(wc, tridex, apex)
        res["tip_clear_worst"] = float(tip_worst)
    res["spike"] = _spike(wc, edges, boundary) if finite else float("nan")
    res["stretch_max"] = _stretch_max(wc, start_w, edges) if finite else float("nan")
    if "layer_top" in extra:
        top = extra["layer_top"]
        res["layer_gap"] = float(np.median(wc[top, 2]) - np.median(wc[~top, 2]))
    ok, msg = scene_def.engaged(res) if finite else (False, "non-finite result")
    res["engaged"], res["engaged_msg"] = bool(ok), msg
    return res


# ----------------------------------------------------------------- baselines
METRICS = ("final_min_z", "final_max_z", "sphere_pen_worst", "floor_pen_worst",
           "plane_gap_max", "plane_gap_min", "layer_gap", "slide", "tip_clear", "tip_clear_worst", "spike", "stretch_max", "oc_contacts", "sc_contacts",
           "friction_calls")


def _path(name):
    return os.path.join(BASE_DIR, "%s.npz" % name)


def has_baseline(name):
    return os.path.isfile(_path(name))


def save_baseline(name, res):
    """Store a result as the reference.  Refuses one that did not engage."""
    if not res["finite"]:
        raise RuntimeError("%s: non-finite result, not stored" % name)
    if not res["engaged"]:
        raise RuntimeError("%s: collision did not engage (%s), not stored"
                           % (name, res["engaged_msg"]))
    os.makedirs(BASE_DIR, exist_ok=True)
    meta = fingerprint()
    payload = {"co": res["co"], "frames": np.array(res["frames"]),
               "time_s": np.array(res["time_s"])}
    for m in METRICS:
        if m in res:
            payload["m_" + m] = np.array(res[m])
    for k, v in meta.items():
        payload["meta_" + k] = np.array(v)
    np.savez(_path(name), **payload)
    return _path(name)


def load_baseline(name):
    z = np.load(_path(name))
    out = {"co": z["co"], "frames": int(z["frames"]),
           "time_s": float(z["time_s"])}
    out["metrics"] = {k[2:]: float(z[k]) for k in z.files if k.startswith("m_")}
    out["meta"] = {k[5:]: str(z[k]) for k in z.files if k.startswith("meta_")}
    return out


def compare(name, res, tol=0.0):
    """How far a run is from its baseline.

    Positions must match within `tol` (0 = bit-identical, right for checking
    the Python against itself; the C++ port will need a small tolerance).  The
    physical metrics are reported alongside, because for the port they are the
    real acceptance test: same rest height, same penetration, same layer gap.
    """
    if not has_baseline(name):
        return {"ok": False, "why": "no baseline"}
    base = load_baseline(name)
    if base["frames"] != res["frames"]:
        return {"ok": False, "why": "baseline ran %d frames, this run %d"
                % (base["frames"], res["frames"])}
    if base["co"].shape != res["co"].shape:
        return {"ok": False, "why": "vertex count changed %s -> %s"
                % (base["co"].shape, res["co"].shape)}
    if not res["finite"]:
        return {"ok": False, "why": "non-finite result"}
    d = np.linalg.norm(res["co"] - base["co"], axis=1)
    metrics = {m: (base["metrics"][m], float(res[m]))
               for m in METRICS if m in base["metrics"] and m in res}
    same_code = base["meta"].get("code") == fingerprint()["code"]
    return {"ok": bool(d.max() <= tol) and res["engaged"],
            "max_dev": float(d.max()), "mean_dev": float(d.mean()),
            "engaged": res["engaged"], "engaged_msg": res["engaged_msg"],
            "metrics": metrics, "same_code": same_code,
            "base_time": base["time_s"], "time": res["time_s"],
            "base_meta": base["meta"]}
