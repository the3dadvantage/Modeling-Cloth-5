"""
Cloth <-> collider-object collision for Modeling Cloth.

Companion to self_collide_2.py -- same broad-phase / narrow-phase shape so the
two stay maintainable together.  The collider is kinematic (infinite mass): the
cloth takes 100% of every correction, there is no triangle/edge reaction side.

Three narrow phases (each gated by an MC_props toggle):
  * point_to_triangle  -- cloth vertex   vs collider triangle   (cl_point_tris)
  * triangle_to_point  -- collider vertex vs cloth triangle      (ob_point_tris)
  * edge_to_edge        -- cloth edge     vs collider edge        (ob_edges)

Rewrite notes vs. the previous version
--------------------------------------
  * inside_triangles() now guards the barycentric denominator (was an unguarded
    1/x -> NaN on any degenerate collider triangle) and uses a float32-safe eps.
  * closest_point_segments() uses the Ericson clamp cascade -> correct closest
    points near segment ends and for parallel edges (cloth over a rod); the old
    edges_to_edges_clip() clamped s and t independently.
  * Per-leaf candidate generation does an AABB-overlap pre-filter instead of the
    full cloth x collider Cartesian product -> far fewer narrow-phase pairs.
  * One shared accumulate buffer + one shared per-vertex count for all three
    phases, normalised once and applied once (the old code ran add_forces()
    three times, each with its own 1/bincount and its own C.co += ).
  * 1/count uses np.divide(where=) -> no divide-by-zero RuntimeWarning per frame.
  * Every response move is magnitude-clamped (per constraint and per step).
  * Continuous (swept) test: sign-of-distance change at the interpolated
    crossing time catches tunnelling regardless of end distance.  Toggle OC_SWEPT.
  * SC-style relaxation loop (OC_ITERATIONS) re-reading C.co each pass.
  * triangle_to_point pushes along the COLLIDER VERTEX normal (a fixed axis),
    not the cloth-triangle normal -- with the cloth normal any small warp tilts
    the push and shears the triangle further, so the surface ripples.  It is
    shared over the corners by w / sum(w^2), so the point over the collider
    vertex moves the full push; one-sided; honours ob_collision_tri_damping.
  * edge_to_edge pushes along the real separation of the closest points,
    turned to the collider's outside by the edge normal (see _edge_contact);
    the edge normal alone is only the fallback when the edges nearly touch.
  * A vertex's contacts are combined by projection, not averaged -- see
    _project_contacts.
  * mc_collide.dll (cpp/mc_collide) is a C++ port of the narrow resolve and
    the broad phase, used when the scene's collision backend is C++ (see
    collide_native.py).  This module stays the reference and owns every
    tunable; tests/shadow_compare.py checks the two agree call by call.
  * Inelastic contact: MC5 folds (co - vel_start) into C.velocity after this
    call.  Per contacted vertex the velocity component heading INTO the collider
    is removed and the push-out is cancelled from that feedback, so a resting
    cloth doesn't buzz.  OC_RESTITUTION > 0 leaves some as bounce.
  * Friction (ob_friction / ob_static_threshold): one positional Coulomb pass
    after the normal solve.  Per contact, the cloth contact point's slip this
    frame is measured RELATIVE to the collider anchor under it (barycentric pt
    for a face, the vertex itself for a poke, the clip point for an edge), so
    sticking == moving with an animated collider.  slip <= budget -> cancel it
    (static); else cancel down to budget and keep (slip-budget)*(1-mu) (kinetic).
    The correction is tangential only and reaches velocity through MC5's
    feedback; it is re-measured every frame, so bend/stretch still move stuck
    points and a hard pull breaks them loose.  recollide() runs no friction.
  * Dead code removed: CPT(), get_edge_pairs__(), combine_forces(), the unused
    DATA/COL/Box globals and the import-time print() notes.

`radius`, `tridex`, `tidx` args of collision_force() are kept only for
call-site compatibility; thickness comes from ob.MC_props.ob_collision_radius
and the index arrays from C.*  (unchanged behaviour).
"""

import numpy as np
import bpy

# Same pattern as every other module: text datablock while developing, package
# import once installed as an addon.  This used to prefer pc_tools.utils and
# had no package fallback, so it could not import as an addon at all, and in
# development it could silently run against a different utils than the rest.
try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U

# ===========================================================================
# Tunables
# ===========================================================================
OC_EPS             = 1e-7     # float32-safe epsilon, for LENGTHS only
# Degeneracy of a triangle or segment pair, measured against the element's own
# size (it is sin^2 of an angle), so the answer does not depend on scene scale.
# The old absolute 1e-7 was applied to quantities that scale with length^4.
OC_GEOM_REL_EPS    = 1e-10
OC_GEOM_TINY       = 1e-30    # a squared length this small really is zero

OC_ITERATIONS      = 2        # narrow-phase relaxation passes per collision_force call
OC_RECOLLIDE_ITERS = 1        # passes per recollide() call
OC_LOOKAHEAD_SLACK = 1.0      # extra broad-phase reach, * thickness, when the
                              #   pairs are cached for the next step's recollide
OC_RECOLLIDE_BAND  = 1.0      # near_pairs keeps pairs within (1 + this) *
                              #   thickness of touching at the start of a step
OC_RELAX           = 0.7      # under-relaxation; lower = gentler, less marginal-contact chatter
OC_CONTACT_ROUNDS  = 4        # per-vertex rounds combining its contacts (see _project_contacts)
OC_EDGE_SEP_MIN    = 0.1      # edge contact: below this * thickness apart the separation's
                              #   direction is noise and the collider edge normal is used

OC_SWEPT           = True     # continuous (sign-change) tests
OC_SWEPT_MARGIN    = 0.05     # barycentric slack (also: cloth-vert-near-collider-face-edge slack)
OC_PBD_EXACT       = True     # True: w/sum(w^2) edge lever weights; False: w

OC_DETECT_SCALE    = 2.0      # broad-phase reach = OC_DETECT_SCALE * thickness
OC_PUSH_CAP        = 1.5      # a single contact never moves a point more than this * thickness
OC_CLAMP_THICK     = 3.0      # per-move clamp = max(OC_CLAMP_THICK*thickness,
OC_CLAMP_EDGE      = 2.0      #                      OC_CLAMP_EDGE*char_edge_len)

OC_MAX_DEPTH       = 16       # spatial split recursion cap
OC_TILE            = 512      # row tile for AABB pair matrices (memory cap)
OC_MIN_SPLIT_GAIN  = 0.98     # bail to a leaf if a split barely shrinks the bigger child

OC_VEL_DAMP        = 0.9      # extra scale on a contacting vertex's velocity (1.0 = none);
                             #   < 1 bleeds the "breathing" of a drape over rough geometry
OC_RESTITUTION     = 0.0      # 0 = inelastic (kill velocity into the collider); 1 = full bounce

OC_FRICTION_FLOOR  = 0.02     # min normal-force proxy for friction = this * thickness
                             #   (numerical floor so a quiet contact isn't exactly frictionless)
OC_FRICTION_MAXMOVE = 1.0     # friction correction per vertex per frame, capped at this * thickness


class _ModuleView:
    """This module's tunables and helpers as attributes, for collide_native.
    Reads the live globals, so a tunable changed at runtime is seen too.
    (Loaded as a text datablock the module is not in sys.modules.)"""
    __slots__ = ("_g",)

    def __init__(self, g):
        self._g = g

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name)


_THIS = _ModuleView(globals())


# ===========================================================================
# Small structures
# ===========================================================================
class Node:
    __slots__ = (
        "clv_min", "clv_max", "obv_min", "obv_max",
        "clt_min", "clt_max", "obt_min", "obt_max",
        "clv_idx", "obv_idx", "clt_idx", "obt_idx",
    )

    def __init__(self, clv_min, clv_max, obv_min, obv_max,
                 clt_min, clt_max, obt_min, obt_max,
                 clv_idx, obv_idx, clt_idx, obt_idx):
        self.clv_min = clv_min; self.clv_max = clv_max
        self.obv_min = obv_min; self.obv_max = obv_max
        self.clt_min = clt_min; self.clt_max = clt_max
        self.obt_min = obt_min; self.obt_max = obt_max
        self.clv_idx = clv_idx; self.obv_idx = obv_idx
        self.clt_idx = clt_idx; self.obt_idx = obt_idx


class Candidates:
    def __init__(self):
        self.pt_a = []; self.pt_b = []      # cloth vert  , collider tri
        self.tp_a = []; self.tp_b = []      # collider vert, cloth tri
        self.ee_a = []; self.ee_b = []      # cloth edge id, collider edge id

    def finalize(self, n_ob_tri, n_cl_tri, n_ob_edge):
        out = {}
        out["pt"] = _dedup_pairs(self.pt_a, self.pt_b, n_ob_tri)
        out["tp"] = _dedup_pairs(self.tp_a, self.tp_b, n_cl_tri)
        out["ee"] = _dedup_pairs(self.ee_a, self.ee_b, n_ob_edge)
        return out


def _dedup_pairs(a_list, b_list, span):
    if not a_list:
        return None, None
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    key = a.astype(np.int64) * span + b
    _, u = np.unique(key, return_index=True)
    return a[u], b[u]


# ===========================================================================
# Per-mesh cache (characteristic edge length for the move clamp)
# ===========================================================================
def _get_cache(C):
    c = getattr(C, "_oc_cache", None)
    nv = int(C.vc)
    nt = int(C.tridex.shape[0])
    if c is not None and c.get("nv") == nv and c.get("nt") == nt:
        return c
    char = None
    try:
        e = np.concatenate([C.tridex[:, [0, 1]], C.tridex[:, [1, 2]], C.tridex[:, [2, 0]]], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)
        el = np.linalg.norm(C.start_co[e[:, 0]] - C.start_co[e[:, 1]], axis=1)
        mean_edge = float(np.mean(el)) if el.size else 1.0
        char = np.zeros(nv)
        cnt = np.zeros(nv)
        np.add.at(char, e[:, 0], el)
        np.add.at(char, e[:, 1], el)
        np.add.at(cnt, e[:, 0], 1.0)
        np.add.at(cnt, e[:, 1], 1.0)
        char = np.where(cnt > 0, char / np.where(cnt > 0, cnt, 1.0), mean_edge)
    except Exception as ex:  # noqa: BLE001
        print("[object_collide] char-length cache failed:", ex)
    c = {"nv": nv, "nt": nt, "char_len": char}
    C._oc_cache = c
    return c


# ===========================================================================
# Vector helpers
# ===========================================================================
def _unit(v):
    n = np.sqrt(np.einsum("ij,ij->i", v, v))
    n = np.where(n > OC_EPS, n, 1.0)
    return v / n[:, None]


def _bary(tri_co, w):
    return np.einsum("mij,mi->mj", tri_co, w)


def _clip_weights(w):
    """Barycentric weights pulled onto the triangle and renormalised.

    Clipping alone can leave them summing to more than 1 near an edge -- e.g.
    (-0.05, 0.5, 0.55) -> (0, 0.5, 0.55) -- which over-distributes a push and
    puts a friction anchor off the triangle.
    """
    w = np.clip(w, 0.0, 1.0)
    s = w.sum(axis=1, keepdims=True)
    ok = s > 1e-12
    return np.where(ok, w / np.where(ok, s, 1.0), 1.0 / 3.0)


def _clamp_rows(vecs, max_norm):
    n = np.sqrt(np.einsum("ij,ij->i", vecs, vecs))
    big = n > max_norm
    if np.any(big):
        vecs[big] *= (max_norm[big] / n[big])[:, None]


def _max_move(cache, vidx, thickness):
    base = OC_CLAMP_THICK * thickness
    char = cache.get("char_len") if cache else None
    if char is not None:
        return np.maximum(base, OC_CLAMP_EDGE * char[vidx])
    return np.full(len(vidx), max(base, OC_CLAMP_EDGE * 10.0 * thickness))


# ===========================================================================
# Bounds / broad phase
# ===========================================================================
def get_poly_bounds(arr):
    return np.min(arr, axis=1), np.max(arr, axis=1)


def overlapping_box(min1, max1, min2, max2):
    lo = np.maximum(min1, min2)
    hi = np.minimum(max1, max2)
    if np.all(lo <= hi):
        return lo, hi
    return None


def _aabb_pairs(amin, amax, bmin, bmax, margin):
    na = amin.shape[0]
    rows = []
    cols = []
    for i0 in range(0, na, OC_TILE):
        i1 = min(na, i0 + OC_TILE)
        sep = ((amin[i0:i1, None, :] - margin > bmax[None, :, :]) |
               (amax[i0:i1, None, :] + margin < bmin[None, :, :]))
        r, c = np.nonzero(~sep.any(axis=2))
        rows.append(r + i0)
        cols.append(c)
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def _edges_from_tris(booler, tri_edge_idxer, edge_ids, tri_idx):
    booler[:] = False
    booler[tri_edge_idxer[tri_idx]] = True
    return edge_ids[booler]


def _edge_bounds_2frame(joined2, ev):
    """joined2 (N,2,3) start+current positions; ev (E,2) endpoint indices."""
    pts = np.concatenate([joined2[ev[:, 0]], joined2[ev[:, 1]]], axis=1)  # (E,4,3)
    return pts.min(axis=1), pts.max(axis=1)


def _leaf(C, data, node, cand, detect, want):
    do_pt, do_tp, do_ee = want

    if do_pt and node.clv_idx.size and node.obt_idx.size:
        r, c = _aabb_pairs(node.clv_min, node.clv_max, node.obt_min, node.obt_max, detect)
        if r.size:
            cand.pt_a.append(node.clv_idx[r])
            cand.pt_b.append(node.obt_idx[c])

    if do_tp and node.obv_idx.size and node.clt_idx.size:
        r, c = _aabb_pairs(node.obv_min, node.obv_max, node.clt_min, node.clt_max, detect)
        if r.size:
            cand.tp_a.append(node.obv_idx[r])
            cand.tp_b.append(node.clt_idx[c])

    if do_ee and node.clt_idx.size and node.obt_idx.size:
        cle = _edges_from_tris(C.tridex_edge_booler, C.tri_edge_idxer,
                               C.tridex_edges, np.unique(node.clt_idx))
        obe = _edges_from_tris(data["tridex_edge_booler"], data["tri_edge_idxer"],
                               data["tridex_edges"], np.unique(node.obt_idx))
        if cle.size and obe.size:
            a_lo, a_hi = _edge_bounds_2frame(C.joined_co, C.tri_eidx[cle])
            b_lo, b_hi = _edge_bounds_2frame(data["joined_co"], data["tridex_eidx"][obe])
            r, c = _aabb_pairs(a_lo, a_hi, b_lo, b_hi, detect)
            if r.size:
                cand.ee_a.append(cle[r])
                cand.ee_b.append(obe[c])


def _count(node):
    return (node.clv_idx.shape[0] * node.obt_idx.shape[0] +
            node.obv_idx.shape[0] * node.clt_idx.shape[0])


def split_box_t(C, data, node, count, bmin, bmax, cand, detect, want,
                t_count=100, depth=0, max_depth=OC_MAX_DEPTH):
    if count <= t_count or depth >= max_depth:
        if count > 0:
            _leaf(C, data, node, cand, detect, want)
        return

    depth += 1
    dif = bmax - bmin
    axis = int(np.argmax(dif))
    mid = bmin[axis] + dif[axis] * 0.5
    hi = mid + detect
    lo = mid - detect

    clv_lo = node.clv_min[:, axis] <= hi
    clv_hi = node.clv_max[:, axis] >= lo
    obv_lo = node.obv_min[:, axis] <= hi
    obv_hi = node.obv_max[:, axis] >= lo
    clt_lo = node.clt_min[:, axis] <= hi
    clt_hi = node.clt_max[:, axis] >= lo
    obt_lo = node.obt_min[:, axis] <= hi
    obt_hi = node.obt_max[:, axis] >= lo

    left = Node(
        node.clv_min[clv_lo], node.clv_max[clv_lo],
        node.obv_min[obv_lo], node.obv_max[obv_lo],
        node.clt_min[clt_lo], node.clt_max[clt_lo],
        node.obt_min[obt_lo], node.obt_max[obt_lo],
        node.clv_idx[clv_lo], node.obv_idx[obv_lo],
        node.clt_idx[clt_lo], node.obt_idx[obt_lo],
    )
    right = Node(
        node.clv_min[clv_hi], node.clv_max[clv_hi],
        node.obv_min[obv_hi], node.obv_max[obv_hi],
        node.clt_min[clt_hi], node.clt_max[clt_hi],
        node.obt_min[obt_hi], node.obt_max[obt_hi],
        node.clv_idx[clv_hi], node.obv_idx[obv_hi],
        node.clt_idx[clt_hi], node.obt_idx[obt_hi],
    )
    lc = _count(left)
    rc = _count(right)
    if max(lc, rc) >= count * OC_MIN_SPLIT_GAIN:
        _leaf(C, data, node, cand, detect, want)
        return

    nbmax = bmax.copy()
    nbmax[axis] = mid
    split_box_t(C, data, left, lc, bmin, nbmax, cand, detect, want, t_count, depth, max_depth)

    nbmin = bmin.copy()
    nbmin[axis] = mid
    split_box_t(C, data, right, rc, nbmin, bmax, cand, detect, want, t_count, depth, max_depth)


# ===========================================================================
# Geometry
# ===========================================================================
def inside_triangles(tris, points, margin=0.0):
    """Barycentric inside test. Returns (check, weights[w0,w1,w2], v2).

    float64, with degeneracy judged RELATIVE to the triangle's own size.  The
    denominator is |v0 x v1|^2, which scales with edge length^4, so the old
    absolute 1e-7 cut-off called every triangle under ~1.8 cm degenerate and
    produced garbage weights: at 1 cm a point five triangle-lengths away read
    as inside.  den / (|v0|^2 |v1|^2) is sin^2 of the corner angle, which does
    not depend on scale at all.
    """
    tris = np.asarray(tris, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    origins = tris[:, 0]
    cv = tris[:, 1:] - origins[:, None]
    v0 = cv[:, 0]
    v1 = cv[:, 1]
    v2 = points - origins

    d = np.einsum("ijk,ijk->ij", cv, cv)
    d00 = d[:, 0]
    d11 = d[:, 1]
    d01 = np.einsum("ij,ij->i", v0, v1)
    d02 = np.einsum("ij,ij->i", v0, v2)
    d12 = np.einsum("ij,ij->i", v1, v2)

    den = d00 * d11 - d01 * d01
    # written as not-greater so a NaN or a zero-area triangle lands in degen
    degen = ~(den > OC_GEOM_REL_EPS * d00 * d11)
    inv = 1.0 / np.where(degen, 1.0, den)

    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv
    w = 1.0 - (u + v)

    weights = np.stack([w, u, v], axis=-1)
    # a degenerate triangle has no inside; neutral weights rather than the
    # enormous ones a vanishing denominator gives, so nothing downstream blows up
    weights[degen] = 1.0 / 3.0
    check = (u >= margin) & (v >= margin) & (w >= margin) & ~degen
    return check, weights, v2


def closest_point_segments(p1, q1, p2, q2):
    """Vectorised closest points between segments p1q1 and p2q2 (Ericson).
    Robust for parallel / near-parallel / degenerate segments.

    float64, with the parallel test relative to the segments' own lengths.
    a*e - b^2 is |d1|^2 |d2|^2 sin^2(angle), which scales with edge length^4,
    so the old absolute 1e-7 called every pair of edges under ~1.8 cm
    "parallel" and forced s = 0 -- which edge_to_edge then rejects as not
    interior, switching edge collision off entirely on fine meshes.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)

    # a segment only collapses to a point when it genuinely has no length
    pt_a = a <= OC_GEOM_TINY
    pt_e = e <= OC_GEOM_TINY
    a_safe = np.where(pt_a, 1.0, a)
    e_safe = np.where(pt_e, 1.0, e)

    denom = a * e - b * b
    nonpar = denom > OC_GEOM_REL_EPS * a * e
    denom_safe = np.where(nonpar, denom, 1.0)
    s = np.where(nonpar, np.clip((b * f - c * e) / denom_safe, 0.0, 1.0), 0.0)

    t = (b * s + f) / e_safe
    t_lt0 = t < 0.0
    t_gt1 = t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(t_lt0, np.clip(-c / a_safe, 0.0, 1.0), s)
    s = np.where(t_gt1, np.clip((b - c) / a_safe, 0.0, 1.0), s)

    # first segment is a point: closest point on the second to it
    s = np.where(pt_a, 0.0, s)
    t = np.where(pt_a, np.clip(f / e_safe, 0.0, 1.0), t)
    # second segment is a point: closest point on the first to it (Ericson's
    # third case, which the previous version was missing)
    only_e = pt_e & ~pt_a
    s = np.where(only_e, np.clip(-c / a_safe, 0.0, 1.0), s)
    t = np.where(only_e, 0.0, t)
    both = pt_a & pt_e
    s = np.where(both, 0.0, s)
    t = np.where(both, 0.0, t)

    cpa = p1 + d1 * s[:, None]
    cpb = p2 + d2 * t[:, None]
    return cpa, cpb, s, t


# ===========================================================================
# Friction   (positional Coulomb, surface-relative anchor)
# ===========================================================================
def _cloth_fric_at(cfg, idx):
    """Cloth friction vertex-group weight at vertex indices `idx` (shape follows
    `idx`; 1.0 everywhere if the group is unset)."""
    idx = np.asarray(idx)
    if cfg is None:
        return np.ones(idx.shape, dtype=np.float64)
    return np.asarray(cfg).reshape(-1)[idx].astype(np.float64)


def _friction_correction(c1, c0, a1, a0, n, normal_push, mu, static_thresh, thickness):
    """Tangential position correction opposing this frame's slip of the cloth
    contact point (c0 -> c1) relative to the collider anchor under it (a0 -> a1).

      stick (slip <= budget)  : cancel all tangential slip
      slide (slip >  budget)  : cancel exactly `budget` (a fixed friction impulse)

    `mu` is per-contact:  cloth.ob_friction  x  cloth MC_friction weight  x
    collider ob_collider_friction x collider MC_ob_friction weight.
    budget = static_thresh (where mu > 0) if given, else mu * max(normal_push,
    floor).  normal_push is |this frame's normal correction along n| -- the PBD
    proxy for normal force, which for a resting contact is ~ m*g*dt and gives the
    exact Coulomb friction angle (stick while tan(slope) <= mu).
    """
    mu = np.asarray(mu, dtype=np.float64)
    d_rel = (c1 - c0) - (a1 - a0)
    d_t = d_rel - np.einsum("ij,ij->i", d_rel, n)[:, None] * n     # tangential part
    mag = np.sqrt(np.einsum("ij,ij->i", d_t, d_t))

    if static_thresh > 0.0:
        budget = np.where(mu > 0.0, static_thresh, 0.0)
    else:
        budget = mu * np.maximum(normal_push, OC_FRICTION_FLOOR * thickness)

    # Coulomb: remove all the slip when stuck, remove exactly `budget` when
    # sliding (a fixed friction impulse, independent of slide speed -- NOT
    # budget + a fraction of the excess, which removes so much of a fast slide
    # that the velocity feedback locks it up and it never breaks loose).
    remove_mag = np.minimum(mag, budget)
    remove = np.where(mag > OC_EPS, remove_mag / np.maximum(mag, OC_EPS), 0.0)
    fric = -d_t * remove[:, None]

    fm = np.sqrt(np.einsum("ij,ij->i", fric, fric))
    cap = OC_FRICTION_MAXMOVE * thickness
    big = fm > cap
    if np.any(big):
        fric[big] *= (cap / fm[big])[:, None]
    return fric


def _closest_on_boundary(tri, p):
    """Closest point to p on each triangle's three edges (p is outside it)."""
    best = None
    best_d2 = None
    for a, b in ((0, 1), (1, 2), (2, 0)):
        A, B = tri[:, a], tri[:, b]
        AB = B - A
        ab2 = np.einsum("ij,ij->i", AB, AB)
        t = np.einsum("ij,ij->i", p - A, AB) / np.maximum(ab2, OC_GEOM_TINY)
        q = A + np.clip(t, 0.0, 1.0)[:, None] * AB
        d2 = np.einsum("ij,ij->i", p - q, p - q)
        if best is None:
            best, best_d2 = q, d2
        else:
            closer = d2 < best_d2
            best[closer] = q[closer]
            best_d2 = np.where(closer, d2, best_d2)
    return best


def _tip_weights(tco, P, vn, margin):
    """Which cloth triangles are over a collider vertex, looking along that
    vertex's normal, and where: returns (check, weights) like
    inside_triangles.

    Looking along the cloth triangle's own normal (plain inside_triangles)
    made the answer depend on the contact's own result: lifting a triangle
    off a sharp tip tilts it, the tilted triangle's projection swings off
    the tip, and the next step the tip belonged to the neighbouring triangle.
    The lifted corner was dropped, its triangle flattened, the tip came back
    to it -- a vertex kicked up and dropped on alternate frames.  Along the
    vertex normal (the direction the push goes), tilting a triangle does not
    change what is over the tip.
    """
    tco = np.asarray(tco, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    vn = np.asarray(vn, dtype=np.float64)
    flat_t = tco - np.einsum("ijk,ik->ij", tco, vn)[:, :, None] * vn[:, None, :]
    flat_p = P - np.einsum("ij,ij->i", P, vn)[:, None] * vn
    check, w, _ = inside_triangles(flat_t, flat_p, margin=margin)
    return check, w


def _edge_contact(cpa, cpb, en, thickness):
    """Signed distance and push direction between a cloth edge and a collider
    edge, from their closest points cpa / cpb and the collider edge's outward
    normal en.  Returns (dist, normal, ok).

    The push goes along the real separation cpb -> cpa, turned to the
    collider's outside by en.  Measuring and pushing along en itself was a
    bug: en's plane runs past the end of the edge, so near a sharp tip the
    planes of the edges on every side overlapped into a peak ~ thickness /
    en_z above it, and cloth near the tip was pushed up to that peak -- out of
    reach of its contacts, so it dropped, got caught, and was pushed up again
    on alternate frames.

    en is still the fallback when the edges nearly touch: there the
    separation's direction is round-off, which is the flat-cloth-on-a-flat-
    plane case en was chosen for.
    """
    en = np.asarray(en, dtype=np.float64)
    enl = np.sqrt(np.einsum("ij,ij->i", en, en))
    ok = enl > OC_EPS
    en = en / np.where(ok, enl, 1.0)[:, None]
    sep = np.asarray(cpa, dtype=np.float64) - np.asarray(cpb, dtype=np.float64)
    sl = np.sqrt(np.einsum("ij,ij->i", sep, sep))
    side = np.where(np.einsum("ij,ij->i", sep, en) < 0.0, -1.0, 1.0)
    real = sl > OC_EDGE_SEP_MIN * thickness
    normal = np.where(real[:, None], sep / np.where(real, sl, 1.0)[:, None] * side[:, None], en)
    dist = np.where(real, sl * side, np.einsum("ij,ij->i", sep, en))
    ok = ok | real
    return dist, normal, ok


def _face_contact(tri, n, p):
    """How a cloth point sits against a collider face.

    Returns (in_zone, distance, normal, weights).  Over the triangle itself
    the distance is to its plane and the push is along the face normal.  The
    zone also takes in a slack band just past the triangle's edges, so a point
    near a ridge is still handled here -- but there the face's plane is NOT
    the surface, so distance and push are taken from the closest point ON the
    triangle (an edge or a corner), straight away from it.

    Using the plane in that band was a bug.  The band is a barycentric
    fraction, so on a large face it is several centimetres wide: past a sharp
    tip it reached a point on the other side, and that far face's plane
    pushed it back across the tip.  Its neighbour got the mirror image, and
    the two were driven into each other -- neighbours pulled onto the
    contact point.
    """
    tri = np.asarray(tri, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    d = np.einsum("ij,ij->i", p - tri[:, 0], n)
    strict, w, _ = inside_triangles(tri, p, margin=0.0)
    slack, _, _ = inside_triangles(tri, p, margin=-OC_SWEPT_MARGIN)
    # past an edge and above the plane: measured from the edge / corner.
    # Past an edge and BELOW the plane is the other side of a ridge -- a
    # neighbouring face's business, not this one's.
    edge = slack & ~strict & (d > 0.0)
    dist = d.copy()
    normal = n.copy()
    if edge.any():
        cp = _closest_on_boundary(tri[edge], p[edge])
        v = p[edge] - cp
        ln = np.sqrt(np.einsum("ij,ij->i", v, v))
        ok = ln > OC_EPS
        dist_e = np.where(ok, ln, d[edge])
        norm_e = np.where(ok[:, None], v / np.where(ok, ln, 1.0)[:, None], n[edge])
        dist[edge] = dist_e
        normal[edge] = norm_e
    return strict | edge, dist, normal, w


# ===========================================================================
# Narrow phase   (each returns (hits, move) or (None, None);
#                 with fric=(mu, static_thresh) it returns the friction move)
# ===========================================================================
def point_to_triangle(data, C, cl_v, ob_t, thickness, cache, fric=None):
    """Cloth vertex pushed out of a collider triangle to the +normal shell."""
    if cl_v is None or len(cl_v) == 0:
        return None, None

    tri1 = C.local_trico[ob_t]                 # (M,3,3) collider tri, current
    n1 = C.local_normals[ob_t]                 # (M,3) unit outward
    p1 = C.co[cl_v]
    d1 = np.einsum("ij,ij->i", p1 - tri1[:, 0], n1)

    # A cloth vertex just past a collider-face edge / ridge is still handled
    # here (the well-behaved single-vertex response) instead of falling
    # through to triangle_to_point / edge_to_edge on spiky colliders -- but
    # measured from the edge, not the face's plane.  See _face_contact.
    zone, dist1, nrm1, w_pt = _face_contact(tri1, n1, p1)
    prox = zone & (dist1 < thickness) & (dist1 > -OC_PUSH_CAP * thickness)

    swept = np.zeros(len(cl_v), dtype=bool)
    if OC_SWEPT:
        tri0 = C.start_local_trico[ob_t]
        n0 = C.start_local_normals[ob_t]
        p0 = C.start_co[cl_v]
        d0 = np.einsum("ij,ij->i", p0 - tri0[:, 0], n0)
        # a real deep crossing this frame: clearly outside -> clearly inside
        # (a sub-thickness wobble around the plane is NOT a tunnel and must not
        #  trigger a full push, or the surface chatters)
        deep = (d0 > thickness) & (d1 < -thickness)
        if np.any(deep):
            # Containment is judged where the vertex CROSSED the plane, not
            # where it finished.  A fast diagonal move can pass through the
            # triangle's interior and end up beside it, and testing the end
            # point let exactly those tunnels through.  Same method as
            # self_collide_2.  d0 - d1 > 2*thickness wherever deep holds, so
            # the division is safe.
            tau = np.clip(d0 / np.where(deep, d0 - d1, 1.0), 0.0, 1.0)[:, None]
            pT = p0 + tau * (p1 - p0)
            triT = tri0 + tau[:, :, None] * (tri1 - tri0)
            checkT, wT, _ = inside_triangles(triT, pT, margin=-OC_SWEPT_MARGIN)
            swept = deep & checkT
            only = swept & ~prox
            if np.any(only):
                w_pt[only] = wT[only]      # anchor friction where it crossed

    hit = prox | swept
    if not np.any(hit):
        return None, None

    rows = np.nonzero(hit)[0]
    vh = cl_v[rows]
    # a proximity contact uses the closest-feature distance and direction; a
    # tunnel caught only by the swept test is pushed back out of the plane
    pr = prox[rows]
    dd = np.where(pr, dist1[rows], d1[rows])
    nn = np.where(pr[:, None], nrm1[rows], n1[rows])
    push = np.clip(thickness - dd, 0.0, OC_PUSH_CAP * thickness)   # one-sided + capped

    if fric is not None:
        mu_, st_, tstep, cfg = fric
        w_h = _clip_weights(w_pt[rows])
        obt_r = ob_t[rows]
        a1 = _bary(C.local_trico[obt_r], w_h)              # collider anchor, now
        a0 = _bary(C.start_local_trico[obt_r], w_h)        #                , frame start
        push_ref = np.abs(np.einsum("ij,ij->i", tstep[vh], nn))
        ftri = data.get("friction_tri")
        mu_row = (mu_ * _cloth_fric_at(cfg, vh) *
                  (ftri[obt_r] if ftri is not None else 1.0))
        return vh, _friction_correction(C.co[vh], C.start_co[vh], a1, a0,
                                        nn, push_ref, mu_row, st_, thickness)

    move = push[:, None] * nn
    _clamp_rows(move, _max_move(cache, vh, thickness))
    return vh, move


def triangle_to_point(data, C, ob_v, cl_t, thickness, cache, fric=None):
    """Collider vertex poking the cloth: push the cloth triangle away from the
    collider vertex, ALONG THE COLLIDER VERTEX NORMAL.

    Pushing along the cloth-triangle normal instead (as before) means any small
    warp tilts that normal, which shears the triangle further -> the warp feeds
    itself and the surface ripples.  The collider vertex normal is a fixed axis.
    """
    if ob_v is None or len(ob_v) == 0:
        return None, None

    tridex = C.tridex[cl_t]
    tco1 = C.co[tridex]                          # (M,3,3) cloth tri, current
    vn = _unit(C.local_vert_norms[ob_v])         # (M,3) collider outward vertex normal

    P1 = data["joined_co"][ob_v][:, 1]
    # small negative margin: keep collider verts near a cloth-tri edge in the
    # contact set consistently frame-to-frame (a hard edge flickers the set,
    # which shows up as surface chatter).  Judged along the vertex normal --
    # see _tip_weights.
    check, w = _tip_weights(tco1, P1, vn, -OC_SWEPT_MARGIN)
    plot1 = _bary(tco1, w)                       # cloth-tri point over the collider vertex
    d1 = np.einsum("ij,ij->i", P1 - plot1, vn)   # < 0 when the cloth is properly outside
    prox = check & (d1 > -thickness) & (d1 < thickness)   # collider vert within the shell

    swept = np.zeros(len(ob_v), dtype=bool)
    if OC_SWEPT:
        tco0 = C.start_co[tridex]
        P0 = data["joined_co"][ob_v][:, 0]
        plot0 = _bary(tco0, w)                   # same bary weights on the start triangle
        d0 = np.einsum("ij,ij->i", P0 - plot0, vn)
        # collider vertex clearly poked through the cloth this frame (below -> above)
        swept = check & (d0 < -thickness) & (d1 > thickness)

    hit = prox | swept
    if not np.any(hit):
        return None, None

    rows = np.nonzero(hit)[0]
    trv = tridex[rows]
    w_h = _clip_weights(w[rows])                 # barycentric weights of the collider vert
    th = trv.reshape(-1)
    # one-sided + capped push, along the (fixed) collider vertex normal
    push = np.clip(d1[rows] + thickness, 0.0, OC_PUSH_CAP * thickness)

    if fric is not None:
        mu_, st_, tstep, cfg = fric
        obv_r = ob_v[rows]
        c1 = _bary(C.co[trv], w_h)                         # cloth contact point, now
        c0 = _bary(C.start_co[trv], w_h)                   #                     , frame start
        a1 = data["joined_co"][obv_r][:, 1]               # anchor = the collider vertex itself
        a0 = data["joined_co"][obv_r][:, 0]
        push_ref = np.abs(np.einsum("ij,ij->i", _bary(tstep[trv], w_h), vn[rows]))
        fvert = data.get("friction_vert")
        mu_row = (mu_ * _cloth_fric_at(cfg, trv).mean(axis=1) *
                  (fvert[obv_r] if fvert is not None else 1.0))
        fmove = _friction_correction(c1, c0, a1, a0, vn[rows], push_ref, mu_row, st_, thickness)
        return th, (fmove[:, None, :] * w_h[:, :, None]).reshape(-1, 3)

    move = push[:, None] * vn[rows]
    _clamp_rows(move, _max_move(cache, trv[:, 0], thickness))

    # Distribute by barycentric weight: the cloth vertex nearest the collider
    # vertex takes most of the move, far corners barely move -- a rigid
    # translation lifts non-contact corners and inflates the sheet on spiky
    # colliders.
    #
    # Plain w moves the point over the collider vertex by only sum(w^2) of the
    # push -- a third of it mid-triangle -- so a sharp tip was never pushed
    # clear of: a sheet on a cone came to rest 1.2 cm above the tip with a
    # 5 cm thickness.  w / sum(w^2) (as edge_to_edge does, OC_PBD_EXACT) keeps
    # the same proportions but moves that point the whole push.
    tri_damp = float(C.ob.MC_props.ob_collision_tri_damping)
    wt = w_h * tri_damp
    if OC_PBD_EXACT:
        wt = wt / np.einsum("ij,ij->i", w_h, w_h)[:, None]
    tm = (move[:, None, :] * wt[:, :, None]).reshape(-1, 3)
    # which contact each row belongs to, and its corner's weight -- so the
    # combine can judge the contact as a whole (see _project_contacts)
    grp = (np.repeat(np.arange(len(rows)), 3), w_h.reshape(-1))
    return th, tm, grp


def edge_to_edge(data, C, cl_e, ob_e, thickness, cache, fric=None):
    """Cloth edge pushed off a collider edge along the collider edge's outward normal."""
    if cl_e is None or len(cl_e) == 0:
        return None, None

    cl_ev = C.tri_eidx[cl_e]
    ob_ev = data["tridex_eidx"][ob_e]
    a0, a1 = cl_ev[:, 0], cl_ev[:, 1]      # cloth edge vertex indices
    b0, b1 = ob_ev[:, 0], ob_ev[:, 1]      # collider edge vertex indices

    cpa, cpb, s, t = closest_point_segments(
        C.co[a0], C.co[a1],
        data["joined_co"][b0][:, 1], data["joined_co"][b1][:, 1])
    sep1 = cpa - cpb
    dist1 = np.sqrt(np.einsum("ij,ij->i", sep1, sep1))
    # strictly interior: contacts at an edge endpoint (where ridge edges meet at
    # a spike apex) are ambiguous -> let point/triangle handle those, not here
    interior = (s > 1e-3) & (s < 1.0 - 1e-3) & (t > 1e-3) & (t < 1.0 - 1e-3)

    # Signed distance and push direction along the real separation, turned to
    # the collider's outside by the edge's outward normal -- see _edge_contact.
    d1, udir, good = _edge_contact(cpa, cpb, data["local_edge_normals"][ob_e], thickness)
    prox = interior & good & (dist1 < thickness)   # genuine gap within the shell (NOT "d1 < 0
                                                   # while far away", which fires all over a
                                                   # spiky collider)

    swept = np.zeros(len(cl_e), dtype=bool)
    if OC_SWEPT:
        scpa, scpb, _, _ = closest_point_segments(
            C.start_co[a0], C.start_co[a1],
            data["joined_co"][b0][:, 0], data["joined_co"][b1][:, 0])
        d0, _, _ = _edge_contact(scpa, scpb, data["start_local_edge_normals"][ob_e],
                                 thickness)
        # a real crossing: flipped sides, still near the collider edge, and not a
        # sub-thickness wobble (both heights must clear a fraction of thickness)
        swept = (interior & (np.sign(d0) != np.sign(d1)) &
                 (dist1 < OC_PUSH_CAP * thickness) &
                 (np.minimum(np.abs(d0), np.abs(d1)) > 0.25 * thickness))

    hit = prox | swept
    if not np.any(hit):
        return None, None

    rows = np.nonzero(hit)[0]
    s_h, t_h = s[rows], t[rows]
    push = np.clip(thickness - d1[rows], 0.0, OC_PUSH_CAP * thickness)   # one-sided + capped
    ai0, ai1 = a0[rows], a1[rows]
    bi0, bi1 = b0[rows], b1[rows]
    hits = np.concatenate([ai0, ai1])

    if OC_PBD_EXACT:
        da = (1.0 - s_h) ** 2 + s_h ** 2 + OC_EPS
        w1 = (1.0 - s_h) / da
        w2 = s_h / da
    else:
        w1, w2 = 1.0 - s_h, s_h

    if fric is not None:
        mu_, st_, tstep, cfg = fric
        c1 = cpa[rows]                                                     # cloth closest pt, now
        c0 = (1.0 - s_h)[:, None] * C.start_co[ai0] + s_h[:, None] * C.start_co[ai1]
        a1p = cpb[rows]                                                    # collider anchor, now
        a0p = ((1.0 - t_h)[:, None] * data["joined_co"][bi0][:, 0] +
               t_h[:, None] * data["joined_co"][bi1][:, 0])
        ts = (1.0 - s_h)[:, None] * tstep[ai0] + s_h[:, None] * tstep[ai1]
        push_ref = np.abs(np.einsum("ij,ij->i", ts, udir[rows]))
        fedge = data.get("friction_edge")
        mu_row = (mu_ * 0.5 * (_cloth_fric_at(cfg, ai0) + _cloth_fric_at(cfg, ai1)) *
                  (fedge[ob_e[rows]] if fedge is not None else 1.0))
        fmove = _friction_correction(c1, c0, a1p, a0p, udir[rows], push_ref,
                                     mu_row, st_, thickness)
        return hits, np.concatenate([fmove * w1[:, None], fmove * w2[:, None]])

    corr = udir[rows] * push[:, None]                       # move cloth closest point out to d1 = thickness
    edge_damp = float(C.ob.MC_props.ob_collision_edge_damping)
    f1 = corr * (w1 * edge_damp)[:, None]
    f2 = corr * (w2 * edge_damp)[:, None]
    moves = np.concatenate([f1, f2])
    _clamp_rows(moves, _max_move(cache, hits, thickness))
    n = len(rows)
    grp = (np.concatenate([np.arange(n), np.arange(n)]), np.concatenate([1.0 - s_h, s_h]))
    return hits, moves, grp


# ===========================================================================
# Accumulation / resolve
# ===========================================================================
def _accum_buffers(C, nv):
    fa = getattr(C, "_oc_accum", None)
    if fa is None or fa[0].shape[0] != nv:
        fa = (np.zeros((nv, 3), dtype=np.float64), np.zeros(nv, dtype=np.float64))
        C._oc_accum = fa
    return fa


def _scatter(force_accum, count_accum, idx, move):
    if idx is None or len(idx) == 0:
        return
    np.add.at(force_accum, idx, np.nan_to_num(move))
    np.add.at(count_accum, idx, 1.0)


def _project_contacts(nv, parts, rounds=None):
    """One move per vertex that satisfies all the contacts pushing it.

    parts: (idx, move) or (idx, move, (contact_id, weight)) from the contact
    functions.  A point contact asks its vertex to move |m| along m/|m|.  A
    triangle or edge contact asks the weighted point of several vertices to
    move its push along its normal, shared out as weight / sum(weight^2) --
    so it is judged as a whole, not as separate demands on each corner.

    In rounds: each contact's shortfall is measured against the moves so far;
    each vertex takes the correction of its most-unmet contact.  Pushes that
    agree give the strongest one exactly; small redundant ones are already
    met and add nothing; both faces of a crease get met; and a triangle or
    edge contact whose corners were already lifted by other contacts asks
    only for what is still missing, rather than stacking on top.

    Contacts used to be averaged by count, which lets many small contacts
    outvote one large one: a cloth vertex one thickness over a cone tip is
    also almost exactly one thickness from all 32 edges that meet there, each
    wanting a millimetre, so the tip's push was divided by ~30 and the sheet
    settled onto the tip.  Averaging also left cloth over a cube ridge at
    2.6-4.0 cm from the ridge line with a 5 cm thickness.
    """
    rounds = OC_CONTACT_ROUNDS if rounds is None else rounds
    x = np.zeros((nv, 3), dtype=np.float64)
    hit = np.zeros(nv, dtype=bool)
    V, M, CID, A = [], [], [], []
    base = 0
    for part in parts:
        if part is None or part[0] is None or len(part[0]) == 0:
            continue
        idx, mv = part[0], np.nan_to_num(np.asarray(part[1], dtype=np.float64))
        if len(part) > 2 and part[2] is not None:
            cid, a = part[2]
        else:
            cid, a = np.arange(len(idx)), np.ones(len(idx))
        V.append(idx)
        M.append(mv)
        CID.append(cid + base)
        A.append(np.asarray(a, dtype=np.float64))
        base += int(cid.max()) + 1
    if not V:
        return x, hit
    v = np.concatenate(V)
    mv = np.concatenate(M)
    cid = np.concatenate(CID)
    a = np.concatenate(A)
    hit[v] = True

    ml = np.sqrt(np.einsum("ij,ij->i", mv, mv))
    live = (ml > OC_GEOM_TINY) & (a > OC_GEOM_TINY)
    v, mv, cid, a, ml = v[live], mv[live], cid[live], a[live], ml[live]
    if not v.size:
        return x, hit
    n = mv / ml[:, None]                                    # contact normal, per row
    nc = int(cid.max()) + 1
    asq = np.bincount(cid, a * a, minlength=nc)            # sum of weight^2 per contact
    # the contact's push, from any of its rows: |m_r| = a_r / sum(a^2) * push
    push = np.zeros(nc)
    np.maximum.at(push, cid, ml * asq[cid] / a)
    share = a / asq[cid]                                   # this row's part of a unit correction

    for _ in range(rounds):
        moved = np.bincount(cid, a * np.einsum("ij,ij->i", x[v], n), minlength=nc)
        deficit = push - moved
        corr = share * deficit[cid]
        pos = corr > OC_EPS
        if not pos.any():
            break
        best = np.full(nv, -np.inf)
        np.maximum.at(best, v[pos], corr[pos])
        sel = np.nonzero(pos & (corr == best[v]))[0]
        _, first = np.unique(v[sel], return_index=True)    # one per vertex
        sel = sel[first]
        x[v[sel]] += corr[sel, None] * n[sel]
    return x, hit

def _friction_applied(count):
    """Called with the number of friction corrections each pass applies, by
    both the Python and the C++ path.  Does nothing; it is a hook for the
    benchmark to count friction contacts without caring which ran."""


def _narrow_resolve(data, C, cands, thickness, iters, relax, do_friction=False):
    # C++ when the scene asks for it and the DLL is loaded -- see
    # collide_native.  Same arguments, same results.
    native = getattr(C, "collide_native", None)
    if native is not None:
        return native.narrow_resolve(_THIS, data, C, cands, thickness,
                                     iters, relax, do_friction)
    props = C.ob.MC_props
    do_pt = bool(props.cl_point_tris)
    do_tp = bool(props.ob_point_tris)
    do_ee = bool(props.ob_edges)

    nv = int(C.vc)
    cache = _get_cache(C)
    force_accum, count_accum = _accum_buffers(C, nv)
    any_hit = np.zeros(nv, dtype=bool)
    total_step = np.zeros((nv, 3), dtype=np.float64)
    step_clamp = _max_move(cache, np.arange(nv), thickness)

    clv, obt = cands.get("pt", (None, None))
    obv, clt = cands.get("tp", (None, None))
    cle, obe = cands.get("ee", (None, None))

    for _ in range(max(1, iters)):
        parts = []
        if do_pt and clv is not None:
            parts.append(point_to_triangle(data, C, clv, obt, thickness, cache))
        if do_tp and obv is not None:
            parts.append(triangle_to_point(data, C, obv, clt, thickness, cache))
        if do_ee and cle is not None:
            parts.append(edge_to_edge(data, C, cle, obe, thickness, cache))

        step, hit = _project_contacts(nv, parts)
        if not np.any(hit):
            break

        step = np.nan_to_num(step * relax)
        _clamp_rows(step, step_clamp)
        C.co += step.astype(C.co.dtype)
        total_step += step
        any_hit |= hit

    # ---- friction: one pass on the now-settled positions ----
    mu = float(getattr(props, "ob_friction", 0.0))          # cloth scalar = the master gate
    fmax = float(data.get("friction_max", 1.0))             # 0 only if every collider is slick
    if do_friction and mu > 0.0 and fmax > 0.0 and np.any(any_hit):
        st = float(getattr(props, "ob_static_threshold", 0.0))
        cfg = getattr(C, "friction_group", None)            # cloth MC_friction weights (vc,1) or None
        fpar = (mu, st, total_step, cfg)
        force_accum[:] = 0.0
        count_accum[:] = 0.0
        if do_pt and clv is not None:
            _scatter(force_accum, count_accum, *point_to_triangle(data, C, clv, obt, thickness, cache, fpar))
        if do_tp and obv is not None:
            _scatter(force_accum, count_accum, *triangle_to_point(data, C, obv, clt, thickness, cache, fpar))
        if do_ee and cle is not None:
            _scatter(force_accum, count_accum, *edge_to_edge(data, C, cle, obe, thickness, cache, fpar))
        fhit = count_accum > 0.0
        _friction_applied(int(count_accum.sum()))
        if np.any(fhit):
            inv = np.zeros(nv)
            np.divide(1.0, count_accum, out=inv, where=fhit)
            fstep = np.nan_to_num(force_accum * inv[:, None])
            _clamp_rows(fstep, np.full(nv, OC_FRICTION_MAXMOVE * thickness))
            C.co += fstep.astype(C.co.dtype)   # tangential; reaches velocity via MC5's feedback

    return any_hit, total_step


# ===========================================================================
# Public entry points
# ===========================================================================
def _broad_phase(data, C, cl_joined, cl_trijoined, ob_joined, obt_lo, obt_hi,
                 detect, want):
    """Every (element, element) pair whose bounding boxes come within `detect`
    of each other, as de-duplicated, sorted candidate lists.

    cl_joined (V,k,3) / ob_joined are each vertex's positions over the time
    span being searched; cl_trijoined (T,3k,3) the same for cloth triangles.
    The splits and the root cull are conservative, so the result is exactly
    the pairwise box test -- _restrict_pairs relies on that, and so does the
    C++ version, which finds the same pairs by sort-and-sweep.
    """
    native = getattr(C, "collide_native", None)
    if native is not None:
        return native.broad_phase(_THIS, data, C, cl_joined, cl_trijoined, ob_joined,
                                  obt_lo, obt_hi, detect, want)
    clv_min, clv_max = get_poly_bounds(cl_joined)
    clt_min, clt_max = get_poly_bounds(cl_trijoined)
    obv_min, obv_max = get_poly_bounds(ob_joined)

    cl_lo, cl_hi = U.get_bounds(cl_joined.reshape(-1, 3))
    ob_lo, ob_hi = U.get_bounds(ob_joined.reshape(-1, 3))
    inter = overlapping_box(ob_lo - detect, ob_hi + detect, cl_lo - detect, cl_hi + detect)
    if inter is None:
        return None
    box_lo, box_hi = inter

    def _in_box(mn, mx):
        return np.all((mn <= box_hi) & (mx >= box_lo), axis=1)

    clv_sel = _in_box(clv_min, clv_max)
    obv_sel = _in_box(obv_min, obv_max)
    clt_sel = _in_box(clt_min, clt_max)
    obt_sel = _in_box(obt_lo, obt_hi)

    root = Node(
        clv_min[clv_sel], clv_max[clv_sel],
        obv_min[obv_sel], obv_max[obv_sel],
        clt_min[clt_sel], clt_max[clt_sel],
        obt_lo[obt_sel], obt_hi[obt_sel],
        C.idxer[clv_sel], data["idxer"][obv_sel],
        C.tidx[clt_sel], data["tidx"][obt_sel],
    )

    cand = Candidates()
    count = _count(root)
    if count > 0:
        # A box stops splitting once it holds count / 2**depth pairs --
        # relative to the root, see split_tuner.py.  MC5 sets oc_split_depth
        # each call (the tuner's pick, or the property).
        depth = int(getattr(C, "oc_split_depth",
                            getattr(C.ob.MC_props, "ob_box_depth", 5)))
        depth = min(max(depth, 0), OC_MAX_DEPTH)
        t_count = max(1, int(count) >> depth)
        # _leaf reads edge positions from C.joined_co / data["joined_co"], so
        # they have to be the arrays for the span being searched
        keep = C.joined_co, data["joined_co"]
        C.joined_co, data["joined_co"] = cl_joined, ob_joined
        try:
            split_box_t(C, data, root, count, box_lo, box_hi, cand, detect, want,
                        t_count=t_count, depth=0, max_depth=OC_MAX_DEPTH)
        finally:
            C.joined_co, data["joined_co"] = keep

    return cand.finalize(int(data["tidx"].size), int(C.tridex.shape[0]),
                         int(data["tridex_edges"].size))


def _pair_overlap(amin, amax, bmin, bmax, margin):
    """Row-by-row version of the _aabb_pairs test, same arithmetic."""
    sep = (amin - margin > bmax) | (amax + margin < bmin)
    return ~sep.any(axis=1)


def _restrict_pairs(data, C, cands, obt_lo, obt_hi, detect):
    """Cut a candidate superset down to exactly the pairs an unpadded broad
    phase over start -> current would have found.  Sorted order is kept, so
    the narrow phase sees the same lists in the same order."""
    out = {}
    a, b = cands.get("pt", (None, None))
    if a is not None:
        clv_min, clv_max = get_poly_bounds(C.joined_co[a])
        k = _pair_overlap(clv_min, clv_max, obt_lo[b], obt_hi[b], detect)
        out["pt"] = (a[k], b[k]) if k.any() else (None, None)
    else:
        out["pt"] = (None, None)
    a, b = cands.get("tp", (None, None))
    if a is not None:
        obv_min, obv_max = get_poly_bounds(data["joined_co"][a])
        clt_min, clt_max = get_poly_bounds(C.joined_trico[b])
        k = _pair_overlap(obv_min, obv_max, clt_min, clt_max, detect)
        out["tp"] = (a[k], b[k]) if k.any() else (None, None)
    else:
        out["tp"] = (None, None)
    a, b = cands.get("ee", (None, None))
    if a is not None:
        a_lo, a_hi = _edge_bounds_2frame(C.joined_co, C.tri_eidx[a])
        b_lo, b_hi = _edge_bounds_2frame(data["joined_co"], data["tridex_eidx"][b])
        k = _pair_overlap(a_lo, a_hi, b_lo, b_hi, detect)
        out["ee"] = (a[k], b[k]) if k.any() else (None, None)
    else:
        out["ee"] = (None, None)
    return out


def _as_recall(cands, want):
    pt = cands["pt"] if (want[0] and cands["pt"][0] is not None) else None
    tp = cands["tp"] if (want[1] and cands["tp"][0] is not None) else None
    ee = cands["ee"] if (want[2] and cands["ee"][0] is not None) else None
    return pt, ee, tp


def collision_force(data, C, ob, tridex, tidx, co_start, co_current, radius=0.03,
                    lookahead=None):
    """
    Main cloth<->collider entry.  `radius`, `tridex`, `tidx` are unused (kept
    for call-site compatibility).  Returns (point_tri, edge_edge, tri_point)
    for recollide(); each is a (cloth_idx, collider_idx) tuple or None.

    lookahead (recollide only): None, or how many times this call's motion to
    extend every box forward.  The broad phase then also finds the pairs the
    cloth is about to reach next step, and those are what is returned -- the
    next step's recollide starts from them instead of searching again.  This
    pass itself still resolves exactly the unpadded pairs, so its result does
    not change.
    """
    props = C.ob.MC_props
    thickness = float(props.ob_collision_radius)
    detect = thickness * OC_DETECT_SCALE
    want = (bool(props.cl_point_tris), bool(props.ob_point_tris), bool(props.ob_edges))

    C.joined_co[:, 0] = co_start
    C.joined_co[:, 1] = co_current

    if data["tidx"].size == 0 or not any(want):
        return None, None, None

    C.joined_trico[:, :3] = co_start[C.tridex]
    C.joined_trico[:, 3:] = co_current[C.tridex]
    obt_lo = np.minimum(C.start_local_trico.min(axis=1), C.local_trico.min(axis=1))
    obt_hi = np.maximum(C.start_local_trico.max(axis=1), C.local_trico.max(axis=1))

    if lookahead is None:
        cands = _broad_phase(data, C, C.joined_co, C.joined_trico, data["joined_co"],
                             obt_lo, obt_hi, detect, want)
        recall = None
    else:
        # each element's box also covers where it will be if it keeps moving
        # as it did this call, `lookahead` times over, plus one thickness of
        # slack for what bend and stretch add
        la = float(lookahead)
        cl_next = co_current + (co_current - co_start) * la
        cl3 = np.concatenate([C.joined_co, cl_next[:, None]], axis=1)
        clt3 = np.concatenate([C.joined_trico, cl_next[C.tridex]], axis=1)
        obj = data["joined_co"]
        ob3 = np.concatenate([obj, (obj[:, 1] + (obj[:, 1] - obj[:, 0]) * la)[:, None]],
                             axis=1)
        tri_next = C.local_trico + (C.local_trico - C.start_local_trico) * la
        obt3_lo = np.minimum(obt_lo, tri_next.min(axis=1))
        obt3_hi = np.maximum(obt_hi, tri_next.max(axis=1))
        wide = _broad_phase(data, C, cl3, clt3, ob3, obt3_lo, obt3_hi,
                            detect + thickness * OC_LOOKAHEAD_SLACK, want)
        if wide is None:
            cands = None
        else:
            cands = _restrict_pairs(data, C, wide, obt_lo, obt_hi, detect)
            recall = _as_recall(wide, want)
    if cands is None:
        return None, None, None
    if lookahead is None:
        recall = _as_recall(cands, want)

    any_hit, total_step = _narrow_resolve(
        data, C, cands, thickness, OC_ITERATIONS, OC_RELAX, do_friction=True)

    # The velocity side of the contact is not applied here: it is gathered and
    # applied ONCE per step by velocity_feedback(), which MC5.collision_stage
    # calls after every substep has run.  Applied per call it ran once per
    # substep and compounded -- the same sliding contact kept 90% of its
    # velocity at one substep and 53% at eight, so the substep count was
    # changing the physics.
    _vel_accumulate(C, any_hit, total_step)
    return recall


# ===========================================================================
# Velocity feedback, once per step
# ===========================================================================
def _vel_state(C, nv):
    acc = getattr(C, "_oc_vel", None)
    if acc is None or acc[0].shape[0] != nv:
        acc = (np.zeros(nv, dtype=bool), np.zeros((nv, 3), dtype=np.float64))
        C._oc_vel = acc
    return acc


def _vel_accumulate(C, any_hit, total_step):
    hit, step = _vel_state(C, int(C.vc))
    hit |= any_hit
    step += total_step


def begin_velocity(C):
    """Start of a step: forget the contacts gathered for the previous one."""
    hit, step = _vel_state(C, int(C.vc))
    hit[:] = False
    step[:] = 0.0


def _rec_state(C, nv):
    acc = getattr(C, "_oc_rec", None)
    if acc is None or acc[0].shape[0] != nv:
        acc = (np.zeros(nv, dtype=bool), np.zeros((nv, 3), dtype=np.float64))
        C._oc_rec = acc
    return acc


def begin_recollide(C):
    """Start of a step: forget which verts recollide pushed last step."""
    hit, push = _rec_state(C, int(C.vc))
    hit[:] = False
    push[:] = 0.0


def _recollide_no_bounce(C):
    """Recollide pushes are position-only, but MC5 turns the step's whole
    displacement into velocity afterwards -- so every push that lifted a
    vertex off a collider left it moving away from it next frame.  On a sharp
    tip that is a kick of several cm a frame: the jumping.

    The pushes can't simply be subtracted: bend and stretch undo most of each
    one and recollide pushes again every iteration, so their sum is far more
    than the vertex actually moved.  Instead, per pushed vertex, whatever
    velocity it would leave the step with AWAY from the collider (along the
    direction it was pushed) is removed.  Sliding and settling are kept; only
    the bounce goes, as the end-of-step pass already does for its contacts.
    """
    rec = getattr(C, "_oc_rec", None)
    if rec is None or not rec[0].any():
        return
    hit, push = rec
    idx = np.nonzero(hit)[0]
    p = push[idx]
    pn = np.sqrt(np.einsum("ij,ij->i", p, p))
    keep = pn > OC_EPS
    idx, p, pn = idx[keep], p[keep], pn[keep]
    if not idx.size:
        return
    u = p / pn[:, None]
    # the velocity this vertex will leave the step with: MC5 adds the step's
    # displacement (co - vel_start) to C.velocity right after this
    v_next = (C.velocity[idx].astype(np.float64) +
              (C.co[idx] - C.vel_start[idx]).astype(np.float64))
    out = np.einsum("ij,ij->i", v_next, u)
    away = out > 0.0
    if np.any(away):
        C.velocity[idx[away]] -= (out[away, None] * u[away]).astype(C.velocity.dtype)


def velocity_feedback(C):
    """Inelastic contact, applied once per step over every substep's contacts.

    MC5 adds (co - vel_start) to C.velocity afterwards, so the positional
    push-out would become bounce velocity (restitution 1) and a resting cloth
    would keep its full inbound velocity -- the surface buzzes.  So per
    contacted vertex: remove the velocity component heading INTO the collider
    (along the net push-out axis) and cancel the push-out from the feedback.

    With one substep this is exactly what collision_force used to do inline.
    With several, the push-outs are summed into one net step first, so the
    damping is applied once rather than once per substep.
    """
    hit, step = _vel_state(C, int(C.vc))
    if not np.any(hit):
        _recollide_no_bounce(C)
        return
    idx = np.nonzero(hit)[0]
    s = step[idx]
    sn = np.sqrt(np.einsum("ij,ij->i", s, s))
    keep = sn > OC_EPS
    idx = idx[keep]
    s = s[keep]
    if idx.size:
        u = s / sn[keep, None]
        v = C.velocity[idx].astype(np.float64)
        vdot = np.einsum("ij,ij->i", v, u)
        v -= vdot[:, None] * u                             # zero the whole normal component
        v -= (1.0 - OC_RESTITUTION) * s                    # cancel push-out feedback
        if OC_VEL_DAMP < 1.0:
            v *= OC_VEL_DAMP
        C.velocity[idx] = v.astype(C.velocity.dtype)
    hit[:] = False
    step[:] = 0.0
    _recollide_no_bounce(C)


def find_pairs(data, C, co_start, co_current):
    """Broad phase on its own, for when there are no cached pairs yet (the
    first step, or the step after the colliders changed).  Same result format
    as collision_force."""
    props = C.ob.MC_props
    want = (bool(props.cl_point_tris), bool(props.ob_point_tris), bool(props.ob_edges))
    if data["tidx"].size == 0 or not any(want):
        return None, None, None
    thickness = float(props.ob_collision_radius)
    C.joined_co[:, 0] = co_start
    C.joined_co[:, 1] = co_current
    C.joined_trico[:, :3] = co_start[C.tridex]
    C.joined_trico[:, 3:] = co_current[C.tridex]
    obt_lo = np.minimum(C.start_local_trico.min(axis=1), C.local_trico.min(axis=1))
    obt_hi = np.maximum(C.start_local_trico.max(axis=1), C.local_trico.max(axis=1))
    cands = _broad_phase(data, C, C.joined_co, C.joined_trico, data["joined_co"],
                         obt_lo, obt_hi,
                         thickness * (OC_DETECT_SCALE + OC_LOOKAHEAD_SLACK), want)
    if cands is None:
        return None, None, None
    return _as_recall(cands, want)


def near_pairs(data, C, point_tri, edge_edge, tri_point):
    """Keep only the cached pairs that are close to touching right now.

    The cached list is everything the broad phase could not rule out -- about
    8,800 pairs for a sheet draped on a cone, of which a handful are in
    contact.  Recollide runs once per bend / stretch iteration, so it pays for
    every pair many times a step.  This runs once, at the start of the step
    after velocity has moved the cloth, and keeps pairs within
    (1 + OC_RECOLLIDE_BAND) * thickness of contact.
    """
    native = getattr(C, "collide_native", None)
    if native is not None:
        return native.near_pairs(_THIS, data, C, point_tri, edge_edge, tri_point)

    thickness = float(C.ob.MC_props.ob_collision_radius)
    reach = thickness * (1.0 + OC_RECOLLIDE_BAND)

    def box_near(p, lo, hi):
        return np.all((p >= lo - reach) & (p <= hi + reach), axis=1)

    pt = None
    if point_tri is not None:
        v, t = point_tri
        tri = C.local_trico[t]
        p = C.co[v].astype(np.float64)
        d = np.einsum("ij,ij->i", p - tri[:, 0], C.local_normals[t])
        k = ((d < reach) & (d > -(OC_PUSH_CAP * thickness + reach)) &
             box_near(p, tri.min(axis=1), tri.max(axis=1)))
        if k.any():
            pt = (v[k], t[k])

    tp = None
    if tri_point is not None:
        v, t = tri_point
        P = data["joined_co"][v, 1].astype(np.float64)
        tco = C.co[C.tridex[t]].astype(np.float64)
        n = np.cross(tco[:, 1] - tco[:, 0], tco[:, 2] - tco[:, 0])
        n = _unit(n)
        d = np.einsum("ij,ij->i", P - tco[:, 0], n)
        k = (np.abs(d) < reach) & box_near(P, tco.min(axis=1), tco.max(axis=1))
        if k.any():
            tp = (v[k], t[k])

    ee = None
    if edge_edge is not None:
        a, b = edge_edge
        ea = C.tri_eidx[a]
        eb = data["tridex_eidx"][b]
        ob_now = data["joined_co"][:, 1]
        c1, c2 = closest_point_segments(C.co[ea[:, 0]], C.co[ea[:, 1]],
                                        ob_now[eb[:, 0]], ob_now[eb[:, 1]])[:2]
        dist = np.linalg.norm(c1 - c2, axis=1)
        k = dist < reach
        if k.any():
            ee = (a[k], b[k])

    return pt, ee, tp


def recollide(data, C, point_tri, edge_edge, tri_point):
    """Re-resolve the stored candidate lists after other solver steps moved C.co.
    Positional only -- no velocity feedback."""
    cands = {}
    if point_tri:
        cands["pt"] = point_tri
    if tri_point:
        cands["tp"] = tri_point
    if edge_edge:
        cands["ee"] = edge_edge
    if not cands:
        return

    thickness = float(C.ob.MC_props.ob_collision_radius)
    any_hit, total_step = _narrow_resolve(data, C, cands, thickness,
                                          OC_RECOLLIDE_ITERS, OC_RELAX)
    # which way each vertex was pushed, for _recollide_no_bounce
    hit, push = _rec_state(C, int(C.vc))
    hit |= any_hit
    push += total_step
