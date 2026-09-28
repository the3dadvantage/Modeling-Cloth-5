"""
Self collision (vertex-triangle + edge-edge) for Modeling Cloth.

Rewrite notes -- what changed vs. the previous version
-----------------------------------------------------
BUGS FIXED
  * One shared force buffer + one shared per-vertex constraint count.  The old
    code called add_forces() three times, each zeroing the buffer, normalising
    by its own bincount and writing C.co immediately, so the vertex side and
    triangle side of the *same* contact were scaled differently -> net impulse
    injected every contact ("swimming").
  * Triangle-side correction now uses the PBD-exact  w / sum(w^2)  distribution
    (the commented-out formula in utils.point_to_triangle_forces), not 1/max(w)
    which is only correct at the centroid.
  * Edge-edge no longer applies the full closest-point correction to *both*
    edges (that was a 2x overshoot at mid-edge contacts).  Each edge now moves
    half; combined relative motion is exactly the target.
  * Velocity of every vertex that received a response is damped
    (C.ob.MC_props.sc_vel_damping) -- the old module dropped this, so a fast
    vertex kept the momentum that caused the tunnel.
  * EPS is now float32-safe (1e-7, was 1e-12 -> below fp32 epsilon).
  * closest_point_segments() has the standard clamp cascade -> handles
    parallel / near-parallel edges (every cloth fold) instead of collapsing
    s = t = 0.
  * Every response move is magnitude-clamped (per constraint and per step) so a
    bad flip can't launch a vertex across the mesh.
  * split_box() count metric is consistent at the root and children;
    max recursion depth raised.
  * The unused positional `radius` arg is ignored (thickness = sc_radius as
    before) and documented; `sc_edges` and `sc_damping` are actually honoured.

NEW
  * Continuous (swept) narrow phase: sign-of-plane-distance change for
    vertex-triangle and sign-of-separation change for edge-edge, evaluated at
    the interpolated crossing time.  Catches tunnelling regardless of the end
    distance.  Toggle: SC_SWEPT.
  * Relaxation loop: SC_ITERATIONS narrow-phase passes per call, re-reading
    C.co, so resolving contact A can't leave a fresh penetration B.
  * Topological neighbour exclusion.  A per-mesh cache (built lazily, rebuilt on
    topology / radius change) stores every vertex pair within K edge-hops.
    Candidate vertex-triangle and edge-edge pairs whose members are within K
    rings are dropped, so you can raise sc_radius without the sheet repelling
    neighbouring-on-the-surface geometry.  K is chosen automatically from
    sc_radius / mean-edge-length (override with SC_RING_DEPTH).
  * detect radius (broad phase) is decoupled from contact thickness
    (SC_DETECT_SCALE) so the broad phase keeps enough candidates for the swept
    tests while the shell stays thin.
  * Per-vertex move clamp uses a characteristic (mean incident edge) length.
  * AABB pre-filter before the O(n^2) pair matrices in each leaf; edge pairing
    is tiled to bound memory.
  * SC_SHEET_SIDE: a contacting vertex is pushed toward the side its own 1-ring
    sits on, rather than trusting sign(current plane distance) which inverts
    once a stack compresses past a triangle height.
  * SC_RESTITUTION: MC5 folds (co - vel_start) back into C.velocity after this
    call, so a positional push-out otherwise becomes bounce velocity.  For
    resting (non-tunnel) contacts that contribution is removed -> inelastic
    contact.  Tunnel recoveries keep it so they don't fall straight back.

C++: mc_collide.dll (cpp/mc_collide) ports the candidate search and the
relaxation loop, used when the scene's collision backend is C++ (see
collide_native.py).  This module stays the reference and owns every tunable.

Tunables are module-level constants just below.  Promote any of them to
MC_props later if you want them on the panel.
"""

import numpy as np
import bpy

# Same pattern as every other module -- see object_collide.py.
try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U

# ===========================================================================
# Tunables
# ===========================================================================
SC_EPS            = 1e-7      # float32-safe epsilon, for LENGTHS only
# Degeneracy relative to the element's own size -- see object_collide.py.
SC_GEOM_REL_EPS   = 1e-10
SC_GEOM_TINY      = 1e-30
EPS               = SC_EPS   # backwards-compat alias

SC_ITERATIONS     = 4        # narrow-phase relaxation passes per call (1 == old behaviour)
SC_RELAX          = 0.9      # under-relaxation; effective = SC_RELAX * sc_damping
                            #   lower it (0.5-0.7) if stacked contacts buzz;
                            #   raise SC_ITERATIONS for faster penetration recovery
SC_BLEND          = 0.5      # vertex vs triangle share of a *proximity* contact
                            #   (a detected full tunnel always pushes the vertex 100%)
FORCE_BLEND       = SC_BLEND # backwards-compat alias

SC_SWEPT          = True     # continuous (sign-change) vertex-tri / edge-edge tests
SC_SWEPT_MARGIN   = 0.05     # barycentric slack for the swept containment test
SC_PBD_EXACT      = True     # True: w/sum(w^2) (constraint-exact). False: w (softer, momentum-conserving)
SC_SHEET_SIDE     = True     # push a contacting vertex toward the side its own 1-ring
                            #   sits on, with sign(start-of-frame distance) as the
                            #   tie-breaker.  More robust than sign(current distance),
                            #   which inverts once a stack compresses past a triangle.

SC_DETECT_SCALE   = 2.0      # broad-phase reach  = SC_DETECT_SCALE * sc_radius
SC_CLAMP_THICK    = 4.0      # per-move clamp = max(SC_CLAMP_THICK*thickness,
SC_CLAMP_EDGE     = 2.0      #                      SC_CLAMP_EDGE*char_edge_len)

SC_RING_DEPTH     = None     # None -> auto from radius/edge length; or force an int
SC_RING_FACTOR    = 1.0      # auto depth so ring reach ~ SC_RING_FACTOR * detect_radius
SC_RING_MIN_DEPTH = 1
SC_RING_MAX_DEPTH = 5

SC_MAX_DEPTH      = 16        # spatial split recursion cap
SC_TILE           = 512      # row tile for AABB / edge pair matrices (memory cap)
SC_MIN_SPLIT_GAIN = 0.98     # bail to a leaf if a split shrinks the bigger child by less than this

SC_RESTITUTION    = 0.0      # 0 = inelastic: the positional push-out of a *resting*
                            #   contact is subtracted back out of the velocity feedback
                            #   (see module docstring).  >0 leaves some as bounce.
                            #   Tunnel push-outs are never cancelled (they must persist).

DEFAULT_MAX_DEPTH = SC_MAX_DEPTH  # backwards-compat alias


class _ModuleView:
    """This module's tunables as attributes, for collide_native (loaded as a
    text datablock the module is not in sys.modules).  Reads live globals."""
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
    __slots__ = ("cl_min", "cl_max", "cl_tri_min", "cl_tri_max", "cl_idxer", "cl_tidxer")

    def __init__(self, cl_min, cl_max, cl_tri_min, cl_tri_max, cl_idxer, cl_tidxer):
        self.cl_min = cl_min
        self.cl_max = cl_max
        self.cl_tri_min = cl_tri_min
        self.cl_tri_max = cl_tri_max
        self.cl_idxer = cl_idxer
        self.cl_tidxer = cl_tidxer


class CollisionCandidates:
    def __init__(self):
        self.verts = []
        self.tris = []
        self.edge_i = []
        self.edge_j = []

    def finalize(self, nv, nt):
        if self.verts:
            v = np.concatenate(self.verts)
            t = np.concatenate(self.tris)
            key = v.astype(np.int64) * nt + t
            _, u = np.unique(key, return_index=True)
            v, t = v[u], t[u]
        else:
            v = t = None

        if self.edge_i:
            ea = np.concatenate(self.edge_i)
            eb = np.concatenate(self.edge_j)
            a = np.sort(ea, axis=1)
            b = np.sort(eb, axis=1)
            ka = a[:, 0].astype(np.int64) * nv + a[:, 1]
            kb = b[:, 0].astype(np.int64) * nv + b[:, 1]
            lo = np.minimum(ka, kb)
            hi = np.maximum(ka, kb)
            _, u = np.unique(np.stack([lo, hi], axis=1), axis=0, return_index=True)
            ea, eb = ea[u], eb[u]
        else:
            ea = eb = None

        return v, t, ea, eb


# ===========================================================================
# Per-mesh cache: topological neighbour keys + characteristic lengths
# ===========================================================================
def _ring_keys(edges, nv, K):
    """Sorted int64 array of  v*nv + u  for every u within K edge-hops of v."""
    ebi = np.concatenate([edges, edges[:, ::-1]], axis=0)
    try:
        import scipy.sparse as sp
        data = np.ones(ebi.shape[0], dtype=np.int8)
        A = sp.csr_matrix((data, (ebi[:, 0], ebi[:, 1])), shape=(nv, nv))
        reach = sp.identity(nv, dtype=np.int8, format="csr") + A
        reach.data[:] = 1
        P = A
        for _ in range(max(0, K - 1)):
            P = P @ A
            P.data[:] = 1
            reach = reach + P
            reach.data[:] = 1
        reach = reach.tocoo()
        return np.sort(reach.row.astype(np.int64) * nv + reach.col.astype(np.int64))
    except Exception:
        adj = [[] for _ in range(nv)]
        for a, b in ebi:
            adj[a].append(b)
        keys = []
        for s in range(nv):
            seen = {s}
            frontier = [s]
            for _ in range(K):
                nxt = []
                for u in frontier:
                    for w in adj[u]:
                        if w not in seen:
                            seen.add(w)
                            nxt.append(w)
                frontier = nxt
                if not frontier:
                    break
            base = s * nv
            keys.extend(base + u for u in seen)
        return np.sort(np.array(keys, dtype=np.int64))


def _build_cache(C, detect):
    nv = int(C.vc)
    T = C.tridex
    nt = int(T.shape[0])
    ref = np.asarray(C.start_co, dtype=np.float64)

    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)

    el = np.linalg.norm(ref[e[:, 0]] - ref[e[:, 1]], axis=1)
    mean_edge = float(np.mean(el)) if el.size else 1.0

    char = np.zeros(nv)
    cnt = np.zeros(nv)
    np.add.at(char, e[:, 0], el)
    np.add.at(char, e[:, 1], el)
    np.add.at(cnt, e[:, 0], 1.0)
    np.add.at(cnt, e[:, 1], 1.0)
    char = np.where(cnt > 0, char / np.where(cnt > 0, cnt, 1.0), mean_edge)

    # padded 1-ring adjacency  (nv, max_deg)  for the "which side is my own
    # sheet on" vote -- lets a contact push a vertex back toward its own layer
    # instead of toward whichever side of the triangle plane it currently sits
    # on (which inverts once layers compress past a triangle height).
    ebi = np.concatenate([e, e[:, ::-1]], axis=0)
    order = np.argsort(ebi[:, 0], kind="stable")
    src = ebi[order, 0]
    dst = ebi[order, 1]
    deg = np.bincount(src, minlength=nv)
    max_deg = int(deg.max()) if deg.size else 1
    starts = np.cumsum(deg) - deg
    pos = np.arange(src.size) - np.repeat(starts, deg)
    adj_pad = np.zeros((nv, max_deg), dtype=np.int32)
    adj_mask = np.zeros((nv, max_deg), dtype=bool)
    adj_pad[src, pos] = dst
    adj_mask[src, pos] = True

    if SC_RING_DEPTH is not None:
        K = int(SC_RING_DEPTH)
    else:
        K = int(np.ceil(SC_RING_FACTOR * detect / max(mean_edge, 1e-9)))
    K = int(np.clip(K, SC_RING_MIN_DEPTH, SC_RING_MAX_DEPTH))

    try:
        reach_keys = _ring_keys(e, nv, K)
    except Exception as ex:  # noqa: BLE001
        print("[self_collide_2] ring key build failed:", ex)
        reach_keys = None

    print("[self_collide_2] neighbour exclusion K=%d  mean_edge=%.5f  detect=%.5f  keys=%s"
          % (K, mean_edge, detect, "none" if reach_keys is None else reach_keys.size))

    return {"nv": nv, "nt": nt, "detect": float(detect), "depth": K,
            "reach_keys": reach_keys, "char_len": char,
            "adj_pad": adj_pad, "adj_mask": adj_mask}


def _get_cache(C, detect):
    cache = getattr(C, "_sc_cache", None)
    nv = int(C.vc)
    nt = int(C.tridex.shape[0])
    ok = cache is not None and cache.get("nv") == nv and cache.get("nt") == nt
    if ok:
        d0 = cache.get("detect", 0.0)
        if d0 <= 0.0 or not (0.66 <= detect / d0 <= 1.5):
            ok = False
    if not ok:
        try:
            cache = _build_cache(C, detect)
        except Exception as ex:  # noqa: BLE001
            print("[self_collide_2] cache build failed, neighbour exclusion off:", ex)
            cache = {"nv": nv, "nt": nt, "detect": float(detect), "depth": 0,
                     "reach_keys": None, "char_len": None,
                     "adj_pad": None, "adj_mask": None}
        C._sc_cache = cache
    return cache


# ===========================================================================
# Vector helpers
# ===========================================================================
def _unit(v):
    n = np.sqrt(np.einsum("ij,ij->i", v, v))
    n = np.where(n > SC_EPS, n, 1.0)
    return v / n[:, None]


def _bary(tri_co, w):
    """tri_co (M,3,3), w (M,3) -> (M,3) barycentric point."""
    return np.einsum("mij,mi->mj", tri_co, w)


def _clamp_rows(vecs, max_norm):
    n = np.sqrt(np.einsum("ij,ij->i", vecs, vecs))
    big = n > max_norm
    if np.any(big):
        vecs[big] *= (max_norm[big] / n[big])[:, None]


def _max_move(cache, vidx, thickness):
    base = SC_CLAMP_THICK * thickness
    char = cache.get("char_len") if cache else None
    if char is not None:
        return np.maximum(base, SC_CLAMP_EDGE * char[vidx])
    return np.full(len(vidx), max(base, SC_CLAMP_EDGE * 10.0 * thickness))


def _sheet_side(C, cache, vidx, plane_pt, n, fallback):
    """+/-1 telling which side of the plane (plane_pt, n) vertex `vidx`'s own
    1-ring sits on.  Returns `fallback` where the vote is inconclusive.
    This is what keeps a squashed lower layer from being driven up through the
    layer above it: the vertex is pushed back toward its own material, not
    toward whichever side sign(distance) currently reports.
    """
    ap = cache.get("adj_pad") if cache else None
    am = cache.get("adj_mask") if cache else None
    if ap is None:
        return fallback
    nbr = C.co[ap[vidx]]                                   # (H, D, 3)
    dn = np.einsum("hkj,hj->hk", nbr - plane_pt[:, None, :], n)
    s = np.sum(dn * am[vidx], axis=1)                      # (H,)
    side = np.sign(s)
    return np.where(np.abs(s) > SC_EPS, side, fallback)


# ===========================================================================
# Broad phase
# ===========================================================================
def get_poly_bounds(arr):
    return np.min(arr, axis=1), np.max(arr, axis=1)


def _aabb_pairs(amin, amax, bmin, bmax, margin):
    """Index pairs (i, j) whose (inflated) AABBs overlap on all 3 axes."""
    na = amin.shape[0]
    rows = []
    cols = []
    for i0 in range(0, na, SC_TILE):
        i1 = min(na, i0 + SC_TILE)
        sep = ((amin[i0:i1, None, :] - margin > bmax[None, :, :]) |
               (amax[i0:i1, None, :] + margin < bmin[None, :, :]))
        r, c = np.nonzero(~sep.any(axis=2))
        rows.append(r + i0)
        cols.append(c)
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def edges_from_tris(C, tidx):
    C.tridex_edge_booler[:] = False
    C.tridex_edge_booler[C.tri_edge_idxer[tidx]] = True
    return C.tridex_edges[C.tridex_edge_booler]


def _filter_vt(C, cache, verts, tri_v):
    labels = getattr(C, "sew_labels", None)
    if labels is None:
        keep = (verts != tri_v[:, 0]) & (verts != tri_v[:, 1]) & (verts != tri_v[:, 2])
    else:
        vl = labels[verts]
        tl = labels[tri_v]
        keep = (vl != tl[:, 0]) & (vl != tl[:, 1]) & (vl != tl[:, 2])

    rk = cache.get("reach_keys") if cache else None
    if rk is not None:
        nv = cache["nv"]
        k = verts.astype(np.int64)[:, None] * nv + tri_v
        keep &= ~np.isin(k, rk).any(axis=1)
    return keep


def _filter_ee(C, cache, ea, eb):
    a0, a1 = ea[:, 0], ea[:, 1]
    b0, b1 = eb[:, 0], eb[:, 1]
    labels = getattr(C, "sew_labels", None)
    if labels is None:
        keep = (a0 != b0) & (a0 != b1) & (a1 != b0) & (a1 != b1)
    else:
        la0, la1 = labels[a0], labels[a1]
        lb0, lb1 = labels[b0], labels[b1]
        keep = (la0 != lb0) & (la0 != lb1) & (la1 != lb0) & (la1 != lb1)

    rk = cache.get("reach_keys") if cache else None
    if rk is not None:
        nv = cache["nv"]
        k = np.stack([a0.astype(np.int64) * nv + b0,
                      a0.astype(np.int64) * nv + b1,
                      a1.astype(np.int64) * nv + b0,
                      a1.astype(np.int64) * nv + b1], axis=1)
        keep &= ~np.isin(k, rk).any(axis=1)
    return keep


def _leaf_edge_pairs(C, eidx, margin):
    E = eidx.shape[0]
    eco = np.concatenate([C.co[eidx], C.start_co[eidx]], axis=1)  # (E,4,3)
    emin = eco.min(axis=1)
    emax = eco.max(axis=1)
    ii = []
    jj = []
    for i0 in range(0, E, SC_TILE):
        i1 = min(E, i0 + SC_TILE)
        sep = ((emin[i0:i1, None, :] - margin > emax[None, :, :]) |
               (emax[i0:i1, None, :] + margin < emin[None, :, :]))
        r, c = np.nonzero(~sep.any(axis=2))
        r = r + i0
        m = r < c
        ii.append(r[m])
        jj.append(c[m])
    ii = np.concatenate(ii) if ii else np.empty(0, np.int64)
    jj = np.concatenate(jj) if jj else np.empty(0, np.int64)
    if ii.size == 0:
        return None, None
    return eidx[ii], eidx[jj]


def generate_leaf_pairs(C, node, cand, cache, detect):
    if node.cl_idxer.shape[0] == 0 or node.cl_tidxer.shape[0] == 0:
        return

    r, c = _aabb_pairs(node.cl_min, node.cl_max, node.cl_tri_min, node.cl_tri_max, detect)
    if r.size:
        verts = node.cl_idxer[r]
        tris_g = node.cl_tidxer[c]
        tri_v = C.tridex[tris_g]

        keep = _filter_vt(C, cache, verts, tri_v)
        verts = verts[keep]
        tris_g = tris_g[keep]
        if verts.size:
            cand.verts.append(verts)
            cand.tris.append(tris_g)

    # Edge pairs come from the edges of EVERY triangle in this box.  They used
    # to come only from triangles that had kept a vertex-triangle pair in this
    # same box, so an edge pair was missed whenever those pairs happened to
    # land in different boxes -- the result depended (slightly) on the split
    # depth, which the auto tuner picks from timings.  Every edge's box lies
    # inside its triangle's box, and the splits never separate two triangles
    # within `detect` of each other, so two edges within `detect` always share
    # a box: the result is now exactly "every edge pair within detect" at any
    # depth, which is also what the C++ version computes.
    edge_ids = edges_from_tris(C, node.cl_tidxer)
    if edge_ids.size < 2:
        return
    eidx = C.tridex_eidx[edge_ids]
    ea, eb = _leaf_edge_pairs(C, eidx, detect)
    if ea is None:
        return
    keep = _filter_ee(C, cache, ea, eb)
    ea = ea[keep]
    eb = eb[keep]
    if ea.size:
        cand.edge_i.append(ea)
        cand.edge_j.append(eb)


def split_box(C, node, count, bmin, bmax, cand, cache, detect,
              t_count=100, depth=0, max_depth=SC_MAX_DEPTH):
    if count <= t_count or depth >= max_depth:
        if count > 0:
            generate_leaf_pairs(C, node, cand, cache, detect)
        return

    depth += 1
    dif = bmax - bmin
    axis = int(np.argmax(dif))
    mid = bmin[axis] + dif[axis] * 0.5
    hi = mid + detect
    lo = mid - detect

    v_lo = node.cl_min[:, axis] <= hi
    v_hi = node.cl_max[:, axis] >= lo
    t_lo = node.cl_tri_min[:, axis] <= hi
    t_hi = node.cl_tri_max[:, axis] >= lo

    left = Node(node.cl_min[v_lo], node.cl_max[v_lo],
                node.cl_tri_min[t_lo], node.cl_tri_max[t_lo],
                node.cl_idxer[v_lo], node.cl_tidxer[t_lo])
    right = Node(node.cl_min[v_hi], node.cl_max[v_hi],
                 node.cl_tri_min[t_hi], node.cl_tri_max[t_hi],
                 node.cl_idxer[v_hi], node.cl_tidxer[t_hi])
    lc = left.cl_idxer.shape[0] * left.cl_tidxer.shape[0]
    rc = right.cl_idxer.shape[0] * right.cl_tidxer.shape[0]

    # No useful partition on any axis (band wider than the feature, or every
    # primitive straddles mid): stop here instead of recursing to max_depth.
    if max(lc, rc) >= count * SC_MIN_SPLIT_GAIN:
        generate_leaf_pairs(C, node, cand, cache, detect)
        return

    nbmax = bmax.copy()
    nbmax[axis] = mid
    split_box(C, left, lc, bmin, nbmax, cand, cache, detect, t_count, depth, max_depth)

    nbmin = bmin.copy()
    nbmin[axis] = mid
    split_box(C, right, rc, nbmin, bmax, cand, cache, detect, t_count, depth, max_depth)


# ===========================================================================
# Geometry
# ===========================================================================
def inside_triangles(tris, points, margin=0.0):
    """Barycentric inside test. Returns (check, weights[w0,w1,w2], v2).

    float64, degeneracy relative to the triangle's own size -- see the
    object_collide.py copy for why the old absolute cut-off broke small
    triangles.  Kept identical to it; the C++ port will have one copy.
    """
    tris = np.asarray(tris, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, None]
    v0 = cross_vecs[:, 0]
    v1 = cross_vecs[:, 1]
    v2 = points - origins

    d00_d11 = np.einsum("ijk,ijk->ij", cross_vecs, cross_vecs)
    d00 = d00_d11[:, 0]
    d11 = d00_d11[:, 1]
    d01 = np.einsum("ij,ij->i", v0, v1)
    d02 = np.einsum("ij,ij->i", v0, v2)
    d12 = np.einsum("ij,ij->i", v1, v2)

    denom = d00 * d11 - d01 * d01
    degen = ~(denom > SC_GEOM_REL_EPS * d00 * d11)
    inv = 1.0 / np.where(degen, 1.0, denom)

    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv
    w = 1.0 - (u + v)

    weights = np.stack([w, u, v], axis=-1)
    weights[degen] = 1.0 / 3.0
    check = (u >= margin) & (v >= margin) & (w >= margin) & ~degen
    return check, weights, v2


def closest_point_segments(p1, q1, p2, q2):
    """Vectorised closest points between segments p1q1 and p2q2 (Ericson).

    Robust for parallel / near-parallel / degenerate segments.
    Returns (cpa, cpb, s, t) with s, t in [0, 1].
    """
    # float64, parallel test relative to the segments' own lengths -- see the
    # object_collide.py copy.  Kept identical to it.
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

    pt_a = a <= SC_GEOM_TINY
    pt_e = e <= SC_GEOM_TINY
    a_safe = np.where(pt_a, 1.0, a)
    e_safe = np.where(pt_e, 1.0, e)

    denom = a * e - b * b
    nonpar = denom > SC_GEOM_REL_EPS * a * e
    denom_safe = np.where(nonpar, denom, 1.0)
    s = np.where(nonpar, np.clip((b * f - c * e) / denom_safe, 0.0, 1.0), 0.0)

    t = (b * s + f) / e_safe

    t_lt0 = t < 0.0
    t_gt1 = t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(t_lt0, np.clip(-c / a_safe, 0.0, 1.0), s)
    s = np.where(t_gt1, np.clip((b - c) / a_safe, 0.0, 1.0), s)

    s = np.where(pt_a, 0.0, s)
    t = np.where(pt_a, np.clip(f / e_safe, 0.0, 1.0), t)
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
# Narrow phase
# ===========================================================================
_TP_NONE = (None, None, None, None, None, None)


def tri_point_response(C, tri_idx, vert_idx, thickness, cache):
    """Static repulsion + swept (tunnelling) response for vertex-triangle.

    Returns (vert_hits, vert_move, vert_swept,
             tri_hits, tri_move, tri_swept)  or  _TP_NONE.
    """
    if tri_idx is None or len(tri_idx) == 0:
        return _TP_NONE

    tri = C.tridex[tri_idx]                       # (M,3)
    tri_co1 = C.co[tri]                           # (M,3,3)
    p1 = C.co[vert_idx]
    A1, B1, D1 = tri_co1[:, 0], tri_co1[:, 1], tri_co1[:, 2]
    n1 = _unit(np.cross(B1 - A1, D1 - A1))
    d1 = np.einsum("ij,ij->i", p1 - A1, n1)

    check1, w1, _ = inside_triangles(tri_co1, p1, margin=0.0)
    plot1 = _bary(tri_co1, w1)
    gap1 = p1 - plot1
    dist1 = np.sqrt(np.einsum("ij,ij->i", gap1, gap1))
    prox = check1 & (dist1 < thickness)

    hit = prox.copy()
    swept = np.zeros(len(tri_idx), dtype=bool)
    d0 = np.zeros(len(tri_idx))
    wT = None

    if SC_SWEPT:
        tri_co0 = C.start_co[tri]
        p0 = C.start_co[vert_idx]
        A0, B0, D0 = tri_co0[:, 0], tri_co0[:, 1], tri_co0[:, 2]
        n0 = _unit(np.cross(B0 - A0, D0 - A0))
        d0 = np.einsum("ij,ij->i", p0 - A0, n0)

        denom = d0 - d1
        crossed = (np.sign(d0) != np.sign(d1)) & (np.abs(denom) > SC_EPS)
        if np.any(crossed):
            tau = np.clip(d0 / np.where(np.abs(denom) > SC_EPS, denom, 1.0), 0.0, 1.0)[:, None]
            pT = p0 + tau * (p1 - p0)
            triT = np.stack([A0 + tau * (A1 - A0),
                             B0 + tau * (B1 - B0),
                             D0 + tau * (D1 - D0)], axis=1)
            checkT, wT, _ = inside_triangles(triT, pT, margin=-SC_SWEPT_MARGIN)
            swept = crossed & checkT
            hit = hit | swept

    if not np.any(hit):
        return _TP_NONE

    w = w1.copy()
    only_swept = swept & ~prox
    if wT is not None and np.any(only_swept):
        w[only_swept] = wT[only_swept]

    plot_end = _bary(tri_co1, w)

    # Which side of the triangle to push the vertex toward.
    #   hist      -- the side it was on at the start of the frame
    #   prox_side -- the side its own 1-ring sits on now (SC_SHEET_SIDE)
    # sign(d1) (current side) is deliberately NOT trusted: once a stack
    # compresses past a triangle height it flips and drives lower layers up.
    hist = np.where(np.abs(d0) > SC_EPS, np.sign(d0), np.sign(d1))
    hist = np.where(hist == 0.0, 1.0, hist)
    if SC_SHEET_SIDE:
        prox_side = _sheet_side(C, cache, vert_idx, plot_end, n1, hist)
    else:
        prox_side = hist
    side = np.where(only_swept, hist, prox_side)          # true tunnels: came-from side
    side = np.where(side == 0.0, 1.0, side)

    target = plot_end + side[:, None] * (thickness * n1)
    move = target - p1

    rows = np.nonzero(hit)[0]
    move = move[rows]
    w_h = w[rows]
    vh = vert_idx[rows]
    th = tri[rows]

    _clamp_rows(move, _max_move(cache, vh, thickness))

    # proximity contacts share the correction 50/50 with the triangle;
    # a detected full tunnel pushes the vertex 100% (leaving it half-through
    # for a frame is worse than a small overshoot).
    bl = np.where(only_swept[rows], 1.0, SC_BLEND)
    v_move = move * bl[:, None]

    if SC_PBD_EXACT:
        dd = np.einsum("ij,ij->i", w_h, w_h) + SC_EPS
        coef = w_h / dd[:, None]
    else:
        coef = w_h
    t_move = ((-move * (1.0 - bl)[:, None])[:, None, :] * coef[:, :, None]).reshape(-1, 3)
    t_hits = th.reshape(-1)

    sw = only_swept[rows]
    return vh, v_move, sw, t_hits, t_move, np.repeat(sw, 3)


def edge_edge_response(C, ea, eb, thickness, cache):
    """Static + swept response for edge-edge. Returns (hits, moves, swept) or (None,)*3."""
    if ea is None or len(ea) == 0:
        return None, None, None

    a0, a1 = ea[:, 0], ea[:, 1]
    b0, b1 = eb[:, 0], eb[:, 1]

    cpa, cpb, s, t = closest_point_segments(C.co[a0], C.co[a1], C.co[b0], C.co[b1])
    sep1 = cpa - cpb
    dist1 = np.sqrt(np.einsum("ij,ij->i", sep1, sep1))
    interior = (s > 0.0) & (s < 1.0) & (t > 0.0) & (t < 1.0)
    prox = interior & (dist1 < thickness)

    hit = prox.copy()
    crossed = np.zeros(len(ea), dtype=bool)
    direction = np.zeros_like(sep1)
    ok = dist1 > SC_EPS
    direction[ok] = sep1[ok] / dist1[ok, None]

    if SC_SWEPT:
        scpa, scpb, _, _ = closest_point_segments(
            C.start_co[a0], C.start_co[a1], C.start_co[b0], C.start_co[b1])
        sep0 = scpa - scpb
        d0n = np.sqrt(np.einsum("ij,ij->i", sep0, sep0))
        crossed = (np.einsum("ij,ij->i", sep0, sep1) < 0.0) & interior
        hit = hit | crossed
        # Lock the push direction to the start-of-frame separation axis -- the
        # stable "which edge is on top" signal, still valid after the sheets
        # have compressed or inverted.  Current-frame axis is used only where
        # the edges started coincident.
        use0 = d0n > SC_EPS
        direction[use0] = sep0[use0] / d0n[use0, None]

    if not np.any(hit):
        return None, None, None

    rows = np.nonzero(hit)[0]
    s_h, t_h = s[rows], t[rows]
    cur = cpa[rows] - cpb[rows]
    corr = direction[rows] * thickness - cur          # relative correction
    half = 0.5 * corr

    if SC_PBD_EXACT:
        da = (1.0 - s_h) ** 2 + s_h ** 2 + SC_EPS
        wa1 = (1.0 - s_h) / da
        wa2 = s_h / da
        db = (1.0 - t_h) ** 2 + t_h ** 2 + SC_EPS
        wb1 = (1.0 - t_h) / db
        wb2 = t_h / db
    else:
        wa1, wa2 = 1.0 - s_h, s_h
        wb1, wb2 = 1.0 - t_h, t_h

    m_a0 = half * wa1[:, None]
    m_a1 = half * wa2[:, None]
    m_b0 = -half * wb1[:, None]
    m_b1 = -half * wb2[:, None]

    hits = np.concatenate([a0[rows], a1[rows], b0[rows], b1[rows]])
    moves = np.concatenate([m_a0, m_a1, m_b0, m_b1])
    _clamp_rows(moves, _max_move(cache, hits, thickness))
    sw = np.tile(crossed[rows], 4)
    return hits, moves, sw


# ===========================================================================
# Accumulation
# ===========================================================================
def _accum_buffers(C, nv):
    fa = getattr(C, "_sc_accum", None)
    if fa is None or fa[0].shape[0] != nv:
        fa = (np.zeros((nv, 3), dtype=np.float64), np.zeros(nv, dtype=np.float64))
        C._sc_accum = fa
    return fa


def _scatter(force_accum, count_accum, idx, move):
    if idx is None or len(idx) == 0:
        return
    np.add.at(force_accum, idx, np.nan_to_num(move))
    np.add.at(count_accum, idx, 1.0)


# ===========================================================================
# Public entry point
# ===========================================================================
def collision_force(C, ob, tridex, tidx, co_start, co_current, radius=0.03):
    """
    Main self-collision entry.

    `radius` (positional) is kept only for call-site compatibility; the contact
    shell is taken from C.ob.MC_props.sc_radius exactly as before.

    Returns (None, (edge_i, edge_j), (tri_idx, vert_idx)) for optional recollide.
    """
    props = C.ob.MC_props
    thickness = float(props.sc_radius)
    detect = thickness * SC_DETECT_SCALE
    relax = SC_RELAX * float(getattr(props, "sc_damping", 1.0))
    do_edges = bool(getattr(props, "sc_edges", True))

    nv = int(C.vc)
    nt = int(C.tridex.shape[0])
    cache = _get_cache(C, detect)

    # ---- swept AABBs (start + current) ----
    C.joined_co[:, 0] = C.start_co
    C.joined_co[:, 1] = C.co
    C.joined_trico[:, :3] = C.start_co[C.tridex]
    C.joined_trico[:, 3:] = C.co[C.tridex]
    cl_min, cl_max = get_poly_bounds(C.joined_co)
    cl_tri_min, cl_tri_max = get_poly_bounds(C.joined_trico)

    root = Node(cl_min, cl_max, cl_tri_min, cl_tri_max, C.idxer, C.tidx)
    root_count = root.cl_idxer.shape[0] * root.cl_tidxer.shape[0]

    # C++ when the scene asks for it and the DLL is loaded -- see collide_native
    native = getattr(C, "collide_native", None)

    cand = CollisionCandidates()
    if native is not None:
        pass
    elif root_count > 0:
        world_min, world_max = U.get_bounds(C.joined_co.reshape(-1, 3))
        # A box stops splitting once it holds root_count / 2**depth pairs.
        # Relative, not absolute: the fastest setting is a fixed number of
        # splits, so an absolute count means a deeper tree on a bigger mesh.
        # See split_tuner.py for the measurements.  MC5 sets sc_split_depth
        # each call (the tuner's pick, or the property).
        depth = int(getattr(C, "sc_split_depth",
                            getattr(props, "sc_box_depth", 5)))
        depth = min(max(depth, 0), SC_MAX_DEPTH)
        t_count = max(1, int(root_count) >> depth)
        split_box(C, root, root_count, world_min, world_max, cand, cache, detect,
                  t_count=t_count, depth=0, max_depth=SC_MAX_DEPTH)

    if native is not None:
        cl_v, cl_t, cl_ea, cl_eb = native.sc_candidates(_THIS, C, cache, detect)
    else:
        cl_v, cl_t, cl_ea, cl_eb = cand.finalize(nv, nt)
    if cl_v is None and cl_ea is None:
        return None, None, None

    if native is not None:
        any_hit, vert_swept, total_step = native.sc_resolve(
            _THIS, C, cache, thickness, detect, relax, do_edges, cl_v, cl_t, cl_ea, cl_eb)
        _vel_accumulate(C, any_hit, vert_swept, total_step)
        return None, (cl_ea, cl_eb), (cl_t, cl_v)

    # ---- relaxation iterations ----
    force_accum, count_accum = _accum_buffers(C, nv)
    any_hit = np.zeros(nv, dtype=bool)
    vert_swept = np.zeros(nv, dtype=bool)      # got a tunnel push-out (keep as velocity)
    total_step = np.zeros((nv, 3), dtype=np.float64)
    step_clamp = _max_move(cache, np.arange(nv), thickness)

    for _ in range(max(1, SC_ITERATIONS)):
        force_accum[:] = 0.0
        count_accum[:] = 0.0

        if cl_v is not None:
            vh, vm, v_sw, th, tm, t_sw = tri_point_response(C, cl_t, cl_v, thickness, cache)
            _scatter(force_accum, count_accum, vh, vm)
            _scatter(force_accum, count_accum, th, tm)
            if vh is not None:
                vert_swept[vh[v_sw]] = True
                vert_swept[th[t_sw]] = True

        if do_edges and cl_ea is not None:
            eh, em, e_sw = edge_edge_response(C, cl_ea, cl_eb, thickness, cache)
            _scatter(force_accum, count_accum, eh, em)
            if eh is not None:
                vert_swept[eh[e_sw]] = True

        hit = count_accum > 0.0
        if not np.any(hit):
            break

        inv = np.zeros(nv)
        inv[hit] = 1.0 / count_accum[hit]
        step = force_accum * (inv[:, None] * relax)
        _clamp_rows(step, step_clamp)

        step = np.nan_to_num(step)
        C.co += step.astype(C.co.dtype)
        total_step += step
        any_hit |= hit

    # Velocity feedback is gathered here and applied once per step by
    # velocity_feedback(), called from MC5.collision_stage after every substep.
    # Applied per call, sc_vel_damping (0.5 by default) compounded once per
    # substep -- 0.5**16 at sixteen substeps, which leaves a self-contacting
    # cloth with essentially no velocity at all.
    _vel_accumulate(C, any_hit, vert_swept, total_step)

    return None, (cl_ea, cl_eb), (cl_t, cl_v)


# ===========================================================================
# Velocity feedback, once per step
# ===========================================================================
def _vel_state(C, nv):
    acc = getattr(C, "_sc_vel", None)
    if acc is None or acc[0].shape[0] != nv:
        acc = (np.zeros(nv, dtype=bool), np.zeros(nv, dtype=bool),
               np.zeros((nv, 3), dtype=np.float64))
        C._sc_vel = acc
    return acc


def _vel_accumulate(C, any_hit, vert_swept, total_step):
    hit, swept, step = _vel_state(C, int(C.vc))
    hit |= any_hit
    swept |= vert_swept
    step += total_step


def begin_velocity(C):
    """Start of a step: forget the contacts gathered for the previous one."""
    hit, swept, step = _vel_state(C, int(C.vc))
    hit[:] = False
    swept[:] = False
    step[:] = 0.0


def velocity_feedback(C):
    """Contact damping and inelastic cancel, once per step.

    Every vertex that touched something this step has its velocity damped by
    sc_vel_damping, once.  MC5 then adds (co - vel_start) to C.velocity, so a
    positional push-out would become bounce velocity: for resting (non-tunnel)
    contacts that push is subtracted back out -> inelastic.  Tunnel recoveries
    keep it so they don't fall straight back through.  The subtraction comes
    after the damping so the cancellation is exact.

    With one substep this is exactly what collision_force used to do inline.
    """
    hit, swept, step = _vel_state(C, int(C.vc))
    if not np.any(hit):
        return
    vel_damp = float(getattr(C.ob.MC_props, "sc_vel_damping", 0.5))
    if vel_damp < 1.0:
        C.velocity[hit] *= vel_damp
    resting = hit & ~swept
    if SC_RESTITUTION < 1.0 and np.any(resting):
        C.velocity[resting] -= ((1.0 - SC_RESTITUTION) * step[resting]).astype(C.velocity.dtype)
    hit[:] = False
    swept[:] = False
    step[:] = 0.0
