"""Sewing tools.

Builds sew edges -- edges with no linked faces, which is what
MC5.basic_sew_springs looks for -- between two selected chains of edges.

Seam bookkeeping lives in two INT point attributes rather than vertex groups
(groups are float weights and you'd need one per loop, which gets out of hand
fast):

    MC_seam_id     signed.  +n = loop A of seam n, -n = loop B of seam n, 0 = none
    MC_seam_order  position along the loop

Pairing is then implicit: (+n, order i) is sewn to (-n, order i).  That makes
flip and offset pure renumbering operations instead of re-deriving chains from
topology.

This module is stage 1: chain parsing, validation, attribute stamping, and
matched-vertex-count 1:1 sewing with flip / offset.  Unequal counts (resampling
by arc length) come later.
"""

import bpy
import bmesh
import numpy as np

try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U


SEAM_ID = "MC_seam_id"
SEAM_ORDER = "MC_seam_order"
SEAM_FACE = "MC_seam_face"      # face domain: which seam filled this face


class ChainError(Exception):
    """Selection can't be read as a simple chain."""
    pass


# --------------------------------------------------------------- attributes
def get_layers(bm, create=False):
    """(id_layer, order_layer) or (None, None) when absent and create is False.

    Prefer ensure_layers() for anything that will hold element references --
    see the warning there.
    """
    il = bm.verts.layers.int.get(SEAM_ID)
    ol = bm.verts.layers.int.get(SEAM_ORDER)
    if create:
        return ensure_layers(bm)
    return il, ol


def ensure_layers(bm):
    """Create the seam layers if missing, and return them.

    MUST be called before collecting any BMVert / BMEdge references: adding a
    CustomData layer reallocates element data, which invalidates every Python
    reference taken beforehand (ReferenceError: BMesh data ... has been removed).
    On a mesh that has never been sewn, the layers don't exist yet, so this is
    the very first thing every operator does.
    """
    il = bm.verts.layers.int.get(SEAM_ID)
    ol = bm.verts.layers.int.get(SEAM_ORDER)
    fl = bm.faces.layers.int.get(SEAM_FACE)
    made = False
    if il is None:
        il = bm.verts.layers.int.new(SEAM_ID)
        made = True
    if ol is None:
        ol = bm.verts.layers.int.new(SEAM_ORDER)
        made = True
    if fl is None:
        bm.faces.layers.int.new(SEAM_FACE)
        made = True
    if made:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    return il, ol


def face_layer(bm):
    return bm.faces.layers.int.get(SEAM_FACE)


def next_seam_id(bm):
    il, _ = get_layers(bm)
    if il is None:
        return 1
    hi = 0
    for v in bm.verts:
        a = abs(v[il])
        if a > hi:
            hi = a
    return hi + 1


def seam_ids(bm):
    """Sorted list of seam ids present on the mesh."""
    il, _ = get_layers(bm)
    if il is None:
        return []
    out = set()
    for v in bm.verts:
        a = abs(v[il])
        if a:
            out.add(a)
    return sorted(out)


def seam_verts(bm, sid):
    """(A, B) ordered vert lists for seam `sid`, or (None, None) if incomplete."""
    il, ol = get_layers(bm)
    if il is None or ol is None:
        return None, None
    A, B = {}, {}
    for v in bm.verts:
        s = v[il]
        if s == sid:
            A[v[ol]] = v
        elif s == -sid:
            B[v[ol]] = v
    if not A or not B:
        return None, None
    n = len(A)
    if len(B) != n or set(A) != set(range(n)) or set(B) != set(range(n)):
        return None, None          # gaps / duplicates -- attributes were disturbed
    return [A[i] for i in range(n)], [B[i] for i in range(n)]


def stamp_seam(bm, A, B_paired, sid):
    """Write the attributes.  B_paired[i] is the vert sewn to A[i].

    Assumes ensure_layers() has already run -- creating a layer here would
    invalidate the A / B_paired references passed in.
    """
    il, ol = get_layers(bm)
    if il is None or ol is None:
        raise RuntimeError("seam layers missing; call ensure_layers() first")
    for i, v in enumerate(A):
        v[il] = sid
        v[ol] = i
    for i, v in enumerate(B_paired):
        v[il] = -sid
        v[ol] = i


def clear_seam_attrs(bm, verts):
    il, ol = get_layers(bm)
    if il is None:
        return
    for v in verts:
        v[il] = 0
        v[ol] = 0


# --------------------------------------------------------------- chain parsing
def selected_edges(bm):
    return [e for e in bm.edges if e.select]


def split_components(edges):
    """Split an edge list into connected components (lists of edges)."""
    by_vert = {}
    for e in edges:
        for v in e.verts:
            by_vert.setdefault(v, []).append(e)

    seen = set()
    comps = []
    for e in edges:
        if e in seen:
            continue
        seen.add(e)
        stack = [e]
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for v in cur.verts:
                for e2 in by_vert[v]:
                    if e2 not in seen:
                        seen.add(e2)
                        stack.append(e2)
        comps.append(comp)
    return comps


def order_chain(edges):
    """(ordered_verts, is_closed) for one connected component.

    Raises ChainError on a branch or a shape that isn't a simple chain/loop.
    """
    adj = {}
    for e in edges:
        v0, v1 = e.verts
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    branch = [v for v, nb in adj.items() if len(nb) > 2]
    if branch:
        raise ChainError("branching vertex at index %d (3 or more selected "
                         "edges meet there)" % branch[0].index)

    ends = [v for v, nb in adj.items() if len(nb) == 1]
    if len(ends) == 0:
        closed = True
        start = min(adj.keys(), key=lambda v: v.index)
    elif len(ends) == 2:
        closed = False
        start = min(ends, key=lambda v: v.index)
    else:
        raise ChainError("chain has %d loose ends (expected 0 or 2)" % len(ends))

    verts = [start]
    prev, cur = None, start
    while True:
        nxt = None
        for n in adj[cur]:
            if n is not prev:
                nxt = n
                break
        if nxt is None or nxt is start:
            break
        verts.append(nxt)
        prev, cur = cur, nxt

    if len(verts) != len(adj):
        raise ChainError("could not walk the whole chain (%d of %d verts)"
                         % (len(verts), len(adj)))
    return verts, closed


def parse_selection(bm, boundary_only=False):
    """(A, B, closed) from the current edge selection.

    Raises ChainError with a message meant for a popup.
    """
    sel = selected_edges(bm)
    if not sel:
        raise ChainError("Nothing selected. Select two groups of connected edges.")

    # existing sew edges would confuse the walk -- they are not chain members
    sel = [e for e in sel if len(e.link_faces) > 0]
    if not sel:
        raise ChainError("The selected edges have no faces. Select two groups "
                         "of connected mesh edges.")

    if boundary_only:
        interior = [e for e in sel if len(e.link_faces) > 1]
        if interior:
            raise ChainError("%d selected edge(s) are interior (2 faces). "
                             "Select boundary edges, or turn off Boundary Only."
                             % len(interior))

    comps = split_components(sel)
    if len(comps) != 2:
        raise ChainError("Select exactly two groups of connected edges "
                         "(found %d)." % len(comps))

    A, a_closed = order_chain(comps[0])
    B, b_closed = order_chain(comps[1])

    if a_closed != b_closed:
        raise ChainError("One selection is a closed loop and the other is open. "
                         "They can't be paired without a cut point.")

    shared = set(A) & set(B)
    if shared:
        raise ChainError("The two chains share %d vertex/vertices -- a vertex "
                         "can't be sewn to itself." % len(shared))

    return A, B, a_closed


# --------------------------------------------------------------- alignment
def _co(verts):
    return np.array([tuple(v.co) for v in verts], dtype=np.float64)


def best_align(A, B, closed):
    """(flip, offset) -- anchor on the closest pair, then take the better flip."""
    ca, cb = _co(A), _co(B)
    n = len(A)

    if not closed:
        fwd = np.linalg.norm(ca - cb, axis=1).sum()
        rev = np.linalg.norm(ca - cb[::-1], axis=1).sum()
        return bool(rev < fwd), 0

    # closest pair anchors the rotation
    d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    ia, ib = np.unravel_index(np.argmin(d2), d2.shape)

    best = None
    for flip in (False, True):
        cbf = cb[::-1] if flip else cb
        jb = (n - 1 - ib) if flip else ib
        off = int((jb - ia) % n)
        tot = np.linalg.norm(ca - np.roll(cbf, -off, axis=0), axis=1).sum()
        if best is None or tot < best[2]:
            best = (flip, off, tot)
    return bool(best[0]), int(best[1])


def pair_order(B, flip, offset, closed):
    """B re-ordered so that result[i] is the vert to sew to A[i]."""
    seq = list(reversed(B)) if flip else list(B)
    if not closed:
        return seq
    n = len(seq)
    return [seq[(i + offset) % n] for i in range(n)]


# --------------------------------------------------------------- arc length
def chain_params(verts, closed):
    """(normalized cumulative arc length per vert, total length)."""
    co = _co(verts)
    if closed:
        seg = np.linalg.norm(np.diff(np.vstack([co, co[:1]]), axis=0), axis=1)
    else:
        seg = np.linalg.norm(np.diff(co, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0.0:
        return np.zeros(len(verts)), 0.0
    return (np.concatenate([[0.0], np.cumsum(seg)])[:len(verts)] / total), total


def edge_spans(params, closed):
    """Per chain-edge (start_param, end_param).  n edges closed, n-1 open."""
    if closed:
        return params, np.concatenate([params[1:], [1.0]])
    return params[:-1], params[1:]


def _locate(starts, ends, t):
    """(edge_index, local_fraction) for a normalized param."""
    i = int(np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(starts) - 1))
    span = ends[i] - starts[i]
    return i, (0.0 if span <= 0.0 else float((t - starts[i]) / span))


def sample_chain(verts, closed, params, ts):
    """World positions at the given normalized params."""
    co = _co(verts)
    starts, ends = edge_spans(params, closed)
    n = len(verts)
    out = np.empty((len(ts), 3))
    for k, t in enumerate(ts):
        i, f = _locate(starts, ends, t)
        a = co[i]
        b = co[(i + 1) % n]
        out[k] = a + (b - a) * f
    return out


def anchor_align(A, B, closed):
    """Rotate/flip so index 0 of each list is the anchor pair.

    Returns (A, B, flip).  Works for any vertex counts -- direction is judged by
    comparing the two chains resampled at the same arc-length fractions, so it
    doesn't need a 1:1 correspondence.
    """
    ca, cb = _co(A), _co(B)

    if not closed:
        fwd = np.linalg.norm(ca[0] - cb[0]) + np.linalg.norm(ca[-1] - cb[-1])
        rev = np.linalg.norm(ca[0] - cb[-1]) + np.linalg.norm(ca[-1] - cb[0])
        if rev < fwd:
            return list(A), list(reversed(B)), True
        return list(A), list(B), False

    # closed: anchor on the closest pair, then pick the better winding
    d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    ia, ib = (int(x) for x in np.unravel_index(np.argmin(d2), d2.shape))
    A2 = list(A[ia:]) + list(A[:ia])

    B_f = list(B[ib:]) + list(B[:ib])
    Br = list(reversed(B))
    ibr = len(B) - 1 - ib
    B_r = Br[ibr:] + Br[:ibr]

    ts = np.linspace(0.0, 1.0, 16, endpoint=False)
    pa = sample_chain(A2, True, chain_params(A2, True)[0], ts)

    def score(X):
        px = sample_chain(X, True, chain_params(X, True)[0], ts)
        return float(np.linalg.norm(pa - px, axis=1).sum())

    if score(B_f) <= score(B_r):
        return A2, B_f, False
    return A2, B_r, True


def merge_walk(tA, tB, eps):
    """Interleave two sorted param lists into clusters.

    Each cluster is (a_index or None, b_index or None):
        (i, j)     both loops already have a vert here -- pair them
        (i, None)  only A has one -- B needs a cut at tA[i]
        (None, j)  only B has one -- A needs a cut at tB[j]

    A cluster never holds two params from the same loop, so both loops end up
    with exactly len(clusters) verts and the 1:1 pairing is guaranteed.
    """
    out = []
    i = j = 0
    na, nb = len(tA), len(tB)
    while i < na or j < nb:
        if i >= na:
            out.append((None, j)); j += 1
        elif j >= nb:
            out.append((i, None)); i += 1
        elif abs(float(tA[i]) - float(tB[j])) < eps:
            out.append((i, j)); i += 1; j += 1
        elif tA[i] < tB[j]:
            out.append((i, None)); i += 1
        else:
            out.append((None, j)); j += 1
    return out


def resample_chain(bm, verts, closed, cut_params):
    """Insert verts at the given normalized params, on the existing edges.

    Cuts land on the edge itself, so the boundary polyline, the surface and the
    UVs are unchanged -- only the face becomes an n-gon.  Returns the new
    ordered vertex list.
    """
    if not cut_params:
        return list(verts)

    params, _ = chain_params(verts, closed)
    starts, ends = edge_spans(params, closed)
    n_edges = len(starts)
    n = len(verts)

    chain_edges = [bm.edges.get((verts[i], verts[(i + 1) % n])) for i in range(n_edges)]

    buckets = [[] for _ in range(n_edges)]
    for t in cut_params:
        i, f = _locate(starts, ends, t)
        if chain_edges[i] is None:
            continue
        buckets[i].append(min(max(f, 1e-6), 1.0 - 1e-6))

    new_order = []
    for i in range(n_edges):
        new_order.append(verts[i])
        fr = sorted(buckets[i])
        if not fr:
            continue
        cur_e = chain_edges[i]
        cur_v = verts[i]
        far = verts[(i + 1) % n]
        prev = 0.0
        for f in fr:
            denom = 1.0 - prev
            if denom <= 1e-9 or cur_e is None:
                break
            local = min(max((f - prev) / denom, 1e-6), 1.0 - 1e-6)
            new_e, new_v = bmesh.utils.edge_split(cur_e, cur_v, local)
            new_order.append(new_v)
            # step onto whichever half still reaches the far end
            nxt = None
            for le in new_v.link_edges:
                if far in le.verts:
                    nxt = le
                    break
            cur_e, cur_v, prev = nxt, new_v, f

    if not closed:
        new_order.append(verts[-1])
    return new_order


def plan_resample(A, B, closed, threshold):
    """(cuts_for_A, cuts_for_B, n_pairs) -- normalized params to insert."""
    tA, _ = chain_params(A, closed)
    tB, _ = chain_params(B, closed)
    eps = float(threshold) / max(len(A), len(B), 1)
    clusters = merge_walk(tA, tB, eps)

    cuts_a, cuts_b = [], []
    for ai, bi in clusters:
        if ai is None:
            cuts_a.append(float(tB[bi]))
        elif bi is None:
            cuts_b.append(float(tA[ai]))
    return cuts_a, cuts_b, len(clusters)


def gather_pairs(A, B, closed, threshold):
    """Many-to-one pairing using only existing verts -- no new geometry.

    MC5.basic_sew_springs already unions sew edges that share a vertex into
    N-way merge groups, so a vert sewn to two others just becomes a 3-way group
    and the solver needs no changes.
    """
    tA, _ = chain_params(A, closed)
    tB, _ = chain_params(B, closed)
    eps = float(threshold) / max(len(A), len(B), 1)
    pairs = []
    for ai, bi in merge_walk(tA, tB, eps):
        if ai is None:
            ai = int(np.argmin(np.abs(tA - tB[bi])))
        elif bi is None:
            bi = int(np.argmin(np.abs(tB - tA[ai])))
        pairs.append((A[ai], B[bi]))
    return pairs


def crossing_count(A, Bp):
    """How many consecutive sew pairs run against each other -- a twist tell."""
    ca, cb = _co(A), _co(Bp)
    ta = np.diff(ca, axis=0)
    tb = np.diff(cb, axis=0)
    if not len(ta):
        return 0
    return int(np.count_nonzero(np.einsum('ij,ij->i', ta, tb) < 0.0))


# --------------------------------------------------------------- edge building
def span_edges(bm, A, B, loose_only=False):
    """Edges with one vert in A and the other in B.

    loose_only keeps just the zero-face ones (true sew edges).  Once Face Fill
    has run the seam edges have faces, so the rebuild path needs them all.
    """
    sa, sb = set(A), set(B)
    out = []
    for e in bm.edges:
        if loose_only and len(e.link_faces):
            continue
        v0, v1 = e.verts
        if (v0 in sa and v1 in sb) or (v0 in sb and v1 in sa):
            out.append(e)
    return out


def sew_edges_between(bm, A, B):
    """Existing zero-face edges spanning the two vert sets."""
    return span_edges(bm, A, B, loose_only=True)


# --------------------------------------------------------------- face fill
def _face_walks(e, f, v_from):
    """True if face f traverses edge e starting at v_from."""
    for loop in f.loops:
        if loop.edge is e:
            return loop.vert is v_from
    return None


def fill_faces(bm, A, Bp, closed, sid, select=True):
    """Quad-fill between consecutive sewn pairs.  Returns the new faces.

    Winding is matched to the panel face already sitting on A[i]-A[i+1], so the
    filled strip agrees with the surrounding surface instead of flipping normals
    along the seam.
    """
    fl = face_layer(bm)
    if fl is None:
        return []
    n = len(A)
    span = n if closed else n - 1
    made = []
    for i in range(span):
        j = (i + 1) % n
        ring = [A[i], A[j], Bp[j], Bp[i]]

        # a gathered seam can repeat a vert -- collapse the quad to a triangle
        uniq = []
        for v in ring:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) < 3:
            continue
        if bm.faces.get(uniq) is not None:
            continue

        ea = bm.edges.get((A[i], A[j]))
        if ea is not None and len(ea.link_faces) == 1:
            if _face_walks(ea, ea.link_faces[0], A[i]) is True:
                uniq.reverse()      # panel already walks A[i]->A[j]; go the other way

        try:
            f = bm.faces.new(uniq)
        except ValueError:
            continue
        f[fl] = sid
        if select:
            f.select_set(True)
        made.append(f)

    if made:
        bm.normal_update()
    return made


def clear_fill_faces(bm, sid=None):
    """Delete fill faces (all, or one seam's).  Edges survive, so the seam goes
    back to being zero-face sew edges."""
    fl = face_layer(bm)
    if fl is None:
        return 0
    doomed = [f for f in bm.faces if f[fl] and (sid is None or f[fl] == sid)]
    n = len(doomed)
    for f in doomed:
        try:
            bm.faces.remove(f)
        except Exception:
            pass
    if n:
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
    return n


def seam_face_count(bm, sid):
    fl = face_layer(bm)
    if fl is None:
        return 0
    return sum(1 for f in bm.faces if f[fl] == sid)


def face_fill_enabled(context=None):
    """The Face Fill scene toggle, read defensively (sew_tools does not import
    MC_ui)."""
    ctx = context or bpy.context
    try:
        return bool(ctx.scene.MC_props.sew_face_fill)
    except Exception:
        return False


def apply_face_fill(context, enable):
    """Turn fill faces on/off for the seam the selection is touching.

    Returns (n_changed, message).  With nothing relevant selected this is a
    no-op -- the property is still meaningful as the default for the next sew.
    """
    ob = getattr(context, "object", None)
    if ob is None or ob.type != 'MESH' or ob.mode != 'EDIT':
        return 0, None

    bm = bmesh.from_edit_mesh(ob.data)
    ensure_layers(bm)

    sid = seam_id_from_selection(bm)
    if sid is None:
        return 0, None
    if sid == -1:
        return 0, "Selection touches more than one seam."

    if not enable:
        n = clear_fill_faces(bm, sid)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        _flush(ob)
        return n, None

    A, B = seam_verts(bm, sid)
    if A is None:
        return 0, ("Seam %d was gathered (verts reused), so it has no 1:1 "
                   "pairing to fill. Re-sew it with Matched or Resample." % sid)

    closed = bm.edges.get((A[0], A[-1])) is not None
    made = fill_faces(bm, A, B, closed, sid)
    n = len(made)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    _flush(ob)
    return n, None


def remove_edges(bm, edges):
    for e in edges:
        try:
            bm.edges.remove(e)
        except Exception:
            pass


def make_sew_edges(bm, pairs, select=True):
    """(created, skipped_existing, skipped_degenerate)."""
    created, existing, degenerate = [], 0, 0
    for va, vb in pairs:
        if va is vb:
            degenerate += 1
            continue
        if bm.edges.get((va, vb)) is not None:
            existing += 1
            continue
        e = bm.edges.new((va, vb))
        if select:
            e.select_set(True)
        created.append(e)
    return created, existing, degenerate


def rebuild_seam(bm, sid, flip=False, delta=0, select=True):
    """Re-pair an existing seam. Returns (created, existing, degenerate) or None."""
    A, B = seam_verts(bm, sid)
    if A is None:
        return None

    # a filled seam has faces on its edges, so drop those first -- otherwise the
    # old edges can't be removed and the new pairing lands on top of the old one
    had_faces = clear_fill_faces(bm, sid) > 0
    closed = bm.edges.get((A[0], A[-1])) is not None
    remove_edges(bm, span_edges(bm, A, B))
    bm.edges.ensure_lookup_table()

    n = len(A)
    seq = list(reversed(B)) if flip else list(B)
    Bp = [seq[(i + delta) % n] for i in range(n)]

    stamp_seam(bm, A, Bp, sid)
    res = make_sew_edges(bm, list(zip(A, Bp)), select=select)
    if had_faces:
        fill_faces(bm, A, Bp, closed, sid, select=select)
    return res


def _selected_sid(bm):
    """seam_id_from_selection, but None for the ambiguous case too."""
    sid = seam_id_from_selection(bm)
    return None if sid in (None, -1) else sid


def seam_id_from_selection(bm):
    """Seam id touched by the current selection, or None / -1 for ambiguous."""
    il, _ = get_layers(bm)
    if il is None:
        return None
    ids = set()
    for v in bm.verts:
        if v.select and v[il]:
            ids.add(abs(v[il]))
    if not ids:
        for e in bm.edges:
            if e.select and not len(e.link_faces):
                for v in e.verts:
                    if v[il]:
                        ids.add(abs(v[il]))
    if not ids:
        return None
    if len(ids) > 1:
        return -1
    return ids.pop()


# --------------------------------------------------------------- operators
def _bm(context):
    """(object, live edit bmesh) with the seam layers guaranteed to exist.

    ensure_layers runs here so it happens before any caller collects element
    references -- creating a layer later would invalidate them.
    """
    ob = context.object
    bm = bmesh.from_edit_mesh(ob.data)
    ensure_layers(bm)
    return ob, bm


def _flush(ob, destructive=True):
    """Push the edit bmesh back to the mesh.

    Every BMVert / BMEdge reference held by the caller is potentially dead
    afterwards, so read anything you need BEFORE calling this.
    """
    try:
        bmesh.update_edit_mesh(ob.data, loop_triangles=True,
                               destructive=destructive)
    except TypeError:
        # older / newer signatures without the keywords
        bmesh.update_edit_mesh(ob.data)


class MC_OT_sew_selected(bpy.types.Operator):
    """Create sew edges between two selected groups of connected edges"""
    bl_idname = "mc.sew_selected"
    bl_label = "Sew Selected Edges"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Counts",
        description="What to do when the two chains have different vertex counts",
        items=[
            ('MATCH', "Matched Only",
             "Require equal vertex counts and pair 1:1. Errors on a mismatch"),
            ('RESAMPLE', "Resample",
             "Subdivide both chains at matched arc length so every edge gathers "
             "by the same ratio. Cuts land on existing edges, so the surface and "
             "UVs are unchanged - faces just become n-gons"),
            ('GATHER', "Gather",
             "Pair to the nearest existing vert, many-to-one where counts "
             "differ. Adds no geometry, but gathering is uneven"),
        ],
        default='RESAMPLE')
    threshold: bpy.props.FloatProperty(
        name="Snap Threshold",
        description="How close (as a fraction of average vertex spacing) two "
                    "params must be to count as the same point. Higher snaps "
                    "more and cuts less, at the cost of even gathering",
        min=0.0, max=0.9, default=0.2)
    flip: bpy.props.BoolProperty(
        name="Flip",
        description="Reverse the second chain (fixes a crossed/twisted seam)",
        default=False)
    offset: bpy.props.IntProperty(
        name="Offset",
        description="Rotate the second loop's pairing (closed loops only)",
        default=0)
    auto_align: bpy.props.BoolProperty(
        name="Auto Align",
        description="Anchor on the closest pair of verts and pick the winding "
                    "that gives the shorter seam. Turn off to use Flip/Offset raw",
        default=True)
    boundary_only: bpy.props.BoolProperty(
        name="Boundary Only",
        description="Refuse interior edges (2 linked faces) instead of allowing them",
        default=False)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        try:
            A, B, closed = parse_selection(bm, self.boundary_only)
        except ChainError as e:
            U.popup_error(str(e), icon='ERROR')
            return {'CANCELLED'}

        n_a0, n_b0 = len(A), len(B)
        mismatch = n_a0 != n_b0
        mode = self.mode
        if not mismatch:
            mode = 'MATCH'
        elif mode == 'MATCH':
            U.popup_error("Vertex counts differ (%d and %d). Switch Counts to "
                          "Resample or Gather." % (n_a0, n_b0), icon='ERROR')
            return {'CANCELLED'}

        sid = next_seam_id(bm)
        added_a = added_b = 0
        gathered = False

        if mode == 'GATHER':
            A, B, flip = anchor_align(A, B, closed)
            pairs = gather_pairs(A, B, closed, self.threshold)
            gathered = True
            offset = 0
            n_pairs = len(pairs)
            cross = 0
            # order can't be a bijection when verts are reused, so tag the side
            # only -- flip/offset can't re-pair a gathered seam
            il, ol = get_layers(bm)
            for v in A:
                v[il] = sid; v[ol] = 0
            for v in B:
                v[il] = -sid; v[ol] = 0

        else:
            if mode == 'RESAMPLE':
                A, B, flip = anchor_align(A, B, closed)
                cuts_a, cuts_b, n_pairs = plan_resample(A, B, closed, self.threshold)
                A = resample_chain(bm, A, closed, cuts_a)
                B = resample_chain(bm, B, closed, cuts_b)
                added_a, added_b = len(cuts_a), len(cuts_b)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                offset = 0
                Bp = list(B)
            else:
                if self.auto_align:
                    flip, offset = best_align(A, B, closed)
                    self.flip, self.offset = flip, offset
                else:
                    flip, offset = self.flip, (self.offset if closed else 0)
                Bp = pair_order(B, flip, offset, closed)

            if len(A) != len(Bp):
                U.popup_error("Resample produced %d and %d verts - try a smaller "
                              "Snap Threshold." % (len(A), len(Bp)), icon='ERROR')
                return {'CANCELLED'}

            n_pairs = len(A)
            cross = crossing_count(A, Bp)
            stamp_seam(bm, A, Bp, sid)
            pairs = list(zip(A, Bp))

        # all bmesh reads are done -- make_sew_edges / _flush can invalidate refs
        created, existing, degenerate = make_sew_edges(bm, pairs)
        n_created = len(created)

        n_faces = 0
        if face_fill_enabled(context):
            if gathered:
                U.popup_error("Face Fill needs a 1:1 seam - a gathered seam "
                              "reuses verts. Sew edges made, faces skipped.",
                              icon='INFO')
            else:
                n_faces = len(fill_faces(bm, A, Bp, closed, sid))

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        _flush(ob)

        msg = "Seam %d: %d sew edge(s), %s, %d pairs" % (
            sid, n_created, "closed loop" if closed else "open chain", n_pairs)
        if mismatch:
            msg += "  |  counts %d/%d" % (n_a0, n_b0)
        if added_a or added_b:
            msg += "  |  +%d/+%d verts" % (added_a, added_b)
        if n_faces:
            msg += "  |  %d fill face(s)" % n_faces
        if gathered:
            msg += "  |  gathered (flip/offset unavailable)"
        if existing:
            msg += "  |  %d already existed" % existing
        if degenerate:
            msg += "  |  %d degenerate" % degenerate
        if cross:
            msg += "  |  %d reversed pair(s) - try Flip" % cross
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MC_OT_seam_offset(bpy.types.Operator):
    """Rotate the pairing of the selected seam by one vertex"""
    bl_idname = "mc.seam_offset"
    bl_label = "Seam Offset"
    bl_options = {'REGISTER', 'UNDO'}

    delta: bpy.props.IntProperty(name="Delta", default=1)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        sid = seam_id_from_selection(bm)
        if sid is None:
            U.popup_error("Select part of a seam first (its verts or sew edges).",
                          icon='INFO')
            return {'CANCELLED'}
        if sid == -1:
            U.popup_error("Selection touches more than one seam. Select one seam.",
                          icon='INFO')
            return {'CANCELLED'}

        res = rebuild_seam(bm, sid, flip=False, delta=self.delta)
        if res is None:
            U.popup_error("Seam %d's attributes are incomplete - re-sew it." % sid,
                          icon='ERROR')
            return {'CANCELLED'}
        n_created = len(res[0])          # read before the flush

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        _flush(ob)
        self.report({'INFO'}, "Seam %d offset %+d (%d edges)" % (sid, self.delta, n_created))
        return {'FINISHED'}


class MC_OT_seam_flip(bpy.types.Operator):
    """Reverse the pairing direction of the selected seam"""
    bl_idname = "mc.seam_flip"
    bl_label = "Flip Seam"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        sid = seam_id_from_selection(bm)
        if sid is None:
            U.popup_error("Select part of a seam first (its verts or sew edges).",
                          icon='INFO')
            return {'CANCELLED'}
        if sid == -1:
            U.popup_error("Selection touches more than one seam. Select one seam.",
                          icon='INFO')
            return {'CANCELLED'}

        res = rebuild_seam(bm, sid, flip=True, delta=0)
        if res is None:
            U.popup_error("Seam %d's attributes are incomplete - re-sew it." % sid,
                          icon='ERROR')
            return {'CANCELLED'}
        n_created = len(res[0])          # read before the flush

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        _flush(ob)
        self.report({'INFO'}, "Seam %d flipped (%d edges)" % (sid, n_created))
        return {'FINISHED'}


class MC_OT_seam_face_fill(bpy.types.Operator):
    """Add or remove the fill faces on the selected seam"""
    bl_idname = "mc.seam_face_fill"
    bl_label = "Seam Face Fill"
    bl_options = {'REGISTER', 'UNDO'}

    enable: bpy.props.BoolProperty(name="Enable", default=True)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        n, warn = apply_face_fill(context, self.enable)
        if warn:
            U.popup_error(warn, icon='INFO')
            return {'CANCELLED'}
        if not n:
            U.popup_error("Select part of a seam first (its verts or sew edges).",
                          icon='INFO')
            return {'CANCELLED'}
        self.report({'INFO'}, "%s %d face(s)"
                    % ("Filled" if self.enable else "Removed", n))
        return {'FINISHED'}


class MC_OT_select_sew_edges(bpy.types.Operator):
    """Select every sew edge (edge with no linked faces) on this mesh"""
    bl_idname = "mc.select_sew_edges"
    bl_label = "Select Sew Edges"
    bl_options = {'REGISTER', 'UNDO'}

    extend: bpy.props.BoolProperty(name="Extend", default=False)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        if not self.extend:
            for e in bm.edges:
                e.select_set(False)
        n = 0
        for e in bm.edges:
            if not len(e.link_faces):
                e.select_set(True)
                n += 1
        bm.select_flush(True)
        _flush(ob)
        self.report({'INFO'}, "Selected %d sew edge(s)" % n)
        return {'FINISHED'}


class MC_OT_delete_sew_edges(bpy.types.Operator):
    """Delete sew edges and clear their seam attributes"""
    bl_idname = "mc.delete_sew_edges"
    bl_label = "Delete Sew Edges"
    bl_options = {'REGISTER', 'UNDO'}

    all_seams: bpy.props.BoolProperty(
        name="All",
        description="Delete every sew edge, not just the selected ones",
        default=False)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        # drop fill faces first, otherwise their seam edges aren't loose and
        # wouldn't be picked up here at all
        n_faces = clear_fill_faces(bm, None if self.all_seams else _selected_sid(bm))
        if n_faces:
            bm.edges.ensure_lookup_table()

        doomed = [e for e in bm.edges
                  if not len(e.link_faces) and (self.all_seams or e.select)]
        n_doomed = len(doomed)
        touched = set()
        for e in doomed:
            touched.update(e.verts)
        remove_edges(bm, doomed)
        bm.edges.ensure_lookup_table()

        # only clear attrs on verts that no longer carry any sew edge
        still = set()
        for e in bm.edges:
            if not len(e.link_faces):
                still.update(e.verts)
        clear_seam_attrs(bm, [v for v in touched if v not in still])

        bm.verts.ensure_lookup_table()
        _flush(ob)
        msg = "Deleted %d sew edge(s)" % n_doomed
        if n_faces:
            msg += " and %d fill face(s)" % n_faces
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MC_OT_seam_report(bpy.types.Operator):
    """Print a summary of the seams on this mesh to the console"""
    bl_idname = "mc.seam_report"
    bl_label = "Seam Report"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.mode == 'EDIT'

    def execute(self, context):
        ob, bm = _bm(context)
        ids = seam_ids(bm)
        loose = sum(1 for e in bm.edges if not len(e.link_faces))
        print("# ===== seam report: %s =====" % ob.name)
        print("  sew edges on mesh: %d   seams tagged: %d" % (loose, len(ids)))
        for sid in ids:
            A, B = seam_verts(bm, sid)
            if A is None:
                print("  seam %-3d  ATTRIBUTES INCOMPLETE" % sid)
                continue
            have = len(span_edges(bm, A, B))
            nf = seam_face_count(bm, sid)
            d = _co(A) - _co(B)
            ln = np.linalg.norm(d, axis=1)
            print("  seam %-3d  %d pairs  %d edges  %s  len min %.4f max %.4f "
                  "mean %.4f  reversed %d"
                  % (sid, len(A), have,
                     ("%d fill faces" % nf) if nf else "no fill",
                     ln.min(), ln.max(), ln.mean(), crossing_count(A, B)))
        print("# =====================================")
        self.report({'INFO'}, "%d seam(s), %d sew edge(s) - see console"
                    % (len(ids), loose))
        return {'FINISHED'}


class MC_PT_panel_sewing(bpy.types.Panel):
    """The edit-mode tools that build seams.

    The forces that pull those seams together are a simulation setting, so they
    sit under Forces on the MC5 tab (MC_PT_panel_sew_forces below) rather than
    following the tools onto the tools tab."""
    bl_label = "Sewing"
    bl_idname = "MC_PT_panel_sewing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TOOLS_TAB
    bl_order = 20

    def draw(self, context):
        layout = self.layout
        if U.needs_mesh(layout, context):
            return
        ob = context.object

        # ---------------------------------------------------------- tools
        if ob.mode != 'EDIT':
            box = layout.box()
            box.label(text="Edit Mode for the sewing tools", icon='INFO')
        else:
            box = layout.box()
            box.label(text="Create", icon='MOD_CLOTH')
            box.label(text="Select two edge chains")
            box.operator("mc.sew_selected", text="Sew Selected Edges")
            box.label(text="Mismatched counts: Counts in the redo panel",
                      icon='INFO')

            sc = context.scene
            if hasattr(sc, "MC_props") and hasattr(sc.MC_props, "sew_face_fill"):
                box.prop(sc.MC_props, "sew_face_fill", toggle=True, icon='FACESEL')
            else:
                row = box.row(align=True)
                row.operator("mc.seam_face_fill", text="Fill Faces").enable = True
                row.operator("mc.seam_face_fill", text="Remove Faces").enable = False

            box = layout.box()
            box.label(text="Adjust Seam", icon='ARROW_LEFTRIGHT')
            row = box.row(align=True)
            row.operator("mc.seam_offset", text="Offset -1").delta = -1
            row.operator("mc.seam_offset", text="Offset +1").delta = 1
            box.operator("mc.seam_flip", text="Flip Seam")

            box = layout.box()
            box.label(text="Manage", icon='RESTRICT_SELECT_OFF')
            box.operator("mc.select_sew_edges", text="Select Sew Edges")
            row = box.row(align=True)
            row.operator("mc.delete_sew_edges", text="Delete Selected").all_seams = False
            row.operator("mc.delete_sew_edges", text="Delete All").all_seams = True
            box.operator("mc.seam_report", text="Seam Report", icon='INFO')


class MC_PT_panel_sew_forces(bpy.types.Panel):
    """How hard the seams pull -- a simulation setting, so it sits under Forces
    beside the other ones rather than on the tools tab with the seam tools."""
    bl_label = "Sewing"
    bl_idname = "MC_PT_panel_sew_forces"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_forces"
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        if U.needs_cloth(layout, context):
            return
        p = context.object.MC_props

        box = layout.box()
        box.label(text="Forces", icon='FORCE_FORCE')
        box.prop(p, "sew_force")
        box.prop(p, "butt_sew_force")
        box.prop(p, "target_sew_length")

        box = layout.box()
        box.label(text="Sew Bend")
        box.prop(p, "sew_bend", toggle=True)
        if p.sew_bend:
            box.prop(p, "sew_bend_force")
            box.prop(p, "sew_bend_from_start")
            if not p.sew_bend_from_start:
                box.prop(p, "sew_target_angle")


CLASSES = [
    MC_OT_sew_selected,
    MC_OT_seam_offset,
    MC_OT_seam_flip,
    MC_OT_seam_face_fill,
    MC_OT_select_sew_edges,
    MC_OT_delete_sew_edges,
    MC_OT_seam_report,
    MC_PT_panel_sewing,
    MC_PT_panel_sew_forces,
]
