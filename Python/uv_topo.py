"""UV shape topology.

Pure python/numpy -- no bpy -- so it can be tested outside Blender.
uv_shape_tools.py wraps this with the Blender glue.

The question this answers is "will this unwrap cleanly, and if not, where do I
cut".  A surface flattens without overlap exactly when every piece is a
topological disc, which is a computable property rather than a judgement call:

    chi = V - E + F

    chi == 1   a disc -- unwraps cleanly
    chi == 2   a closed shell (sphere-like) -- no boundary, needs a cut
    chi <= 0   handles -- genus (2 - chi - boundary_loops) / 2, each needs a cut

So a closed cube reports chi 2 and "needs 1 cut", a torus reports chi 0 and
"needs 3", and a flat grid reports chi 1 and passes.
"""

import numpy as np
from collections import deque


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def edge_faces(faces):
    """{edge: [face indices]} for a list of vertex-index faces."""
    out = {}
    for fi, f in enumerate(faces):
        n = len(f)
        for i in range(n):
            k = _edge_key(int(f[i]), int(f[(i + 1) % n]))
            out.setdefault(k, []).append(fi)
    return out


def face_components(faces, ef=None):
    """Groups of face indices connected through shared edges."""
    if ef is None:
        ef = edge_faces(faces)
    adj = {}
    for k, fs in ef.items():
        for a in fs:
            for b in fs:
                if a != b:
                    adj.setdefault(a, set()).add(b)

    seen = set()
    comps = []
    for start in range(len(faces)):
        if start in seen:
            continue
        seen.add(start)
        q = deque([start])
        comp = []
        while q:
            f = q.popleft()
            comp.append(f)
            for n in adj.get(f, ()):
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        comps.append(sorted(comp))
    return comps


def boundary_loops_verts(faces, ef=None):
    """Vertex lists, one per separate boundary loop."""
    if ef is None:
        ef = edge_faces(faces)
    adj = {}
    for (a, b), fs in ef.items():
        if len(fs) != 1:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = set()
    loops = []
    for v in sorted(adj):
        if v in seen:
            continue
        comp = []
        seen.add(v)
        q = deque([v])
        while q:
            cur = q.popleft()
            comp.append(cur)
            for n in adj[cur]:
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        loops.append(comp)
    return loops


def boundary_loop_count(faces, ef=None):
    """How many separate boundary loops the face set has."""
    return len(boundary_loops_verts(faces, ef))


def analyze(faces):
    """Per-component topology report.

    Each entry: faces, verts, edges, boundary_edges, boundary_loops, chi,
    genus, closed, is_disc, cuts_needed, note.
    """
    if not len(faces):
        return []
    ef = edge_faces(faces)
    out = []
    for comp in face_components(faces, ef):
        cf = [faces[i] for i in comp]
        cef = edge_faces(cf)
        verts = set()
        for f in cf:
            verts.update(int(v) for v in f)
        V, E, F = len(verts), len(cef), len(cf)
        chi = V - E + F
        b_edges = sum(1 for fs in cef.values() if len(fs) == 1)
        b_loops = boundary_loop_count(cf, cef)
        closed = b_edges == 0
        # chi = 2 - 2g - b  for an orientable surface with b boundary loops
        genus = max(0, int(round((2 - chi - b_loops) / 2)))
        is_disc = (chi == 1 and b_loops == 1)

        if is_disc:
            cuts, note = 0, "disc - unwraps cleanly"
        elif closed and genus == 0:
            cuts, note = 1, "closed shell - needs 1 cut to open it"
        elif genus > 0:
            cuts = 2 * genus + (1 if closed else 0)
            note = "genus %d - needs %d cuts" % (genus, cuts)
        elif b_loops > 1:
            cuts = b_loops - 1
            note = "%d boundary loops - needs %d cut(s) to join them" % (b_loops, cuts)
        else:
            cuts, note = 1, "not a disc (chi %d)" % chi

        out.append({"faces": comp, "n_faces": F, "n_verts": V, "n_edges": E,
                    "boundary_edges": b_edges, "boundary_loops": b_loops,
                    "chi": chi, "genus": genus, "closed": closed,
                    "is_disc": is_disc, "cuts_needed": cuts, "note": note})
    return out


def needs_cutting(faces):
    """(bool, list of component reports that are not discs)."""
    rep = analyze(faces)
    bad = [r for r in rep if not r["is_disc"]]
    return bool(bad), bad


# ------------------------------------------------------------------- cutting
def handle_cuts(faces, ef=None):
    """Edges that must be cut to remove handles, via the classic cut graph.

    Build a spanning tree of the DUAL graph (faces joined across shared edges);
    the edges that tree never crosses form a graph carrying all the topology,
    and pruning its dangling ends leaves exactly the handle loops.  On a
    genus-0 shell this prunes away to nothing, which is correct -- a sphere has
    no handles, it just needs opening (see open_cut).
    """
    if ef is None:
        ef = edge_faces(faces)

    manifold = {k: fs for k, fs in ef.items() if len(fs) == 2}
    adj = {}
    for k, (a, b) in manifold.items():
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))

    crossed = set()
    seen = set()
    for start in range(len(faces)):
        if start in seen:
            continue
        seen.add(start)
        q = deque([start])
        while q:
            f = q.popleft()
            for nf, k in adj.get(f, ()):
                if nf not in seen:
                    seen.add(nf)
                    crossed.add(k)
                    q.append(nf)

    g = {k for k in manifold if k not in crossed}

    # prune dangling ends until only loops remain
    deg = {}
    for a, b in g:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    changed = True
    while changed:
        changed = False
        for k in list(g):
            a, b = k
            if deg.get(a, 0) == 1 or deg.get(b, 0) == 1:
                g.discard(k)
                deg[a] -= 1
                deg[b] -= 1
                changed = True
    return g


def open_cut(faces, ef=None):
    """A path of edges that opens a closed shell into a disc.

    Cutting a closed genus-0 surface along an arc (not a loop) turns it into a
    disc, so this walks the graph diameter -- farthest vertex from an arbitrary
    start, then farthest from that -- and returns the path between them.
    """
    if ef is None:
        ef = edge_faces(faces)
    adj = {}
    for (a, b) in ef:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj:
        return set()

    def bfs(src):
        prev = {src: None}
        q = deque([src])
        last = src
        while q:
            v = q.popleft()
            last = v
            for n in adj[v]:
                if n not in prev:
                    prev[n] = v
                    q.append(n)
        return last, prev

    start = next(iter(adj))
    a, _ = bfs(start)
    b, prev = bfs(a)

    out = set()
    cur = b
    while prev.get(cur) is not None:
        p = prev[cur]
        out.add(_edge_key(cur, p))
        cur = p
    return out


def connect_boundaries(faces, ef=None):
    """Edges joining separate boundary loops, turning an annulus into a disc.

    A cylinder is the case that needs this: it is not closed, so open_cut does
    not apply, and it has no handles, so handle_cuts finds nothing -- but with
    two boundary loops it still is not a disc.  One path between the loops
    fixes it.
    """
    if ef is None:
        ef = edge_faces(faces)
    loops = boundary_loops_verts(faces, ef)
    if len(loops) < 2:
        return set()

    adj = {}
    for (a, b) in ef:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    out = set()
    merged = set(loops[0])
    for nxt in loops[1:]:
        target = set(nxt)
        prev = {v: None for v in merged}
        q = deque(merged)
        hit = None
        while q and hit is None:
            v = q.popleft()
            for n in adj.get(v, ()):
                if n in prev:
                    continue
                prev[n] = v
                if n in target:
                    hit = n
                    break
                q.append(n)
        merged |= target
        if hit is None:
            continue
        cur = hit
        while prev.get(cur) is not None:
            p = prev[cur]
            out.add(_edge_key(cur, p))
            merged.add(cur)
            cur = p
    return out


def auto_cut(faces):
    """Seam edges that make every component of `faces` unwrappable.

    Handle cuts, then an opening arc for anything still closed, then a path
    joining any remaining extra boundary loops.
    """
    ef = edge_faces(faces)
    seams = set(handle_cuts(faces, ef))

    for r in analyze(faces):
        if r["is_disc"]:
            continue
        cf = [faces[i] for i in r["faces"]]
        cef = edge_faces(cf)
        if r["closed"]:
            seams |= open_cut(cf, cef)
        elif r["boundary_loops"] > 1:
            seams |= connect_boundaries(cf, cef)
    return seams


def sharp_edges(faces, normals, angle_deg=40.0, ef=None):
    """Edges whose two faces meet at more than `angle_deg`."""
    if ef is None:
        ef = edge_faces(faces)
    normals = np.asarray(normals, dtype=np.float64)
    limit = np.cos(np.radians(angle_deg))
    out = set()
    for k, fs in ef.items():
        if len(fs) != 2:
            continue
        d = float(np.dot(normals[fs[0]], normals[fs[1]]))
        if d < limit:
            out.add(k)
    return out


def plane_edges(co, faces, axis=0, offset=0.0, tol=1e-4, ef=None):
    """Edges lying on a plane -- a clean symmetry cut for garments."""
    if ef is None:
        ef = edge_faces(faces)
    co = np.asarray(co, dtype=np.float64)
    out = set()
    for (a, b) in ef:
        if abs(co[a][axis] - offset) <= tol and abs(co[b][axis] - offset) <= tol:
            out.add((a, b))
    return out


# --------------------------------------------------------------- flat pattern
def fit_scale(co3, co2, edges, mode='LEAST_SQUARES'):
    """Uniform scale taking the flat layout onto real-world size.

    LEAST_SQUARES minimises squared edge-length error, which is what you want
    when islands stretch unevenly; EDGE_MEAN matches the average edge and AREA
    matches total area.
    """
    co3 = np.asarray(co3, dtype=np.float64)
    co2 = np.asarray(co2, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64)
    if not len(e):
        return 1.0
    l3 = np.linalg.norm(co3[e[:, 1]] - co3[e[:, 0]], axis=1)
    l2 = np.linalg.norm(co2[e[:, 1]] - co2[e[:, 0]], axis=1)
    good = (l2 > 1e-12) & (l3 > 1e-12)
    if not np.any(good):
        return 1.0
    l3, l2 = l3[good], l2[good]

    if mode == 'EDGE_MEAN':
        s = l3.mean() / l2.mean()
    elif mode == 'AREA':
        s = np.sqrt(max((l3 ** 2).sum(), 1e-24) / max((l2 ** 2).sum(), 1e-24))
    else:
        s = (l2 @ l3) / max(l2 @ l2, 1e-24)

    # a collapsed mesh can produce 0 or nan; scaling the flat pattern by that
    # would squash it to a point, so fall back to identity
    s = float(s)
    return s if np.isfinite(s) and s > 1e-12 else 1.0


def stretch_report(co3, co2, edges, scale=1.0):
    """How badly the flattening distorts edge lengths."""
    co3 = np.asarray(co3, dtype=np.float64)
    co2 = np.asarray(co2, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64)
    if not len(e):
        return {"min": 1.0, "max": 1.0, "mean": 1.0, "rms": 0.0}
    l3 = np.linalg.norm(co3[e[:, 1]] - co3[e[:, 0]], axis=1)
    l2 = np.linalg.norm(co2[e[:, 1]] - co2[e[:, 0]], axis=1) * scale
    good = l3 > 1e-12
    if not np.any(good):
        # every 3D edge is zero length -- a collapsed or scale-0 mesh.  There is
        # no stretch to measure, and reducing an empty array raises.
        return {"min": 1.0, "max": 1.0, "mean": 1.0, "rms": 0.0, "edges": 0}
    r = l2[good] / l3[good]
    r = r[np.isfinite(r)]
    if not len(r):
        return {"min": 1.0, "max": 1.0, "mean": 1.0, "rms": 0.0, "edges": 0}
    return {"min": float(r.min()), "max": float(r.max()),
            "mean": float(r.mean()),
            "rms": float(np.sqrt(((r - 1.0) ** 2).mean())),
            "edges": int(len(r))}
