"""Grid fill geometry core.

Pure numpy -- no bpy, no bmesh, no globals -- so it can be tested outside
Blender.  grid_fill.py wraps this with the Blender glue.

The pipeline this supports:

    3D loop -> flatten to 2D -> resample evenly -> lattice / samples
            -> keep what's inside -> stitch to the border -> lift back to 3D

Flattening offers two methods.  'project' drops the loop onto its best-fit
plane, which is exact when the loop is planar.  'unroll' rebuilds the loop in 2D
from its 3D edge lengths and turning angles, so edge lengths survive even when
the loop is badly non-planar -- that matters because the grid gets relaxed back
onto the real 3D border afterwards.
"""

import numpy as np

try:
    from scipy.spatial import Delaunay
    HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    Delaunay = None
    HAVE_SCIPY = False


EPS = 1e-12

# Guards against a too-small spacing.  The point count grows with the SQUARE of
# 1/spacing, so a slider nudge can take a lattice from thousands of points to
# hundreds of millions and lock Blender up.  These raise with the actual numbers
# instead, so the message says what to change.
MAX_LATTICE_POINTS = 500000

# Bridson is a sequential Python loop and measures at roughly 330 samples/sec,
# so this is a time budget (~60s worst case), not a memory one.  The lattice
# modes are vectorised and far cheaper -- prefer them for dense meshes.
POISSON_RATE = 330.0
MAX_POISSON_SAMPLES = 20000


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


# --------------------------------------------------------------------- planes
def best_fit_plane(co):
    """(origin, basis) with basis rows (e1, e2, normal), orthonormal.

    SVD rather than an average of cross products -- the latter is unstable for
    near-degenerate or strongly non-planar loops.
    """
    co = np.asarray(co, dtype=np.float64)
    origin = co.mean(axis=0)
    _, _, vt = np.linalg.svd(co - origin, full_matrices=True)
    basis = vt[:3]
    if np.linalg.det(basis) < 0.0:          # keep it right-handed
        basis = basis.copy()
        basis[2] *= -1.0
    return origin, basis


def planarity(co, origin=None, basis=None):
    """Max distance from the best-fit plane, over the loop's diameter.

    0 is perfectly flat.  Around 0.05 already looks visibly domed.
    """
    co = np.asarray(co, dtype=np.float64)
    if origin is None:
        origin, basis = best_fit_plane(co)
    d = np.abs((co - origin) @ basis[2])
    diam = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0)))
    return float(d.max() / diam) if diam > EPS else 0.0


def to_plane(co, origin, basis):
    return (np.asarray(co, dtype=np.float64) - origin) @ basis[:2].T


def from_plane(co2, origin, basis):
    return origin + np.asarray(co2, dtype=np.float64) @ basis[:2]


# ------------------------------------------------------------------ flattening
def loop_edge_lengths(co, closed=True):
    co = np.asarray(co, dtype=np.float64)
    nxt = np.roll(co, -1, axis=0) if closed else co[1:]
    cur = co if closed else co[:-1]
    return np.linalg.norm(nxt - cur, axis=1)


def turning_angles(co, normal=None):
    """Signed turn at each vertex of a closed 3D polyline.

    Magnitude comes from the 3D angle so curvature is preserved; the sign comes
    from the best-fit plane normal so the winding stays consistent.
    """
    co = np.asarray(co, dtype=np.float64)
    if normal is None:
        _, basis = best_fit_plane(co)
        normal = basis[2]
    e_prev = _unit(co - np.roll(co, 1, axis=0))
    e_next = _unit(np.roll(co, -1, axis=0) - co)
    dot = np.clip(np.einsum('ij,ij->i', e_prev, e_next), -1.0, 1.0)
    ang = np.arccos(dot)
    sgn = np.sign(np.cross(e_prev, e_next) @ normal)
    sgn[sgn == 0.0] = 1.0
    return ang * sgn


def relax_lengths(p, target, iters=60, closed=True):
    """Gauss-Seidel-ish distance constraint pass on a 2D polyline."""
    p = np.array(p, dtype=np.float64, copy=True)
    n = len(p)
    i0 = np.arange(n if closed else n - 1)
    i1 = (i0 + 1) % n
    for _ in range(iters):
        d = p[i1] - p[i0]
        cur = np.linalg.norm(d, axis=1)
        cur = np.maximum(cur, EPS)
        corr = (((cur - target) / cur) * 0.5)[:, None] * d
        acc = np.zeros_like(p)
        np.add.at(acc, i0, corr)
        np.add.at(acc, i1, -corr)
        p += acc * 0.5
    return p


def unroll_loop(co, iters=60):
    """Flatten a closed 3D polyline to 2D, preserving edge lengths.

    Integrates the 3D turning angles (normalised so the total turn is exactly
    2*pi), removes the leftover closure drift, then relaxes the edges back to
    their true 3D lengths while the loop stays closed.
    """
    co = np.asarray(co, dtype=np.float64)
    n = len(co)
    if n < 3:
        return np.zeros((n, 2))

    lengths = loop_edge_lengths(co, closed=True)
    _, basis = best_fit_plane(co)
    theta = turning_angles(co, basis[2])

    total = theta.sum()
    if abs(total) < 1e-9:
        theta = np.full(n, 2.0 * np.pi / n)
    else:
        theta = theta * (2.0 * np.pi / total)

    phi = np.cumsum(theta)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)

    steps = dirs * lengths[:, None]
    p = np.zeros((n, 2))
    p[1:] = np.cumsum(steps[:-1], axis=0)

    # the walk generally doesn't land back on the start -- spread that out
    drift = p[-1] + steps[-1] - p[0]
    p -= np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * drift

    return relax_lengths(p, lengths, iters=iters, closed=True)


def flatten_loop(co, method='auto', planarity_limit=0.02, iters=60):
    """(co2, method_used).  'auto' picks project for flat loops, unroll if not."""
    co = np.asarray(co, dtype=np.float64)
    origin, basis = best_fit_plane(co)
    if method == 'auto':
        method = 'project' if planarity(co, origin, basis) <= planarity_limit else 'unroll'
    if method == 'project':
        return to_plane(co, origin, basis), 'project'
    return unroll_loop(co, iters=iters), 'unroll'


# ------------------------------------------------------------------- polygon
def signed_area(poly):
    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def ensure_ccw(poly):
    return poly if signed_area(poly) >= 0.0 else poly[::-1].copy()


def _seg_cross(p1, p2, q1, q2):
    """Proper-intersection test for two batches of 2D segments."""
    def side(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) -
                       (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    d1 = side(q1, q2, p1)
    d2 = side(q1, q2, p2)
    d3 = side(p1, p2, q1)
    d4 = side(p1, p2, q2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def self_intersections(poly):
    """Indices of edge pairs that properly cross.  Non-adjacent pairs only."""
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    a1 = p
    a2 = np.roll(p, -1, axis=0)
    hits = []
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[:-1]                     # last edge is adjacent to edge 0
        if not len(j):
            continue
        cross = _seg_cross(np.repeat(a1[i][None], len(j), 0),
                           np.repeat(a2[i][None], len(j), 0),
                           a1[j], a2[j])
        for k in j[cross]:
            hits.append((i, int(k)))
    return hits


def winding_number(pts, poly, chunk=4096, boundary_tol=None):
    """Integer winding number of each point about a closed 2D polygon.

    Chosen over even-odd ray casting because it needs no ray direction and so
    can't be tripped by a ray that grazes a vertex -- which is exactly what
    happens constantly when the points form an axis-aligned lattice.

    The winding number is undefined for a point lying ON the polygon, and a
    lattice point landing exactly on a border vertex is common, so points within
    boundary_tol of the outline are detected and reported as INSIDE.  Without
    that, a vertex hit contributes arctan2(0, 0) == 0 instead of its real turn
    and the total rounds to whatever happens to be nearest.
    """
    pts = np.asarray(pts, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    nxt = np.roll(poly, -1, axis=0)
    elen = np.maximum(np.linalg.norm(nxt - poly, axis=1), EPS)
    if boundary_tol is None:
        diag = float(np.linalg.norm(poly.max(axis=0) - poly.min(axis=0)))
        boundary_tol = diag * 1e-9

    out = np.empty(len(pts), dtype=np.int32)
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk][:, None, :]
        a = poly[None, :, :] - q
        b = nxt[None, :, :] - q
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
        wn = np.rint(np.arctan2(cross, dot).sum(axis=1)
                     / (2.0 * np.pi)).astype(np.int32)
        # |cross| / edge_length is the perpendicular distance to the edge line;
        # dot <= 0 means the foot of that perpendicular falls inside the segment
        on = (np.abs(cross) <= boundary_tol * elen[None, :]) & (dot <= boundary_tol ** 2)
        wn[on.any(axis=1)] = 1
        out[s:s + chunk] = wn
    return out


def inside_region(pts, outer, holes=()):
    """Inside the outer loop and outside every hole."""
    keep = winding_number(pts, outer) != 0
    for h in holes:
        keep &= winding_number(pts, h) == 0
    return keep


def distance_to_loop(pts, poly, chunk=4096):
    """Shortest distance from each point to a closed polyline."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    poly = np.asarray(poly, dtype=np.float64)
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    denom = np.maximum(np.einsum('ij,ij->i', ab, ab), EPS)
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        ap = q[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum('pnj,nj->pn', ap, ab) / denom[None, :], 0.0, 1.0)
        near = a[None, :, :] + ab[None, :, :] * t[..., None]
        out[s:s + chunk] = np.linalg.norm(q[:, None, :] - near, axis=2).min(axis=1)
    return out


def project_to_polyline(pts, poly, chunk=4096):
    """(segment index, t along that segment) for each point.

    Picks the segment the point is actually closest to.  Finding the nearest
    VERTEX instead and then projecting onto the segment that follows it is
    wrong for any point past a vertex's midpoint: it lands on the next segment,
    t clamps to 0, and the point collapses onto the vertex.  On a square that
    piles half of every side onto the corners.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    poly = np.asarray(poly, dtype=np.float64)
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    denom = np.maximum(np.einsum('ij,ij->i', ab, ab), EPS)

    idx = np.empty(len(pts), dtype=np.int64)
    tt = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        ap = q[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum('pnj,nj->pn', ap, ab) / denom[None, :], 0.0, 1.0)
        near = a[None, :, :] + ab[None, :, :] * t[..., None]
        d = np.linalg.norm(q[:, None, :] - near, axis=2)
        j = np.argmin(d, axis=1)
        idx[s:s + chunk] = j
        tt[s:s + chunk] = t[np.arange(len(q)), j]
    return idx, tt


def distance_to_border(pts, outer, holes=()):
    """Shortest distance to the outer loop or any hole."""
    d = distance_to_loop(pts, outer)
    for h in holes:
        d = np.minimum(d, distance_to_loop(pts, h))
    return d


# ------------------------------------------------------------------ resampling
def corner_indices(poly, angle_limit_deg=20.0):
    """Vertices whose turn exceeds the limit, i.e. real corners to preserve."""
    p = np.asarray(poly, dtype=np.float64)
    e_prev = _unit(p - np.roll(p, 1, axis=0))
    e_next = _unit(np.roll(p, -1, axis=0) - p)
    dot = np.clip(np.einsum('ij,ij->i', e_prev, e_next), -1.0, 1.0)
    turn = np.degrees(np.arccos(dot))
    idx = np.nonzero(turn > angle_limit_deg)[0]
    return idx if len(idx) else np.array([0], dtype=np.int64)


def _resample_open(pts, count):
    """count points along an open polyline, first included, last excluded."""
    pts = np.asarray(pts, dtype=np.float64)
    if count < 1:
        return pts[:1].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= EPS:
        return np.repeat(pts[:1], count, axis=0)
    t = np.linspace(0.0, total, count, endpoint=False)
    i = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, len(seg) - 1)
    f = (t - cum[i]) / np.maximum(seg[i], EPS)
    return pts[i] + (pts[i + 1] - pts[i]) * f[:, None]


def resample_loop(poly, spacing, angle_limit_deg=20.0):
    """Evenly spaced points around a closed 2D loop, corners kept exactly.

    Each corner-to-corner span gets a whole number of steps as close to
    `spacing` as possible, so spacing stays even and the corners land on
    vertices.  Fully vectorised -- the old walk could index past the end of its
    length array.
    """
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 3 or spacing <= EPS:
        return p.copy()

    corners = corner_indices(p, angle_limit_deg)
    if len(corners) < 2:
        # no real corners: treat the whole loop as one closed span
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        total = seg.sum()
        count = max(int(round(total / spacing)), 3)
        closed = np.vstack([p, p[:1]])
        return _resample_open(closed, count)

    out = []
    for k in range(len(corners)):
        a = corners[k]
        b = corners[(k + 1) % len(corners)]
        span = np.arange(a, b + 1) if b > a else np.concatenate(
            [np.arange(a, n), np.arange(0, b + 1)])
        pts = p[span]
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        count = max(int(round(length / spacing)), 1)
        out.append(_resample_open(pts, count))
    return np.vstack(out)


# -------------------------------------------------------------------- lattice
def lattice(poly2, spacing, rot_deg=0.0, scale=1.0, stagger=0.0, margin=1.0,
            phase=(0.0, 0.0)):
    """(points, shape) for an axis-aligned lattice covering the polygon.

    The polygon is transformed into grid space first and the bounding box taken
    there, so rotation and scale can never leave part of the border uncovered --
    the old code sized the grid to the untransformed bbox and rotated it
    afterwards, which did.

    stagger offsets alternate rows by that fraction of a cell (0.5 gives the
    triangle layout).

    phase slides the whole lattice by that fraction of a cell in each axis, for
    nudging the grid lines until they sit nicely against the shape.  The bounds
    are widened by a full cell so any phase in [-1, 1] still covers the border.
    """
    poly2 = np.asarray(poly2, dtype=np.float64)
    if spacing <= EPS:
        raise ValueError("spacing must be positive")
    scale = max(float(scale), EPS)

    a = np.radians(rot_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])         # grid space -> border space

    q = (poly2 @ R) / scale                 # border into grid space
    ph = np.asarray(phase, dtype=np.float64).reshape(2)
    lo = q.min(axis=0) - spacing * (margin + 1.0) + ph * spacing
    hi = q.max(axis=0) + spacing * (margin + 1.0)

    nx = max(int(np.ceil((hi[0] - lo[0]) / spacing)) + 1, 2)
    ny = max(int(np.ceil((hi[1] - lo[1]) / spacing)) + 1, 2)

    if nx * ny > MAX_LATTICE_POINTS:
        safe = spacing * np.sqrt(float(nx * ny) / MAX_LATTICE_POINTS)
        raise ValueError(
            "Spacing %.6g needs a %d x %d lattice (%s points, limit %s). "
            "Try %.4g or larger." % (spacing, nx, ny, f"{nx * ny:,}",
                                     f"{MAX_LATTICE_POINTS:,}", safe))
    xs = lo[0] + np.arange(nx) * spacing
    ys = lo[1] + np.arange(ny) * spacing

    gx, gy = np.meshgrid(xs, ys)
    if stagger:
        gx = gx.copy()
        gx[1::2] += spacing * stagger

    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return (pts * scale) @ R.T, (ny, nx)


def lattice_quads(shape):
    """Quad indices for a (rows, cols) lattice, CCW."""
    ny, nx = shape
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    a = (j * nx + i).ravel()
    return np.stack([a, a + 1, a + nx + 1, a + nx], axis=1)


def lattice_tris(shape, stagger=True):
    """Triangle indices for a (rows, cols) lattice.

    With a staggered lattice the diagonal alternates per row, which keeps every
    triangle close to the same shape instead of stretching them one way.
    """
    ny, nx = shape
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    a = (j * nx + i).ravel()
    flip = (j.ravel() % 2 == 1) if stagger else np.zeros(len(a), dtype=bool)
    t1 = np.where(flip[:, None],
                  np.stack([a, a + 1, a + nx], axis=1),
                  np.stack([a, a + 1, a + nx + 1], axis=1))
    t2 = np.where(flip[:, None],
                  np.stack([a + 1, a + nx + 1, a + nx], axis=1),
                  np.stack([a, a + nx + 1, a + nx], axis=1))
    return np.vstack([t1, t2])


# ------------------------------------------------------- poisson disk sampling
def poisson_disk(outer, radius, holes=(), seed=0, k=30, border_margin=None):
    """Bridson sampling inside a polygon.

    Every sample is at least `radius` from its neighbours, which makes the
    Delaunay triangles come out at similar areas without the directional
    stiffness a regular lattice gives a cloth sim.

    border_margin keeps samples that far off the border.  Bridson spaces
    samples from each other but nothing stops one landing a hair from the
    outline, and a sample that close makes the unconstrained Delaunay prefer
    crossing the border edge -- the edge gets swallowed and the mesh boundary
    stops matching the loop.  Defaults to 0.7 * radius.
    """
    outer = np.asarray(outer, dtype=np.float64)
    holes = [np.asarray(h, dtype=np.float64) for h in holes]
    if border_margin is None:
        border_margin = 0.7 * radius
    rng = np.random.default_rng(seed)
    lo = outer.min(axis=0)
    hi = outer.max(axis=0)
    if np.any(hi <= lo) or radius <= EPS:
        return np.zeros((0, 2))

    # Poisson packs roughly one sample per 0.75 * radius^2, so check before
    # sampling rather than grinding through millions of rejections
    est = abs(signed_area(outer)) / max(0.75 * radius * radius, EPS)
    if est > MAX_POISSON_SAMPLES:
        safe = radius * np.sqrt(est / MAX_POISSON_SAMPLES)
        raise ValueError(
            "Spacing %.6g would need about %s samples (~%d min at this "
            "sampler's speed, limit %s). Try %.4g or larger, or use a "
            "lattice fill." % (radius, f"{int(est):,}",
                               max(1, int(est / POISSON_RATE / 60)),
                               f"{MAX_POISSON_SAMPLES:,}", safe))

    def allowed(p):
        if not inside_region(p[None], outer, holes)[0]:
            return False
        if border_margin > 0.0:
            return bool(distance_to_border(p[None], outer, holes)[0] >= border_margin)
        return True

    cell = radius / np.sqrt(2.0)
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    grid = -np.ones(tuple(dims), dtype=np.int64)

    samples = []
    active = []

    def fits(p):
        gi = np.clip(((p - lo) / cell).astype(int), 0, dims - 1)
        i0 = np.maximum(gi - 2, 0)
        i1 = np.minimum(gi + 3, dims)
        near = grid[i0[0]:i1[0], i0[1]:i1[1]]
        near = near[near >= 0]
        if not len(near):
            return True, gi
        # gather only the handful of neighbours in range.  Building
        # np.array(samples) here instead makes every candidate test O(N) and
        # the whole sampler O(N^2) -- fine at a few hundred points, minutes at
        # a few tens of thousands.
        nb = np.empty((len(near), 2))
        for k, s_i in enumerate(near):
            nb[k] = samples[int(s_i)]
        return bool(np.linalg.norm(nb - p, axis=1).min() >= radius), gi

    start = None
    for _ in range(400):
        cand = lo + rng.random(2) * (hi - lo)
        if allowed(cand):
            start = cand
            break
    if start is None:
        return np.zeros((0, 2))

    ok, gi = fits(start)
    samples.append(start)
    grid[gi[0], gi[1]] = 0
    active.append(0)

    while active:
        ai = int(rng.integers(len(active)))
        origin = samples[active[ai]]
        placed = False
        for _ in range(k):
            ang = rng.random() * 2.0 * np.pi
            rad = radius * (1.0 + rng.random())
            p = origin + np.array([np.cos(ang), np.sin(ang)]) * rad
            if np.any(p < lo) or np.any(p > hi):
                continue
            if not allowed(p):
                continue
            ok, gi = fits(p)
            if not ok:
                continue
            grid[gi[0], gi[1]] = len(samples)
            samples.append(p)
            active.append(len(samples) - 1)
            placed = True
            break
        if not placed:
            active.pop(ai)

    return np.array(samples)


# ------------------------------------------------------------------ delaunay
def circumradius(tri_co):
    """Circumradius per triangle, (T,3,2) -> (T,).  R = abc / 4A."""
    a = np.linalg.norm(tri_co[:, 1] - tri_co[:, 0], axis=1)
    b = np.linalg.norm(tri_co[:, 2] - tri_co[:, 1], axis=1)
    c = np.linalg.norm(tri_co[:, 0] - tri_co[:, 2], axis=1)
    area2 = np.abs((tri_co[:, 1, 0] - tri_co[:, 0, 0]) * (tri_co[:, 2, 1] - tri_co[:, 0, 1]) -
                   (tri_co[:, 1, 1] - tri_co[:, 0, 1]) * (tri_co[:, 2, 0] - tri_co[:, 0, 0]))
    return (a * b * c) / np.maximum(2.0 * area2, EPS)


def relax_samples(samples, outer, holes=(), iters=5, strength=0.5,
                  qhull_options=None, border_margin=0.0):
    """Laplacian relaxation of interior samples over the Delaunay neighbourhood.

    Evens out the triangle areas.  Any sample that a step would push out of the
    region -- or closer to the border than border_margin -- keeps its old
    position.  The margin matters: the neighbourhood includes border vertices,
    so without it relaxation walks samples steadily toward the outline and
    undoes the spacing that poisson_disk's own margin established.
    """
    if not HAVE_SCIPY or iters < 1 or len(samples) < 3:
        return np.asarray(samples, dtype=np.float64)

    outer = np.asarray(outer, dtype=np.float64)
    fixed = [outer] + [np.asarray(h, dtype=np.float64) for h in holes]
    n_fixed = sum(len(f) for f in fixed)
    p = np.array(samples, dtype=np.float64, copy=True)

    for _ in range(iters):
        pts = np.vstack(fixed + [p])
        try:
            tri = Delaunay(pts, qhull_options=qhull_options)
        except Exception:
            break
        e = np.vstack([tri.simplices[:, [0, 1]],
                       tri.simplices[:, [1, 2]],
                       tri.simplices[:, [2, 0]]])
        e = np.vstack([e, e[:, ::-1]])
        acc = np.zeros_like(pts)
        cnt = np.zeros(len(pts))
        np.add.at(acc, e[:, 0], pts[e[:, 1]])
        np.add.at(cnt, e[:, 0], 1.0)
        mean = acc / np.maximum(cnt, 1.0)[:, None]

        moved = p + (mean[n_fixed:] - p) * strength
        good = inside_region(moved, outer, holes)
        if border_margin > 0.0:
            good &= distance_to_border(moved, outer, holes) >= border_margin
        p[good] = moved[good]

    return p


def longest_edge(tri_co):
    """Longest side per triangle, (T,3,2) -> (T,)."""
    a = np.linalg.norm(tri_co[:, 1] - tri_co[:, 0], axis=1)
    b = np.linalg.norm(tri_co[:, 2] - tri_co[:, 1], axis=1)
    c = np.linalg.norm(tri_co[:, 0] - tri_co[:, 2], axis=1)
    return np.maximum(np.maximum(a, b), c)


def delaunay_fill(outer, holes=(), samples=None, max_edge=0.0, spacing=None,
                  qhull_options=None, min_area=1e-12):
    """Triangulate a region from its border plus interior samples.

    scipy's Delaunay is unconstrained, so the raw result covers the convex hull.
    Two filters carve the region back out:

      * centroid test -- the triangle's centroid must be inside the outer loop
        and outside every hole.  This catches most of it.
      * max_edge test -- optional, OFF by default.  No side longer than
        max_edge * spacing.  The centroid test turned out to carve every shape
        tried here exactly, star tips included, so this is only a guard for
        pathological loops.  It is only meaningful when the interior really is
        sampled at `spacing`: on a sparsely sampled region the legitimate
        triangles are large and this throws them away.

    If max_edge is used, note it measures the longest side, not the
    circumradius: circumradius blows up for any obtuse triangle, so filtering
    on that throws away the thin-but-legitimate slivers along the border.

    Returns (points, tris, info).  points is border-first, so indices
    [0, len(outer)) are the outer loop in order.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for the Delaunay fill")

    outer = np.asarray(outer, dtype=np.float64)
    holes = [np.asarray(h, dtype=np.float64) for h in holes]
    parts = [outer] + holes
    if samples is not None and len(samples):
        parts.append(np.asarray(samples, dtype=np.float64))
    pts = np.vstack(parts)

    tri = Delaunay(pts, qhull_options=qhull_options)
    simp = tri.simplices
    co = pts[simp]

    keep = inside_region(co.mean(axis=1), outer, holes)

    area2 = np.abs((co[:, 1, 0] - co[:, 0, 0]) * (co[:, 2, 1] - co[:, 0, 1]) -
                   (co[:, 1, 1] - co[:, 0, 1]) * (co[:, 2, 0] - co[:, 0, 0]))
    keep &= area2 > min_area

    if spacing is None:
        spacing = float(np.median(loop_edge_lengths(outer)))
    if max_edge and max_edge > 0.0:
        keep &= longest_edge(co) <= max_edge * spacing

    tris = simp[keep]

    # every border edge should end up in exactly one kept triangle; anything
    # else means the fill cut a corner or doubled back
    info = {"n_points": len(pts), "n_tris": int(len(tris)),
            "dropped": int(len(simp) - len(tris)),
            "spacing": spacing}
    info["border_gaps"] = int(_unmatched_border_edges(outer, tris))
    return pts, tris, info


# ------------------------------------------------------------------ assembly
def boundary_loops(faces, min_len=3):
    """Ordered vertex loops around the boundary of a face set.

    An edge used by exactly one face is a boundary edge.  Returns each loop as
    a list of vertex indices in walk order.
    """
    count = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = int(f[i]), int(f[(i + 1) % n])
            k = (a, b) if a < b else (b, a)
            count[k] = count.get(k, 0) + 1

    adj = {}
    for (a, b), c in count.items():
        if c != 1:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops = []
    seen = set()
    for start in sorted(adj):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = None
            for n in adj[cur]:
                if n != prev and n not in seen:
                    nxt = n
                    break
            if nxt is None:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= min_len:
            loops.append(loop)
    return loops


def compact(pts, faces, keep):
    """Drop unkept points and reindex the faces that survive entirely."""
    keep = np.asarray(keep, dtype=bool)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    out = [ [int(remap[v]) for v in f] for f in faces
            if all(keep[int(v)] for v in f) ]
    return pts[keep], out


def lattice_patch(outer, holes=(), spacing=1.0, rot_deg=0.0, scale=1.0,
                  tris=False, inset=0.6, stitch=True, qhull_options=None,
                  phase=(0.0, 0.0)):
    """Lattice interior + triangulated band out to the border.

    Lattice points closer to the border than `inset * spacing` are dropped, so
    the band always has room and the lattice never leaves a sliver pinned
    against the outline.  The band itself is the Delaunay fill of the region
    between the border and the lattice's own boundary loops -- the same tested
    path the Poisson mode uses, rather than a second hand-rolled stitcher.

    Returns (points2, faces, n_border, info).  Points are border-first:
    [0, len(outer)) is the outer loop in order, then the hole loops, then the
    lattice.  faces are index lists of 3 or 4.
    """
    outer = np.asarray(outer, dtype=np.float64)
    holes = [np.asarray(h, dtype=np.float64) for h in holes]

    pts, shape = lattice(outer, spacing, rot_deg=rot_deg, scale=scale,
                         stagger=0.5 if tris else 0.0, phase=phase)
    cells = lattice_tris(shape) if tris else lattice_quads(shape)

    keep = inside_region(pts, outer, holes)
    if inset > 0.0:
        keep &= distance_to_border(pts, outer, holes) >= inset * spacing
    lat_pts, lat_faces = compact(pts, cells, keep)

    info = {"lattice_points": int(len(lat_pts)),
            "lattice_faces": int(len(lat_faces)), "band_tris": 0,
            "border_gaps": 0, "bands": 0}

    n_border = len(outer) + sum(len(h) for h in holes)

    if not stitch or not len(lat_faces):
        # nothing to stitch to -- fill the whole region with triangles
        all_pts, tri, tinfo = delaunay_fill(outer, holes, samples=lat_pts,
                                            spacing=spacing,
                                            qhull_options=qhull_options)
        info["band_tris"] = int(len(tri))
        info["border_gaps"] = tinfo["border_gaps"]
        return all_pts, [list(map(int, t)) for t in tri], n_border, info

    # global index space: outer, then each hole, then the lattice
    g_outer = np.arange(len(outer))
    g_holes = []
    cursor = len(outer)
    for h in holes:
        g_holes.append(np.arange(cursor, cursor + len(h)))
        cursor += len(h)
    g_lat = np.arange(n_border, n_border + len(lat_pts))
    out_pts = np.vstack([outer] + holes + [lat_pts])

    inner_loops = [np.asarray(l, dtype=np.int64) for l in boundary_loops(lat_faces)]

    # A lattice boundary loop that encircles a hole bounds a band on its INSIDE
    # (the gap between the hole and the lattice), so it has to be that band's
    # outer boundary.  Treating it as another hole of `outer` -- which is what
    # "inside outer, outside everything" would do -- drops that band entirely.
    # Every lattice loop that contains the hole is a candidate -- the lattice's
    # OUTER loop contains it too -- so take the tightest one by area.  Picking
    # the first match grabs the outer loop and re-fills the whole lattice
    # region on top of itself.
    loop_area = [abs(signed_area(lat_pts[l])) for l in inner_loops]
    hole_of = {}
    for hi, h in enumerate(holes):
        cands = [li for li in range(len(inner_loops))
                 if li not in hole_of
                 and winding_number(h[:1], lat_pts[inner_loops[li]])[0] != 0]
        if cands:
            hole_of[min(cands, key=lambda x: loop_area[x])] = hi

    bands = []
    free = [li for li in range(len(inner_loops)) if li not in hole_of]
    if free:
        bands.append((g_outer, [g_lat[inner_loops[li]] for li in free]))
    for li, hi in hole_of.items():
        bands.append((g_lat[inner_loops[li]], [g_holes[hi]]))

    faces = [[int(g_lat[v]) for v in f] for f in lat_faces]
    gaps = 0
    n_band = 0
    for a_idx, b_idx_list in bands:
        _, tris, binfo = delaunay_fill(out_pts[a_idx],
                                       holes=[out_pts[b] for b in b_idx_list],
                                       spacing=spacing,
                                       qhull_options=qhull_options)
        gmap = np.concatenate([a_idx] + b_idx_list)
        faces += [[int(gmap[i]) for i in t] for t in tris]
        gaps += binfo["border_gaps"]
        n_band += len(tris)

    info["band_tris"] = int(n_band)
    info["border_gaps"] = int(gaps)
    info["bands"] = len(bands)
    return out_pts, faces, n_border, info


def poisson_patch(outer, holes=(), spacing=1.0, seed=0, relax_iters=0,
                  border_margin=None, max_edge=0.0, qhull_options=None):
    """Isotropic triangle fill: Poisson samples + Delaunay.

    Unlike a lattice this has no preferred direction, so a cloth made from it
    stretches the same way on every axis instead of being stiffer along the
    grid lines.

    relax_iters defaults to 0.  Laplacian relaxation over the Delaunay
    neighbourhood is the obvious thing to reach for, but measured on these
    shapes it does not improve triangle-area uniformity -- Bridson sampling is
    already even (nearest-neighbour spread under 0.1), and averaging positions
    is not the same as equalising areas.  It is kept available; turning it up
    is a visual choice, not a quality win.

    Returns (points2, faces, n_border, info).
    """
    outer = np.asarray(outer, dtype=np.float64)
    holes = [np.asarray(h, dtype=np.float64) for h in holes]
    if border_margin is None:
        border_margin = 0.7 * spacing
    s = poisson_disk(outer, spacing, holes=holes, seed=seed,
                     border_margin=border_margin)
    if relax_iters and len(s):
        s = relax_samples(s, outer, holes, iters=relax_iters,
                          qhull_options=qhull_options,
                          border_margin=border_margin)
    pts, tris, info = delaunay_fill(outer, holes, samples=s, spacing=spacing,
                                    max_edge=max_edge,
                                    qhull_options=qhull_options)
    n_border = len(outer) + sum(len(h) for h in holes)
    info["samples"] = int(len(s))
    return pts, [list(map(int, t)) for t in tris], n_border, info


def procrustes_2d(A, B, with_scale=True):
    """Best rigid (optionally scaled) map taking ordered set A onto B.

    Used to line an unrolled loop up with the same loop projected onto its
    best-fit plane, so interior points computed in unrolled space can be lifted
    to a sensible 3D starting position before the border is pinned and the
    patch is relaxed onto it.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    X, Y = A - ca, B - cb
    u, s, vt = np.linalg.svd(X.T @ Y)
    R = u @ vt
    if np.linalg.det(R) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        R = u @ vt
    scale = (s.sum() / max(float((X ** 2).sum()), EPS)) if with_scale else 1.0

    def apply(P):
        return (np.asarray(P, dtype=np.float64) - ca) @ R * scale + cb
    return apply


def laplacian_smooth(co, faces, locked, iters=10, factor=0.5):
    """Smooth a mesh in place-ish, holding `locked` vertices still.

    Stateless by design: always run from the unsmoothed coordinates, so the
    iteration count can be changed freely without accumulating.
    """
    co = np.array(co, dtype=np.float64, copy=True)
    locked = np.asarray(locked, dtype=bool)
    if iters < 1 or not len(faces):
        return co

    e = []
    for f in faces:
        n = len(f)
        for i in range(n):
            e.append((int(f[i]), int(f[(i + 1) % n])))
    e = np.array(e, dtype=np.int64)
    e = np.vstack([e, e[:, ::-1]])

    free = ~locked
    for _ in range(iters):
        acc = np.zeros_like(co)
        cnt = np.zeros(len(co))
        np.add.at(acc, e[:, 0], co[e[:, 1]])
        np.add.at(cnt, e[:, 0], 1.0)
        mean = acc / np.maximum(cnt, 1.0)[:, None]
        co[free] += (mean[free] - co[free]) * factor
    return co


def _unmatched_border_edges(outer, tris):
    """How many outer-loop edges are not an edge of exactly one kept triangle."""
    n = len(outer)
    want = set()
    for i in range(n):
        a, b = i, (i + 1) % n
        want.add((min(a, b), max(a, b)))
    seen = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if a < n and b < n:
                k = (min(int(a), int(b)), max(int(a), int(b)))
                seen[k] = seen.get(k, 0) + 1
    return sum(1 for k in want if seen.get(k, 0) != 1)
