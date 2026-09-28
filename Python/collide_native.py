"""Bridge to mc_collide.dll, the C++ port of the collision code.

object_collide.py and self_collide_2.py stay the reference and own every
tunable; this module only moves arrays across.  MC5 sets C.collide_native to
the loaded bridge each step when the scene's collision backend is C++, and
the Python modules hand over to it where a native version exists:

    object_collide._broad_phase      -> broad_phase    (pair search)
    object_collide._narrow_resolve   -> narrow_resolve (contacts, combine, friction)
    self_collide_2.collision_force   -> sc_candidates + sc_resolve

Each returns exactly what its Python counterpart returns, so switching back is
a property change.  tests/shadow_compare.py checks them call by call.

Loading
  * The DLL is found the same way as the solver DLL (U.find_dll).
  * A timestamped copy is loaded rather than the file itself, so the DLL can
    be rebuilt (cpp/mc_collide/build.bat) while Blender has it open.
  * The DLL reports its interface version and struct sizes; a mismatch is
    refused, so a stale build can never be called with the wrong layout.
"""
import ctypes
import os
import shutil
import tempfile
import numpy as np
import bpy

try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U

ABI = 4
DLL_NAME = "mc_collide" + U.LIB_SUFFIX

LAST_ERROR = None        # why the DLL isn't loaded, for the settings panel
_BRIDGE = None
_TRIED = False

f64p = ctypes.POINTER(ctypes.c_double)
f32p = ctypes.POINTER(ctypes.c_float)
i32p = ctypes.POINTER(ctypes.c_int32)
u8p = ctypes.POINTER(ctypes.c_uint8)


class OcParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in (
        "thickness", "relax", "push_cap", "swept_margin",
        "clamp_thick", "clamp_edge", "geom_rel_eps", "geom_tiny", "eps",
        "edge_sep_min", "friction_floor", "friction_maxmove",
        "tri_damp", "edge_damp", "mu", "static_thresh", "friction_max")] + \
        [(n, ctypes.c_int32) for n in (
            "iters", "contact_rounds", "do_pt", "do_tp", "do_ee", "do_friction",
            "swept", "pbd_exact")]


class OcMesh(ctypes.Structure):
    _fields_ = [
        ("co", f32p), ("start_co", f32p), ("tridex", i32p), ("cl_eidx", i32p),
        ("char_len", f64p), ("fric_group", f64p),
        ("nv", ctypes.c_int32), ("nt", ctypes.c_int32), ("ne", ctypes.c_int32),
        ("trico1", f64p), ("trico0", f64p), ("tnorm1", f64p), ("tnorm0", f64p),
        ("vnorm", f64p), ("joined", f64p), ("ob_eidx", i32p),
        ("enorm1", f64p), ("enorm0", f64p),
        ("fric_tri", f64p), ("fric_vert", f64p), ("fric_edge", f64p),
        ("to", ctypes.c_int32), ("vo", ctypes.c_int32), ("eo", ctypes.c_int32),
        ("pt_v", i32p), ("pt_t", i32p), ("npt", ctypes.c_int32),
        ("tp_v", i32p), ("tp_t", i32p), ("ntp", ctypes.c_int32),
        ("ee_c", i32p), ("ee_o", i32p), ("nee", ctypes.c_int32),
    ]


class ScParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in (
        "thickness", "detect", "relax", "blend", "swept_margin",
        "eps", "geom_rel_eps", "geom_tiny", "clamp_thick", "clamp_edge")] + \
        [(n, ctypes.c_int32) for n in ("iters", "do_edges", "swept", "pbd_exact", "sheet_side")]


class ScMesh(ctypes.Structure):
    _fields_ = [
        ("co", f32p), ("start_co", f32p), ("tridex", i32p), ("edges", i32p),
        ("reach_keys", ctypes.POINTER(ctypes.c_int64)), ("labels", i32p),
        ("char_len", f64p), ("adj_pad", i32p), ("adj_mask", u8p),
        ("nk", ctypes.c_int64),
        ("nv", ctypes.c_int32), ("nt", ctypes.c_int32), ("ne", ctypes.c_int32),
        ("max_deg", ctypes.c_int32),
    ]


class _Keep:
    """Arrays handed to C must stay alive until the call returns."""
    def __init__(self):
        self.refs = []

    def f64(self, a):
        if a is None:
            return None
        a = np.ascontiguousarray(a, dtype=np.float64)
        self.refs.append(a)
        return a.ctypes.data_as(f64p)

    def i32(self, a):
        if a is None:
            return None
        a = np.ascontiguousarray(a, dtype=np.int32)
        self.refs.append(a)
        return a.ctypes.data_as(i32p)

    def f32_in(self, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        self.refs.append(a)
        return a.ctypes.data_as(f32p)


def _inout_f32(a):
    """C writes into this array in place: it must already be the real thing."""
    if a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
        raise TypeError("in/out array must be C-contiguous float32")
    return a.ctypes.data_as(f32p)


class Bridge:
    def __init__(self, lib, path):
        self.lib = lib
        self.path = path
        lib.oc_narrow_resolve.argtypes = [ctypes.POINTER(OcParams), ctypes.POINTER(OcMesh),
                                          u8p, f64p]
        lib.oc_narrow_resolve.restype = ctypes.c_int32
        lib.mc_box_pairs.argtypes = [f32p, f32p, ctypes.c_int32, f32p, f32p, ctypes.c_int32,
                                     ctypes.c_float]
        lib.mc_box_pairs.restype = ctypes.c_int64
        lib.mc_box_pairs_fetch.argtypes = [i32p, i32p]
        lib.mc_box_pairs_fetch.restype = None
        i64p = ctypes.POINTER(ctypes.c_int64)
        lib.sc_candidates.argtypes = [ctypes.POINTER(ScParams), ctypes.POINTER(ScMesh), i64p, i64p]
        lib.sc_candidates.restype = None
        lib.sc_candidates_fetch.argtypes = [i32p, i32p, i32p, i32p]
        lib.sc_candidates_fetch.restype = None
        lib.sc_resolve.argtypes = [ctypes.POINTER(ScParams), ctypes.POINTER(ScMesh),
                                   i32p, i32p, ctypes.c_int64, i32p, i32p, ctypes.c_int64,
                                   u8p, u8p, f64p]
        lib.sc_resolve.restype = None
        lib.oc_near_pairs.argtypes = [ctypes.POINTER(OcParams), ctypes.POINTER(OcMesh),
                                      ctypes.c_double, u8p, u8p, u8p]
        lib.oc_near_pairs.restype = None

    # ------------------------------------------------------- self collision
    def _sc_structs(self, SC, C, cache, thickness, detect, relax, do_edges, keep):
        P = ScParams(thickness=float(thickness), detect=float(detect), relax=float(relax),
                     blend=SC.SC_BLEND, swept_margin=SC.SC_SWEPT_MARGIN, eps=SC.SC_EPS,
                     geom_rel_eps=SC.SC_GEOM_REL_EPS, geom_tiny=SC.SC_GEOM_TINY,
                     clamp_thick=SC.SC_CLAMP_THICK, clamp_edge=SC.SC_CLAMP_EDGE,
                     iters=int(SC.SC_ITERATIONS), do_edges=int(bool(do_edges)),
                     swept=int(bool(SC.SC_SWEPT)), pbd_exact=int(bool(SC.SC_PBD_EXACT)),
                     sheet_side=int(bool(SC.SC_SHEET_SIDE)))
        edges = cache.get("native_edges")
        if edges is None:
            # the edges that belong to a triangle, in edge-id order -- the
            # ones the Python's per-box edge search can find
            ids = np.unique(C.tri_edge_idxer)
            edges = np.ascontiguousarray(C.tridex_eidx[ids], dtype=np.int32)
            cache["native_edges"] = edges
        M = ScMesh()
        M.co = _inout_f32(C.co)
        M.start_co = keep.f32_in(C.start_co)
        M.tridex = keep.i32(C.tridex)
        M.edges = keep.i32(edges)
        rk = cache.get("reach_keys")
        if rk is not None:
            rk = np.ascontiguousarray(rk, dtype=np.int64)
            keep.refs.append(rk)
            M.reach_keys = rk.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            M.nk = int(rk.size)
        labels = getattr(C, "sew_labels", None)
        M.labels = keep.i32(labels)
        M.char_len = keep.f64(cache.get("char_len"))
        if cache.get("adj_pad") is not None:
            # converted once per mesh, not every call
            if cache.get("native_adj") is None:
                cache["native_adj"] = (
                    np.ascontiguousarray(cache["adj_pad"], dtype=np.int32),
                    np.ascontiguousarray(cache["adj_mask"], dtype=np.uint8))
            ap, am = cache["native_adj"]
            M.adj_pad = ap.ctypes.data_as(i32p)
            M.adj_mask = am.ctypes.data_as(u8p)
            M.max_deg = int(ap.shape[1])
        M.nv, M.nt, M.ne = int(C.vc), int(C.tridex.shape[0]), int(len(edges))
        return P, M, edges

    def sc_candidates(self, SC, C, cache, detect):
        """Native self-collision broad phase and filters.  Returns what
        CollisionCandidates.finalize returns: (verts, tris, edge_a, edge_b),
        each None when empty; edges as endpoint pairs."""
        keep = _Keep()
        P, M, edges = self._sc_structs(SC, C, cache, 0.0, detect, 0.0, True, keep)
        n_vt, n_ee = ctypes.c_int64(), ctypes.c_int64()
        self.lib.sc_candidates(ctypes.byref(P), ctypes.byref(M), ctypes.byref(n_vt), ctypes.byref(n_ee))
        v = np.empty(n_vt.value, dtype=np.int32)
        t = np.empty(n_vt.value, dtype=np.int32)
        a = np.empty(n_ee.value, dtype=np.int32)
        b = np.empty(n_ee.value, dtype=np.int32)
        self.lib.sc_candidates_fetch(v.ctypes.data_as(i32p), t.ctypes.data_as(i32p),
                                     a.ctypes.data_as(i32p), b.ctypes.data_as(i32p))
        vt = (v, t) if len(v) else (None, None)
        ee = (edges[a], edges[b]) if len(a) else (None, None)
        # remember the edge-list positions, so sc_resolve (called next with
        # these very arrays) needn't search for them again
        C._sc_native_ee = (ee[0], a, b)
        return vt[0], vt[1], ee[0], ee[1]

    def sc_resolve(self, SC, C, cache, thickness, detect, relax, do_edges, cl_v, cl_t, cl_ea, cl_eb):
        """Native relaxation loop.  Returns (any_hit, vert_swept, total_step)."""
        keep = _Keep()
        P, M, edges = self._sc_structs(SC, C, cache, thickness, detect, relax, do_edges, keep)
        nv = int(C.vc)
        vp = tp = ap = bp = None
        n_vt = n_ee = 0
        if cl_v is not None:
            vp, tp, n_vt = keep.i32(cl_v), keep.i32(cl_t), len(cl_v)
        known = getattr(C, "_sc_native_ee", None)
        if cl_ea is not None and known is not None and known[0] is cl_ea:
            ap, bp, n_ee = keep.i32(known[1]), keep.i32(known[2]), len(cl_ea)
        elif cl_ea is not None:
            # endpoint pairs from elsewhere (the Python candidate search):
            # back to positions in the native edge list
            key = edges[:, 0].astype(np.int64) * nv + edges[:, 1]
            order = np.argsort(key)
            def pos(e):
                k = e[:, 0].astype(np.int64) * nv + e[:, 1]
                return order[np.searchsorted(key[order], k)]
            ap, bp, n_ee = keep.i32(pos(cl_ea)), keep.i32(pos(cl_eb)), len(cl_ea)
        any_hit = np.zeros(nv, dtype=np.uint8)
        swept = np.zeros(nv, dtype=np.uint8)
        total = np.zeros((nv, 3), dtype=np.float64)
        self.lib.sc_resolve(ctypes.byref(P), ctypes.byref(M), vp, tp, n_vt, ap, bp, n_ee,
                            any_hit.ctypes.data_as(u8p), swept.ctypes.data_as(u8p),
                            total.ctypes.data_as(f64p))
        return any_hit.astype(bool), swept.astype(bool), total

    # ------------------------------------------------------------ broad phase
    def box_pairs(self, amin, amax, bmin, bmax, margin):
        """Every (i, j) with box i of a within `margin` of box j of b, sorted
        -- the same pairs, in the same order, as object_collide's split tree."""
        keep = _Keep()
        pa = (keep.f32_in(amin), keep.f32_in(amax))
        pb = (keep.f32_in(bmin), keep.f32_in(bmax))
        n = self.lib.mc_box_pairs(pa[0], pa[1], int(len(amin)), pb[0], pb[1], int(len(bmin)),
                                  ctypes.c_float(margin))
        a = np.empty(n, dtype=np.int32)
        b = np.empty(n, dtype=np.int32)
        if n:
            self.lib.mc_box_pairs_fetch(a.ctypes.data_as(i32p), b.ctypes.data_as(i32p))
        return a, b

    def broad_phase(self, OC, data, C, cl_joined, cl_trijoined, ob_joined, obt_lo, obt_hi,
                    detect, want):
        """Native _broad_phase.  Same arguments; returns the same candidate
        dict (never None -- no pairs at all comes back as empty lists, which
        every caller treats the same way)."""
        out = {"pt": (None, None), "tp": (None, None), "ee": (None, None)}

        def put(key, pair, ids_a=None, ids_b=None):
            a, b = pair
            if len(a):
                out[key] = (a if ids_a is None else ids_a[a], b if ids_b is None else ids_b[b])

        if want[0]:
            clv_min, clv_max = OC.get_poly_bounds(cl_joined)
            put("pt", self.box_pairs(clv_min, clv_max, obt_lo, obt_hi, detect))
        if want[1]:
            obv_min, obv_max = OC.get_poly_bounds(ob_joined)
            clt_min, clt_max = OC.get_poly_bounds(cl_trijoined)
            put("tp", self.box_pairs(obv_min, obv_max, clt_min, clt_max, detect))
        if want[2]:
            # only edges that belong to a triangle, as the split tree finds them
            cle = np.unique(C.tri_edge_idxer)
            obe = np.unique(data["tri_edge_idxer"])
            a_lo, a_hi = OC._edge_bounds_2frame(cl_joined, C.tri_eidx[cle])
            b_lo, b_hi = OC._edge_bounds_2frame(ob_joined, data["tridex_eidx"][obe])
            put("ee", self.box_pairs(a_lo, a_hi, b_lo, b_hi, detect), cle, obe)
        return out

    # ------------------------------------------------------ object collision
    def _oc_params(self, OC, data, C, thickness, iters, relax, do_friction):
        props = C.ob.MC_props
        return OcParams(
            thickness=float(thickness), relax=float(relax), push_cap=OC.OC_PUSH_CAP,
            swept_margin=OC.OC_SWEPT_MARGIN, clamp_thick=OC.OC_CLAMP_THICK,
            clamp_edge=OC.OC_CLAMP_EDGE, geom_rel_eps=OC.OC_GEOM_REL_EPS,
            geom_tiny=OC.OC_GEOM_TINY, eps=OC.OC_EPS, edge_sep_min=OC.OC_EDGE_SEP_MIN,
            friction_floor=OC.OC_FRICTION_FLOOR, friction_maxmove=OC.OC_FRICTION_MAXMOVE,
            tri_damp=float(props.ob_collision_tri_damping),
            edge_damp=float(props.ob_collision_edge_damping),
            mu=float(getattr(props, "ob_friction", 0.0)),
            static_thresh=float(getattr(props, "ob_static_threshold", 0.0)),
            friction_max=float(data.get("friction_max", 1.0)),
            iters=int(iters), contact_rounds=int(OC.OC_CONTACT_ROUNDS),
            do_pt=int(bool(props.cl_point_tris)), do_tp=int(bool(props.ob_point_tris)),
            do_ee=int(bool(props.ob_edges)), do_friction=int(bool(do_friction)),
            swept=int(bool(OC.OC_SWEPT)), pbd_exact=int(bool(OC.OC_PBD_EXACT)))

    def _oc_mesh(self, OC, data, C, pairs):
        """The OcMesh for these candidate pairs, reusing the last one built
        when nothing it points at can have changed.

        Recollide calls this 20+ times a step with the same arrays; packing
        them (float64 copies of the collider geometry) every time cost more
        than a third of each call.  The key is a per-step token (MC5 bumps
        C._native_step in prepare_recollide) plus the identity of every array
        read.  The collider arrays are REPLACED, not edited, whenever the
        colliders are refreshed, so an unchanged identity within a step means
        unchanged contents; the cache holds the originals so an id can never
        be recycled into a false match.
        """
        pt, tp, ee = pairs
        sources = (C.co, C.start_co, C.local_trico, C.start_local_trico, C.local_normals,
                   C.start_local_normals, C.local_vert_norms, data["joined_co"],
                   data["local_edge_normals"], data["start_local_edge_normals"],
                   pt[0], pt[1], tp[0], tp[1], ee[0], ee[1])
        key = (getattr(C, "_native_step", None),) + tuple(id(s) for s in sources)
        hit = getattr(C, "_native_mesh", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        keep = _Keep()
        cache = OC._get_cache(C)
        cfg = getattr(C, "friction_group", None)
        M = OcMesh()
        M.co = _inout_f32(C.co)
        M.start_co = keep.f32_in(C.start_co)
        M.tridex = keep.i32(C.tridex)
        M.cl_eidx = keep.i32(C.tri_eidx)
        M.char_len = keep.f64(cache.get("char_len"))
        M.fric_group = keep.f64(None if cfg is None else np.asarray(cfg).reshape(-1))
        M.nv, M.nt, M.ne = int(C.vc), int(C.tridex.shape[0]), int(len(C.tri_eidx))
        M.trico1 = keep.f64(C.local_trico)
        M.trico0 = keep.f64(C.start_local_trico)
        M.tnorm1 = keep.f64(C.local_normals)
        M.tnorm0 = keep.f64(C.start_local_normals)
        M.vnorm = keep.f64(C.local_vert_norms)
        M.joined = keep.f64(data["joined_co"])
        M.ob_eidx = keep.i32(data["tridex_eidx"])
        M.enorm1 = keep.f64(data["local_edge_normals"])
        M.enorm0 = keep.f64(data["start_local_edge_normals"])
        M.fric_tri = keep.f64(data.get("friction_tri"))
        M.fric_vert = keep.f64(data.get("friction_vert"))
        M.fric_edge = keep.f64(data.get("friction_edge"))
        M.to = int(len(C.local_trico))
        M.vo = int(len(data["joined_co"]))
        M.eo = int(len(data["tridex_eidx"]))
        for (a, b), (fa, fb, fn) in ((pt, ("pt_v", "pt_t", "npt")),
                                     (tp, ("tp_v", "tp_t", "ntp")),
                                     (ee, ("ee_c", "ee_o", "nee"))):
            if a is None or len(a) == 0:
                setattr(M, fn, 0)
                continue
            setattr(M, fa, keep.i32(a))
            setattr(M, fb, keep.i32(b))
            setattr(M, fn, int(len(a)))
        M._keep = keep                     # the arrays live as long as M
        M._sources = sources               # and ids can't be recycled
        C._native_mesh = (key, M)
        return M

    def narrow_resolve(self, OC, data, C, cands, thickness, iters, relax, do_friction):
        """Native _narrow_resolve.  Same arguments, same return value."""
        nv = int(C.vc)
        P = self._oc_params(OC, data, C, thickness, iters, relax, do_friction)
        M = self._oc_mesh(OC, data, C, (cands.get("pt", (None, None)),
                                        cands.get("tp", (None, None)),
                                        cands.get("ee", (None, None))))
        any_hit = np.zeros(nv, dtype=np.uint8)
        total_step = np.zeros((nv, 3), dtype=np.float64)
        n_fric = self.lib.oc_narrow_resolve(ctypes.byref(P), ctypes.byref(M),
                                            any_hit.ctypes.data_as(u8p),
                                            total_step.ctypes.data_as(f64p))
        if n_fric:
            OC._friction_applied(int(n_fric))
        return any_hit.astype(bool), total_step

    def near_pairs(self, OC, data, C, point_tri, edge_edge, tri_point):
        """Native object_collide.near_pairs.  Same arguments, same result."""
        none = (None, None)
        pt, ee, tp = point_tri or none, edge_edge or none, tri_point or none
        thickness = float(C.ob.MC_props.ob_collision_radius)
        P = self._oc_params(OC, data, C, thickness, 1, 1.0, False)
        # packed fresh: the recollide calls after this pass other pairs
        C._native_mesh = None
        M = self._oc_mesh(OC, data, C, (pt, tp, ee))
        C._native_mesh = None
        n = [0 if x[0] is None else len(x[0]) for x in (pt, tp, ee)]
        k_pt, k_tp, k_ee = (np.zeros(max(1, m), dtype=np.uint8) for m in n)
        self.lib.oc_near_pairs(ctypes.byref(P), ctypes.byref(M),
                               ctypes.c_double(thickness * (1.0 + OC.OC_RECOLLIDE_BAND)),
                               k_pt.ctypes.data_as(u8p), k_tp.ctypes.data_as(u8p),
                               k_ee.ctypes.data_as(u8p))

        def kept(pair, k, m):
            if not m:
                return None
            k = k[:m].astype(bool)
            return (pair[0][k], pair[1][k]) if k.any() else None

        return kept(pt, k_pt, n[0]), kept(ee, k_ee, n[2]), kept(tp, k_tp, n[1])


def _load():
    global LAST_ERROR
    path = U.find_dll(names=(DLL_NAME,), required=False)
    if path is None:
        LAST_ERROR = "%s not found (build it with cpp/mc_collide/build.bat)" % DLL_NAME
        return None
    try:
        # load a copy so the real file can be rebuilt while Blender runs
        stamp = int(os.path.getmtime(path))
        copy = os.path.join(tempfile.gettempdir(),
                            "mc_collide_%d%s" % (stamp, U.LIB_SUFFIX))
        if not os.path.isfile(copy):
            shutil.copy2(path, copy)
        lib = ctypes.CDLL(copy)
        lib.mc_collide_abi.restype = ctypes.c_int32
        lib.mc_collide_sizeof_params.restype = ctypes.c_int32
        lib.mc_collide_sizeof_mesh.restype = ctypes.c_int32
        lib.mc_collide_sizeof_sc_params.restype = ctypes.c_int32
        lib.mc_collide_sizeof_sc_mesh.restype = ctypes.c_int32
        abi = lib.mc_collide_abi()
        sp, sm = lib.mc_collide_sizeof_params(), lib.mc_collide_sizeof_mesh()
        ssp, ssm = lib.mc_collide_sizeof_sc_params(), lib.mc_collide_sizeof_sc_mesh()
    except Exception as e:                                  # noqa: BLE001
        LAST_ERROR = "could not load %s: %s" % (path, e)
        return None
    if (abi != ABI or sp != ctypes.sizeof(OcParams) or sm != ctypes.sizeof(OcMesh)
            or ssp != ctypes.sizeof(ScParams) or ssm != ctypes.sizeof(ScMesh)):
        LAST_ERROR = ("%s is a different version (interface %d, want %d; struct sizes %d/%d, "
                      "want %d/%d) -- rebuild it" % (path, abi, ABI, sp, sm,
                                                    ctypes.sizeof(OcParams), ctypes.sizeof(OcMesh)))
        return None
    LAST_ERROR = None
    return Bridge(lib, path)


def get():
    """The loaded bridge, or None (see LAST_ERROR).  Loads once per session."""
    global _BRIDGE, _TRIED
    if not _TRIED:
        _TRIED = True
        _BRIDGE = _load()
        if _BRIDGE is None:
            print("MC: C++ collision unavailable, using Python -- %s" % LAST_ERROR)
    return _BRIDGE
