"""Shadow comparison: Python and C++ on identical inputs, call by call.

Runs a scene with the Python collision code.  Every time _narrow_resolve is
called, the C++ version runs on an identical copy of the inputs and the two
outputs are compared.  The run always continues with the Python result, so
the two never drift apart: a difference here is a difference in the code, not
round-off amplified by a chaotic scene.

    blender -b --factory-startup --python tests/shadow_compare.py -- [scenes]

Positions are float32, so a few ulp of difference (~1e-7 relative) is
expected.  Anything larger names the call, the vertex, and what each side
thought.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collide_bench as B    # noqa: E402

bpy = B.bpy
TOL = 1e-5          # a real mismatch, not float32 round-off


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    names = args or list(B.SCENES)
    B.load()
    MC5 = B.mc5()
    OC = MC5.OC
    native = MC5.NATIVE.get()
    if native is None:
        print("SHADOW C++ unavailable: %s" % MC5.NATIVE.LAST_ERROR, flush=True)
        return 1
    orig = OC._narrow_resolve
    stats = {}

    def shadow(data, C, cands, thickness, iters, relax, do_friction=False):
        co_in = C.co.copy()
        # C++ on a copy of the inputs
        C.co = co_in.copy()
        n_hit, n_step = native.narrow_resolve(OC._THIS, data, C, cands, thickness,
                                              iters, relax, do_friction)
        co_native = C.co.copy()
        # Python on the real thing (the run continues with this)
        C.co = co_in.copy()
        C.collide_native = None
        p_hit, p_step = orig(data, C, cands, thickness, iters, relax, do_friction)
        d = np.linalg.norm(C.co.astype(np.float64) - co_native, axis=1)
        s = stats.setdefault("calls", [0, 0.0, None])
        s[0] += 1
        hit_diff = int((p_hit != n_hit).sum())
        if d.max() > s[1]:
            s[1] = float(d.max())
            v = int(np.argmax(d))
            s[2] = (s[0], v, C.co[v].copy(), co_native[v], hit_diff,
                    float(np.abs(p_step - n_step).max()))
        if hit_diff:
            stats.setdefault("hit_mismatch", []).append((s[0], hit_diff))
        return p_hit, p_step

    orig_broad = OC._broad_phase

    def as_lists(c):
        if c is None:
            return {k: (np.zeros(0, int), np.zeros(0, int)) for k in ("pt", "tp", "ee")}
        return {k: tuple(np.zeros(0, int) if x is None else np.asarray(x, dtype=np.int64)
                         for x in c.get(k, (None, None))) for k in ("pt", "tp", "ee")}

    def shadow_broad(data, C, cl_joined, cl_trijoined, ob_joined, obt_lo, obt_hi, detect, want):
        keep = C.collide_native
        C.collide_native = None
        py = orig_broad(data, C, cl_joined, cl_trijoined, ob_joined, obt_lo, obt_hi, detect, want)
        C.collide_native = keep
        cc = native.broad_phase(OC._THIS, data, C, cl_joined, cl_trijoined, ob_joined,
                                obt_lo, obt_hi, detect, want)
        P, Q = as_lists(py), as_lists(cc)
        b = stats.setdefault("broad", [0, 0, ""])
        b[0] += 1
        for k in ("pt", "tp", "ee"):
            same = (len(P[k][0]) == len(Q[k][0]) and np.array_equal(P[k][0], Q[k][0])
                    and np.array_equal(P[k][1], Q[k][1]))
            if not same:
                b[1] += 1
                if not b[2]:
                    b[2] = "%s: python %d pairs, c++ %d" % (k, len(P[k][0]), len(Q[k][0]))
        return py

    SC = MC5.SC
    orig_sc = SC.collision_force

    def same_pairs(p, q):
        if p is None or q is None:
            return (p is None or len(p) == 0) and (q is None or len(q) == 0)
        return np.array_equal(np.asarray(p, dtype=np.int64), np.asarray(q, dtype=np.int64))

    def shadow_sc(C, ob, tridex, tidx, co_start, co_current, radius=0.03):
        props = C.ob.MC_props
        thickness = float(props.sc_radius)
        detect = thickness * SC.SC_DETECT_SCALE
        relax = SC.SC_RELAX * float(getattr(props, "sc_damping", 1.0))
        do_edges = bool(getattr(props, "sc_edges", True))
        cache = SC._get_cache(C, detect)
        co_in = C.co.copy()
        nat = native.sc_candidates(SC._THIS, C, cache, detect)
        cap = {}
        orig_fin = SC.CollisionCandidates.finalize

        def fin(self, nv, nt):
            r = orig_fin(self, nv, nt)
            cap["r"] = r
            return r

        SC.CollisionCandidates.finalize = fin
        keep = C.collide_native
        C.collide_native = None
        try:
            out = orig_sc(C, ob, tridex, tidx, co_start, co_current, radius)
        finally:
            SC.CollisionCandidates.finalize = orig_fin
            C.collide_native = keep
        s = stats.setdefault("sc", [0, 0, 0.0, ""])
        s[0] += 1
        py = cap.get("r")
        if py is not None:
            same = all(same_pairs(a, b) for a, b in zip(py, nat))
            if not same:
                s[1] += 1
                if not s[3]:
                    s[3] = "vt %s/%s, ee %s/%s" % tuple(
                        "-" if x is None else len(x) for x in (py[0], nat[0], py[2], nat[2]))
            if py[0] is not None or py[2] is not None:
                co_py = C.co.copy()
                C.co = co_in.copy()
                native.sc_resolve(SC._THIS, C, cache, thickness, detect, relax, do_edges, *py)
                s[2] = max(s[2], float(np.abs(C.co.astype(np.float64) - co_py).max()))
                C.co = co_py
        return out

    orig_near = OC.near_pairs

    def shadow_near(data, C, pt, ee, tp):
        keep = C.collide_native
        C.collide_native = None
        py = orig_near(data, C, pt, ee, tp)
        C.collide_native = keep
        cc = native.near_pairs(OC._THIS, data, C, pt, ee, tp)
        n = stats.setdefault("near", [0, 0, ""])
        n[0] += 1
        for kind, a, b in zip(("pt", "ee", "tp"), py, cc):
            if a is None or b is None:
                same = a is None and b is None
            else:
                same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            if not same:
                n[1] += 1
                if not n[2]:
                    n[2] = "%s: python %s, c++ %s" % (kind, "-" if a is None else len(a[0]),
                                                      "-" if b is None else len(b[0]))
        return py

    OC._narrow_resolve = shadow
    OC._broad_phase = shadow_broad
    SC.collision_force = shadow_sc
    OC.near_pairs = shadow_near
    bad = 0
    try:
        for name in names:
            stats.clear()
            B.BACKEND = "PYTHON"
            B.run_scene(name)
            n, worst, info = stats.get("calls", [0, 0.0, None])
            hm = stats.get("hit_mismatch", [])
            nb, bdiff, bwhy = stats.get("broad", [0, 0, ""])
            ns, sdiff, sworst, swhy = stats.get("sc", [0, 0, 0.0, ""])
            nn, ndiff, nwhy = stats.get("near", [0, 0, ""])
            ok = worst <= TOL and not hm and not bdiff and not sdiff and sworst <= TOL and not ndiff
            bad += not ok
            print("SHADOW %-4s %-20s %4d resolves, worst position difference %.3g%s | "
                  "%d broad phases, %s"
                  % ("ok" if ok else "DIFF", name, n, worst,
                     ("  (%d calls with different hit sets)" % len(hm)) if hm else "",
                     nb, "pair lists identical" if not bdiff else
                     "%d differ (first: %s)" % (bdiff, bwhy)),
                  flush=True)
            if ns:
                print("                                self collision: %d calls, candidates %s, "
                      "worst position difference %.3g"
                      % (ns, "identical" if not sdiff else "differ in %d (first: %s)" % (sdiff, swhy),
                         sworst), flush=True)
            if nn:
                print("                                recollide pair filter: %d calls, %s"
                      % (nn, "kept pairs identical" if not ndiff else
                         "%d differ (first: %s)" % (ndiff, nwhy)), flush=True)
            if info and not ok:
                call, v, py, cc, hd, sd = info
                print("         at call %d, vertex %d: python %s  c++ %s  (hit sets differ in %d, "
                      "total_step differs by %.3g)" % (call, v, np.round(py, 6), np.round(cc, 6), hd, sd),
                      flush=True)
    finally:
        OC._narrow_resolve = orig
        OC._broad_phase = orig_broad
        SC.collision_force = orig_sc
        OC.near_pairs = orig_near
    print("SHADOW %s" % ("ALL MATCH" if not bad else "%d scene(s) differ" % bad), flush=True)
    return 1 if bad else 0


code = 1
try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
sys.stdout.flush()
os._exit(code)
