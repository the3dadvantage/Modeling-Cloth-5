"""The collision backend switch and the C++ port.

  * mc_collide.dll loads and passes its interface check.
  * The scene setting picks the implementation each step.
  * A missing or out-of-date DLL falls back to Python and says why.
  * On identical inputs C++ and Python agree: contacts to float32 round-off,
    broad-phase pair lists exactly (see tests/shadow_compare.py for all
    scenes).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("TEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


B.load()
MC5 = B.mc5()
NATIVE = MC5.NATIVE
OC = MC5.OC
scene = B.bpy.context.scene

native = NATIVE.get()
check("mc_collide.dll loads and passes the interface check", native is not None,
      "(%s)" % (native.path if native else NATIVE.LAST_ERROR))

scene.MC_props.collision_backend = 'CPP'
check("setting C++ -> the native bridge", MC5.collision_backend() is native)
scene.MC_props.collision_backend = 'PYTHON'
check("setting Python -> no bridge", MC5.collision_backend() is None)

# an out-of-date DLL is refused, and the reason is kept for the panel
saved = (NATIVE.ABI, NATIVE._TRIED, NATIVE._BRIDGE)
NATIVE.ABI = saved[0] + 100
NATIVE._TRIED = False
refused = NATIVE.get()
why = NATIVE.LAST_ERROR or ""
NATIVE.ABI, NATIVE._TRIED, NATIVE._BRIDGE = saved
check("an out-of-date DLL is refused", refused is None and "different version" in why,
      "(%s)" % why[:80])

# a missing DLL falls back
saved_name = NATIVE.DLL_NAME
NATIVE.DLL_NAME = "no_such_mc_collide.dll"
NATIVE._TRIED = False
missing = NATIVE.get()
why = NATIVE.LAST_ERROR or ""
NATIVE.DLL_NAME = saved_name
NATIVE._TRIED, NATIVE._BRIDGE = saved[1], saved[2]
NATIVE.LAST_ERROR = None
check("a missing DLL falls back to Python", missing is None and "not found" in why,
      "(%s)" % why[:80])

# shadow: both implementations on identical inputs, every call
if native is not None:
    orig_nr, orig_bp = OC._narrow_resolve, OC._broad_phase
    worst = {"pos": 0.0, "hits": 0, "pairs": 0, "calls": 0, "broad": 0}

    def nr(data, C, cands, thickness, iters, relax, do_friction=False):
        co_in = C.co.copy()
        C.co = co_in.copy()
        n_hit, _ = native.narrow_resolve(OC._THIS, data, C, cands, thickness, iters, relax,
                                         do_friction)
        co_n = C.co.copy()
        C.co = co_in.copy()
        C.collide_native = None
        p_hit, p_step = orig_nr(data, C, cands, thickness, iters, relax, do_friction)
        worst["pos"] = max(worst["pos"], float(np.abs(C.co - co_n).max()))
        worst["hits"] += int((p_hit != n_hit).sum())
        worst["calls"] += 1
        return p_hit, p_step

    def bp(data, C, *a):
        keep = C.collide_native
        C.collide_native = None
        py = orig_bp(data, C, *a)
        C.collide_native = keep
        cc = native.broad_phase(OC._THIS, data, C, *a)
        worst["broad"] += 1
        for k in ("pt", "tp", "ee"):
            pa = (py or {}).get(k, (None, None))
            ca = cc.get(k, (None, None))
            pa = [np.zeros(0) if x is None else np.asarray(x, dtype=np.int64) for x in pa]
            ca = [np.zeros(0) if x is None else np.asarray(x, dtype=np.int64) for x in ca]
            if not (np.array_equal(pa[0], ca[0]) and np.array_equal(pa[1], ca[1])):
                worst["pairs"] += 1
        return py

    OC._narrow_resolve, OC._broad_phase = nr, bp
    try:
        for name in ("oc_floor", "oc_friction", "oc_small"):
            B.BACKEND = "PYTHON"
            B.run_scene(name)
    finally:
        OC._narrow_resolve, OC._broad_phase = orig_nr, orig_bp
    check("shadow: contacts agree to float32 round-off", worst["pos"] < 1e-5 and not worst["hits"],
          "(%d calls, worst %.2g, hit-set differences %d)" % (worst["calls"], worst["pos"], worst["hits"]))
    check("shadow: broad-phase pair lists identical", worst["pairs"] == 0 and worst["broad"] > 0,
          "(%d broad phases, %d differing)" % (worst["broad"], worst["pairs"]))

    # ---- self collision: candidates identical, resolve to round-off, with
    # and without sew labels (sewn verts count as one point and never repel)
    SC = MC5.SC
    B.BACKEND = "PYTHON"
    B.clear()
    ob, extra = B.SCENES["sc_folded"].build()
    MC5.update_ob_colliders(MC5.OC_DATA)
    C = MC5.get_cloth(ob)
    for fr in range(12):
        B.bpy.context.scene.frame_set(ob.MC_props.reset_frame + 1 + fr)
        MC5.physics(C)

    def compare_sc(label):
        props = C.ob.MC_props
        thickness = float(props.sc_radius)
        detect = thickness * SC.SC_DETECT_SCALE
        relax = SC.SC_RELAX * float(props.sc_damping)
        cache = SC._get_cache(C, detect)
        co_saved, sco_in = C.co.copy(), C.start_co.copy()
        # press the top layer 2.5 cm down into the bottom one, so there is
        # real work for both resolves (and a swept crossing for some verts)
        C.start_co[:] = C.co
        C.co[extra["layer_top"], 2] -= 0.025
        co_in = C.co.copy()
        nat = native.sc_candidates(SC._THIS, C, cache, detect)
        cap = {}
        orig_fin = SC.CollisionCandidates.finalize

        def fin(self, nv, nt):
            cap["r"] = orig_fin(self, nv, nt)
            return cap["r"]

        SC.CollisionCandidates.finalize = fin
        C.collide_native = None
        try:
            SC.collision_force(C, C.ob, C.tridex, C.tidx, C.start_co, C.co)
        finally:
            SC.CollisionCandidates.finalize = orig_fin
        py = cap["r"]
        co_py = C.co.copy()
        C.co = co_in.copy()
        native.sc_resolve(SC._THIS, C, cache, thickness, detect, relax, True, *py)
        diff = float(np.abs(C.co.astype(np.float64) - co_py).max())
        moved = float(np.abs(co_py.astype(np.float64) - co_in).max())
        same = all((a is None and b is None) or
                   (a is not None and b is not None and np.array_equal(np.asarray(a), np.asarray(b)))
                   for a, b in zip(py, nat))
        C.co, C.start_co[:] = co_saved, sco_in
        n = lambda x: 0 if x is None else len(x)
        check("self collision %s: candidates identical" % label, same,
              "(vt %d/%d, ee %d/%d)" % (n(py[0]), n(nat[0]), n(py[2]), n(nat[2])))
        check("self collision %s: resolve agrees to round-off" % label, diff < 1e-5 and moved > 1e-3,
              "(worst %.2g; the resolve moved verts up to %.3g)" % (diff, moved))
        return py

    py_plain = compare_sc("")
    # sew the two layers together along x: vert i on the top layer shares a
    # label with the vert it was folded onto
    co = np.array(C.co)
    labels = np.arange(C.vc, dtype=np.int32)
    order = np.lexsort((co[:, 1], np.round(co[:, 0], 3)))
    labels[order[1::2]] = labels[order[0::2][:len(order[1::2])]]
    C.sew_labels = labels
    py_sewn = compare_sc("with sew labels")
    C.sew_labels = None
    fewer = (0 if py_sewn[0] is None else len(py_sewn[0])) < (0 if py_plain[0] is None else len(py_plain[0]))
    check("sew labels remove candidates between sewn verts", fewer)

    # ---- recollide on a moving collider, whole run, both backends.  The C++
    # bridge packs its arrays once per step and reuses them for every
    # recollide call; if that cache ever served last step's collider the two
    # runs would split apart here.
    out = {}
    for backend in ("PYTHON", "CPP"):
        B.BACKEND = backend
        out[backend] = B.run_scene("oc_moving", {"ob_recollide": True})
    B.BACKEND = "PYTHON"
    d = float(np.abs(out["PYTHON"]["co"] - out["CPP"]["co"]).max())
    check("recollide on a moving collider: C++ run matches Python", d < 1e-4,
          "(max difference %.2g over the run; lifted to %.4f / %.4f)"
          % (d, out["PYTHON"]["final_max_z"], out["CPP"]["final_max_z"]))

    # and the C++ backend really runs a scene, engaged
    B.BACKEND = "CPP"
    r = B.run_scene("oc_friction")
    B.BACKEND = "PYTHON"
    check("C++ backend runs a scene (friction on a slope)", r["engaged"], "(%s)" % r["engaged_msg"])

print("================ %s ================"
      % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
