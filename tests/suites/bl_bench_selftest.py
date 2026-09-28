"""The benchmark must be able to fail.

A harness that always passes is worse than none.  This checks that:
  * a small physics change (the velocity damping on contact) is caught,
  * a run whose collision never engaged is refused as a baseline,
  * a frame-count mismatch is reported instead of compared.
Needs baselines captured first (run_bench.py -- capture).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print("SELFTEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


B.load()
OC = B.mc5().OC

# 1. a physics change is caught
old = OC.OC_VEL_DAMP
OC.OC_VEL_DAMP = old * 0.98
try:
    res = B.run_scene("oc_sphere")
finally:
    OC.OC_VEL_DAMP = old
c = B.compare("oc_sphere", res)
check("2% change to contact damping is caught", not c["ok"] and c["max_dev"] > 0,
      "(max deviation %.3g)" % c.get("max_dev", -1))

# 2. the untouched code still matches (the patch was undone)
c = B.compare("oc_sphere", B.run_scene("oc_sphere"))
check("unpatched run matches again", c["ok"], "(max deviation %.3g)" % c.get("max_dev", -1))

# 3. a run where the collider was switched off is refused as a baseline
orig_build = B.SCENES["oc_floor"].build


def build_no_collider():
    ob, extra = orig_build()
    for o in B.bpy.data.objects:
        if o is not ob:
            o.MC_props.ob_collision = False
    return ob, extra


B.SCENES["oc_floor"].build = build_no_collider
try:
    res = B.run_scene("oc_floor")
finally:
    B.SCENES["oc_floor"].build = orig_build
check("fall-through is flagged not engaged", not res["engaged"], "(%s)" % res["engaged_msg"])
try:
    B.save_baseline("selftest_should_not_exist", res)
    refused = False
except RuntimeError:
    refused = True
check("and refused as a baseline", refused and not B.has_baseline("selftest_should_not_exist"))

# 4. frame mismatch is reported, not compared
res = B.run_scene("oc_floor")
res["frames"] += 1
c = B.compare("oc_floor", res)
check("frame-count mismatch reported", not c["ok"] and "frames" in c.get("why", ""),
      "(%s)" % c.get("why"))

print("SELFTEST %s" % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
