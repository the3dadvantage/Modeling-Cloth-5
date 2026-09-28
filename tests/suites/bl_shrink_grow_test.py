"""Shrink Grow: the multiplier on the linear springs' rest lengths.

Shrink Grow scales the target distance of every linear spring, so the cloth
settles at `shrink_grow` times its target edge lengths: 0.5 gathers it to half,
2.0 lets it out to double.  It has a floor of zero (a negative rest length
would turn the springs inside out) and no ceiling.

Both solvers have to agree, because which one runs is not the user's choice.

Pre-shrink was an experiment -- a ramp that started the springs short and let
them back out over the solver iterations -- and is gone.  The checks at the end
make sure it left nothing behind.
"""
import bpy, os, sys, glob
import numpy as np

sys.excepthook = lambda *e: (sys.__excepthook__(*e), sys.stdout.flush(), os._exit(1))

# the addon's modules, found relative to this file so the tests run
# wherever the repository is checked out (MC_SRC overrides)
SRC = os.environ.get("MC_SRC") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Python")
# the modules are loaded as text datablocks with no path of their own, so say
# where the per-platform libraries are (cpp/build_msvc.bat, cpp/build_unix.sh)
os.environ.setdefault("MC_LIB_DIR",
                      os.path.join(os.path.dirname(SRC), "addon", "lib"))
for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
    n = os.path.basename(f)
    if n not in bpy.data.texts:
        bpy.data.texts.new(n).from_string(open(f, encoding="utf-8", errors="replace").read())
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
U = MC5.U
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def build(shrink=1.0):
    """A cloth with nothing acting on it but its own springs."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select', np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    p = ob.MC_props
    p.cloth = True
    p.gravity = 0.0
    p.velocity = 0.0          # no carried momentum: pure relaxation
    p.bend_force = 0.0
    p.stretch = 20.0          # enough iterations to actually converge
    p.shrink_grow = shrink
    p.animated = True
    return ob


def ratio(C):
    """Mean edge length as a fraction of the spring's target length."""
    d = np.linalg.norm(C.co[C.es_0] - C.co[C.es_1], axis=1)
    return float(np.mean(d / C.target_dists))


def settle(ob, frames=40):
    for _ in range(frames):
        scene.frame_set(scene.frame_current + 1)
    return MC5.get_cloth(ob)


print("== the springs settle at shrink_grow times their target ==", flush=True)
for want in (1.0, 0.5, 2.0, 0.75):
    ob = build(want)
    C = settle(ob)
    got = ratio(C)
    check(abs(got - want) < 0.03 * max(want, 1.0),
          "shrink_grow %.2f -> edges at %.3f of target" % (want, got))
    check(np.isfinite(C.co).all(), "  and the cloth is finite at %.2f" % want)

print("\n== it is the rest length, not a transient ==", flush=True)
ob = build(0.5)
C = settle(ob, 40)
first = ratio(C)
C = settle(ob, 40)
check(abs(ratio(C) - first) < 0.01,
      "it stays put once settled (%.3f then %.3f)" % (first, ratio(C)))

print("\n== both solvers agree ==", flush=True)
results = {}
for want in (0.5, 1.5):
    ob = build(want)
    C = settle(ob, 5)
    C.b_iters = 0
    start = C.co.copy()
    for name, fn in (("numpy", MC5.stretch_force_np), ("c++", MC5.stretch_force_cpp)):
        C.co[:] = start
        for _ in range(30):
            fn(C)
        results[(want, name)] = ratio(C)
    a, b = results[(want, "numpy")], results[(want, "c++")]
    check(abs(a - b) < 0.02,
          "shrink_grow %.2f: numpy %.3f vs c++ %.3f" % (want, a, b))
    check(abs(b - want) < 0.05 * max(want, 1.0),
          "  and the c++ solver is the one that honours it (%.3f)" % b)

print("\n== the property itself ==", flush=True)
ob = build(1.0)
p = ob.MC_props
rna = p.bl_rna.properties["shrink_grow"]
check(rna.hard_min == 0.0, "shrink_grow cannot go below zero (min %s)" % rna.hard_min)
check(rna.hard_max > 1e6, "and has no upper limit (max %s)" % rna.hard_max)
p.shrink_grow = -5.0
check(p.shrink_grow == 0.0, "a negative value clamps to zero (%.2f)" % p.shrink_grow)
p.shrink_grow = 100.0
check(p.shrink_grow == 100.0, "a large value is accepted (%.2f)" % p.shrink_grow)

print("\n== pre-shrink is gone ==", flush=True)
check("pre_shrink" not in p.bl_rna.properties, "the property is removed")
check("pre_shrink" not in UI.PRESET_KEYS, "presets no longer store it")
src = open(os.path.join(SRC, "MC5.py"), encoding="utf-8").read()
check("pre_shrink" not in src, "no mention left in MC5.py")
ui_src = open(os.path.join(SRC, "MC_ui.py"), encoding="utf-8").read()
check("pre_shrink" not in ui_src, "no mention left in MC_ui.py")

# an old preset that still carries the key must not break
preset = scene.MC_props.mc_presets.add()
preset.name = "old"
preset.version = UI.PRESET_VERSION
import json
preset.data = json.dumps({"pre_shrink": 0.5, "shrink_grow": 0.8, "stretch": 3.0})
UI.preset_apply(preset, p)
check(abs(p.shrink_grow - 0.8) < 1e-6 and abs(p.stretch - 3.0) < 1e-6,
      "a preset holding the old key still applies the rest (%.2f, %.2f)"
      % (p.shrink_grow, p.stretch))

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
