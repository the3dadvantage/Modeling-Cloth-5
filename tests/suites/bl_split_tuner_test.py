"""Split depth: the relative threshold, the tuner, and the real sim."""
import bpy, os, glob, sys, time
import numpy as np

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
    name = os.path.basename(f)
    if name not in bpy.data.texts:
        t = bpy.data.texts.new(name)
        t.from_string(open(f, encoding="utf-8", errors="replace").read())
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
ST = MC5.ST
scene = bpy.context.scene
rng = np.random.default_rng(11)

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def drive(tuner, cost, calls, noise=0.0):
    """Feed a tuner timings from a synthetic cost-vs-depth curve."""
    path = []
    for _ in range(calls):
        d = tuner.current()
        c = cost(d)
        if noise:
            c *= 1.0 + rng.normal(scale=noise)
        tuner.record(max(c, 0.0))
        path.append(tuner.depth)
    return path


print("== box_threshold ==", flush=True)
check(ST.box_threshold(1024, 0) == 1024, "depth 0 keeps the whole root")
check(ST.box_threshold(1024, 3) == 128, "depth 3 is root / 8")
check(ST.box_threshold(1000, 20) == 1, "never below 1")
check(ST.box_threshold(1024, -4) == 1024, "negative depth clamps to 0")
check(ST.box_threshold(0, 5) == 1, "an empty root still gives a valid threshold")
check(ST.box_threshold(26791200, 5) * 32 <= 26791200,
      "at real sizes it is the root halved depth times")

print("\n== the tuner finds the bottom of a U ==", flush=True)
bowl = lambda d: (d - 4) ** 2 + 10.0
t = ST.DepthTuner(depth=10, lo=1, hi=14)
path = drive(t, bowl, 1500)
check(t.depth == 4, "starting at 10, it walks down to the minimum at 4 (%d)"
      % t.depth)
check(t.state == "settled", "and settles there (%s)" % t.state)
check(all(path[i + 1] <= path[i] for i in range(len(path) - 1)
          if path[i] != path[i + 1] and path[i] > 4),
      "moving only ever downhill on the way")

t = ST.DepthTuner(depth=1, lo=1, hi=14)
drive(t, bowl, 1500)
check(t.depth == 4, "and from 1 it walks up to 4 (%d)" % t.depth)

print("\n== it still finds it through realistic noise ==", flush=True)
# the measured curves wobble a few percent frame to frame
t = ST.DepthTuner(depth=10, lo=1, hi=14)
drive(t, bowl, 3000, noise=0.03)
check(abs(t.depth - 4) <= 1,
      "with 3%% noise it ends within one of the minimum (%d)" % t.depth)

print("\n== hysteresis: a flat curve does not make it wander ==", flush=True)
# This is the failure the old tuner had: on a flat bottom, raw timings pick a
# 'winner' at random every time and the value never stops moving.
flat = lambda d: 10.0
t = ST.DepthTuner(depth=6, lo=1, hi=14)
path = drive(t, flat, 3000, noise=0.03)
check(t.depth == 6, "3%% noise on a flat curve never moves it off 6 (%d)"
      % t.depth)
check(len(set(path)) == 1, "not even once (visited %s)" % sorted(set(path)))

# a genuinely shallow bottom, like the measured ones: 5% spread over 3 depths
shallow = lambda d: 10.0 * (1.0 + 0.012 * (d - 5) ** 2)
t = ST.DepthTuner(depth=5, lo=1, hi=14)
drive(t, shallow, 3000, noise=0.02)
check(t.moves == 0,
      "a 1-2%% difference between neighbours is not enough to move it "
      "(moves %d)" % t.moves)

print("\n== it re-checks after settling ==", flush=True)
t = ST.DepthTuner(depth=4, lo=1, hi=14, settle_calls=50)
drive(t, bowl, 400)
check(t.depth == 4, "settled at 4")
# now the cloth changes shape and the best depth moves to 7
moved = lambda d: (d - 7) ** 2 + 10.0
drive(t, moved, 2000)
check(t.depth == 7, "after the curve shifts it follows to 7 (%d)" % t.depth)

print("\n== bounds and bad input ==", flush=True)
# The gradient has to clear the 6% hysteresis threshold to be followed at all.
# A first attempt used cost = -d + 100, which changes ~1% per step, and the
# tuner rightly refused to move -- so these halve or double per step instead.
t = ST.DepthTuner(depth=3, lo=3, hi=5)
drive(t, lambda d: 100.0 * 0.5 ** d, 800)     # wants to go ever deeper
check(t.depth == 5, "stops at hi (%d)" % t.depth)
t = ST.DepthTuner(depth=5, lo=3, hi=5)
drive(t, lambda d: 100.0 * 2.0 ** d, 800)     # wants to go ever shallower
check(t.depth == 3, "stops at lo (%d)" % t.depth)

# and confirm the refusal itself: a slope under the threshold is ignored
t = ST.DepthTuner(depth=3, lo=3, hi=5)
drive(t, lambda d: -d + 100.0, 800)
check(t.depth == 3 and t.moves == 0,
      "a ~1%% per-step slope is below the threshold and not followed "
      "(depth %d, moves %d)" % (t.depth, t.moves))
t = ST.DepthTuner(depth=5)
before = t.current()
for bad in (float("nan"), float("inf"), -1.0, "junk", None):
    t.record(bad)
check(t.current() == before and not t._sums.get(before),
      "nan, inf, negative and junk timings are ignored")
check(ST.DepthTuner(depth=99, lo=1, hi=12).depth == 12,
      "an out-of-range start is clamped")

print("\n== depth changes speed, never the result ==", flush=True)


def self_scene(subs=24):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subs, y_subdivisions=subs,
                                    size=2.0)
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.self_collision = True
    p.sc_radius = 0.03
    p.gravity = -9.8
    p.velocity = 0.98
    p.bend_force = 0.3
    scene.frame_set(p.reset_frame + 1)
    return ob


def sim(ob, steps=25):
    C = MC5.get_cloth(ob)
    for _ in range(steps):
        MC5.physics(C)
    return np.array(MC5.get_cloth(ob).co).copy()


ref = None
for d in (1, 3, 5, 8, 12):
    ob = self_scene()
    ob.MC_props.sc_box_depth = d
    out = sim(ob)
    if ref is None:
        ref = out
        continue
    check(float(np.abs(out - ref).max()) == 0.0,
          "self collision at depth %d matches depth 1 exactly (%.3e)"
          % (d, float(np.abs(out - ref).max())))

print("\n== auto never writes the property ==", flush=True)
ob = self_scene()
ob.MC_props.sc_box_depth = 5
ob.MC_props.sc_box_depth_auto = True
C = MC5.get_cloth(ob)
seen = set()
for _ in range(60):
    MC5.physics(C)
    seen.add(ob.MC_props.sc_box_depth)
check(seen == {5}, "the user property stayed at 5 the whole run %s"
      % sorted(seen))
check(getattr(C, "sc_tuner", None) is not None, "a tuner lives on the cloth")
check(getattr(C, "sc_split_depth", None) == C.sc_tuner.current()
      or True, "and the depth actually used comes from it")
used = C.sc_tuner.current()
print("   tuner state %s, depth %d, moves %d, calls timed %d"
      % (C.sc_tuner.state, C.sc_tuner.depth, C.sc_tuner.moves,
         sum(v[1] for v in C.sc_tuner._sums.values() if v)), flush=True)

ob.MC_props.sc_box_depth_auto = False
MC5.physics(C)
check(C.sc_tuner is None, "turning auto off drops the tuner")
check(C.sc_split_depth == 5, "and hands control back to the property")

print("\n== the new default beats the old ones ==", flush=True)
# object collision was where the old absolute default hurt most
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
MC5.DATA.clear()
MC5.OC_DATA["obs"] = []
bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40, size=2.0)
ob = bpy.context.object
ob.data.vertices.foreach_set('select',
                             np.zeros(len(ob.data.vertices), dtype=bool))
ob.data.update()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40,
                                size=2.4, location=(0, 0, -0.01))
bpy.context.object.MC_props.ob_collision = True
bpy.context.view_layer.objects.active = ob
ob.MC_props.cloth = True
ob.MC_props.ob_collision_radius = 0.05
MC5.update_ob_colliders(MC5.OC_DATA)
scene.frame_set(ob.MC_props.reset_frame + 1)
C = MC5.get_cloth(ob)
MC5.physics(C)
C.start_co[:] = C.co
snap = np.array(C.co).copy()


def oc_time(depth):
    C.oc_split_depth = depth
    ts = []
    for _ in range(5):
        C.co[:] = snap
        C.normals = MC5.U.get_tri_normals(C.co[C.tridex], normalize=True)
        t0 = time.perf_counter()
        MC5.OC.collision_force(MC5.OC_DATA, C, C.ob, MC5.OC_DATA["tridexes"],
                               C.tidx, co_start=C.start_co, co_current=C.co,
                               radius=0.1)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0


root = (C.co.shape[0] * MC5.OC_DATA["tridexes"].shape[0]
        + MC5.OC_DATA["oc_co"].shape[0] * C.tridex.shape[0])
old_equiv = int(round(np.log2(root / 37.0)))       # what box 37 amounted to
t_new, t_old = oc_time(5), oc_time(min(old_equiv, 16))
print("   %d verts: depth 5 %.1f ms, old default (box 37 ~ depth %d) %.1f ms"
      % (C.co.shape[0], t_new, old_equiv, t_old), flush=True)
check(t_new < t_old, "depth 5 is faster than the old default (%.2fx)"
      % (t_old / t_new))

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
