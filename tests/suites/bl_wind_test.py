"""Wind tests."""
import bpy, os, glob, sys
import numpy as np
from mathutils import Euler

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
W = UI.WIND
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def close(a, b, tol=1e-6):
    # <= so that tol=0.0 means "exactly equal" rather than being impossible
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= tol


def build(**settings):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=2.0, location=(0, 0, 0))
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = 0.0            # wind only, so nothing else muddies the result
    p.velocity = 1.0
    p.stretch = 1.25
    p.bend_force = 0.5
    for k, v in settings.items():
        setattr(p, k, v)
    scene.frame_set(p.reset_frame + 1)
    return ob


def sim(ob, steps=30):
    C = MC5.get_cloth(ob)
    for _ in range(steps):
        MC5.physics(C)
    return np.array(MC5.get_cloth(ob).co).copy()


print("== off by default ==", flush=True)
ob = build()
check(tuple(ob.MC_props.wind) == (0.0, 0.0, 0.0), "wind defaults to zero")
check(W.is_active(ob) is False, "is_active False with no wind")
a = sim(build())
b = sim(build(wind=(0.0, 0.0, 0.0)))
check(close(a, b, 0.0), "zero wind is bit-identical to no wind")

print("\n== the vertex group exists at 1.0 ==", flush=True)
ob = build()
C = MC5.get_cloth(ob)
check("MC_wind" in C.group_data, "MC_wind group was created")
check("MC_wind" in [g.name for g in ob.vertex_groups],
      "and it is a real vertex group on the object")
check(close(C.group_data["MC_wind"], 1.0), "defaults to 1.0 everywhere")

print("\n== wind pushes the cloth ==", flush=True)
flat = sim(build())
blown = sim(build(wind=(0.0, 0.0, 5.0)))       # grid lies in XY, normal is Z
check(float(np.abs(blown - flat).max()) > 1e-4, "wind moved the cloth")
check(blown[:, 2].mean() > flat[:, 2].mean(), "and moved it downwind (+Z)")

print("\n== facing: edge-on catches nothing, square-on catches all ==",
      flush=True)
# the grid's normals are +Z, so wind along Z is square on and wind along X is
# edge-on
square = sim(build(wind=(0.0, 0.0, 5.0)))
edge = sim(build(wind=(5.0, 0.0, 0.0)))
base = sim(build())
d_square = float(np.abs(square - base).max())
d_edge = float(np.abs(edge - base).max())
print("   square-on moved %.5f, edge-on moved %.5f" % (d_square, d_edge),
      flush=True)
check(d_square > 1e-4, "square-on wind moves it (%.5f)" % d_square)
check(d_edge < d_square * 0.05,
      "edge-on wind barely does (%.5f vs %.5f)" % (d_edge, d_square))

# and the facing term itself: normals must be one per vertex, so plant the
# three cases at the front of a full-size array
ob = build(wind=(0.0, 0.0, 1.0))
C = MC5.get_cloth(ob)
C.wind_step = 0
n = np.zeros((C.co.shape[0], 3))
n[0] = (0.0, 0.0, 1.0)      # square on
n[1] = (1.0, 0.0, 0.0)      # edge on
n[2] = (0.0, 0.0, -1.0)     # square on, flipped
f = W.wind_force(C, n)
mag = np.linalg.norm(f, axis=1)
check(mag[1] < 1e-12, "perpendicular normal -> zero force (%.3e)" % mag[1])
check(abs(mag[0] - mag[2]) < 1e-12,
      "a flipped normal catches the same wind, not the opposite")
check(mag[0] > 0.0, "parallel normal -> full force")

print("\n== the empty aims the wind, strength is a single value ==", flush=True)
ob = build(wind=(0.0, 0.0, 99.0))      # vector must be ignored once aimed
bpy.ops.object.empty_add()
e = bpy.context.object
e.rotation_euler = Euler((0.0, np.pi / 2.0, 0.0))     # +Z now points +X
bpy.context.view_layer.update()
ob.MC_props.wind_object = e
ob.MC_props.wind_strength = 3.0
v = W.gust_vector(ob, 0)
check(close(v, [3.0, 0.0, 0.0], 1e-5),
      "empty rotated 90deg aims the wind down +X %s" % np.round(v, 4))
check(abs(np.linalg.norm(v) - 3.0) < 1e-5,
      "wind_strength sets the magnitude, not the vector's length")
ob.MC_props.wind_strength = -2.0
v = W.gust_vector(ob, 0)
check(close(v, [-2.0, 0.0, 0.0], 1e-5),
      "a negative strength blows the other way %s" % np.round(v, 4))
ob.MC_props.wind_object = None
check(abs(np.linalg.norm(W.gust_vector(ob, 0)) - 99.0) < 1e-4,
      "clearing the object hands direction and strength back to the vector")

print("\n== strength is mirrored between cloth and wind object ==", flush=True)
ob = build()
bpy.ops.object.empty_add()
e = bpy.context.object
ob.MC_props.wind_object = e
check(W.cloths_using(e) == [ob], "the empty knows which cloth uses it")

ob.MC_props.wind_strength = 4.25
check(abs(e.MC_props.wind_strength - 4.25) < 1e-6,
      "cloth -> object (%.4f)" % e.MC_props.wind_strength)
e.MC_props.wind_strength = -1.5
check(abs(ob.MC_props.wind_strength + 1.5) < 1e-6,
      "object -> cloth (%.4f)" % ob.MC_props.wind_strength)

# two cloths sharing one wind object
ob2 = bpy.data.objects.new("second", ob.data.copy())
bpy.context.collection.objects.link(ob2)
ob2.MC_props.wind_object = e
check(len(W.cloths_using(e)) == 2, "two cloths on one wind object")
e.MC_props.wind_strength = 7.0
check(abs(ob.MC_props.wind_strength - 7.0) < 1e-6
      and abs(ob2.MC_props.wind_strength - 7.0) < 1e-6,
      "the object updates both of them")
ob2.MC_props.wind_strength = 2.0
check(abs(e.MC_props.wind_strength - 2.0) < 1e-6,
      "and either cloth updates the object")
bpy.data.objects.remove(ob2)

# the bracket write must not fire the callback back at us
ob.MC_props.wind_strength = 5.0
check(abs(ob.MC_props.wind_strength - 5.0) < 1e-6
      and abs(e.MC_props.wind_strength - 5.0) < 1e-6,
      "repeated edits settle instead of recursing")

print("\n== a deleted wind object does not break anything ==", flush=True)
ob = build(wind_strength=2.0)
bpy.ops.object.empty_add()
e = bpy.context.object
ob.MC_props.wind_object = e
bpy.data.objects.remove(e)
check(ob.MC_props.wind_object is None,
      "Blender cleared the pointer when the object went away")
v = W.gust_vector(ob, 0)
check(np.all(np.isfinite(v)), "gust_vector still returns something finite")

print("\n== the cloth's own rotation is accounted for ==", flush=True)
# World wind along +Z on a cloth pitched 90deg about X.  That rotation carries
# the cloth's local +Y round to world +Z, so world +Z must read as local +Y --
# not still as local Z, which is what ignoring the object's rotation would give.
ob = build(wind=(0.0, 0.0, 1.0))
ob.rotation_euler = Euler((np.pi / 2.0, 0.0, 0.0))
bpy.context.view_layer.update()
local = W.to_local(ob, np.array([0.0, 0.0, 1.0]))
check(close(local, [0.0, 1.0, 0.0], 1e-6),
      "world +Z becomes local +Y on a cloth pitched 90deg %s" % np.round(local, 4))
check(abs(local[2]) < 1e-6, "and no longer lies along local Z")

print("\n== turbulence is uncapped ==", flush=True)
ob = build(wind=(0.0, 0.0, 1.0), wind_turbulence_min=-2.0,
           wind_turbulence_max=6.0, wind_transition=5)
mults = [float(W.gust_vector(ob, s)[2]) for s in range(400)]
lo, hi = min(mults), max(mults)
print("   multiplier ranged %.3f to %.3f" % (lo, hi), flush=True)
check(lo < 0.0, "it goes negative, reversing the wind (%.3f)" % lo)
check(hi > 2.0, "and well above 1, with no ceiling (%.3f)" % hi)
check(lo >= -2.0 - 1e-9 and hi <= 6.0 + 1e-9, "stays inside the stated range")
check(any(abs(m) < 0.2 for m in mults), "and passes through near-zero lulls")

ob = build(wind=(0.0, 0.0, 1.0), wind_turbulence_min=0.0,
           wind_turbulence_max=0.0)
check(close(W.gust_vector(ob, 7), [0, 0, 0]),
      "a 0-0 range parks the wind at nothing")

print("\n== gusts are smooth, not jitter ==", flush=True)
ob = build(wind=(0.0, 0.0, 1.0), wind_turbulence_min=0.0,
           wind_turbulence_max=1.0, wind_transition=40, wind_smoothness=1.0)
seq = np.array([float(W.gust_vector(ob, s)[2]) for s in range(400)])
jumps = np.abs(np.diff(seq))
check(jumps.max() < 0.12,
      "no sudden jumps between steps (largest %.4f)" % jumps.max())
check(seq.std() > 0.05, "but it really does vary (std %.4f)" % seq.std())

print("\n== smoothness changes the curve ==", flush=True)
lin = np.array([W.wobble(s, 40, 1, 7, 0.0) for s in range(400)])
sm = np.array([W.wobble(s, 40, 1, 7, 1.0) for s in range(400)])
check(float(np.abs(lin - sm).max()) > 1e-3,
      "smoothness 0 and 1 give different curves")
# a linear ramp has constant slope inside a gust; an eased one does not
seg_l = np.diff(lin[0:40])
seg_s = np.diff(sm[0:40])
check(seg_l.std() < 1e-9, "smoothness 0 moves at a constant rate")
check(seg_s.std() > 1e-6, "smoothness 1 eases in and out")
check(W.ease(0.0, 1.0) == 0.0 and abs(W.ease(1.0, 1.0) - 1.0) < 1e-12,
      "the easing curve still hits both ends")

print("\n== direction variation swings the wind ==", flush=True)
ob = build(wind=(0.0, 0.0, 1.0), wind_direction_variation=0.0,
           wind_transition=10)
steady = np.array([W.gust_vector(ob, s) for s in range(200)])
ob.MC_props.wind_direction_variation = 45.0
swung = np.array([W.gust_vector(ob, s) for s in range(200)])
off_axis = np.abs(swung[:, :2]).max()
check(np.abs(steady[:, :2]).max() < 1e-12, "0 variation stays on axis")
check(off_axis > 0.1, "45 deg variation swings off axis (%.4f)" % off_axis)

print("\n== spatial variation ==", flush=True)
ob = build(wind=(0.0, 0.0, 2.0), wind_spatial_amount=0.0)
C = MC5.get_cloth(ob)
C.wind_step = 0
nrm = np.zeros((C.co.shape[0], 3))
nrm[:, 2] = 1.0
uniform = W.wind_force(C, nrm)
check(float(np.linalg.norm(uniform, axis=1).std()) < 1e-12,
      "0 amount gives every vertex the same wind")
ob.MC_props.wind_spatial_amount = 0.8
ob.MC_props.wind_spatial_scale = 0.5
varied = W.wind_force(C, nrm)
check(float(np.linalg.norm(varied, axis=1).std()) > 1e-6,
      "amount 0.8 makes it differ across the surface")

print("\n== the vertex group scales it ==", flush=True)
ob = build(wind=(0.0, 0.0, 3.0))
C = MC5.get_cloth(ob)
C.wind_step = 0
nrm = np.zeros((C.co.shape[0], 3))
nrm[:, 2] = 1.0
full = W.wind_force(C, nrm)
C.group_data["MC_wind"][:] = 0.0
none = W.wind_force(C, nrm)
check(float(np.abs(none).max()) == 0.0, "a zero weight removes the wind")
C.group_data["MC_wind"][:] = 0.5
half = W.wind_force(C, nrm)
check(close(half, full * 0.5), "a 0.5 weight halves it")

print("\n== repeatable ==", flush=True)
kw = dict(wind=(1.0, 0.0, 2.0), wind_turbulence_min=-1.0,
          wind_turbulence_max=3.0, wind_direction_variation=30.0,
          wind_spatial_amount=0.5, wind_transition=12)
r1 = sim(build(**kw), steps=40)
r2 = sim(build(**kw), steps=40)
check(close(r1, r2, 0.0), "the same settings give the same result")
r3 = sim(build(wind_seed=9, **kw), steps=40)
check(not close(r3, r1, 1e-9), "a different seed gives a different result")

print("\n== animated and continuous agree ==", flush=True)


def run(mode, ticks=40):
    ob = build(**kw)
    reset = ob.MC_props.reset_frame
    if mode == "animated":
        ob.MC_props.animated = True
    else:
        ob.MC_props.continuous = True
    for f in range(ticks):
        if mode == "animated":
            scene.frame_set(reset + 2 + f)
        else:
            MC5.mc_handler_continuous()
    return np.array(MC5.get_cloth(ob).co).copy()


an, co = run("animated"), run("continuous")
check(close(an, co, 0.0),
      "wind lands identically in both modes (%.3e)"
      % float(np.abs(an - co).max()))

print("\n== presets survive the vector property ==", flush=True)
ob = build(wind=(1.5, -2.0, 0.5), wind_turbulence_min=-0.5,
           wind_turbulence_max=3.0, wind_smoothness=0.25)
pr = scene.MC_props.mc_presets.add()
UI.preset_store(pr, ob.MC_props)
ob2 = build()
UI.preset_apply(pr, ob2.MC_props)
check(close(list(ob2.MC_props.wind), [1.5, -2.0, 0.5], 1e-5),
      "the wind vector round-trips %s" % list(ob2.MC_props.wind))
check(abs(ob2.MC_props.wind_turbulence_min + 0.5) < 1e-5,
      "a negative turbulence floor round-trips")
check(abs(ob2.MC_props.wind_smoothness - 0.25) < 1e-5,
      "smoothness round-trips")

print("\n== vectorised noise matches the scalar reference ==", flush=True)
MASK = 0xFFFFFFFFFFFFFFFF


def ref_hash(x):
    """The original python-int avalanche, kept as ground truth."""
    x &= MASK
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK
    x ^= x >> 33
    return x


def ref_rand01(index, seed=0, stream=0):
    return ref_hash(int(index) * 0x9E3779B97F4A7C15
                    ^ (int(seed) * 0xBF58476D1CE4E5B9)
                    ^ (int(stream) * 0x94D049BB133111EB)) / 2.0 ** 64


# uint64 wrap-around must reproduce the python-int arithmetic exactly, or
# every existing gust pattern would quietly change
worst = 0.0
for i in (0, 1, 2, 7, 99, 1234, 99999):
    for s in (0, 1, 9):
        for st in (0, 7, 11, 13):
            worst = max(worst, abs(float(W.rand01(i, s, st))
                                   - ref_rand01(i, s, st)))
check(worst == 0.0, "rand01 is bit-identical to the scalar version (%.3e)"
      % worst)

# and it vectorises over both index and stream
many = W.rand01(np.arange(50), 3, 7)
one = np.array([W.rand01(i, 3, 7) for i in range(50)])
check(close(many, one, 0.0), "rand01 vectorises over index")
streams = W.rand01(5, 3, np.array([11, 12, 13]))
check(close(streams, [W.rand01(5, 3, s) for s in (11, 12, 13)], 0.0),
      "rand01 vectorises over stream")

# the eight-corner broadcast must match a plain corner loop
def ref_value_noise(p, seed):
    i = np.floor(p)
    f = W.smoothstep(p - i)
    i = i.astype(np.int64)
    out = np.zeros(p.shape[0])
    for dz in (0, 1):
        wz = f[:, 2] if dz else 1.0 - f[:, 2]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dx in (0, 1):
                wx = f[:, 0] if dx else 1.0 - f[:, 0]
                out += W._hash_grid(i[:, 0] + dx, i[:, 1] + dy,
                                    i[:, 2] + dz, seed) * wx * wy * wz
    return out


pts = np.random.default_rng(0).normal(size=(500, 3)) * 3.0
check(close(W.value_noise(pts, 4), ref_value_noise(pts, 4), 1e-12),
      "value_noise matches a corner-by-corner loop")
check(W.value_noise(np.zeros((0, 3)), 1).shape == (0,),
      "value_noise handles an empty array")
nz = W.value_noise(pts, 4)
check(nz.min() >= 0.0 and nz.max() < 1.0,
      "and stays in [0, 1) (%.4f..%.4f)" % (nz.min(), nz.max()))

print("\n== no python loops left over the data ==", flush=True)
# the module comes from a text datablock, so inspect cannot reach its source;
# parse the file on disk instead
import ast
tree = ast.parse(open(os.path.join(SRC, "wind.py"), encoding="utf-8").read())
funcs = {n.name: n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef)}
for name in ("value_noise", "rand01", "_hash_u64", "wobble", "wind_force",
             "gust_vector", "_hash_grid", "apply"):
    node = funcs.get(name)
    check(node is not None, "%s exists" % name)
    if node is None:
        continue
    loops = [type(n).__name__ for n in ast.walk(node)
             if isinstance(n, (ast.For, ast.While, ast.ListComp,
                               ast.GeneratorExp))]
    check(not loops, "%s has no python iteration %s" % (name, loops))

print("\n== aimed wind actually runs through physics ==", flush=True)
# is_active used to test only the vector, so an aimed wind with the default
# zero vector never ran at all
ob = build()
bpy.ops.object.empty_add()
e = bpy.context.object
# unrotated, so its +Z blows straight at a grid whose normals are +Z.  Aiming
# it along the sheet instead would correctly produce no force at all.
e.rotation_euler = Euler((0.0, 0.0, 0.0))
bpy.context.view_layer.update()
ob.MC_props.wind_object = e
ob.MC_props.wind_strength = 6.0
check(tuple(ob.MC_props.wind) == (0.0, 0.0, 0.0),
      "the vector is left at zero, as it would be in practice")
check(W.is_active(ob) is True, "is_active sees the aimed wind")
before = np.array(MC5.get_cloth(ob).co).copy()
after = sim(ob, steps=25)
check(float(np.abs(after - before).max()) > 1e-4,
      "and the cloth is actually blown (%.5f)"
      % float(np.abs(after - before).max()))

print("\n== speed ==", flush=True)
import time
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
MC5.DATA.clear()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=120, y_subdivisions=120,
                                size=2.0)
big = bpy.context.object
big.data.vertices.foreach_set('select',
                              np.zeros(len(big.data.vertices), dtype=bool))
big.data.update()
bpy.context.view_layer.objects.active = big
big.MC_props.cloth = True
big.MC_props.wind = (1.0, 0.0, 2.0)
big.MC_props.wind_spatial_amount = 0.7
big.MC_props.wind_direction_variation = 30.0
C = MC5.get_cloth(big)
nrm = np.zeros((C.co.shape[0], 3))
nrm[:, 2] = 1.0
W.wind_force(C, nrm)                      # warm up
t0 = time.time()
for s in range(50):
    C.wind_step = s
    W.wind_force(C, nrm)
dt = (time.time() - t0) / 50.0
print("   %d verts, spatial noise on: %.3f ms per step"
      % (C.co.shape[0], dt * 1000.0), flush=True)
check(dt < 0.05, "a 14k-vert cloth costs under 50ms per step (%.3f ms)"
      % (dt * 1000.0))

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
