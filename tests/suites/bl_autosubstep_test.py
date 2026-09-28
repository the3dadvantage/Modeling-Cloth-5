"""Does collision_auto_substeps pick the right count for SELF collision?

Auto substeps exists to keep the movement in one collision test below the
collision margin.  With two margins in play -- the object collision radius and
the self collision radius -- the binding constraint is the SMALLER of them, and
a margin that is not in use should not be consulted at all.
"""
import bpy, os, glob, sys, math
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
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def build(collider=False, **settings):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=2.0)
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    if collider:
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2,
                                        size=8.0, location=(0, 0, -4))
        bpy.context.object.MC_props.ob_collision = True
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = -9.8
    p.velocity = 0.98
    p.stretch = 1.25
    for k, v in settings.items():
        setattr(p, k, v)
    if collider:
        MC5.update_ob_colliders(MC5.OC_DATA)
    scene.frame_set(p.reset_frame + 1)
    return ob


def steps_for(ob, move, actual=None):
    """Ask collision_steps how it would slice a movement of `move`.

    The real movement is measured back out of the float32 arrays rather than
    assumed: adding 0.1 to a float32 does not land exactly 0.1 away, and the
    ceil() in collision_steps turns that into an extra substep.
    """
    C = MC5.get_cloth(ob)
    C.start_co[:] = C.co
    C.co[0, 0] += move             # one vertex moves, which is the worst case
    real = float(np.abs(C.co - C.start_co).max())
    n = MC5.collision_steps(C)
    C.co[0, 0] -= move
    if actual is not None:
        actual.append(real)
    return n


def expected(move, margin, safety):
    return max(1, math.ceil(move / (margin * safety)))


def steps_and_want(ob, move, margin, safety):
    got_move = []
    n = steps_for(ob, move, got_move)
    return n, expected(got_move[0], margin, safety)


print("== the plain object-collision case still works ==", flush=True)
ob = build(collider=True, collision_auto_substeps=True,
           collision_max_substeps=64, ob_collision_radius=0.05,
           collision_substep_margin=0.5, self_collision=False)
got, want = steps_and_want(ob, 0.5, 0.05, 0.5)
check(got == want, "0.5 move, 0.05 object margin -> %d substeps (want %d)"
      % (got, want))

print("\n== self collision alone, no colliders in the scene ==", flush=True)
# sc_radius is the only margin that applies here.  ob_collision_radius is set
# deliberately large; it must not be consulted, because nothing is using it.
ob = build(collider=False, collision_auto_substeps=True,
           collision_max_substeps=64, self_collision=True,
           sc_radius=0.01, ob_collision_radius=0.5,
           collision_substep_margin=0.5)
got, want = steps_and_want(ob, 0.1, 0.01, 0.5)
check(got == want,
      "0.1 move, 0.01 self margin -> %d substeps (want %d, driven by "
      "sc_radius not the unused object radius)" % (got, want))

print("\n== both active: the tighter margin must win ==", flush=True)
ob = build(collider=True, collision_auto_substeps=True,
           collision_max_substeps=64, self_collision=True,
           sc_radius=0.01, ob_collision_radius=0.5,
           collision_substep_margin=0.5)
got, want = steps_and_want(ob, 0.1, 0.01, 0.5)
check(got == want,
      "tight self margin 0.01 beside loose object margin 0.5 -> %d (want %d)"
      % (got, want))

ob = build(collider=True, collision_auto_substeps=True,
           collision_max_substeps=64, self_collision=True,
           sc_radius=0.5, ob_collision_radius=0.01,
           collision_substep_margin=0.5)
got, want = steps_and_want(ob, 0.1, 0.01, 0.5)
check(got == want,
      "and the other way round, tight object margin -> %d (want %d)"
      % (got, want))

print("\n== the knobs behave ==", flush=True)
ob = build(collider=False, collision_auto_substeps=True,
           collision_max_substeps=8, self_collision=True,
           sc_radius=0.01, ob_collision_radius=0.01,
           collision_substep_margin=0.5)
check(steps_for(ob, 10.0) == 8, "the cap is respected (%d)" % steps_for(ob, 10.0))

ob.MC_props.collision_max_substeps = 64
ob.MC_props.collision_substep_margin = 1.0
loose, want = steps_and_want(ob, 0.1, 0.01, 1.0)
check(loose == want, "a looser safety factor asks for fewer (%d, want %d)"
      % (loose, want))
ob.MC_props.collision_substep_margin = 0.25
tight, want = steps_and_want(ob, 0.1, 0.01, 0.25)
check(tight == want, "a tighter one asks for more (%d, want %d)"
      % (tight, want))
check(tight > loose, "and tighter really is more than looser (%d > %d)"
      % (tight, loose))

print("\n== off, and degenerate inputs ==", flush=True)
ob = build(collider=False, collision_auto_substeps=False,
           collision_substeps=3, self_collision=True)
check(steps_for(ob, 5.0) == 3, "auto off uses the explicit count")

ob = build(collider=False, collision_auto_substeps=True,
           self_collision=True, sc_radius=0.01)
check(steps_for(ob, 0.0) == 1, "no movement -> 1 substep")

ob = build(collider=False, collision_auto_substeps=True,
           self_collision=False)
check(steps_for(ob, 1.0) == 1,
      "nothing colliding at all -> 1 substep, whatever the radii say")

print("\n== auto substeps engage in a real self-collision-only sim ==",
      flush=True)
# The question that matters: with self collision on, no colliders, and a small
# sc_radius beside a large ob_collision_radius, does the solver actually take
# more than one substep?  Under the old max() it took exactly one, which is
# what made auto substeps look like it did nothing for self collision.


def substeps_used(ob, steps=40):
    C = MC5.get_cloth(ob)
    used = []
    for _ in range(steps):
        MC5.physics(C)
        used.append(int(getattr(C, "collision_substeps_used", 1)))
    return max(used), used


kw = dict(collider=False, self_collision=True, sc_radius=0.02,
          ob_collision_radius=0.5, collision_substep_margin=0.5,
          gravity=-9.8, sc_edges=True)

off = build(collision_auto_substeps=False, collision_substeps=1, **kw)
n_off, _ = substeps_used(off)
on = build(collision_auto_substeps=True, collision_max_substeps=16, **kw)
n_on, seq = substeps_used(on)
print("   substeps used: auto off max %d, auto on max %d (last few %s)"
      % (n_off, n_on, seq[-5:]), flush=True)
check(n_off == 1, "with auto off it stays at 1, as asked (%d)" % n_off)
check(n_on > 1,
      "with auto on and only self collision active, it really does subdivide "
      "(max %d)" % n_on)
check(n_on <= 16, "and respects the cap (%d)" % n_on)

# the large, unused object radius must not be what is driving that number
on2 = build(collision_auto_substeps=True, collision_max_substeps=16,
            **dict(kw, ob_collision_radius=0.001))
n_on2, _ = substeps_used(on2)
check(n_on2 == n_on,
      "changing the unused object radius changes nothing (%d vs %d)"
      % (n_on2, n_on))

# but changing the self radius, which IS in use, must change it
on3 = build(collision_auto_substeps=True, collision_max_substeps=16,
            **dict(kw, sc_radius=0.2))
n_on3, _ = substeps_used(on3)
print("   sc_radius 0.02 -> %d substeps, sc_radius 0.2 -> %d"
      % (n_on, n_on3), flush=True)
check(n_on3 < n_on,
      "a looser self radius needs fewer substeps (%d < %d)" % (n_on3, n_on))

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)

