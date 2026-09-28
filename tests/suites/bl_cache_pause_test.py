"""Pausing a cache bake.

The bake is a modal operator, so headless there are no real events to send it.
Its modal() is driven directly with a stand-in context and event instead, which
is enough to pin down what matters: while paused nothing is baked and the sim
does not advance, and resuming carries on from the same frame.
"""
import bpy, os, glob, sys, types, tempfile, shutil
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
CACHE = MC5.CACHE
scene = bpy.context.scene

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


TMP = tempfile.mkdtemp(prefix="mc_cache_")
scene.MC_props.cache_dir = TMP


class Ctx:
    """Enough of a context for modal() and _apply_pause()."""
    def __init__(self, ob):
        self.scene = scene
        self.object = ob
        self.window_manager = bpy.context.window_manager
        self.status = []
        self.workspace = types.SimpleNamespace(
            status_text_set=lambda s: self.status.append(s))


class Ev:
    def __init__(self, type='TIMER', value='PRESS'):
        self.type = type
        self.value = value


def build():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6,
                                    size=2.0)
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = -9.8
    p.velocity = 0.98
    p.stretch = 1.25
    return ob


class FakeOp:
    """Borrows the operator's methods.

    bpy_struct subclasses cannot be constructed from python, so the methods
    are taken off the class and bound to a plain object carrying the same
    attributes invoke() would have set.
    """
    modal = UI.MC_OT_cache_bake.modal
    _modal = UI.MC_OT_cache_bake._modal
    _apply_pause = UI.MC_OT_cache_bake._apply_pause
    _finish = UI.MC_OT_cache_bake._finish

    def report(self, *a, **k):
        pass


def fake_bake(ob, start=1, end=20):
    """An operator instance positioned as if invoke() had just run."""
    op = FakeOp()
    op.ob = ob
    op.start, op.end, op.cur = start, end, start
    op._paused = False
    op._timer = None
    op._saved = (False, False)
    sc = scene.MC_props
    sc.cache_baking = True
    sc.cache_bake_paused = False
    ob.MC_props["animated"] = True
    CACHE.bake_heartbeat()
    return op


print("== registration ==", flush=True)
check(hasattr(bpy.types, "MC_OT_cache_bake_pause"), "the pause operator exists")
check(scene.MC_props.cache_baking is False, "cache_baking starts False")
check(scene.MC_props.cache_bake_paused is False, "cache_bake_paused starts False")

print("\n== the pause button only works during a bake ==", flush=True)
check(UI.MC_OT_cache_bake_pause.poll(bpy.context) is False,
      "poll is False when nothing is baking")
scene.MC_props.cache_baking = True
CACHE.bake_heartbeat()
check(UI.MC_OT_cache_bake_pause.poll(bpy.context) is True,
      "poll is True while baking")
res = bpy.ops.mc.cache_bake_pause()
check(res == {'FINISHED'} and scene.MC_props.cache_bake_paused is True,
      "it pauses (%s)" % (res,))
bpy.ops.mc.cache_bake_pause()
check(scene.MC_props.cache_bake_paused is False, "and toggles back to running")
scene.MC_props.cache_baking = False

print("\n== paused, nothing is baked and the sim does not move ==", flush=True)
ob = build()
CACHE.wipe(ob)
op = fake_bake(ob)
ctx = Ctx(ob)

# a couple of normal ticks first
for _ in range(3):
    op.modal(ctx, Ev('TIMER'))
baked_before = sorted(CACHE.cached_frames(ob))
cur_before = op.cur
co_before = np.array(MC5.get_cloth(ob).co).copy()
check(len(baked_before) == 3, "three frames baked so far (%d)" % len(baked_before))

# now pause, via the panel button's route
scene.MC_props.cache_bake_paused = True
for _ in range(5):
    r = op.modal(ctx, Ev('TIMER'))
check(r == {'RUNNING_MODAL'}, "the modal keeps running while paused (%s)" % (r,))
check(op._paused is True, "it noticed the flag")
check(op.cur == cur_before, "the frame did not advance (%d)" % op.cur)
check(sorted(CACHE.cached_frames(ob)) == baked_before,
      "no new frames were written")
check(float(np.abs(np.array(MC5.get_cloth(ob).co) - co_before).max()) == 0.0,
      "and the cloth did not move")
check(ob.MC_props.animated is False,
      "animated is off while paused, so scrubbing cannot step the sim")

print("\n== scrubbing while paused does not advance the sim ==", flush=True)
scene.frame_set(scene.frame_current + 5)
MC5.mc_handler()
check(float(np.abs(np.array(MC5.get_cloth(ob).co) - co_before).max()) == 0.0,
      "the frame handler left the cloth alone")

print("\n== resuming carries on from the same frame ==", flush=True)
scene.MC_props.cache_bake_paused = False
op.modal(ctx, Ev('TIMER'))
check(ob.MC_props.animated is True, "animated is back on")
check(op._paused is False, "no longer paused")
after = sorted(CACHE.cached_frames(ob))
check(len(after) == len(baked_before) + 1,
      "one more frame baked (%d -> %d)" % (len(baked_before), len(after)))
check(op.cur == cur_before + 1, "and it moved on by one frame (%d)" % op.cur)
check(after[:len(baked_before)] == baked_before,
      "the frames baked before the pause are untouched")

print("\n== the keyboard toggles it too ==", flush=True)
for key in ('SPACE', 'P'):
    was = scene.MC_props.cache_bake_paused
    r = op.modal(ctx, Ev(key, 'PRESS'))
    check(scene.MC_props.cache_bake_paused is (not was),
          "%s toggles pause" % key)
    check(r == {'RUNNING_MODAL'}, "%s keeps the modal alive" % key)
    check(op._paused is scene.MC_props.cache_bake_paused,
          "%s applied it immediately" % key)
    # put it back
    op.modal(ctx, Ev(key, 'PRESS'))
check(scene.MC_props.cache_bake_paused is False, "left running after toggling")

# key releases must not double-toggle
was = scene.MC_props.cache_bake_paused
op.modal(ctx, Ev('SPACE', 'RELEASE'))
check(scene.MC_props.cache_bake_paused is was,
      "a key RELEASE does not toggle")

print("\n== other events still pass through ==", flush=True)
check(op.modal(ctx, Ev('MOUSEMOVE', 'ANY')) == {'PASS_THROUGH'},
      "mouse moves pass through, so the viewport stays usable")

print("\n== finishing clears the flags ==", flush=True)
r = op._finish(ctx, cancelled=True)
check(scene.MC_props.cache_baking is False, "cache_baking cleared")
check(scene.MC_props.cache_bake_paused is False, "cache_bake_paused cleared")
check(r == {'CANCELLED'}, "reports cancelled (%s)" % (r,))

print("\n== a bake that finishes while paused is not left stuck ==", flush=True)
ob = build()
CACHE.wipe(ob)
op = fake_bake(ob, start=1, end=2)
ctx = Ctx(ob)
op.modal(ctx, Ev('TIMER'))
scene.MC_props.cache_bake_paused = True
op.modal(ctx, Ev('TIMER'))
r = op._finish(ctx, cancelled=True)
check(scene.MC_props.cache_bake_paused is False,
      "stopping from a paused bake clears the pause")
check(ob.MC_props.animated is False or ob.MC_props.animated is True,
      "and animated is restored to its saved value (%s)"
      % ob.MC_props.animated)

shutil.rmtree(TMP, ignore_errors=True)

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.exit(0 if ok else 1)
