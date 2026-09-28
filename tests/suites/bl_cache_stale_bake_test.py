"""A bake flag left over from a bake that no longer exists.

The scene's cache_baking flag is saved with the .blend.  A file saved mid-bake,
or a bake that died on an error, used to leave it on with no bake running: the
Cache panel then showed only a red "Resume Bake" button that did nothing.

  * the panel shows Bake / From Current unless a bake is really alive
  * the pause button can't be used without a live bake
  * opening a file clears the flags and restores Animated / Continuous
  * starting a new bake over a stale flag just works
  * an error inside the bake ends it cleanly instead of sticking the flag
"""
import bpy, os, sys, glob, types, tempfile, shutil
import numpy as np

# an uncaught error in a -b --python script still exits 0; make it fail
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
    name = os.path.basename(f)
    if name not in bpy.data.texts:
        t = bpy.data.texts.new(name)
        t.from_string(open(f, encoding="utf-8", errors="replace").read())
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
CACHE = MC5.CACHE

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


TMP = tempfile.mkdtemp(prefix="mc_stale_")
bpy.context.scene.MC_props.cache_dir = TMP + os.sep


class Rec:
    """A layout that records operator calls and button labels."""
    def __init__(self, ops):
        self.ops = ops
        self.scale_y = self.scale_x = 1.0
        self.alert = False
        self.enabled = self.active = True
        self.use_property_split = False

    def _chain(self, *a, **k):
        return Rec(self.ops)

    box = row = column = split = column_flow = grid_flow = _chain

    def prop(self, *a, **k):
        return self

    def operator(self, idname, text="", **k):
        self.ops.append((idname, text))
        return types.SimpleNamespace()

    def label(self, *a, **k):
        return None

    separator = template_list = menu = prop_search = label


def cache_panel_buttons():
    P = [c for c in UI.CLASSES if isinstance(c, type)
         and issubclass(c, bpy.types.Panel) and "cache" in c.__name__.lower()]
    ops = []
    for p in P:
        p.draw(types.SimpleNamespace(layout=Rec(ops)), bpy.context)
    return ops


def build():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select', np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    return ob


ob = build()
scene = bpy.context.scene
sc = scene.MC_props

print("== a stale flag (no bake alive) ==", flush=True)
CACHE.bake_ended()
sc.cache_baking = True
sc.cache_bake_paused = True
ops = cache_panel_buttons()
ids = [o for o, _ in ops]
check("mc.cache_bake" in ids and "mc.cache_bake_pause" not in ids,
      "panel shows Bake / From Current, not Resume (%s)" % ops)
check(UI.MC_OT_cache_bake_pause.poll(bpy.context) is False,
      "the pause button can't be pressed")

print("\n== a live bake ==", flush=True)
CACHE.bake_heartbeat()
ids = [o for o, _ in cache_panel_buttons()]
check("mc.cache_bake_pause" in ids and "mc.cache_bake" not in ids,
      "panel shows Pause / Resume while the bake is alive")
check(UI.MC_OT_cache_bake_pause.poll(bpy.context) is True, "and it can be pressed")
check(CACHE.clear_stale_bake() == 0 and sc.cache_baking,
      "clear_stale_bake leaves a live bake alone")
CACHE.bake_ended()

print("\n== file saved mid-bake, then reopened ==", flush=True)
# what invoke leaves while baking: flags on, Animated forced on, restore data
ob.MC_props["animated"] = True
ob.MC_props["continuous"] = False
ob[CACHE.RESTORE_KEY] = [0, 1]
sc.cache_baking = True
sc.cache_bake_paused = True
name = ob.name
path = os.path.join(TMP, "midbake.blend")
bpy.ops.wm.save_as_mainfile(filepath=path)
CACHE.bake_heartbeat()          # even with a heartbeat left in this session
bpy.ops.wm.open_mainfile(filepath=path)
scene = bpy.context.scene
sc = scene.MC_props
ob = bpy.data.objects[name]
check(sc.cache_baking is False and sc.cache_bake_paused is False,
      "opening the file cleared the flags")
check(ob.MC_props.animated is False and ob.MC_props.continuous is True,
      "Animated / Continuous back to what they were before the bake (%s, %s)"
      % (ob.MC_props.animated, ob.MC_props.continuous))
check(CACHE.RESTORE_KEY not in ob, "restore data removed")
check(not CACHE.bake_running(), "no bake counted as running")

print("\n== an error inside the bake ends it cleanly ==", flush=True)
bpy.context.view_layer.objects.active = ob


class FakeOp:
    modal = UI.MC_OT_cache_bake.modal
    _modal = UI.MC_OT_cache_bake._modal
    _apply_pause = UI.MC_OT_cache_bake._apply_pause
    _finish = UI.MC_OT_cache_bake._finish

    def report(self, *a, **k):
        pass


class Ctx:
    def __init__(self):
        self.scene = bpy.context.scene
        self.object = ob
        self.window_manager = bpy.context.window_manager
        self.workspace = types.SimpleNamespace(status_text_set=lambda s: None)


op = FakeOp()
op.ob = ob
op.start, op.end, op.cur = 1, 5, 1
op._paused = False
op._timer = None
op._saved = (False, True)
ob[CACHE.RESTORE_KEY] = [0, 1]
ob.MC_props["animated"] = True
sc.cache_baking = True
CACHE.bake_heartbeat()
real = CACHE.save_frame
CACHE.save_frame = lambda *a, **k: 1 / 0
try:
    r = op.modal(Ctx(), types.SimpleNamespace(type='TIMER', value='NOTHING'))
finally:
    CACHE.save_frame = real
check(r == {'CANCELLED'}, "the modal ended (%s)" % (r,))
check(sc.cache_baking is False and not CACHE.bake_running(), "flag and heartbeat cleared")
check(ob.MC_props.animated is False and ob.MC_props.continuous is True
      and CACHE.RESTORE_KEY not in ob, "object settings restored")

print("\n== deleting the cache of a paused bake stops the bake ==", flush=True)


def paused_bake(target):
    o = FakeOp()
    o.ob = target
    o.start, o.end, o.cur = 1, 20, 1
    o._paused = False
    o._timer = None
    o._saved = (False, True)
    target[CACHE.RESTORE_KEY] = [0, 1]
    target.MC_props["cache_playback"] = False
    target.MC_props["animated"] = True
    CACHE.bake_ended()
    CACHE.bake_heartbeat(target.name)
    sc.cache_baking = True
    sc.cache_bake_paused = False
    for _ in range(3):
        o.modal(Ctx(), types.SimpleNamespace(type='TIMER', value='NOTHING'))
    sc.cache_bake_paused = True
    o.modal(Ctx(), types.SimpleNamespace(type='TIMER', value='NOTHING'))
    return o


def delete_cache_of(target):
    bpy.context.view_layer.objects.active = target
    with bpy.context.temp_override(object=target, active_object=target):
        return bpy.ops.mc.cache_delete(use_confirm=False)


# deleting a different object's cache leaves the bake alone
other = ob.copy()
other.data = ob.data.copy()
other.name = "OtherCloth"
bpy.context.scene.collection.objects.link(other)
op = paused_bake(ob)
check(len(CACHE.cached_frames(ob)) == 3 and op._paused, "bake is paused with 3 frames")
delete_cache_of(other)
check(sc.cache_baking and sc.cache_bake_paused and not CACHE.bake_stop_requested(),
      "deleting another object's cache doesn't touch the bake")
r = op.modal(Ctx(), types.SimpleNamespace(type='TIMER', value='NOTHING'))
check(r == {'RUNNING_MODAL'}, "and it stays paused (%s)" % (r,))

delete_cache_of(ob)
check(CACHE.cached_frames(ob) == [], "the cache is gone")
check(sc.cache_baking is False and sc.cache_bake_paused is False,
      "flags cleared at once")
ids = [o for o, _ in cache_panel_buttons()]
check("mc.cache_bake" in ids and "mc.cache_bake_pause" not in ids,
      "panel shows Bake again, not Resume")
r = op.modal(Ctx(), types.SimpleNamespace(type='TIMER', value='NOTHING'))
check(r == {'CANCELLED'}, "the paused bake ends on its next tick (%s)" % (r,))
check(not CACHE.bake_running() and CACHE.cached_frames(ob) == [],
      "no bake left running, and it wrote nothing back")
check(ob.MC_props.animated is False and ob.MC_props.continuous is True
      and CACHE.RESTORE_KEY not in ob, "object settings restored")

shutil.rmtree(TMP, ignore_errors=True)
print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
