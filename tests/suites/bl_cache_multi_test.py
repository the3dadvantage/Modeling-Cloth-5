"""Cache with several objects and across file saves / loads.

  * Two cloth objects bake and play back independently.
  * A duplicated cloth (Shift+D copies cache_id) gets its own cache the first
    time it is baked, and never overwrites the original's; until then it shows
    no cache rather than the original's.
  * A reopened file plays back straight away (the frame handler used to be
    lost on load).
  * A cache baked before the file was first saved is moved beside the .blend
    on save (it used to stay in Blender's temp folder and look lost).

The Bake operator is modal, so bake() performs the same steps it does.
"""
import os
import sys
import shutil
import numpy as np
import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("TEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


UI = B.load()
MC5 = B.mc5()
CACHE = MC5.CACHE
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "mc_cache_multi_test")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)


def make(name, x, gravity=-2.0):
    ob = B.grid(8, 1.0, (x, 0, 1.0))
    ob.name = name
    B.make_cloth(ob, gravity=gravity)
    return ob


def bake(ob, start=1, end=10):
    scene = bpy.context.scene
    p = ob.MC_props
    p["cache_playback"] = False
    p["continuous"] = False
    p["animated"] = True
    MC5.install_handler()
    CACHE.wipe(ob)
    MC5.reset_cloth_ob(ob)
    for fr in range(start, end + 1):
        scene.frame_set(fr)
        CACHE.save_frame(ob, fr, MC5.get_cloth(ob).co)
    CACHE.write_manifest(ob, len(ob.data.vertices), start, end)
    CACHE.free(ob)
    p["animated"] = False


def shown(ob):
    k = ob.data.shape_keys.key_blocks["MC_current"]
    co = np.empty(len(k.data) * 3, dtype=np.float32)
    k.data.foreach_get("co", co)
    return co.reshape(-1, 3)


def plays_back(ob, frame):
    want = CACHE.load_frame_disk(ob, frame)
    bpy.context.scene.frame_set(frame)
    return want is not None and np.allclose(shown(ob), want)


scene = bpy.context.scene
scene.MC_props.cache_dir = OUT + os.sep
B.clear()
scene.frame_set(0)

# ---- two objects
a, b = make("ClothA", -1.5), make("ClothB", 1.5)
bake(a)
bake(b)
a.MC_props.cache_playback = True
b.MC_props.cache_playback = True
check("two objects: separate cache folders", a.MC_props.cache_id != b.MC_props.cache_id,
      "(ids %d, %d)" % (a.MC_props.cache_id, b.MC_props.cache_id))
check("two objects: both play back together", plays_back(a, 5) and plays_back(b, 5))

# ---- a duplicate
a.MC_props.cache_playback = False
bpy.ops.object.select_all(action='DESELECT')
a.select_set(True)
bpy.context.view_layer.objects.active = a
bpy.ops.object.duplicate()
dup = bpy.context.object
check("duplicate: shows no cache before its own bake (not the original's)",
      CACHE.cached_frames(dup) == [] and CACHE.info_string(dup) == "cache: empty",
      "(%s)" % CACHE.info_string(dup))
check("duplicate: the original still sees its cache", len(CACHE.cached_frames(a)) == 10)
before = CACHE.load_frame_disk(a, 5).copy()
dup.location.x += 5.0
dup.MC_props.gravity = -6.0
bake(dup)
after = CACHE.load_frame_disk(a, 5)
check("duplicate: baking it leaves the original's cache alone", np.array_equal(before, after))
check("duplicate: got its own folder",
      dup.MC_props.cache_id not in (a.MC_props.cache_id, b.MC_props.cache_id),
      "(ids: original %d, duplicate %d)" % (a.MC_props.cache_id, dup.MC_props.cache_id))
check("duplicate: its cache is its own sim",
      not np.allclose(CACHE.load_frame_disk(dup, 5) - [5.0, 0, 0], before, atol=1e-3))
# and the original re-baked after the duplicate exists keeps its own folder
old_id = a.MC_props.cache_id
bake(a)
check("original re-baked after duplicating keeps its folder", a.MC_props.cache_id == old_id)

# ---- save, reopen, play back
a.MC_props.cache_playback = True
path = os.path.join(OUT, "multi.blend")
bpy.ops.wm.save_as_mainfile(filepath=path)
bpy.ops.wm.open_mainfile(filepath=path)
a = bpy.data.objects["ClothA"]
names = [h.__name__ for h in bpy.app.handlers.frame_change_post]
check("reopened file: frame handler is back", "mc_handler" in names, "(%s)" % names)
check("reopened file: playback works without touching anything", plays_back(a, 7))

# ---- baked unsaved, then saved
bpy.ops.wm.read_homefile(use_empty=True)
scene = bpy.context.scene
scene.MC_props.cache_dir = "//mc_cache/"
check("unsaved file: cache reported as temporary", CACHE.is_temporary(scene))
c = make("ClothC", 0.0)
bake(c)
n_temp = len(CACHE.cached_frames(c))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "later.blend"))
n_saved = len(CACHE.cached_frames(c))
check("unsaved file: after saving, the cache is found beside the .blend",
      n_temp == 10 and n_saved == 10 and not CACHE.is_temporary(scene),
      "(%d frames before saving, %d after, now in %s)" % (n_temp, n_saved, CACHE.cache_root(scene)))
check("relative cache folder kept as '//' (Blender 5 path option)",
      scene.MC_props.cache_dir.startswith("//"), "(%s)" % scene.MC_props.cache_dir)

print("================ %s ================"
      % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
