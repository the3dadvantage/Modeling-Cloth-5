"""Cache playback while the mesh is open in edit mode.

Playback used to be skipped entirely in edit mode, so the mesh sat on whichever
frame it happened to be on when edit mode started: scrubbing did nothing.  Edit
mode works on a bmesh of its own, so the frame has to be written there.

  * scrubbing in edit mode shows the cached frames
  * what it showed survives leaving edit mode -- the edit bmesh does not
    overwrite it on the way out
  * the solver stays out of it: playback is the cache, not a simulation
  * entering edit mode part-way through playback picks up from the right frame
  * changing the topology in edit mode neither crashes nor switches playback
    off under the user
  * object-mode playback still behaves
"""
import bpy, bmesh, os, sys, glob, shutil, tempfile
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
UI.U.popup_error = lambda *a, **k: print("POPUP:", a[0] if a else "", flush=True)
MC5 = UI.MC5
U = MC5.U
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


OUT = tempfile.mkdtemp(prefix="mc_editmode_")
scene.MC_props.cache_dir = OUT + os.sep

START, END = 1, 12


def build():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=2.0)
    ob = bpy.context.object
    ob.name = "Cloth"
    ob.data.vertices.foreach_set('select', np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    p = ob.MC_props
    p.cloth = True
    p.gravity = -4.0
    return ob


def bake(ob):
    """What the modal bake operator does, without needing a window."""
    p = ob.MC_props
    p["cache_playback"] = False
    p["continuous"] = False
    p["animated"] = True
    MC5.install_handler()
    CACHE.wipe(ob)
    MC5.reset_cloth_ob(ob)
    for fr in range(START, END + 1):
        scene.frame_set(fr)
        CACHE.save_frame(ob, fr, MC5.get_cloth(ob).co)
    CACHE.write_manifest(ob, len(ob.data.vertices), START, END)
    CACHE.free(ob)
    p["animated"] = False


def edit_co(ob):
    """What the user is actually looking at while in edit mode."""
    obm = bmesh.from_edit_mesh(ob.data)
    obm.verts.ensure_lookup_table()
    return np.array([v.co[:] for v in obm.verts], dtype=np.float32)


def key_co(ob, key='MC_current'):
    k = ob.data.shape_keys.key_blocks[key]
    co = np.empty((len(k.data), 3), dtype=np.float32)
    k.data.foreach_get('co', co.ravel())
    return co


cloth = build()
bake(cloth)
frames = CACHE.cached_frames(cloth)
check(len(frames) == END - START + 1, "baked %d frames" % len(frames))
cached = {fr: CACHE.load_frame_disk(cloth, fr) for fr in frames}
spread = float(np.abs(cached[END] - cached[START]).max())
check(spread > 0.05, "the cached frames differ from each other (%.3g)" % spread)

print("\n== scrubbing in edit mode ==", flush=True)
cloth.MC_props.cache_playback = True
scene.frame_set(START)
bpy.ops.object.mode_set(mode='EDIT')
check(cloth.data.is_editmode, "in edit mode")

seen = {}
for fr in (4, 9, 2, END):
    scene.frame_set(fr)
    seen[fr] = edit_co(cloth)
worst = max(float(np.abs(seen[fr] - cached[fr]).max()) for fr in seen)
check(worst < 1e-5, "every frame scrubbed to showed its cached shape (%.3g)" % worst)
check(float(np.abs(seen[4] - seen[9]).max()) > 1e-3,
      "and the frames were genuinely different (not stuck on one)")

print("\n== leaving edit mode keeps it ==", flush=True)
scene.frame_set(7)
in_edit = edit_co(cloth)
bpy.ops.object.mode_set(mode='OBJECT')
check(np.abs(key_co(cloth) - cached[7]).max() < 1e-5,
      "the shape key holds frame 7 after leaving edit mode")
check(np.abs(in_edit - key_co(cloth)).max() < 1e-5,
      "and it matches what was on screen in edit mode")

print("\n== the solver stays out of it ==", flush=True)
bpy.ops.object.mode_set(mode='EDIT')
scene.frame_set(5)
first = edit_co(cloth)
for _ in range(3):
    scene.frame_set(5)          # same frame again
again = edit_co(cloth)
check(np.abs(first - again).max() < 1e-6,
      "re-entering the same frame does not advance anything")
C = MC5.get_cloth(cloth)
check(float(np.abs(C.velocity).max()) == 0.0, "velocity stays at rest during playback")

print("\n== entering edit mode part-way through ==", flush=True)
bpy.ops.object.mode_set(mode='OBJECT')
scene.frame_set(3)
bpy.ops.object.mode_set(mode='EDIT')
scene.frame_set(10)
check(np.abs(edit_co(cloth) - cached[10]).max() < 1e-5,
      "playback carried on into edit mode")

print("\n== editing the topology while playing back ==", flush=True)
obm = bmesh.from_edit_mesh(cloth.data)
obm.edges.ensure_lookup_table()
bmesh.ops.subdivide_edges(obm, edges=[obm.edges[0]], cuts=1, use_grid_fill=True)
bmesh.update_edit_mesh(cloth.data)
n_edit = len(bmesh.from_edit_mesh(cloth.data).verts)
scene.frame_set(6)
check(cloth.MC_props.cache_playback,
      "playback was not switched off under the user mid-edit")
check(len(bmesh.from_edit_mesh(cloth.data).verts) == n_edit,
      "and the edit itself was left alone (%d verts)" % n_edit)
bpy.ops.object.mode_set(mode='OBJECT')

print("\n== object mode still works ==", flush=True)
cloth = build()
bake(cloth)
cached = {fr: CACHE.load_frame_disk(cloth, fr) for fr in CACHE.cached_frames(cloth)}
cloth.MC_props.cache_playback = True
worst = 0.0
for fr in (2, 8, 11):
    scene.frame_set(fr)
    worst = max(worst, float(np.abs(key_co(cloth) - cached[fr]).max()))
check(worst < 1e-5, "object-mode playback unchanged (%.3g)" % worst)

shutil.rmtree(OUT, ignore_errors=True)
print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if ok else 1)
