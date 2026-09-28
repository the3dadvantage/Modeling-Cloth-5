"""Cloth animation cache.

Bake a slow simulation to disk once, then scrub / play it back / render from the
cached vertex positions instead of re-simulating.

On disk (per cloth object):

    <cache_dir>/mc_<id>/frame_<000042>.npy      # (vc, 3) float32, object-local space
    <cache_dir>/mc_<id>/frame_<000042>.npy.bak  # previous version, kept by "Update This Frame"
    <cache_dir>/mc_<id>/manifest.json           # {vc, start, end, name, created}

Nothing here writes Blender data during a panel draw: id assignment only happens
on an explicit bake (create=True).  Frame lookup is a pure function of
scene.frame_current so playing backwards and scrubbing just work.
"""

import bpy
import bmesh
import numpy as np
import os
import glob
import json
import time
import bisect
import shutil

try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U


FRAME_GLOB = "frame_*.npy"
BAK_GLOB = "frame_*.npy.bak"
MANIFEST = "manifest.json"

# cache_id -> {"frames": {frame:int -> (vc,3) float32}, "vc": int,
#              "start": int, "end": int, "dir": str}
MEM = {}


# --------------------------------------------------------------------- bake state
# Whether a bake is really running.  The scene's cache_baking flag is a saved
# property, so it could outlive the bake -- a file saved mid-bake, a bake that
# died on an error, scripts reloaded underneath it -- and the panel then showed
# only Pause / Resume, wired to nothing.  The running bake refreshes this
# heartbeat on every tick (paused or not).  It lives in driver_namespace:
# never saved, and it survives the scripts being reloaded.
_BAKE_KEY = "MC_cache_bake_heartbeat"
BAKE_STALE = 5.0            # seconds without a tick before a bake counts as gone
RESTORE_KEY = "mc_bake_restore"   # on the object: [animated, continuous] before the bake


def _bake_state():
    s = bpy.app.driver_namespace.get(_BAKE_KEY)
    return s if isinstance(s, dict) else None


def bake_heartbeat(ob_name=None):
    s = _bake_state()
    if s is None:
        s = {"ob": None, "stop": False}
        bpy.app.driver_namespace[_BAKE_KEY] = s
    if ob_name is not None:
        s["ob"] = ob_name
    s["t"] = time.monotonic()


def bake_ended():
    bpy.app.driver_namespace.pop(_BAKE_KEY, None)


def bake_running():
    s = _bake_state()
    return s is not None and (time.monotonic() - s["t"]) < BAKE_STALE


def baking_object():
    """Name of the object the live bake is writing, or None."""
    return _bake_state()["ob"] if bake_running() else None


def stop_bake(ob=None):
    """Ask the running bake to stop (only if it is baking `ob`, when given).
    The modal ends itself on its next tick; the scene flags are cleared now so
    the panel shows the Bake buttons straight away.  Returns True if a bake
    was asked to stop."""
    if not bake_running():
        return False
    s = _bake_state()
    if ob is not None and s["ob"] != ob.name:
        return False
    s["stop"] = True
    for sc in bpy.data.scenes:
        sc.MC_props["cache_baking"] = False
        sc.MC_props["cache_bake_paused"] = False
    return True


def bake_stop_requested():
    s = _bake_state()
    return bool(s and s.get("stop"))


def clear_stale_bake():
    """Undo what an interrupted bake left behind: the scenes' baking / paused
    flags, and the Animated / Continuous settings it overrides on the object
    it was baking.  Does nothing while a bake is actually running."""
    if bake_running():
        return 0
    n = 0
    for sc in bpy.data.scenes:
        p = sc.MC_props
        if p.cache_baking or p.cache_bake_paused:
            p["cache_baking"] = False
            p["cache_bake_paused"] = False
            n += 1
    for ob in bpy.data.objects:
        saved = ob.get(RESTORE_KEY)
        if saved is not None:
            # bracket-assign: don't fire cb_cloth and rebuild the cloth
            ob.MC_props["animated"] = bool(saved[0])
            ob.MC_props["continuous"] = bool(saved[1])
            del ob[RESTORE_KEY]
            n += 1
    return n


# --------------------------------------------------------------------- paths
def temp_root():
    """Where caches go while the .blend has never been saved.  Blender's
    session temp folder -- deleted when Blender quits."""
    return os.path.join(bpy.app.tempdir, "mc_cache")


def is_temporary(scene=None):
    """True when caches are being written to the session temp folder."""
    return os.path.abspath(cache_root(scene)) == os.path.abspath(temp_root())


@bpy.app.handlers.persistent
def on_save_post(*_args):
    """Move caches baked before the .blend was first saved next to it.

    Until the file is saved, a '//' cache folder can't resolve, so the cache
    goes to Blender's session temp folder -- which is deleted on quit.  After
    the first save the same setting points beside the .blend, where nothing had
    been written, so the bake looked lost.  Existing caches at the new place
    are never overwritten.
    """
    try:
        scene = bpy.context.scene
        src_root = temp_root()
        if not os.path.isdir(src_root) or is_temporary(scene):
            return
        dst_root = cache_root(scene)
        moved = 0
        for name in os.listdir(src_root):
            src = os.path.join(src_root, name)
            dst = os.path.join(dst_root, name)
            if not (name.startswith("mc_") and os.path.isdir(src)) or os.path.exists(dst):
                continue
            os.makedirs(dst_root, exist_ok=True)
            shutil.move(src, dst)
            moved += 1
        if moved:
            MEM.clear()
            print("MC cache: moved %d cache folder(s) from the temp folder to %s" % (moved, dst_root))
    except Exception as e:                                  # noqa: BLE001
        print("MC cache: could not move the temporary cache:", e)


def cache_root(scene=None):
    """Absolute folder that holds every object's cache subfolder."""
    scene = scene or bpy.context.scene
    raw = (scene.MC_props.cache_dir or "//mc_cache/").strip()
    path = bpy.path.abspath(raw) if raw else ""
    if not path or (raw.startswith("//") and not bpy.data.filepath):
        # .blend not saved yet -> "//" can't resolve, fall back to temp
        path = os.path.join(bpy.app.tempdir, "mc_cache")
    return path


def _owner_name(cid, scene=None):
    """Name of the object a cache folder was baked for (from its manifest)."""
    try:
        with open(os.path.join(cache_root(scene), "mc_%04d" % cid, MANIFEST), "r") as f:
            return json.load(f).get("name")
    except Exception:
        return None


def _obj_id(ob, assign=False, scene=None):
    """Stable per-object cache id.  Returns 0 (a folder that never exists) when
    the object has no usable id and assign is False, so callers in a draw() are
    safe.

    Duplicating an object (Shift+D) copies every property, cache_id included,
    so the copy used to share the original's folder: baking the copy
    overwrote the original's cache.  When another object has the same id, the
    one named in the folder's manifest owns it; the other is treated as having
    no cache and gets an id of its own the first time it is baked.
    """
    cid = ob.MC_props.cache_id
    if cid > 0:
        sharing = [o for o in bpy.data.objects
                   if o is not ob and o.MC_props.cache_id == cid]
        if not sharing:
            return cid
        owner = _owner_name(cid, scene)
        names = [o.name for o in sharing]
        if owner == ob.name:
            return cid
        if owner not in names and ob.name < min(names):
            # the manifest names neither (nothing baked yet, or the owner was
            # renamed): the first by name keeps the id
            return cid
        if not assign:
            return 0
    elif not assign:
        return 0
    ids = [o.MC_props.cache_id for o in bpy.data.objects]
    cid = max(max(ids) + 1, 1) if ids else 1
    ob.MC_props.cache_id = cid
    return cid


def object_dir(ob, create=False, scene=None):
    cid = _obj_id(ob, assign=create, scene=scene)
    path = os.path.join(cache_root(scene), "mc_%04d" % cid)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _frame_name(frame):
    return "frame_%06d.npy" % int(frame)


def _frame_path(ob, frame, scene=None, create=False):
    return os.path.join(object_dir(ob, create=create, scene=scene), _frame_name(frame))


# --------------------------------------------------------------------- manifest
def read_manifest(ob, scene=None):
    try:
        with open(os.path.join(object_dir(ob, scene=scene), MANIFEST), "r") as f:
            return json.load(f)
    except Exception:
        return None


def write_manifest(ob, vc, start, end, scene=None):
    d = object_dir(ob, create=True, scene=scene)
    data = {"vc": int(vc), "start": int(start), "end": int(end),
            "name": ob.name, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(os.path.join(d, MANIFEST), "w") as f:
            json.dump(data, f, indent=1)
    except Exception as e:
        print("MC cache: manifest write failed:", e)
    return data


# --------------------------------------------------------------------- listing
def cached_frames(ob, scene=None):
    """Sorted list of the integer frame numbers present on disk."""
    out = []
    for p in glob.glob(os.path.join(object_dir(ob, scene=scene), FRAME_GLOB)):
        base = os.path.basename(p)[len("frame_"):-len(".npy")]
        try:
            out.append(int(base))
        except ValueError:
            pass
    out.sort()
    return out


def cache_stats(ob, scene=None):
    """(frame_count, size_in_MB) for the frame files on disk."""
    n = 0
    total = 0
    for p in glob.glob(os.path.join(object_dir(ob, scene=scene), FRAME_GLOB)):
        n += 1
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return n, total / (1024.0 * 1024.0)


def info_string(ob, scene=None):
    """One-line status for the panel.  Pure read, safe to call from draw()."""
    try:
        frames = cached_frames(ob, scene=scene)
    except Exception:
        return "cache: n/a"
    if not frames:
        return "cache: empty"
    n, mb = cache_stats(ob, scene=scene)
    contiguous = (frames[-1] - frames[0] + 1) == len(frames)
    tag = "" if contiguous else "  (gaps)"
    return "cache: %d fr  %d–%d%s  %.1f MB" % (n, frames[0], frames[-1], tag, mb)


# --------------------------------------------------------------------- io
def save_frame(ob, frame, co, scene=None):
    d = object_dir(ob, create=True, scene=scene)
    if not os.path.isfile(os.path.join(d, MANIFEST)):
        # claim the folder from the first frame on, so a copy sharing the id
        # can tell whose it is even if the bake is stopped early
        write_manifest(ob, len(co), frame, frame, scene=scene)
    arr = np.ascontiguousarray(co, dtype=np.float32)
    np.save(os.path.join(d, _frame_name(frame)), arr)
    m = MEM.get(_obj_id(ob))
    if m is not None:
        m["frames"][int(frame)] = arr.copy()


def overwrite_frame(ob, frame, co, backup=True, scene=None):
    """Replace one cached frame with `co`, keeping the old file as .bak."""
    d = object_dir(ob, create=True, scene=scene)
    path = os.path.join(d, _frame_name(frame))
    if backup and os.path.isfile(path):
        try:
            if os.path.isfile(path + ".bak"):
                os.remove(path + ".bak")
            os.replace(path, path + ".bak")
        except OSError:
            pass
    np.save(path, np.ascontiguousarray(co, dtype=np.float32))
    m = MEM.get(_obj_id(ob))
    if m is not None:
        m["frames"][int(frame)] = np.ascontiguousarray(co, dtype=np.float32)


def load_frame_disk(ob, frame, scene=None):
    p = _frame_path(ob, frame, scene=scene)
    if not os.path.isfile(p):
        return None
    try:
        return np.load(p)
    except Exception as e:
        print("MC cache: load failed", p, e)
        return None


# --------------------------------------------------------------------- in memory
def preload(ob, scene=None):
    cid = _obj_id(ob, assign=True)
    frames = cached_frames(ob, scene=scene)
    m = {"frames": {}, "dir": object_dir(ob, scene=scene)}
    for fr in frames:
        a = load_frame_disk(ob, fr, scene=scene)
        if a is not None:
            m["frames"][fr] = a
    man = read_manifest(ob, scene=scene) or {}
    first_vc = m["frames"][frames[0]].shape[0] if frames else 0
    m["vc"] = int(man.get("vc", first_vc))
    m["start"] = int(man.get("start", frames[0] if frames else 0))
    m["end"] = int(man.get("end", frames[-1] if frames else 0))
    MEM[cid] = m
    return m


def free(ob):
    MEM.pop(_obj_id(ob), None)


# --------------------------------------------------------------------- guards
def vcount_matches(ob, scene=None, vc=None):
    """(ok, cached_vc).  ok is None when there is no cache at all.

    `vc` overrides the mesh's own count, for edit mode: ob.data is a topology
    behind whatever the user is editing."""
    man = read_manifest(ob, scene=scene)
    if man is not None:
        cached_vc = int(man.get("vc", 0))
    else:
        frames = cached_frames(ob, scene=scene)
        if not frames:
            return None, 0
        a = load_frame_disk(ob, frames[0], scene=scene)
        cached_vc = 0 if a is None else a.shape[0]
    if vc is None:
        vc = len(ob.data.vertices)
    return (cached_vc == vc), cached_vc


# --------------------------------------------------------------------- lookup
def get_playback_co(ob, frame, scene=None):
    """(vc,3) float32 for `frame`, or None if the cache is empty.

    Out of the cached range -> clamp to the nearest end (HOLD).
    In range but that exact frame is missing -> nearest cached frame <= frame.
    """
    m = MEM.get(_obj_id(ob))
    frames_map = m["frames"] if m is not None else None

    def _load(fr):
        if frames_map is not None and fr in frames_map:
            return frames_map[fr]
        return load_frame_disk(ob, fr, scene=scene)

    avail = sorted(frames_map.keys()) if frames_map is not None \
        else cached_frames(ob, scene=scene)
    if not avail:
        return None

    lo, hi = avail[0], avail[-1]
    if frame <= lo:
        return _load(lo)
    if frame >= hi:
        return _load(hi)
    exact = _load(frame)
    if exact is not None:
        return exact
    i = bisect.bisect_right(avail, frame) - 1
    return _load(avail[i])


# --------------------------------------------------------------------- resume seed
def seed_resume(C, frame, scene=None):
    """Seed C.co / C.start_co / C.vel_start / C.velocity from cached frames so a
    live sim can carry on from `frame` as if it had never stopped.

    Velocity is a finite difference of cached positions (MC5 convention:
    velocity == co - previous_co):
        frame-2 cached -> 2nd-order backward  1.5 c0 - 2 c(-1) + 0.5 c(-2)
        frame-1 cached -> 1st-order           c0 - c(-1)
        neither        -> zero
    Returns True when at least the `frame` file was loaded.
    """
    ob = C.ob
    c0 = load_frame_disk(ob, frame, scene=scene)
    if c0 is None or c0.shape[0] != C.vc:
        return False
    cm1 = load_frame_disk(ob, frame - 1, scene=scene)
    cm2 = load_frame_disk(ob, frame - 2, scene=scene)

    C.co[:] = c0
    if cm1 is not None and cm1.shape == c0.shape:
        if cm2 is not None and cm2.shape == c0.shape:
            vel = 1.5 * c0 - 2.0 * cm1 + 0.5 * cm2
        else:
            vel = c0 - cm1
        C.start_co[:] = cm1
    else:
        vel = np.zeros_like(c0)
        C.start_co[:] = c0
    C.velocity[:] = vel.astype(C.velocity.dtype)
    C.vel_start[:] = c0
    return True


# --------------------------------------------------------------------- delete
def wipe(ob, scene=None):
    """Delete every cached frame (and .bak) but keep the folder / manifest."""
    d = object_dir(ob, scene=scene)
    for pat in (FRAME_GLOB, BAK_GLOB):
        for p in glob.glob(os.path.join(d, pat)):
            try:
                os.remove(p)
            except OSError:
                pass
    free(ob)


def wipe_all(ob, scene=None):
    """Delete the object's whole cache subfolder.

    Only ever removes files matching our own patterns and then the (now empty)
    folder, and only when that folder actually sits under the configured cache
    root -- a misconfigured cache_dir can't take anything else down with it.
    """
    d = os.path.abspath(object_dir(ob, scene=scene))
    root = os.path.abspath(cache_root(scene))
    for pat in (FRAME_GLOB, BAK_GLOB, MANIFEST):
        for p in glob.glob(os.path.join(d, pat)):
            try:
                os.remove(p)
            except OSError:
                pass
    if d != root and d.startswith(root + os.sep):
        try:
            if not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass
    free(ob)


def delete_frame(ob, frame, forward=False, scene=None):
    """Delete frame `frame` (and every cached frame after it when forward).
    Returns the number of files removed."""
    d = object_dir(ob, scene=scene)
    frame = int(frame)
    if forward:
        targets = [f for f in cached_frames(ob, scene=scene) if f >= frame]
    else:
        targets = [frame]

    m = MEM.get(_obj_id(ob))
    n = 0
    for f in targets:
        p = os.path.join(d, _frame_name(f))
        if os.path.isfile(p):
            try:
                os.remove(p)
                n += 1
            except OSError:
                pass
        if m is not None:
            m["frames"].pop(f, None)

    if n:
        rem = cached_frames(ob, scene=scene)
        man = read_manifest(ob, scene=scene) or {}
        if rem:
            write_manifest(ob, man.get("vc", len(ob.data.vertices)),
                           man.get("start", rem[0]), rem[-1], scene=scene)
    return n


# --------------------------------------------------------------------- playback
def playback_step(C):
    """Called from MC5.physics when cache_playback is on.  Loads the cached
    frame for scene.frame_current into the cloth and returns True so the caller
    skips the solver.  Turns cache_playback off (and says why) on any problem."""
    ob = C.ob
    p = ob.MC_props

    scene = bpy.context.scene
    frame = scene.frame_current

    # In edit mode ob.data is a topology behind what the user is editing, so
    # the counts come from the edit bmesh.  Deliberately NOT
    # ob.update_from_editmode(): pushing the edit mesh back into the datablock
    # from inside a frame-change handler crashes Blender outright once the
    # topology has changed.
    obm = bmesh.from_edit_mesh(ob.data) if ob.data.is_editmode else None
    live_vc = len(obm.verts) if obm is not None else len(ob.data.vertices)

    ok, cvc = vcount_matches(ob, scene, vc=live_vc)
    if ok is None:
        if obm is not None:
            return True
        U.popup_error("No cache for '%s' yet – bake first." % ob.name, icon='INFO')
        p["cache_playback"] = False
        return True
    if not ok:
        if obm is not None:
            # part-way through an edit the counts go in and out of step: show
            # nothing rather than switching playback off under the user.  It is
            # settled when they leave edit mode and the cloth is rebuilt.
            return True
        U.popup_error("Cache has %d verts, mesh has %d – playback off for '%s'."
                      % (cvc, live_vc, ob.name), icon='ERROR')
        p["cache_playback"] = False
        return True

    cid = _obj_id(ob)
    m = MEM.get(cid)
    if m is not None and m.get("dir") != object_dir(ob, scene=scene):
        # loaded from another folder (the cache folder setting changed, or a
        # temp cache moved on save): stale
        MEM.pop(cid, None)
    if p.cache_in_memory and cid > 0 and cid not in MEM:
        preload(ob, scene)

    co = get_playback_co(ob, frame, scene)
    if co is None or co.shape[0] != C.vc:
        if obm is not None:
            return True
        p["cache_playback"] = False
        return True

    C.co[:] = co
    C.velocity[:] = 0.0
    C.start_co[:] = co
    if obm is not None:
        write_edit_mode(ob, co, obm)
        return True
    keys = ob.data.shape_keys.key_blocks
    keys['MC_current'].data.foreach_set('co', np.ascontiguousarray(co, np.float32).ravel())
    ob.data.update()
    return True


def write_edit_mode(ob, co, obm=None):
    """Show `co` on a mesh that is open in edit mode.

    Edit mode works on a bmesh of its own.  Writing the shape key datablock
    there does nothing visible, and is then overwritten by that bmesh when the
    user leaves -- which is why playback used to be skipped in edit mode
    altogether, leaving the mesh stuck on whichever frame it was on when edit
    mode started.  Writing the bmesh instead shows the cached frame, and the
    write survives leaving edit mode.

    Returns False when the mesh no longer matches the cache, which happens
    while the user is part-way through adding geometry.
    """
    if obm is None:
        obm = bmesh.from_edit_mesh(ob.data)
    obm.verts.ensure_lookup_table()
    if len(obm.verts) != co.shape[0]:
        return False

    layer = obm.verts.layers.shape.get('MC_current')
    key = ob.active_shape_key
    # a vert's own co is whichever shape key is active, so it is only ours to
    # write when that key is MC_current
    live = key is not None and key.name == 'MC_current'
    for i, v in enumerate(obm.verts):
        pt = co[i]
        if layer is not None:
            v[layer] = pt
        if live:
            v.co = pt
    bmesh.update_edit_mesh(ob.data, loop_triangles=False, destructive=False)
    return True
