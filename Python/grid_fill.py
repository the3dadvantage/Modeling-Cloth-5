"""Grid fill -- Blender glue.

Reads boundary loops off a source object, builds a filled patch for each, and
writes the result into its own object so the source loop stays editable.  All
the geometry lives in grid_geo.py; this module only does Blender.

Pipeline per loop:

    3D loop -> flatten (project / unroll / uv) -> resample evenly
            -> lattice or poisson fill -> lift to 3D
            -> pin the border to its true 3D positions -> relax the interior

The relax at the end is what makes a non-planar border work: the interior is
placed on a flat plane first, the border is snapped back to where it really is,
and smoothing with the border locked pulls the interior onto the real shape.

Smoothing state lives in two mesh attributes on the RESULT object rather than a
cached bmesh, so changing the iteration count always recomputes from the
unsmoothed positions and nothing goes stale across undo or a file reload.
"""

import bpy
import bmesh
import numpy as np

try:
    U = bpy.data.texts['utils.py'].as_module()
    GEO = bpy.data.texts['grid_geo.py'].as_module()
except Exception:
    from . import utils as U
    from . import grid_geo as GEO


ATTR_FLAT = "MC_grid_flat"      # FLOAT_VECTOR: position before smoothing
ATTR_LOCK = "MC_grid_locked"    # INT: 1 for border verts
PROP_SRC = "MC_grid_source"
PROP_LOOP = "MC_grid_loop"


# ------------------------------------------------------------ source reading
def source_loops(ob, selected_only=False, refresh=True):
    """Ordered vertex-index loops from the object's boundary edges.

    refresh=False when the caller has already pushed edit-mode changes into
    ob.data -- doing it here as well is harmless but the caller must have done
    it before reading coordinates.
    """
    if refresh and ob.data.is_editmode:
        ob.update_from_editmode()
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()

    edges = [e for e in bm.edges if len(e.link_faces) < 2]
    if selected_only:
        sel = [e for e in edges if e.select]
        if sel:
            edges = sel

    adj = {}
    for e in edges:
        a, b = e.verts[0].index, e.verts[1].index
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops = []
    seen = set()
    for start in sorted(adj):
        if start in seen or len(adj[start]) != 2:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = None
            for n in adj[cur]:
                if n != prev and n not in seen:
                    nxt = n
                    break
            if nxt is None:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    bm.free()
    return loops


def nest_loops(loops, co):
    """Group loops into (outer, [holes]) by plane agreement and containment."""
    if len(loops) < 2:
        return [(l, []) for l in loops]

    info = []
    for l in loops:
        c = co[l]
        o, b = GEO.best_fit_plane(c)
        flat = GEO.to_plane(c, o, b)
        info.append((o, b, flat, abs(GEO.signed_area(flat))))

    order = sorted(range(len(loops)), key=lambda i: -info[i][3])
    parent = {}
    for k, i in enumerate(order):
        for j in order[:k]:
            if j in parent:
                continue
            oj, bj, _, _ = info[j]
            if abs(float(bj[2] @ info[i][1][2])) < 0.94:      # planes disagree
                continue
            probe = GEO.to_plane(co[loops[i]][:1], oj, bj)
            if GEO.winding_number(probe, info[j][2])[0] != 0:
                parent[i] = j
                break

    out = []
    for i in order:
        if i in parent:
            continue
        out.append((loops[i], [loops[k] for k, p in parent.items() if p == i]))
    return out


# ---------------------------------------------------------------- patch build
def build_patch(co3, hole_co3, s):
    """(verts3, faces, locked) for one loop.  s is a settings dict."""
    origin, basis = GEO.best_fit_plane(co3)

    flat2, method = GEO.flatten_loop(co3, s["flatten"], s["planarity_limit"])
    proj2 = GEO.to_plane(co3, origin, basis)
    lift = GEO.procrustes_2d(flat2, proj2) if method != 'project' else (lambda p: p)

    holes2 = [GEO.to_plane(h, origin, basis) for h in hole_co3]

    # Scale multiplies the cell size, so the border has to be resampled at the
    # same effective size -- otherwise the border keeps its original density
    # while the lattice coarsens, and the stitched band has to fan a fine
    # border into a coarse interior.
    cell = max(s["spacing"] * s["scale"], 1e-5)

    border2 = GEO.resample_loop(flat2, cell, s["angle_limit"])
    holes_rs = [GEO.resample_loop(h, cell, s["angle_limit"]) for h in holes2]

    if s["fill"] == 'TRIS_POISSON':
        pts2, faces, n_border, info = GEO.poisson_patch(
            border2, holes_rs, spacing=cell, seed=s["seed"],
            relax_iters=s["relax_iters"], max_edge=s["max_edge"],
            qhull_options=s["qhull"] or None)
    else:
        pts2, faces, n_border, info = GEO.lattice_patch(
            border2, holes_rs, spacing=cell, rot_deg=s["rotate"],
            scale=1.0, tris=(s["fill"] == 'TRIS_GRID'),
            inset=s["inset"], qhull_options=s["qhull"] or None,
            phase=s["offset"])

    info["cell"] = cell
    info["border_verts"] = len(border2)

    # to 3D: everything onto the plane, then pin the border where it really is
    verts3 = GEO.from_plane(lift(pts2), origin, basis)
    locked = np.zeros(len(verts3), dtype=bool)
    locked[:n_border] = True

    nb = len(border2)
    verts3[:nb] = _resample_3d(co3, flat2, border2)
    cur = nb
    for h3, h2, hr in zip(hole_co3, holes2, holes_rs):
        verts3[cur:cur + len(hr)] = _resample_3d(h3, h2, hr)
        cur += len(hr)

    info["method"] = method
    return verts3, faces, locked, info


def _resample_3d(co3, flat2, resampled2):
    """Where each resampled 2D border point sits on the original 3D loop.

    The resampled points lie on the flattened polyline, so find the segment
    each one is on and carry its parameter straight across to the matching 3D
    segment.
    """
    if len(flat2) < 2:
        return np.repeat(co3[:1], len(resampled2), axis=0)
    j, t = GEO.project_to_polyline(resampled2, flat2)
    nxt = (j + 1) % len(flat2)
    return co3[j] + (co3[nxt] - co3[j]) * t[:, None]


# ------------------------------------------------------------- result objects
def result_name(src, i):
    return "%s_grid_%d" % (src.name, i)


def find_results(src):
    return [o for o in bpy.data.objects if o.get(PROP_SRC) == src.name]


def source_of(ob):
    """The object this patch was built from, or None.

    PROP_SRC holds a name rather than a pointer so the link survives a reload,
    which means it can dangle: the object may never have had one, or the source
    may since have been deleted.  Both read as "no source" here.
    """
    if ob is None:
        return None
    name = ob.get(PROP_SRC)
    if not isinstance(name, str):
        return None
    return bpy.data.objects.get(name)


def ensure_result(src, i, verts, faces):
    """Create or refresh the result object, geometry included.

    New objects go through U.ob_from_py_data.  An existing one keeps its mesh
    datablock and just has the geometry swapped -- that avoids churning a mesh
    datablock on every Live rebuild and keeps material slots and anything else
    hanging off the mesh intact.
    """
    name = result_name(src, i)
    vlist = [tuple(map(float, v)) for v in verts]
    flist = [list(map(int, f)) for f in faces]

    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'MESH':
        # note the argument order here: (co, faces, edges)
        ob = U.ob_from_py_data(vlist, flist, edges=[], name=name)
    else:
        me = ob.data
        me.clear_geometry()
        me.from_pydata(vlist, [], flist)
    ob.data.update()

    # keep the result alongside its source rather than in whatever collection
    # happened to be active
    target = src.users_collection[0] if src.users_collection else None
    if target is not None and target not in list(ob.users_collection):
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        target.objects.link(ob)

    ob[PROP_SRC] = src.name
    ob[PROP_LOOP] = i
    ob.matrix_world = src.matrix_world.copy()
    return ob


def write_attrs(ob, verts, locked):
    """Store the unsmoothed positions and the locked mask.

    This is the whole of the smoothing state -- no cached bmesh, nothing keyed
    by object name in a module global.
    """
    me = ob.data
    for name, kind in ((ATTR_FLAT, 'FLOAT_VECTOR'), (ATTR_LOCK, 'INT')):
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])
        me.attributes.new(name, kind, 'POINT')
    me.attributes[ATTR_FLAT].data.foreach_set(
        'vector', np.ascontiguousarray(verts, dtype=np.float32).ravel())
    me.attributes[ATTR_LOCK].data.foreach_set(
        'value', np.ascontiguousarray(locked, dtype=np.int32))
    me.update()


def read_patch(ob):
    """(flat_co, locked, faces) or None if this isn't a grid-fill result."""
    me = ob.data
    if ATTR_FLAT not in me.attributes or ATTR_LOCK not in me.attributes:
        return None
    n = len(me.vertices)
    flat = np.empty(n * 3, dtype=np.float32)
    me.attributes[ATTR_FLAT].data.foreach_get('vector', flat)
    lock = np.empty(n, dtype=np.int32)
    me.attributes[ATTR_LOCK].data.foreach_get('value', lock)
    faces = [list(p.vertices) for p in me.polygons]
    return flat.reshape(n, 3).astype(np.float64), lock.astype(bool), faces


def apply_smooth(ob, iters, factor=0.5):
    """Recompute from the stored unsmoothed positions -- never accumulates."""
    got = read_patch(ob)
    if got is None:
        return False
    flat, locked, faces = got
    co = GEO.laplacian_smooth(flat, faces, locked, iters=iters, factor=factor)
    ob.data.vertices.foreach_set('co', np.ascontiguousarray(co, np.float32).ravel())
    ob.data.update()
    return True


# ------------------------------------------------------------------ settings
def settings_from(ob):
    p = ob.MC_props
    return {
        "fill": p.gf_fill_type,
        "spacing": max(float(p.gf_spacing), 1e-5),
        "angle_limit": float(p.gf_angle_limit),
        "flatten": p.gf_flatten.lower(),
        "planarity_limit": float(p.gf_planarity_limit),
        "rotate": float(p.gf_rotate),
        "scale": float(p.gf_scale),
        "offset": tuple(float(x) for x in p.gf_offset),
        "inset": float(p.gf_inset),
        "seed": int(p.gf_seed),
        "relax_iters": int(p.gf_relax_iters),
        "max_edge": float(p.gf_max_edge),
        "qhull": p.gf_qhull.strip(),
        "smooth": int(p.gf_smooth_iters),
        "holes": bool(p.gf_holes),
        "selected_only": bool(p.gf_selected_only),
    }


def run(ob, report=None):
    """Build every patch for `ob`.  Returns a list of info dicts."""
    # Push the edit-mode bmesh into ob.data FIRST.  Everything below reads
    # ob.data, so doing this later means the coordinates come from the stale
    # mesh while the topology comes from the fresh one -- the first run after an
    # edit silently uses old positions, and if verts were added the indices run
    # past the end of the coordinate array.
    if ob.data.is_editmode:
        ob.update_from_editmode()

    s = settings_from(ob)
    co = U.get_co(ob).astype(np.float64)
    loops = source_loops(ob, s["selected_only"], refresh=False)
    if not loops:
        return []

    n = len(co)
    for l in loops:
        if l.max() >= n:
            raise RuntimeError("mesh changed mid-build (%d verts, index %d)"
                               % (n, int(l.max())))

    groups = nest_loops(loops, co) if s["holes"] else [(l, []) for l in loops]

    out = []
    for i, (outer, holes) in enumerate(groups):
        verts, faces, locked, info = build_patch(
            co[outer], [co[h] for h in holes], s)
        res = ensure_result(ob, i, verts, faces)
        write_attrs(res, verts, locked)
        if s["smooth"]:
            apply_smooth(res, s["smooth"])
        info["object"] = res.name
        info["holes"] = len(holes)
        info["faces"] = len(faces)
        out.append(info)

    # drop results left over from loops that no longer exist
    for extra in find_results(ob):
        if extra.get(PROP_LOOP, 0) >= len(groups):
            me = extra.data
            bpy.data.objects.remove(extra)
            if me.users == 0:
                bpy.data.meshes.remove(me)
    return out


_PENDING = set()
_LAST_ERROR = {}        # object name -> message from the most recent Live build


def _deferred_rebuild():
    """Run queued rebuilds outside the property callback.

    run() creates, links and removes objects.  Doing that directly from an
    update callback is not safe in Blender -- it can leave the depsgraph in a
    bad state or crash -- so the callback only queues a name and a one-shot
    timer does the work.  Queuing also coalesces a slider drag into a single
    rebuild instead of one per frame.
    """
    names = list(_PENDING)
    _PENDING.clear()
    # names are not references, so a deleted object leaves its message behind;
    # a new object reusing the name would inherit it
    for gone in [k for k in _LAST_ERROR if k not in bpy.data.objects]:
        _LAST_ERROR.pop(gone, None)
    for n in names:
        ob = bpy.data.objects.get(n)
        if ob is None:
            continue
        try:
            run(ob)
            _LAST_ERROR.pop(n, None)
        except Exception as e:
            # a Live rebuild has no operator to report through, so stash the
            # message for the panel -- otherwise it only reaches the console and
            # the tool just looks inert
            _LAST_ERROR[n] = str(e)
            print("grid fill:", e)
    return None                     # returning None unregisters the timer


def cb_live(self, context):
    """Property update: queue a rebuild, but only when Live is on."""
    if not getattr(self, "gf_live", False):
        return
    ob = self.id_data
    if ob is None or ob.type != 'MESH':
        return
    if ob.get(PROP_SRC):
        return                      # this is a generated patch, not a source
    _PENDING.add(ob.name)
    if not bpy.app.timers.is_registered(_deferred_rebuild):
        bpy.app.timers.register(_deferred_rebuild, first_interval=0.0)


def cb_smooth_live(self, context):
    """Smoothing only rewrites vertex positions on meshes that already exist,
    so unlike a rebuild it is safe to do straight from the callback."""
    if not getattr(self, "gf_live", False):
        return
    ob = self.id_data
    if ob is None or ob.type != 'MESH':
        return
    targets = find_results(ob) or ([ob] if read_patch(ob) else [])
    for res in targets:
        try:
            apply_smooth(res, int(self.gf_smooth_iters))
        except Exception as e:
            print("grid smooth:", e)


# ----------------------------------------------------------------- operators
class MC_OT_grid_fill(bpy.types.Operator):
    """Fill each boundary loop with a grid, into its own object"""
    bl_idname = "mc.grid_fill"
    bl_label = "Grid Fill"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.object
        if ob.get(PROP_SRC):
            # filling a generated patch would use its own boundary and spawn
            # <name>_grid_0 off it
            U.popup_error("'%s' is a generated patch. Select its source '%s' "
                          "to rebuild." % (ob.name, ob[PROP_SRC]), icon='INFO')
            return {'CANCELLED'}
        try:
            infos = run(ob)
        except Exception as e:
            U.popup_error("Grid fill failed: %s" % e, icon='ERROR')
            return {'CANCELLED'}

        if not infos:
            U.popup_error("No boundary loops found. The mesh needs open edges.",
                          icon='INFO')
            return {'CANCELLED'}

        gaps = sum(i.get("border_gaps", 0) for i in infos)
        parts = []
        for i in infos:
            bit = "%s: %d faces [%s]" % (i["object"], i["faces"], i["method"])
            if i["holes"]:
                bit += " +%d hole(s)" % i["holes"]
            parts.append(bit)
        msg = "%d patch(es)  |  %s" % (len(infos), "   ".join(parts))
        if gaps:
            msg += "  |  %d border gap(s) - the fill cut a corner" % gaps
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MC_OT_grid_smooth(bpy.types.Operator):
    """Re-apply smoothing to this object's grid patches"""
    bl_idname = "mc.grid_smooth"
    bl_label = "Grid Smooth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.object
        targets = find_results(ob) or ([ob] if read_patch(ob) else [])
        if not targets:
            U.popup_error("No grid patches found for this object.", icon='INFO')
            return {'CANCELLED'}
        n = int(ob.MC_props.gf_smooth_iters)
        for t in targets:
            apply_smooth(t, n)
        self.report({'INFO'}, "Smoothed %d patch(es) at %d iterations"
                    % (len(targets), n))
        return {'FINISHED'}


class MC_OT_grid_select_source(bpy.types.Operator):
    """Make the source object of this grid patch active"""
    bl_idname = "mc.grid_select_source"
    bl_label = "Select Source"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return source_of(context.object) is not None

    def execute(self, context):
        src = source_of(context.object)
        if src is None:
            U.popup_error("The source object for this patch is gone.",
                          icon='INFO')
            return {'CANCELLED'}
        for o in bpy.data.objects:
            o.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        return {'FINISHED'}


def _wrap(text, width):
    """Break a message into panel-width lines (labels don't wrap)."""
    out, line = [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return out[:6]


# --------------------------------------------------------------------- panel
class MC_PT_panel_grid_fill(bpy.types.Panel):
    """Builds mesh, so it lives on the tools tab rather than among the
    simulation settings."""
    bl_label = "Grid Fill"
    bl_idname = "MC_PT_panel_grid_fill"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TOOLS_TAB
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        if U.needs_mesh(layout, context):
            return
        ob = context.object
        p = ob.MC_props

        if ob.get(PROP_SRC):
            # a generated patch: the build settings belong to the source, so
            # only show what actually applies here
            box = layout.box()
            box.label(text="Patch of '%s'" % ob[PROP_SRC], icon='FILE_REFRESH')
            box.operator("mc.grid_select_source", icon='BACK')
            box = layout.box()
            box.label(text="Smoothing")
            box.prop(p, "gf_smooth_iters")
            box.operator("mc.grid_smooth", icon='MOD_SMOOTH')
            return

        err = _LAST_ERROR.get(ob.name)
        if err:
            box = layout.box()
            box.alert = True
            box.label(text="Last build failed:", icon='ERROR')
            for chunk in _wrap(err, 34):
                box.label(text=chunk)

        box = layout.box()
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator("mc.grid_fill", icon='MESH_GRID')
        row.prop(p, "gf_live", text="", icon='TEMP', toggle=True)
        box.prop(p, "gf_fill_type", text="")
        row = box.row(align=True)
        row.prop(p, "gf_spacing")
        row.prop(p, "gf_scale")

        box = layout.box()
        box.label(text="Border")
        box.prop(p, "gf_flatten", text="Flatten")
        if p.gf_flatten == 'AUTO':
            box.prop(p, "gf_planarity_limit")
        box.prop(p, "gf_angle_limit")
        row = box.row(align=True)
        row.prop(p, "gf_holes", toggle=True)
        row.prop(p, "gf_selected_only", toggle=True)

        if p.gf_fill_type in {'QUADS', 'TRIS_GRID'}:
            box = layout.box()
            box.label(text="Lattice")
            box.prop(p, "gf_rotate")
            box.prop(p, "gf_offset")
            box.prop(p, "gf_inset")
        else:
            box = layout.box()
            box.label(text="Poisson")
            box.prop(p, "gf_seed")
            box.prop(p, "gf_relax_iters")

        box = layout.box()
        box.label(text="Smoothing")
        box.prop(p, "gf_smooth_iters")
        box.operator("mc.grid_smooth", icon='MOD_SMOOTH')

        box = layout.box()
        box.label(text="Advanced")
        box.prop(p, "gf_max_edge")
        box.prop(p, "gf_qhull")


CLASSES = [
    MC_OT_grid_fill,
    MC_OT_grid_smooth,
    MC_OT_grid_select_source,
    MC_PT_panel_grid_fill,
]
