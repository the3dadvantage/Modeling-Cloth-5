"""UV shape tools.

Flattens a mesh into a `uv_shape` shape key, so a garment modelled in 3D can be
laid out as its flat sewing pattern.

Deliberately independent of the cloth data structures -- it only touches the
mesh, its UV layers and its shape keys, so it works on any mesh whether or not
it has been set up as cloth.

The hard part is meshes that will not unwrap: a closed solid has no boundary,
so without seams the unwrapper has to overlap.  uv_topo answers that precisely
(V - E + F per component) and can mark the seams automatically.  Cutting the
mesh open along those seams -- "ripping" -- is offered separately because it
changes the vertex count and so has to happen before the shape keys exist.
"""

import bpy
import bmesh
import numpy as np

try:
    U = bpy.data.texts['utils.py'].as_module()
    TOPO = bpy.data.texts['uv_topo.py'].as_module()
except Exception:
    from . import utils as U
    from . import uv_topo as TOPO


SHAPE_BASIS = "Basis"
SHAPE_UV = "uv_shape"


# ------------------------------------------------------------------ mesh read
def mesh_faces(ob):
    return [list(p.vertices) for p in ob.data.polygons]


def mesh_edges(ob):
    e = np.empty((len(ob.data.edges), 2), dtype=np.int64)
    ob.data.edges.foreach_get('vertices', e.ravel())
    return e


def face_normals(ob):
    n = np.empty((len(ob.data.polygons), 3), dtype=np.float32)
    ob.data.polygons.foreach_get('normal', n.ravel())
    return n.astype(np.float64)


def refresh(ob):
    if ob.data.is_editmode:
        ob.update_from_editmode()


# -------------------------------------------------------------------- uv maps
def uv_map_items(self, context):
    """Enum items for the object's UV layers.  Dynamic, so it tracks renames."""
    ob = context.object if context else None
    if ob is None or ob.type != 'MESH' or not ob.data.uv_layers:
        return [('NONE', "<no UV maps>", "This mesh has no UV maps yet")]
    return [(l.name, l.name, "Use the '%s' UV map" % l.name)
            for l in ob.data.uv_layers]


def get_uv_map(ob, name):
    uvs = ob.data.uv_layers
    if name and name in uvs:
        return uvs[name]
    return uvs.active


def uv_face_areas(ob, layer):
    """UV-space area per face."""
    me = ob.data
    n_loops = len(me.loops)
    uv = np.empty(n_loops * 2, dtype=np.float32)
    layer.data.foreach_get('uv', uv)
    uv = uv.reshape(n_loops, 2).astype(np.float64)

    out = np.zeros(len(me.polygons))
    for i, p in enumerate(me.polygons):
        q = uv[p.loop_start:p.loop_start + p.loop_total]
        if len(q) < 3:
            continue
        r = np.roll(q, -1, axis=0)
        out[i] = 0.5 * abs(float(np.sum(q[:, 0] * r[:, 1] - r[:, 0] * q[:, 1])))
    return out


def collapsed_faces(ob, layer, rel_tol=1e-4):
    """How many faces have essentially no UV area.

    One of the two ways a failed unwrap shows up: the solver gives up and
    leaves the whole island at zero area.  See uv_tears for the other.
    """
    a = uv_face_areas(ob, layer)
    if not len(a):
        return 0, 0.0
    ref = float(np.median(a[a > 0])) if np.any(a > 0) else 0.0
    if ref <= 0.0:
        return int(len(a)), 0.0
    return int(np.count_nonzero(a < ref * rel_tol)), ref


def uv_tears(ob, layer, tol=1e-5):
    """Interior, non-seam edges whose two faces disagree in UV.

    bpy.ops.uv.unwrap returns FINISHED even when it cannot solve an island --
    it only emits a warning, which script cannot see.  What it actually does
    then is abandon the solve and pack the faces as a plain grid, so every
    interior edge ends up torn even though no seam asked for it.  Measuring
    the layout catches that whatever the operator claims.

    Returns (torn, interior).  A healthy unwrap tears only on seams, so torn
    is 0; the grid fallback tears every single interior edge.
    """
    me = ob.data
    n_loops = len(me.loops)
    uv = np.empty(n_loops * 2, dtype=np.float32)
    layer.data.foreach_get('uv', uv)
    uv = uv.reshape(n_loops, 2).astype(np.float64)

    lv = np.empty(n_loops, dtype=np.int64)
    me.loops.foreach_get('vertex_index', lv)

    corner = {}          # (face, vert) -> uv
    faces_on = {}        # edge key -> face indices
    for p in me.polygons:
        for li in range(p.loop_start, p.loop_start + p.loop_total):
            corner[(p.index, int(lv[li]))] = uv[li]
        vs = list(p.vertices)
        for j, a in enumerate(vs):
            b = vs[(j + 1) % len(vs)]
            faces_on.setdefault((a, b) if a < b else (b, a), []).append(p.index)

    torn = interior = 0
    for e in me.edges:
        a, b = e.vertices
        f = faces_on.get((a, b) if a < b else (b, a), ())
        if len(f) != 2 or e.use_seam:
            continue
        interior += 1
        for v in (a, b):
            p, q = corner.get((f[0], v)), corner.get((f[1], v))
            if (p is None or q is None
                    or abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol):
                torn += 1
                break
    return torn, interior


def uv_to_verts(ob, layer):
    """Per-vertex UV, averaged over the loops that touch each vertex.

    A vertex split across islands has several UVs; averaging them would drag it
    between islands, so the caller should rip first if that matters.  The count
    of split verts is reported so it can say so.
    """
    me = ob.data
    n_loops = len(me.loops)
    uv = np.empty(n_loops * 2, dtype=np.float32)
    layer.data.foreach_get('uv', uv)
    uv = uv.reshape(n_loops, 2).astype(np.float64)

    lv = np.empty(n_loops, dtype=np.int64)
    me.loops.foreach_get('vertex_index', lv)

    n = len(me.vertices)
    acc = np.zeros((n, 2))
    cnt = np.zeros(n)
    np.add.at(acc, lv, uv)
    np.add.at(cnt, lv, 1.0)
    out = acc / np.maximum(cnt, 1.0)[:, None]

    # a vertex whose loop UVs disagree sits on an island boundary
    spread = np.zeros(n)
    np.add.at(spread, lv, np.linalg.norm(uv - out[lv], axis=1))
    split = int(np.count_nonzero(spread > 1e-6))
    return out, split


# --------------------------------------------------------------- shape keys
def ensure_shape_keys(ob, name=SHAPE_UV):
    """Make sure Basis and `name` exist; return the target key."""
    me = ob.data
    if me.shape_keys is None or SHAPE_BASIS not in me.shape_keys.key_blocks:
        ob.shape_key_add(name=SHAPE_BASIS, from_mix=False)
    if name not in me.shape_keys.key_blocks:
        ob.shape_key_add(name=name, from_mix=False)
    return me.shape_keys.key_blocks[name]


# ------------------------------------------------------------------- seams
def seam_edges(ob, source, sharp_angle=40.0, axis=2, offset=0.0, tol=1e-4):
    """Edge keys to mark as seams, for the chosen source."""
    faces = mesh_faces(ob)
    if source == 'AUTO':
        return TOPO.auto_cut(faces)
    if source == 'SHARP':
        return TOPO.sharp_edges(faces, face_normals(ob), sharp_angle)
    if source == 'PLANE':
        co = U.get_co(ob).astype(np.float64)
        return TOPO.plane_edges(co, faces, axis=axis, offset=offset, tol=tol)
    if source == 'SELECTION':
        out = set()
        for e in ob.data.edges:
            if e.select:
                a, b = e.vertices
                out.add((a, b) if a < b else (b, a))
        return out
    if source == 'MC_SEAMS':
        me = ob.data
        if "MC_seam_id" not in me.attributes:
            return set()
        sid = np.empty(len(me.vertices), dtype=np.int32)
        me.attributes["MC_seam_id"].data.foreach_get('value', sid)
        out = set()
        for e in me.edges:
            a, b = e.vertices
            if sid[a] and sid[a] == sid[b]:
                out.add((a, b) if a < b else (b, a))
        return out
    return set()


def apply_seams(ob, keys, replace=True):
    """Write seam flags onto the mesh."""
    me = ob.data
    n = 0
    for e in me.edges:
        a, b = e.vertices
        k = (a, b) if a < b else (b, a)
        want = k in keys
        if replace:
            e.use_seam = want
        elif want:
            e.use_seam = True
        if want:
            n += 1
    me.update()
    return n


def rip_seams(ob):
    """Split the mesh along its seam edges.

    Changes the vertex count, so this must happen before the shape keys are
    built -- a key has to match the vertex count it was created with.  Blender
    also refuses to split a mesh that already has shape keys, so they are
    removed first and rebuilt afterwards.
    """
    me = ob.data
    if me.shape_keys is not None:
        ob.shape_key_clear()

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    doomed = [e for e in bm.edges if e.seam and len(e.link_faces) == 2]
    n = len(doomed)
    if n:
        bmesh.ops.split_edges(bm, edges=doomed)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return n


# --------------------------------------------------------------------- build
def unwrap(ob, method):
    """Run Blender's unwrapper, restoring mode and selection afterwards."""
    if method == 'NONE':
        return
    prev_mode = ob.mode
    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = ob
    was = ob.select_get()
    ob.select_set(True)
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        if method == 'SMART':
            bpy.ops.uv.smart_project(island_margin=0.02)
        else:
            bpy.ops.uv.unwrap(method=method)
        bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        ob.select_set(was)
        bpy.context.view_layer.objects.active = prev_active
        if ob.mode != prev_mode:
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass


def build_uv_shape(ob, uv_name="", scale_mode='LEAST_SQUARES',
                   shape_name=SHAPE_UV, flat_axis=2):
    """Write the flattened layout into a shape key.  Returns an info dict."""
    refresh(ob)
    layer = get_uv_map(ob, uv_name)
    if layer is None:
        raise RuntimeError("no UV map to read - create or unwrap one first")

    co3 = U.get_co(ob).astype(np.float64)
    uv, split = uv_to_verts(ob, layer)
    edges = mesh_edges(ob)

    scale = TOPO.fit_scale(co3, uv, edges, scale_mode)
    flat = np.zeros((len(co3), 3))
    a, b = [i for i in range(3) if i != flat_axis]
    flat[:, a] = uv[:, 0] * scale
    flat[:, b] = uv[:, 1] * scale
    flat += co3.mean(axis=0) - flat.mean(axis=0)

    key = ensure_shape_keys(ob, shape_name)
    key.data.foreach_set('co', np.ascontiguousarray(flat, np.float32).ravel())
    ob.data.update()

    bad_faces, _ = collapsed_faces(ob, layer)
    torn, interior = uv_tears(ob, layer)

    rep = TOPO.stretch_report(co3, uv, edges, scale)
    rep.update({"uv_map": layer.name, "scale": scale, "split_verts": split,
                "shape": shape_name, "collapsed_faces": bad_faces,
                "n_faces": len(ob.data.polygons),
                "torn_edges": torn, "interior_edges": interior})
    return rep


# ----------------------------------------------------------------- operators
def _ob(context):
    ob = context.object
    return ob if ob is not None and ob.type == 'MESH' else None


class MC_OT_uvs_check(bpy.types.Operator):
    """Report whether this mesh can be flattened, and where it needs cutting"""
    bl_idname = "mc.uvs_check"
    bl_label = "Check Unwrappable"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _ob(context) is not None

    def execute(self, context):
        ob = _ob(context)
        refresh(ob)
        rep = TOPO.analyze(mesh_faces(ob))
        if not rep:
            U.popup_error("This mesh has no faces.", icon='INFO')
            return {'CANCELLED'}

        print("# ===== uv shape check: %s =====" % ob.name)
        bad = 0
        for i, r in enumerate(rep):
            print("  piece %-3d %4d faces %4d verts  chi %+d  genus %d  "
                  "boundary loops %d  ->  %s"
                  % (i, r["n_faces"], r["n_verts"], r["chi"], r["genus"],
                     r["boundary_loops"], r["note"]))
            if not r["is_disc"]:
                bad += 1
        print("# =============================")

        if bad:
            need = sum(r["cuts_needed"] for r in rep)
            U.popup_error("%d of %d piece(s) will overlap - about %d cut(s) "
                          "needed. Use Mark Seams (Auto), then Rip. See the "
                          "console for detail." % (bad, len(rep), need),
                          icon='ERROR')
        self.report({'INFO'}, "%d piece(s), %d need cutting - see console"
                    % (len(rep), bad))
        return {'FINISHED'}


class MC_OT_uvs_new_map(bpy.types.Operator):
    """Add a UV map and make it active"""
    bl_idname = "mc.uvs_new_map"
    bl_label = "New UV Map"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty(name="Name", default="MC_flat")

    @classmethod
    def poll(cls, context):
        return _ob(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ob = _ob(context)
        refresh(ob)
        nm = self.name.strip() or "MC_flat"
        if nm in ob.data.uv_layers:
            U.popup_error("'%s' already exists." % nm, icon='INFO')
            return {'CANCELLED'}
        layer = ob.data.uv_layers.new(name=nm)
        ob.data.uv_layers.active = layer
        try:
            ob.MC_props.uvs_map = layer.name
        except Exception:
            pass
        self.report({'INFO'}, "Created UV map '%s'" % layer.name)
        return {'FINISHED'}


class MC_OT_uvs_mark_seams(bpy.types.Operator):
    """Mark UV seams from the chosen source"""
    bl_idname = "mc.uvs_mark_seams"
    bl_label = "Mark Seams"
    bl_options = {'REGISTER', 'UNDO'}

    replace: bpy.props.BoolProperty(
        name="Replace", description="Clear existing seams first", default=True)

    @classmethod
    def poll(cls, context):
        return _ob(context) is not None

    def execute(self, context):
        ob = _ob(context)
        refresh(ob)
        p = ob.MC_props
        keys = seam_edges(ob, p.uvs_seam_source, p.uvs_sharp_angle,
                          int(p.uvs_plane_axis), p.uvs_plane_offset,
                          p.uvs_plane_tol)
        if not keys:
            U.popup_error("That source marked no edges.", icon='INFO')
            return {'CANCELLED'}
        n = apply_seams(ob, keys, self.replace)
        self.report({'INFO'}, "Marked %d seam edge(s) from %s"
                    % (n, p.uvs_seam_source))
        return {'FINISHED'}


class MC_OT_uvs_rip(bpy.types.Operator):
    """Split the mesh along its seams. Changes the vertex count"""
    bl_idname = "mc.uvs_rip"
    bl_label = "Rip Seams"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _ob(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=340)

    def draw(self, context):
        ob = _ob(context)
        col = self.layout.column()
        n = sum(1 for e in ob.data.edges if e.use_seam) if ob else 0
        col.label(text="Split '%s' along %d seam edge(s)?"
                  % (ob.name if ob else "?", n), icon='UNLINKED')
        col.label(text="This changes the vertex count and clears")
        col.label(text="existing shape keys.")

    def execute(self, context):
        ob = _ob(context)
        refresh(ob)
        before = len(ob.data.vertices)
        n = rip_seams(ob)
        after = len(ob.data.vertices)
        if not n:
            U.popup_error("No interior seam edges to split. Mark seams first.",
                          icon='INFO')
            return {'CANCELLED'}
        self.report({'INFO'}, "Split %d edge(s), %d -> %d verts"
                    % (n, before, after))
        return {'FINISHED'}


class MC_OT_uvs_build(bpy.types.Operator):
    """Unwrap if asked, then write the flat layout into the uv_shape key"""
    bl_idname = "mc.uvs_build"
    bl_label = "Build UV Shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _ob(context) is not None

    def execute(self, context):
        ob = _ob(context)
        refresh(ob)
        p = ob.MC_props

        if not len(ob.data.polygons):
            U.popup_error("This mesh has no faces, so there is nothing to "
                          "unwrap. For a bare edge loop use Grid Fill.",
                          icon='INFO')
            return {'CANCELLED'}

        bad, which = TOPO.needs_cutting(mesh_faces(ob))
        if bad and p.uvs_warn_closed and p.uvs_unwrap != 'SMART':
            U.popup_error("%d piece(s) are not flat-unwrappable (%s). Mark "
                          "Seams (Auto) and Rip, or use Smart Project."
                          % (len(which), which[0]["note"]), icon='ERROR')

        if p.uvs_unwrap != 'NONE':
            target = p.uvs_map if p.uvs_map in ob.data.uv_layers else ""
            if not target:
                layer = ob.data.uv_layers.new(name="MC_flat")
                target = layer.name
                try:
                    p.uvs_map = target
                except Exception:
                    pass
            ob.data.uv_layers.active = ob.data.uv_layers[target]
            unwrap(ob, p.uvs_unwrap)

        try:
            rep = build_uv_shape(ob, p.uvs_map, p.uvs_scale_mode,
                                 p.uvs_shape_name, int(p.uvs_flat_axis))
        except Exception as e:
            U.popup_error("UV shape failed: %s" % e, icon='ERROR')
            return {'CANCELLED'}

        bad = rep["collapsed_faces"]
        torn = rep["torn_edges"]
        if bad:
            U.popup_error(
                "%d of %d face(s) came out with no UV area - the unwrap could "
                "not solve them. Mark Seams (Auto) and Rip, or use Smart "
                "Project." % (bad, rep["n_faces"]), icon='ERROR')
        elif torn:
            U.popup_error(
                "The unwrap did not solve: %d of %d interior edge(s) are split "
                "in UV space without a seam asking for it, so the layout is a "
                "grid of loose faces rather than a pattern. Mark Seams (Auto) "
                "and Rip, or use Smart Project."
                % (torn, rep["interior_edges"]), icon='ERROR')

        msg = ("'%s' from '%s'  |  scale %.4f  |  stretch %.2f-%.2f (rms %.3f)"
               % (rep["shape"], rep["uv_map"], rep["scale"],
                  rep["min"], rep["max"], rep["rms"]))
        if bad:
            msg += "  |  %d collapsed face(s)" % bad
        if torn:
            msg += "  |  unwrap unsolved, %d torn edge(s)" % torn
        if rep["split_verts"]:
            msg += "  |  %d vert(s) span islands - rip for a clean pattern" % rep["split_verts"]
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MC_PT_panel_uv_shape(bpy.types.Panel):
    """Builds mesh from a UV layout: a tools-tab job, not a sim setting."""
    bl_label = "UV Shape"
    bl_idname = "MC_PT_panel_uv_shape"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TOOLS_TAB
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        ob = _ob(context)
        if ob is None:
            U.panel_note(layout, "Select a mesh object")
            return
        p = ob.MC_props

        box = layout.box()
        box.label(text="UV Map")
        row = box.row(align=True)
        row.prop(p, "uvs_map", text="")
        row.operator("mc.uvs_new_map", text="", icon='ADD')
        box.prop(p, "uvs_unwrap", text="Unwrap")

        box = layout.box()
        box.label(text="Seams")
        box.operator("mc.uvs_check", icon='QUESTION')
        box.prop(p, "uvs_seam_source", text="")
        if p.uvs_seam_source == 'SHARP':
            box.prop(p, "uvs_sharp_angle")
        elif p.uvs_seam_source == 'PLANE':
            row = box.row(align=True)
            row.prop(p, "uvs_plane_axis", text="")
            row.prop(p, "uvs_plane_offset", text="Offset")
            box.prop(p, "uvs_plane_tol")
        box.operator("mc.uvs_mark_seams", icon='MOD_UVPROJECT')
        row = box.row()
        row.alert = True
        row.operator("mc.uvs_rip", icon='UNLINKED')

        box = layout.box()
        box.label(text="Shape")
        box.prop(p, "uvs_shape_name")
        box.prop(p, "uvs_scale_mode", text="Scale")
        box.prop(p, "uvs_flat_axis", text="Flat Axis")
        box.prop(p, "uvs_warn_closed")
        row = box.row()
        row.scale_y = 1.4
        row.operator("mc.uvs_build", icon='MOD_UVPROJECT')


CLASSES = [
    MC_OT_uvs_check,
    MC_OT_uvs_new_map,
    MC_OT_uvs_mark_seams,
    MC_OT_uvs_rip,
    MC_OT_uvs_build,
    MC_PT_panel_uv_shape,
]
