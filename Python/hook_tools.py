"""Hook objects for cloth.

A hook is an Empty bound to a set of vertices on a cloth object.  Each frame
the bound vertices are pulled toward where the Empty says they should be, by an
amount set on the Empty itself, so dragging the Empty drags the cloth.

Why bind an offset per vertex rather than just pulling everything at the Empty:
"move the vertex toward the hook" taken literally would collapse every vertex
of a multi-vertex hook onto one point as the force approaches 1.  Instead each
vertex records where it sat in the Empty's local space at bind time, and its
target is that offset carried through the Empty's current matrix.  A force of 1
then makes the group follow the Empty rigidly, keeping its shape, which is what
Blender's own hook modifier does; lower forces lag softly behind it.

All the binding data lives on the Empty:

    ob["MC_hook_cloth"]   name of the cloth object it drives
    ob["MC_hook_verts"]   vertex indices, ints
    ob["MC_hook_rest"]    flat xyz, each vertex in the Empty's local space
    ob.MC_props.hook_force   how hard it pulls, 0..1 (can exceed 1)

Keeping it on the Empty means deleting the Empty deletes the hook, and the data
survives save/load without a parallel registry to keep in sync.  Vertex indices
are stored, so editing topology can invalidate a hook -- the same caveat the
hook modifier carries -- which is why every read validates the indices against
the current vertex count instead of trusting them.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector

try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U


KEY_CLOTH = "MC_hook_cloth"
KEY_VERTS = "MC_hook_verts"
KEY_REST = "MC_hook_rest"
SIM_KEY = "MC_current"          # the shape key the solver reads and writes


# ------------------------------------------------------------------ liveness
def is_live(ob):
    """Whether this reference still points at real data.

    A removed object leaves a Python wrapper that raises on any access rather
    than going None, so trying is the only way to ask.
    """
    if ob is None:
        return False
    try:
        ob.name
    except ReferenceError:
        return False
    return True


# --------------------------------------------------------------- hook lookup
def is_hook(ob):
    return is_live(ob) and KEY_CLOTH in ob.keys() and KEY_VERTS in ob.keys()


def all_hooks():
    return [ob for ob in bpy.data.objects if is_hook(ob)]


def hooks_for(cloth_ob):
    """Every hook bound to this cloth object."""
    if not is_live(cloth_ob):
        return []
    name = cloth_ob.name
    return [ob for ob in bpy.data.objects
            if is_hook(ob) and ob[KEY_CLOTH] == name]


def cloth_of(hook):
    """The cloth object a hook drives, or None if it has gone away."""
    if not is_hook(hook):
        return None
    return bpy.data.objects.get(hook[KEY_CLOTH])


def hook_data(hook, vert_count):
    """(indices, rest) for a hook, dropping anything that no longer fits.

    Stored indices can outlive the topology they were taken from -- a decimate,
    a merge by distance, a rebuilt grid -- so they are filtered against the
    current vertex count rather than used raw.  Returns empty arrays if nothing
    usable is left, which reads as "this hook does nothing" everywhere.
    """
    if not is_hook(hook):
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    idx = np.array(list(hook[KEY_VERTS]), dtype=np.int64)
    rest = np.array(list(hook[KEY_REST]), dtype=np.float64)
    if idx.shape[0] == 0 or rest.shape[0] != idx.shape[0] * 3:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    rest = rest.reshape(-1, 3)

    keep = (idx >= 0) & (idx < vert_count)
    return idx[keep], rest[keep]


# ------------------------------------------------------------------ the math
def targets(rest_local, hook_matrix, cloth_matrix_inv):
    """Where the bound vertices want to be, in cloth local space.

    rest_local is each vertex in the hook's local space, so carrying it through
    the hook's current matrix is what makes the group follow rotation and scale
    and not just translation.
    """
    if rest_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    m = np.array(cloth_matrix_inv, dtype=np.float64) @ np.array(
        hook_matrix, dtype=np.float64)
    out = rest_local @ m[:3, :3].T
    out += m[:3, 3]
    return out


def rest_from_co(co_local, cloth_matrix, hook_matrix_inv):
    """The inverse of targets(): bind positions from where the vertices are."""
    if co_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    m = np.array(hook_matrix_inv, dtype=np.float64) @ np.array(
        cloth_matrix, dtype=np.float64)
    out = co_local @ m[:3, :3].T
    out += m[:3, 3]
    return out


# ------------------------------------------------------------------ solver
def hooked_mask(ob, vert_count):
    """Vertices driven by a hook that actually pulls (force above zero)."""
    mask = np.zeros(vert_count, dtype=bool)
    for h in hooks_for(ob):
        if float(getattr(h.MC_props, "hook_force", 1.0)) == 0.0:
            continue
        idx, _ = hook_data(h, vert_count)
        mask[idx] = True
    return mask


def pin_selection(C):
    """The selection the solver should pin this step: selected minus hooked.

    A selected vertex is a grabbed vertex -- its velocity is zeroed and it is
    put back where the step started.  Making a hook means selecting its
    vertices, and they stay selected on the mesh after leaving edit mode, where
    C.selected is never refreshed.  Left alone, that pin silently beat every
    hook made the ordinary way, so the verts sat still while the Empty moved.

    A hook is the more deliberate constraint, so its vertices are taken out of
    the pin.  Every other selected vertex pins exactly as before, and with no
    hooks this returns C.selected itself, so hook-free cloth is unchanged.
    """
    hooks = hooks_for(C.ob)
    if not hooks:
        return C.selected
    return C.selected & ~hooked_mask(C.ob, C.selected.shape[0])


def hook_force(C):
    """Pull each hooked vertex toward its target.  Called from physics().

    Hooks are gathered from the scene every step rather than cached on C: an
    Empty can be added, deleted, re-bound or re-parented at any moment, and the
    existing forces (magnetic, colliders) already scan bpy.data the same way.
    """
    ob = C.ob
    hooks = hooks_for(ob)
    if not hooks:
        return 0

    vert_count = C.co.shape[0]
    cloth_inv = np.linalg.inv(np.array(ob.matrix_world, dtype=np.float64))
    moved = 0

    for h in hooks:
        force = float(getattr(h.MC_props, "hook_force", 1.0))
        if force == 0.0:
            continue
        idx, rest = hook_data(h, vert_count)
        if idx.shape[0] == 0:
            continue

        goal = targets(rest, h.matrix_world, cloth_inv)
        C.co[idx] += (goal - C.co[idx]).astype(C.co.dtype) * force
        moved += idx.shape[0]

    return moved


# ------------------------------------------------------------------- binding
def selected_verts(ob):
    """Indices of the selected vertices, edit mode or object mode.

    update_from_editmode first: without it the first press after an edit reads
    the stale mesh and appears to do nothing.
    """
    if ob.data.is_editmode:
        ob.update_from_editmode()
    n = len(ob.data.vertices)
    sel = np.zeros(n, dtype=bool)
    ob.data.vertices.foreach_get('select', sel)
    return np.arange(n)[sel]


def cloth_co(ob):
    """Where the vertices actually are: the simulated shape, not the Basis.

    Once an object is cloth, the solver lives in the MC_current shape key and
    ob.data.vertices holds the Basis.  Binding against the Basis placed the
    Empty at the undeformed position and recorded offsets from it, so as soon
    as the sim ran the hooked verts snapped by however far the cloth had moved
    since.  Edit mode writes its edits into the active key, so the caller must
    update_from_editmode first (selected_verts does).
    """
    me = ob.data
    if me.shape_keys is not None and SIM_KEY in me.shape_keys.key_blocks:
        co = np.empty((len(me.vertices), 3), dtype=np.float32)
        me.shape_keys.key_blocks[SIM_KEY].data.foreach_get('co', co.ravel())
        return co.astype(np.float64)
    return U.get_co(ob).astype(np.float64)


def bind(cloth_ob, hook, idx):
    """Record which vertices this hook drives and where they sit on it."""
    if cloth_ob.data.is_editmode:
        cloth_ob.update_from_editmode()
    co = cloth_co(cloth_ob)[idx]
    hook_inv = np.linalg.inv(np.array(hook.matrix_world, dtype=np.float64))
    rest = rest_from_co(co, cloth_ob.matrix_world, hook_inv)

    hook[KEY_CLOTH] = cloth_ob.name
    hook[KEY_VERTS] = [int(i) for i in idx]
    hook[KEY_REST] = [float(v) for v in rest.ravel()]
    return len(idx)


def unbind(hook):
    """Strip the hook data, leaving the Empty itself alone."""
    n = 0
    for k in (KEY_CLOTH, KEY_VERTS, KEY_REST):
        if k in hook.keys():
            del hook[k]
            n += 1
    return n


def make_hook(cloth_ob, idx, name=None, size=0.25):
    """Create an Empty at the median of the given vertices and bind it."""
    if cloth_ob.data.is_editmode:
        cloth_ob.update_from_editmode()
    co = cloth_co(cloth_ob)[idx]
    mw = np.array(cloth_ob.matrix_world, dtype=np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    centre = world.mean(axis=0)

    if name is None:
        name = "%s_hook" % cloth_ob.name
    hook = bpy.data.objects.new(name, None)
    hook.empty_display_type = 'SPHERE'
    hook.empty_display_size = size
    bpy.context.collection.objects.link(hook)
    # matrix_world, not location: Blender does not fold a new location into
    # matrix_world until the depsgraph next evaluates, and bind() reads
    # matrix_world.  Setting it directly means the bind sees where the Empty
    # actually is instead of binding against an identity matrix.
    hook.matrix_world = Matrix.Translation(Vector(centre))

    bind(cloth_ob, hook, idx)
    return hook


# ----------------------------------------------------------------- operators
def context_cloth(context):
    """The cloth object the panel is talking about.

    Selecting a hook to work on it is at least as common as selecting the
    cloth, so an active Empty resolves to the cloth it drives.
    """
    ob = context.object
    if not is_live(ob):
        return None
    if ob.type == 'MESH':
        return ob
    c = cloth_of(ob)
    return c if is_live(c) else None


def selected_hooks(context):
    picked = [o for o in context.selected_objects if is_hook(o)]
    if picked:
        return picked
    ob = context.object
    return [ob] if is_hook(ob) else []


class MC_OT_hook_add(bpy.types.Operator):
    """Add a hook Empty driving the selected vertices"""
    bl_idname = "mc.hook_add"
    bl_label = "Hook Selected Verts"
    bl_options = {'REGISTER', 'UNDO'}

    size: bpy.props.FloatProperty(
        name="Display Size", default=0.25, min=0.001, soft_max=5.0)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return is_live(ob) and ob.type == 'MESH'

    def execute(self, context):
        ob = context.object
        idx = selected_verts(ob)
        if idx.shape[0] == 0:
            U.popup_error("Select some vertices first.", icon='INFO')
            return {'CANCELLED'}

        hook = make_hook(ob, idx, size=self.size)
        self.report({'INFO'}, "Hook '%s' driving %d vert(s)"
                    % (hook.name, idx.shape[0]))
        return {'FINISHED'}


class MC_OT_hook_rebind(bpy.types.Operator):
    """Re-record the bind offsets at the current positions"""
    bl_idname = "mc.hook_rebind"
    bl_label = "Rebind"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(selected_hooks(context))

    def execute(self, context):
        n = 0
        for h in selected_hooks(context):
            c = cloth_of(h)
            if not is_live(c):
                continue
            idx, _ = hook_data(h, len(c.data.vertices))
            if idx.shape[0]:
                bind(c, h, idx)
                n += 1
        if not n:
            U.popup_error("Nothing to rebind.", icon='INFO')
            return {'CANCELLED'}
        self.report({'INFO'}, "Rebound %d hook(s)" % n)
        return {'FINISHED'}


class MC_OT_hook_select_cloth_hooks(bpy.types.Operator):
    """Select the hooks belonging to this cloth object"""
    bl_idname = "mc.hook_select_cloth_hooks"
    bl_label = "Select Hooks For This Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context_cloth(context) is not None

    def execute(self, context):
        cloth = context_cloth(context)
        hooks = hooks_for(cloth)
        if not hooks:
            U.popup_error("'%s' has no hooks." % cloth.name, icon='INFO')
            return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        for h in hooks:
            h.select_set(True)
        context.view_layer.objects.active = hooks[0]
        self.report({'INFO'}, "Selected %d hook(s) on '%s'"
                    % (len(hooks), cloth.name))
        return {'FINISHED'}


class MC_OT_hook_select_all(bpy.types.Operator):
    """Select every hook in the scene"""
    bl_idname = "mc.hook_select_all"
    bl_label = "Select All Hooks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        hooks = all_hooks()
        if not hooks:
            U.popup_error("There are no hooks in this file.", icon='INFO')
            return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        for h in hooks:
            h.select_set(True)
        context.view_layer.objects.active = hooks[0]
        self.report({'INFO'}, "Selected %d hook(s)" % len(hooks))
        return {'FINISHED'}


class MC_OT_hook_select_verts(bpy.types.Operator):
    """Select the vertices the chosen hooks drive"""
    bl_idname = "mc.hook_select_verts"
    bl_label = "Select Hooked Verts"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(selected_hooks(context))

    def execute(self, context):
        hooks = selected_hooks(context)
        by_cloth = {}
        for h in hooks:
            c = cloth_of(h)
            if is_live(c):
                by_cloth.setdefault(c.name, []).append(h)
        if not by_cloth:
            U.popup_error("Those hooks have no cloth object.", icon='INFO')
            return {'CANCELLED'}

        total = 0
        for name, hs in by_cloth.items():
            c = bpy.data.objects[name]
            if c.data.is_editmode:
                c.update_from_editmode()
            n = len(c.data.vertices)
            sel = np.zeros(n, dtype=bool)
            for h in hs:
                idx, _ = hook_data(h, n)
                sel[idx] = True
            c.data.vertices.foreach_set('select', sel)
            c.data.update()
            total += int(sel.sum())

        self.report({'INFO'}, "Selected %d vert(s) across %d object(s)"
                    % (total, len(by_cloth)))
        return {'FINISHED'}


class MC_OT_hook_remove_data(bpy.types.Operator):
    """Strip the hook data, keeping the Empty"""
    bl_idname = "mc.hook_remove_data"
    bl_label = "Remove Hook Data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(selected_hooks(context))

    def execute(self, context):
        hooks = selected_hooks(context)
        for h in hooks:
            unbind(h)
        self.report({'INFO'}, "Cleared %d hook(s) - the Empties are still here"
                    % len(hooks))
        return {'FINISHED'}


class MC_OT_hook_delete(bpy.types.Operator):
    """Delete hook Empties"""
    bl_idname = "mc.hook_delete"
    bl_label = "Delete Hooks"
    bl_options = {'REGISTER', 'UNDO'}

    scope: bpy.props.EnumProperty(
        name="Scope",
        items=[('SELECTED', "Selected", "Only the selected hooks"),
               ('CLOTH', "This Cloth", "Every hook on this cloth object"),
               ('ALL', "All In File", "Every hook in the blend file")],
        default='CLOTH')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=330)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "scope")
        col.label(text="Deletes %d Empt(y/ies)." % len(self._doomed(context)),
                  icon='TRASH')
        col.label(text="The cloth mesh itself is not touched.")

    def _doomed(self, context):
        if self.scope == 'SELECTED':
            return selected_hooks(context)
        if self.scope == 'ALL':
            return all_hooks()
        cloth = context_cloth(context)
        return hooks_for(cloth) if cloth else []

    def execute(self, context):
        doomed = self._doomed(context)
        if not doomed:
            U.popup_error("No hooks to delete for that scope.", icon='INFO')
            return {'CANCELLED'}
        n = len(doomed)
        for h in doomed:
            bpy.data.objects.remove(h)
        self.report({'INFO'}, "Deleted %d hook(s)" % n)
        return {'FINISHED'}


class MC_OT_hook_report(bpy.types.Operator):
    """List the hooks on this cloth object in the console"""
    bl_idname = "mc.hook_report"
    bl_label = "Hook Report"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context_cloth(context) is not None

    def execute(self, context):
        cloth = context_cloth(context)
        hooks = hooks_for(cloth)
        n = len(cloth.data.vertices)
        print("# ===== hooks on %s =====" % cloth.name)
        stale = 0
        for h in hooks:
            idx, _ = hook_data(h, n)
            raw = len(list(h[KEY_VERTS])) if KEY_VERTS in h.keys() else 0
            lost = raw - idx.shape[0]
            stale += 1 if lost else 0
            print("  %-28s %4d vert(s)  force %.3f%s"
                  % (h.name, idx.shape[0],
                     getattr(h.MC_props, "hook_force", 1.0),
                     "  (%d index(es) no longer exist)" % lost if lost else ""))
        print("# %d hook(s), %d with stale indices" % (len(hooks), stale))
        self.report({'INFO'}, "%d hook(s), %d stale - see console"
                    % (len(hooks), stale))
        return {'FINISHED'}


class MC_PT_panel_hooks(bpy.types.Panel):
    bl_label = "Hooks"
    bl_idname = "MC_PT_panel_hooks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["hooks"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        ob = context.object
        cloth = context_cloth(context)

        box = layout.box()
        box.label(text="Create")
        if is_live(ob) and ob.type == 'MESH':
            box.operator("mc.hook_add", icon='HOOK')
        else:
            box.label(text="Select a mesh to add hooks", icon='INFO')

        picked = selected_hooks(context)
        if picked:
            box = layout.box()
            box.label(text="Hook" if len(picked) == 1 else
                      "%d Hooks" % len(picked))
            for h in picked[:4]:
                row = box.row(align=True)
                row.label(text=h.name, icon='HOOK')
                row.prop(h.MC_props, "hook_force", text="")
            if len(picked) > 4:
                box.label(text="...and %d more" % (len(picked) - 4))
            box.operator("mc.hook_rebind", icon='FILE_REFRESH')
            box.operator("mc.hook_select_verts", icon='VERTEXSEL')

        box = layout.box()
        box.label(text="Select")
        box.operator("mc.hook_select_cloth_hooks", icon='RESTRICT_SELECT_OFF')
        box.operator("mc.hook_select_all", icon='SELECT_EXTEND')

        box = layout.box()
        box.label(text="Manage")
        if cloth is not None:
            box.label(text="Cloth: %s (%d hook(s))"
                      % (cloth.name, len(hooks_for(cloth))))
        box.operator("mc.hook_report", icon='INFO')
        box.operator("mc.hook_remove_data", icon='UNLINKED')
        row = box.row()
        row.alert = True
        row.operator("mc.hook_delete", icon='TRASH')


CLASSES = [
    MC_OT_hook_add,
    MC_OT_hook_rebind,
    MC_OT_hook_select_cloth_hooks,
    MC_OT_hook_select_all,
    MC_OT_hook_select_verts,
    MC_OT_hook_remove_data,
    MC_OT_hook_delete,
    MC_OT_hook_report,
    MC_PT_panel_hooks,
]
