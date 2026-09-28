"""Wind.

A wind vector pushes the cloth, scaled per vertex by how squarely that part of
the surface faces the wind: nothing when the surface normal is perpendicular to
the wind, full strength when it is parallel.  That is what makes a sheet luff
and flap instead of simply sliding downwind.

Turbulence varies both the strength and the direction, because wind that only
pulses in magnitude reads as a throb rather than as weather.  Both are driven
by the same smoothly interpolated noise, so `wind_transition` controls how
quickly gusts arrive for both at once.

Time comes from a step counter that resets with the cloth, never from
scene.frame_current.  Under Continuous the frame never advances, so frame-based
wind would sit frozen; a step counter advances in both modes, and since
animated mode runs exactly one step per frame the counter equals frames since
reset, which keeps a render reproducible.
"""

import bpy
import numpy as np
from mathutils import Vector

try:
    U = bpy.data.texts['utils.py'].as_module()
except Exception:
    from . import utils as U


# Wind values are multiplied by this before reaching velocity, matching the
# 0.001 gravity uses, so the properties sit in the same familiar range.
SCALE = 0.001

MASK64 = 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------------- noise
# uint64 arithmetic in numpy wraps modulo 2**64, which is exactly what the
# avalanche wants, so the masking a python-int version needs disappears.  The
# multipliers are typed np.uint64 so numpy keeps the whole thing in uint64
# rather than promoting to float.
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SEED_MIX = np.uint64(0xBF58476D1CE4E5B9)
_STREAM_MIX = np.uint64(0x94D049BB133111EB)
_SH33 = np.uint64(33)
_TWO64 = 18446744073709551616.0


def _hash_u64(x):
    """64-bit avalanche hash over a uint64 array."""
    x = np.asarray(x, dtype=np.uint64)
    x = x ^ (x >> _SH33)
    x = x * _M1
    x = x ^ (x >> _SH33)
    x = x * _M2
    x = x ^ (x >> _SH33)
    return x


def rand01(index, seed=0, stream=0):
    """Deterministic float in [0, 1) per integer index.

    index and stream may be arrays, so every noise stream a step needs is
    hashed in one pass instead of one call each.
    """
    idx = np.asarray(index, dtype=np.int64).astype(np.uint64)
    sd = np.asarray(seed, dtype=np.int64).astype(np.uint64)
    st = np.asarray(stream, dtype=np.int64).astype(np.uint64)
    h = _hash_u64((idx * _GOLDEN) ^ (sd * _SEED_MIX) ^ (st * _STREAM_MIX))
    return h.astype(np.float64) / _TWO64


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease(t, smoothness=1.0):
    """Blend the interpolation curve from linear to very gentle.

    Linear moves between gusts at a constant rate, so each target arrives with
    a visible corner.  smootherstep leaves and arrives at a standstill, which
    reads as air rather than as a machine.  `smoothness` picks between them.
    """
    s = min(max(float(smoothness), 0.0), 1.0)
    if s <= 0.0:
        return t
    if s >= 1.0:
        return smootherstep(t)
    return t + (smootherstep(t) - t) * s


def wobble(step, transition, seed=0, stream=0, smoothness=1.0):
    """A value in [0, 1) that drifts smoothly, re-aiming every `transition`.

    Picking a fresh random number every step would be white noise -- visually
    just jitter.  Interpolating between successive targets is what turns it
    into gusts: `transition` is how many steps a gust takes to arrive and
    `smoothness` is how gently it gets there.
    """
    transition = max(1, int(transition))
    step = np.asarray(step, dtype=np.int64)
    k = step // transition
    rem = step - k * transition
    a = rand01(k, seed, stream)
    b = rand01(k + 1, seed, stream)
    return a + (b - a) * ease(rem / float(transition), smoothness)


def _hash_grid(ix, iy, iz, seed):
    """Vectorised lattice hash -> float in [0, 1). int64 throughout."""
    h = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) \
        ^ (iz * np.int64(83492791)) ^ np.int64(seed * 2654435761 & 0x7FFFFFFF)
    h &= np.int64(0x7FFFFFFF)
    h ^= (h >> np.int64(13))
    h = (h * np.int64(1274126177)) & np.int64(0x7FFFFFFF)
    return (h & np.int64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


# the eight corners of a lattice cell, as a (8, 3) constant
_CORNERS = np.array([(dx, dy, dz)
                     for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                    dtype=np.int64)


def value_noise(p, seed=0):
    """Trilinear value noise at Nx3 points.  Returns N values in [0, 1).

    All eight cell corners are hashed and weighted in one broadcast pass --
    (8, N) throughout -- rather than looping the corners in python.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape[0] == 0:
        return np.zeros(0)

    base = np.floor(p)
    f = smoothstep(p - base)                      # (N, 3)
    base = base.astype(np.int64)

    corner = _CORNERS[:, None, :]                 # (8, 1, 3)
    idx = base[None, :, :] + corner               # (8, N, 3)
    h = _hash_grid(idx[..., 0], idx[..., 1], idx[..., 2], seed)   # (8, N)

    # weight is f on the corners that sit at +1 and (1 - f) on the others
    w = np.where(corner == 1, f[None, :, :], 1.0 - f[None, :, :])  # (8, N, 3)
    return np.einsum('cn,cn->n', h, w[..., 0] * w[..., 1] * w[..., 2])


# ----------------------------------------------------------------- direction
# one noise stream per axis for the direction wander, hashed together
_DIR_STREAMS = np.array([11, 12, 13], dtype=np.int64)


def cloths_using(ob):
    """Every cloth aiming its wind with this object."""
    out = []
    for o in bpy.data.objects:
        p = getattr(o, "MC_props", None)
        if p is None:
            continue
        try:
            if p.wind_object == ob and o != ob:
                out.append(o)
        except (ReferenceError, AttributeError):
            continue
    return out


def cb_wind_strength(self, context):
    """Keep the strength the same on the cloth and on the object aiming it.

    The value is one thing shown in two places, so editing it anywhere writes
    it everywhere: from a cloth to the object it aims with, and from that
    object back to every cloth using it.

    The writes go through props['wind_strength'] rather than
    props.wind_strength, because assigning the IDProperty directly does not
    fire the update callback -- so the partners are updated without each of
    them turning round and updating us back.
    """
    ob = self.id_data
    value = float(self.wind_strength)

    targets = list(cloths_using(ob))
    wob = getattr(self, "wind_object", None)
    if wob is not None:
        targets.append(wob)

    for t in targets:
        if t is None or t == ob:
            continue
        try:
            t.MC_props['wind_strength'] = value
        except (ReferenceError, AttributeError, TypeError):
            continue


def base_vector(ob):
    """The wind in world space, before turbulence.

    With an object aiming the wind, direction comes from its local +Z -- which
    is how a Single Arrow empty draws itself -- and strength is the single
    wind_strength value, because a vector would be saying the direction twice.
    Without one, the XYZ vector carries both.
    """
    p = ob.MC_props

    wob = getattr(p, "wind_object", None)
    if wob is not None:
        strength = float(getattr(p, "wind_strength", 1.0))
        try:
            axis = wob.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
        except ReferenceError:
            return np.zeros(3), 0.0
        n = axis.length
        if n > 1e-12:
            d = np.array(axis / n, dtype=np.float64)
            return d * strength, abs(strength)
        return np.zeros(3), 0.0

    vec = np.array(p.wind, dtype=np.float64)
    return vec, float(np.linalg.norm(vec))


def to_local(ob, world_vec):
    """World direction into the cloth's own space.

    C.co and the vertex normals are both local, so the wind has to come with
    them or rotating the cloth object would swing the weather around with it.
    """
    m = ob.matrix_world.to_3x3()
    try:
        inv = m.inverted()
    except ValueError:
        return world_vec
    return np.array(inv @ Vector(world_vec), dtype=np.float64)


def gust_vector(ob, step):
    """The wind for this step, direction wander and strength both varied.

    The turbulence range is deliberately not clamped.  A floor of 0 lets the
    wind die away to nothing between gusts, a negative floor lets it reverse
    and blow back, and there is no ceiling, so a range like -0.5 to 4 gives
    gusts that slam, drop out, and occasionally push the other way -- which is
    what real wind does.  The multiplier scales the vector, so the sign simply
    flips it.
    """
    p = ob.MC_props
    world, strength = base_vector(ob)
    if strength <= 1e-12:
        return np.zeros(3)

    d = world / strength
    seed = int(getattr(p, "wind_seed", 1))
    trans = int(getattr(p, "wind_transition", 30))
    smooth = float(getattr(p, "wind_smoothness", 1.0))

    # direction wander: nudge the unit direction by a smoothly drifting offset
    # and renormalise, which bounds the swing without any trig
    swing = float(getattr(p, "wind_direction_variation", 0.0))
    if swing > 0.0:
        # all three streams hashed in one pass
        off = wobble(step, trans, seed, _DIR_STREAMS, smooth) * 2.0 - 1.0
        d = d + off * np.tan(np.radians(min(swing, 89.0)))
        n = np.linalg.norm(d)
        d = d / n if n > 1e-12 else world / strength

    lo = float(getattr(p, "wind_turbulence_min", 1.0))
    hi = float(getattr(p, "wind_turbulence_max", 1.0))
    if hi == lo:
        mult = lo
    else:
        if hi < lo:
            lo, hi = hi, lo
        mult = lo + (hi - lo) * wobble(step, trans, seed, 7, smooth)

    return d * (strength * mult)


# -------------------------------------------------------------------- force
def is_active(ob):
    """Is there any wind to apply.

    Which property carries the strength depends on whether an object is aiming
    the wind: with one, the vector is unused and wind_strength is the whole
    story, so testing the vector alone would switch the wind off exactly when
    it is being aimed.
    """
    p = getattr(ob, "MC_props", None)
    if p is None:
        return False
    if getattr(p, "wind_object", None) is not None:
        return float(getattr(p, "wind_strength", 0.0)) != 0.0
    return float(np.linalg.norm(np.array(p.wind, dtype=np.float64))) > 0.0


def wind_force(C, vert_normals):
    """Per-vertex wind, in the cloth's local space.

    vert_normals are the ones physics() already computes for air drag and
    inflate, so wind costs no extra normal evaluation.

    The facing term is abs(dot(normal, wind direction)): zero when the surface
    is edge-on, one when it is square to the wind.  It is absolute because a
    cloth's normals point whichever way the sheet happens to face, and a flag
    should not be sucked upwind when it flips over.
    """
    ob = C.ob
    if not is_active(ob):
        return None

    step = int(getattr(C, "wind_step", 0))
    world = gust_vector(ob, step)

    local = to_local(ob, world)
    n = float(np.linalg.norm(local))
    if n <= 1e-12:
        return None             # this gust happens to be a lull, not an error
    direction = local / n

    nrm = np.asarray(vert_normals, dtype=np.float64)
    lens = np.sqrt(np.einsum('ij,ij->i', nrm, nrm))
    np.maximum(lens, 1e-12, out=lens)
    # facing per vertex without normalising the whole array first: the dot with
    # the direction divided by the normal's own length is the same thing
    facing = np.abs(nrm @ direction) / lens

    # spatial variation folds into the same per-vertex scalar as the facing
    # term and the vertex group, so the force is built in one multiply at the
    # end rather than being rewritten three times
    amount = float(getattr(ob.MC_props, "wind_spatial_amount", 0.0))
    if amount > 0.0:
        scale = float(getattr(ob.MC_props, "wind_spatial_scale", 1.0))
        if scale > 1e-9:
            trans = max(1, int(getattr(ob.MC_props, "wind_transition", 30)))
            drift = direction * (step / float(trans))
            pts = np.asarray(C.co, dtype=np.float64) / scale + drift
            nz = value_noise(pts, int(getattr(ob.MC_props, "wind_seed", 1)))
            facing *= 1.0 + amount * (nz * 2.0 - 1.0)

    group = C.group_data.get("MC_wind")
    if group is not None:
        facing *= group[:, 0] if group.ndim == 2 else group

    return local[None, :] * (facing * SCALE)[:, None]


def apply(C, vert_normals):
    """Add the wind to velocity and advance this cloth's wind clock."""
    force = wind_force(C, vert_normals)
    C.wind_step = int(getattr(C, "wind_step", 0)) + 1
    if force is None:
        return 0
    C.velocity += force.astype(C.velocity.dtype)
    return force.shape[0]


# ----------------------------------------------------------------- operators
class MC_OT_wind_add_object(bpy.types.Operator):
    """Add an arrow Empty and use it to aim the wind"""
    bl_idname = "mc.wind_add_object"
    bl_label = "Add Wind Empty"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.object
        empty = bpy.data.objects.new("%s_wind" % ob.name, None)
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = 1.0
        empty.location = ob.matrix_world.translation
        context.collection.objects.link(empty)
        ob.MC_props.wind_object = empty
        self.report({'INFO'}, "Wind aimed by '%s' - rotate it to steer, its "
                              "+Z is downwind" % empty.name)
        return {'FINISHED'}


class MC_OT_gravity_add_object(bpy.types.Operator):
    """Add an arrow Empty and use it to aim gravity"""
    bl_idname = "mc.gravity_add_object"
    bl_label = "Add Gravity Empty"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.object
        empty = bpy.data.objects.new("%s_gravity" % ob.name, None)
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = 1.0
        # pointing down by default, since that is what gravity usually does
        empty.rotation_euler = (np.pi, 0.0, 0.0)
        empty.location = ob.matrix_world.translation
        context.collection.objects.link(empty)
        ob.MC_props.gravity_object = empty
        self.report({'INFO'}, "Gravity aimed by '%s' - its +Z is the pull "
                              "direction" % empty.name)
        return {'FINISHED'}


class MC_PT_panel_wind(bpy.types.Panel):
    bl_label = "Wind"
    bl_idname = "MC_PT_panel_wind"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["wind"]

    def _settings(self, layout, p):
        """Everything except direction and strength, which differ by context."""
        box = layout.box()
        box.label(text="Turbulence")
        row = box.row(align=True)
        row.prop(p, "wind_turbulence_min", text="Min")
        row.prop(p, "wind_turbulence_max", text="Max")
        col = box.column()
        col.scale_y = 0.7
        col.label(text="Multiples of the vector. 0 lets the", icon='INFO')
        col.label(text="wind drop out, negative reverses it.")
        box.prop(p, "wind_direction_variation")
        box.prop(p, "wind_transition")
        box.prop(p, "wind_smoothness")

        box = layout.box()
        box.label(text="Across the Cloth")
        box.prop(p, "wind_spatial_amount")
        sub = box.row()
        sub.enabled = p.wind_spatial_amount > 0.0
        sub.prop(p, "wind_spatial_scale")
        box.prop(p, "wind_seed")

    def draw(self, context):
        layout = self.layout
        ob = context.object
        if ob is None:
            U.panel_note(layout, "Select a cloth or a wind object")
            return

        # The object aiming the wind is the one you grab to steer it, so the
        # settings are shown while it is active too, rather than making you
        # reselect the cloth to change anything.
        driven = cloths_using(ob)
        if driven:
            box = layout.box()
            box.label(text="Wind Object", icon='EMPTY_SINGLE_ARROW')
            box.prop(ob.MC_props, "wind_strength")
            col = box.column()
            col.scale_y = 0.7
            col.label(text="Rotate this object to steer; its", icon='INFO')
            col.label(text="+Z is the way the wind blows.")
            if len(driven) == 1:
                col.label(text="Driving: %s" % driven[0].name)
                self._settings(layout, driven[0].MC_props)
            else:
                col.label(text="Driving %d cloths:" % len(driven))
                for d in driven[:6]:
                    col.label(text="   %s" % d.name)
                box = layout.box()
                box.label(text="Select one to edit its turbulence",
                          icon='INFO')
            return

        if ob.type != 'MESH':
            layout.box().label(text="No mesh selected")
            return
        p = ob.MC_props

        box = layout.box()
        box.label(text="Direction and Strength")
        row = box.row(align=True)
        row.prop(p, "wind_object", text="Aim")
        row.operator("mc.wind_add_object", text="", icon='ADD')
        if p.wind_object is not None:
            # direction comes from the object, so a vector here would be
            # stating it twice -- one number is the whole story
            box.prop(p, "wind_strength")
        else:
            box.prop(p, "wind", text="")

        self._settings(layout, p)


CLASSES = [
    MC_OT_wind_add_object,
    MC_OT_gravity_add_object,
    MC_PT_panel_wind,
]
