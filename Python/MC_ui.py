import bpy
from bpy.types import PropertyGroup, Panel
from bpy.props import (BoolProperty, IntProperty, FloatProperty,
                       CollectionProperty, FloatVectorProperty,
                       PointerProperty, EnumProperty)
import numpy as np
import os
import time
import json


try:
    MC5 = bpy.data.texts['MC5.py'].as_module()
    U = MC5.U
    SEW = bpy.data.texts['sew_tools.py'].as_module()
    GFILL = bpy.data.texts['grid_fill.py'].as_module()
    UVS = bpy.data.texts['uv_shape_tools.py'].as_module()
    HOOK = bpy.data.texts['hook_tools.py'].as_module()
    WIND = bpy.data.texts['wind.py'].as_module()

except:
    from . import MC5
    U = MC5.U
    from . import sew_tools as SEW
    from . import grid_fill as GFILL
    from . import uv_shape_tools as UVS
    from . import hook_tools as HOOK
    from . import wind as WIND


# ===== SEW BEND ===== #
def update_sew_bend_scalers(cloth):
    """Recompute lf/rf_normal_scalers from the current property values."""
    SH = getattr(cloth, 'sew_hinges', None)
    if SH is None or SH.count == 0:
        return

    props = cloth.ob.MC_props
    # ... paste the matrix-construction + scaler code from build_sew_hinges here ...
    # (the block that builds u_hinge, lf_matrix, rf_matrix and then
    #  either measures from start or builds from props.sew_target_angle)
    print("we resetting SH here?????")
    SH.sew_bend_force = np.float32(props.sew_bend_force)


def update_sew_bend(self, context):
    """Called when the enable checkbox changes."""
    cloth = MC5.get_cloth(context.object)          # your existing helper that returns the Cloth object
    if cloth is None:
        return
    # Just force a rebuild of the hinge list (or set a dirty flag)
    # Easiest: call the full builder again - it already early-outs when disabled
    MC5.build_sew_hinges(cloth)


def update_sew_bend_force(self, context):
    """Called when the strength slider changes."""
    cloth = MC5.get_cloth(context.object)
    if cloth is None or not hasattr(cloth, 'sew_hinges') or cloth.sew_hinges is None:
        MC5.dprint("MC: no sew hinges to apply the strength to")
        return
    
    # Strength is read live every frame, but we can also cache it
    cloth.sew_hinges.sew_bend_force = np.float32(self.sew_bend_force)


def update_sew_target_angle(self, context):
    """Called when the target angle changes."""
    cloth = MC5.get_cloth(context.object)
    print("changing target angle")
    if cloth is None or not hasattr(cloth, 'sew_hinges') or cloth.sew_hinges is None:
        return

    print("cloth was not None")
    # Recompute only the scalers
    MC5.build_sew_hinges(cloth)


def update_sew_bend_from_start(self, context):
    """Called when the 'from starting state' checkbox changes."""
    cloth = MC5.get_cloth(context.object)
    if cloth is None or not hasattr(cloth, 'sew_hinges') or cloth.sew_hinges is None:
        return
    # Recompute only the scalers
    MC5.build_sew_hinges(cloth)
# ===== SEW BEND ===== #


def cb_boundary_edges(self, context):
    ob = bpy.context.object
    obm = U.get_bmesh(ob)
    vc = len(obm.verts)
    boundary_mask = np.zeros(vc, dtype=bool)
    boundary_verts = [[e.verts[0].index, e.verts[1].index] for e in obm.edges if len(e.link_faces) == 1]
    boundary_mask[boundary_verts] = True
        

def cb_magnetic_data(self, context):
    ob = self.id_data
    if self.magnetic_target:
        M = MC5.get_magnetic(ob, start=True)
        M.ob = ob
    else:
        MC5.get_magnetic(ob, clear=True)
    for ob in bpy.data.objects:
        if not ob.MC_props.magnetic_target:
            ob.MC_props.magnetic_id = -1


def cb_magnetic_mode_low(self, context):
    if self["magnetic_target_low"]:
        self["magnetic_target_high"] = False
        self["magnetic_target_near"] = False


def cb_magnetic_mode_high(self, context):
    if self["magnetic_target_high"]:
        self["magnetic_target_low"] = False
        self["magnetic_target_near"] = False


def cb_magnetic_mode_near(self, context):
    if self["magnetic_target_near"]:
        self["magnetic_target_low"] = False
        self["magnetic_target_high"] = False


def cb_ob_collision_data(self, context):
    ob = self.id_data
    if self.ob_collision:
        OC = MC5.get_ob_collision(ob, start=True)
        OC.ob = ob
    else:
        MC5.get_ob_collision(ob, clear=True)
    for ob in bpy.data.objects:
        if not ob.MC_props.ob_collision:
            ob.MC_props.ob_collision_id = -1

    MC5.update_ob_colliders(MC5.OC_DATA)


def cb_collider_friction(self, context):
    # re-bake the joined friction arrays when this collider's friction scalar
    # changes (vertex-group *painting* is picked up per frame in edit / weight
    # paint mode instead)
    if self.ob_collision and MC5.OC_DATA.get("obs"):
        MC5.read_collider_friction(MC5.OC_DATA)


def cloth_refresh(ob):
    C = MC5.get_cloth(ob, start=True)
    C.obm = U.get_bmesh(ob)
    MC5.install_handler()


def cb_cloth(self, context):
    ob = self.id_data
    if self.cloth:
        ob.update_from_editmode()
        bad_choice = True
        msg = "How can I make cloth from an " + ob.type + "? Seriously, some people..."
        if ob.type == "MESH":
            msg = "Mesh has no faces. How can it function as a person in civilized society without a face? What is this, Stranger Things?"
            if len(ob.data.polygons) > 0:
                C = MC5.get_cloth(ob, start=True)
                if C is None:
                    return
                C.obm = U.get_bmesh(ob)
                MC5.install_handler()
                bad_choice = False
        if bad_choice:
            U.popup_error(msg)
            self['cloth'] = False    
    else:
        MC5.get_cloth(ob, clear=True)
    
    for ob in bpy.data.objects:
        if not ob.MC_props.cloth:
            ob.MC_props.cloth_id = -1


def cb_continuous(self, context):
    # Continuous and cache playback are mutually exclusive; playback wins.
    if self.continuous and self.cache_playback:
        self["continuous"] = False
        U.popup_error("Disable Cache Playback to run Continuous.", icon='INFO')
        return
    cb_cloth(self, context)


def cb_cache_playback(self, context):
    ob = self.id_data

    if not self.cache_playback:
        return

    if self.continuous:
        self["continuous"] = False
        U.popup_error("Disabled Continuous - cache playback takes over.", icon='INFO')

    # playback rides the frame-change handler, which only visits objects whose
    # 'cloth' is on -- make sure the Cloth data + handler exist
    if not self.cloth:
        self.cloth = True          # runs cb_cloth: builds C, installs the handler
    else:
        MC5.install_handler()

    ok, cvc = MC5.CACHE.vcount_matches(ob)
    if ok is None:
        U.popup_error("No cache for '%s' yet - bake first." % ob.name, icon='INFO')
        self["cache_playback"] = False
    elif not ok:
        U.popup_error("Cache has %d verts, mesh has %d - re-bake '%s'."
                      % (cvc, len(ob.data.vertices), ob.name), icon='ERROR')
        self["cache_playback"] = False


def cb_sew_face_fill(self, context):
    """Toggling Face Fill adds/removes the fill faces on the seam the current
    selection is touching.  With nothing relevant selected it is just the
    default for the next sew, so a no-op is fine."""
    try:
        n, warn = SEW.apply_face_fill(context, bool(self.sew_face_fill))
    except Exception as e:
        print("sew face fill:", e)
        return
    if warn:
        U.popup_error(warn, icon='INFO')


def cb_target_update(self, context):
    MC5.cloth_refresh(context.object)


# ===================================================================== PRESETS
# The single source of truth for which MC_props fields a preset captures.  To
# make a new property preset-able, add its name here -- nothing else to touch.
#
# Deliberately excluded: identity / data-dict keys (cloth, *_id, ob_collision,
# magnetic_target, cache_id), run mode (animated, continuous, reset_frame), the
# whole cache_* set, collider-side ob_collider_friction, magnetic *target*
# tuning, dev scaffolding (debug*, e_bend_force), every grid_* / uv tool (their
# update callbacks rebuild the mesh), and target_object (an object pointer).
PRESET_KEYS = (
    # integration / forces
    "gravity", "velocity", "stretch", "bend_force", "bend_stabilize",
    "shrink_grow",
    # sewing
    "sew_force", "butt_sew_force", "target_sew_length",
    "sew_bend", "sew_bend_from_start", "sew_bend_force", "sew_target_angle",
    # self collision
    "self_collision", "sc_edges", "sc_radius",
    "sc_damping", "sc_vel_damping", "sc_box_depth",
    "sc_box_depth_auto", "auto_radius",
    # object collision
    "cl_point_tris", "ob_point_tris", "ob_edges", "ob_recollide",
    "ob_recollide_every",
    "ob_box_depth", "ob_box_depth_auto", "ob_collision_radius",
    "ob_collision_tri_damping",
    "ob_collision_edge_damping", "ob_friction", "ob_static_threshold",
    # collision substeps
    "collision_substeps", "collision_auto_substeps", "collision_max_substeps",
    "collision_substep_margin",
    # air effects
    "inflate", "air_drag", "inverted_air_drag",
    # wind (wind_object is a pointer, so it is not preset-able)
    "wind", "wind_strength",
    "wind_turbulence_min", "wind_turbulence_max", "wind_transition",
    "wind_smoothness", "wind_direction_variation", "wind_spatial_amount",
    "wind_spatial_scale", "wind_seed",
    # axis lock / boundary
    "lock_axis_x", "lock_axis_y", "lock_axis_z", "preserve_boundary_edges",
    # magnetic (cloth side)
    "magnetic_force", "magnetic_range",
)

PRESET_VERSION = 1

# Fields the pre-JSON MC_Preset stored as real sub-properties.  Old presets keep
# these as custom props on the collection item; preset_apply migrates them.
PRESET_LEGACY_KEYS = ("gravity", "velocity", "stretch", "bend_force",
                      "shrink_grow", "sew_force", "butt_sew_force", "sew_bend")


def _preset_value(v):
    """JSON-safe form of a property value.

    Vector properties come back as bpy_prop_array, which json cannot encode,
    so they are stored as plain lists.
    """
    if hasattr(v, "__len__") and not isinstance(v, str):
        return [_preset_value(x) for x in v]
    return v


def preset_store(preset, props):
    """Snapshot the preset-able MC_props fields into the preset as a JSON blob."""
    preset.version = PRESET_VERSION
    preset.data = json.dumps({k: _preset_value(getattr(props, k))
                              for k in PRESET_KEYS if hasattr(props, k)})


def preset_apply(preset, props):
    """Write a preset's stored values back onto an MC_props.

    - reads the JSON blob, or migrates a pre-JSON preset from its custom props
    - coerces each value to the live property's type (so a legacy float landing
      on a bool/int field just works)
    - unknown keys (property removed since) and out-of-range values are skipped
      individually rather than aborting the whole load
    """
    saved = {}
    if preset.data:
        try:
            saved = json.loads(preset.data)
        except ValueError:
            print("preset '%s': corrupt data, nothing applied" % preset.name)
            return
    else:
        for k in PRESET_LEGACY_KEYS:
            if k in preset.keys():
                saved[k] = preset[k]
        if saved:
            print("preset '%s': migrated %d legacy value(s)" % (preset.name, len(saved)))

    for k, v in saved.items():
        if not hasattr(props, k):
            continue
        try:
            cur = getattr(props, k)
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):                 # bool already handled above
                v = int(round(v))
            elif isinstance(cur, float):
                v = float(v)
            elif hasattr(cur, "__len__"):              # vector property
                v = [float(x) for x in v][:len(cur)]
            setattr(props, k, v)
        except Exception as e:
            print("preset '%s': skipped %s (%s)" % (preset.name, k, e))


class MC_Preset(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="Untitled Preset")
    data: bpy.props.StringProperty()                       # json blob, see PRESET_KEYS
    version: bpy.props.IntProperty(default=PRESET_VERSION)


class MC_props_scene(PropertyGroup):
    # Removed unused properties: sc_name, name_2, cloth_id, mc_preset_index
    # If needed, they can be added back

    # Hides developer-only controls from normal use.  Nothing in the solver
    # reads it -- it only decides what the panels draw.
    debug_mode: BoolProperty(
        name="Debug Mode",
        description="Show developer settings that normal use never needs",
        default=False,
    )
    
    collision_backend: EnumProperty(
        name="Collision",
        description="Which implementation runs object and self collision. C++ is "
                    "faster; Python is the reference it is checked against",
        items=[('CPP', "C++", "mc_collide.dll -- falls back to Python if it is missing"),
               ('PYTHON', "Python", "The reference implementation")],
        default='CPP',
    )

    mc_presets: CollectionProperty(
        type=MC_Preset,
        name="MC_Preset",
    )

    dll_path: bpy.props.StringProperty(
        name="Solver DLL",
        description="Where the cloth solver DLL lives. Leave empty to search "
                    "the addon folder automatically. Only set this if the "
                    "addon cannot find it",
        default="",
        subtype='FILE_PATH',
    )

    # ===== CACHE BAKE STATE ===== #
    # Baking is a modal operator, so the panel cannot reach into it directly.
    # These two let it: the operator publishes that it is running, and watches
    # the pause flag, which both the panel button and the keyboard shortcut set.
    cache_baking: BoolProperty(
        name="Baking",
        description="A cache bake is running",
        default=False,
    )

    cache_bake_paused: BoolProperty(
        name="Bake Paused",
        description="Hold the bake where it is. Resume carries on from the "
                    "same frame with the sim state intact",
        default=False,
    )
    # ===== CACHE BAKE STATE ===== #

    all_magnetic_targets: BoolProperty(
        name="All Magnetic Targets",
        description="Apply magnetic force to all",
        default=True,
        update=cb_magnetic_data,
    )
    
    reset_selected: BoolProperty(
        name="Reset Selected Vertices",
        description="Reset only affectes selected verts (if those verts are sober and not kittens)",
        default=False,
    )    
    
    text_export_path: bpy.props.StringProperty(
        name="Export Folder",
        description="Folder to save internal text files as .py files",
        default="",
        subtype='DIR_PATH',
    )

    sew_face_fill: BoolProperty(
        name="Face Fill",
        description=("Build faces across the seam instead of leaving bare sew "
                     "edges, joining the two panels into one surface. Toggling "
                     "this adds or removes the faces on the currently selected "
                     "seam, and sets what new seams do. Note that filled edges "
                     "have faces, so the cloth solver stops treating them as "
                     "sew edges and no longer pulls the panels together"),
        default=False,
        update=cb_sew_face_fill,
    )

    cache_dir: bpy.props.StringProperty(
        name="Cache Folder",
        description=("Where cloth caches are written.  '//' is relative to the "
                     ".blend file; if the file is unsaved the system temp folder "
                     "is used instead"),
        default="//mc_cache/",
        subtype='DIR_PATH',
        # Blender 5 warns about (and won't keep) '//' paths without this
        options={'PATH_SUPPORTS_BLEND_RELATIVE'},
    )


class MC_props(PropertyGroup):
    cloth: BoolProperty(
        name="Cloth",
        description="Enable cloth simulation",
        default=False,
        update=cb_cloth,
    )

    reset_frame: IntProperty(
        name="Reset Frame",
        description="Auto reset at this frame",
        default= -1,
    )

    animated: BoolProperty(
        name="Animated",
        description="Runs with blender animation",
        default=False,
        update=cb_cloth,
    )

    continuous: BoolProperty(
        name="Continuous",
        description="Runs as fast as possible",
        default=False,
        update=cb_continuous,
    )

    cloth_id: IntProperty(
        name="Cloth ID",
        description="Key into the data dictionary.",
        default=-1
    )

    # ===== CACHE ===== #
    cache_id: IntProperty(
        name="Cache ID",
        description="Stable key for this object's cache folder (mc_<id>)",
        default=0,
    )

    cache_playback: BoolProperty(
        name="Cache Playback",
        description=("Play the baked cache instead of simulating.  Scrubs and "
                     "plays backwards.  Disables Continuous while on"),
        default=False,
        update=cb_cache_playback,
    )

    cache_use_scene_range: BoolProperty(
        name="Use Scene Range",
        description="Bake over the scene frame range instead of Cache Start/End",
        default=True,
    )

    cache_start: IntProperty(
        name="Cache Start",
        description="First frame to bake",
        default=1,
    )

    cache_end: IntProperty(
        name="Cache End",
        description="Last frame to bake",
        default=100,
    )

    cache_reset_on_bake: BoolProperty(
        name="Reset On Bake",
        description="Delete existing cached frames before baking",
        default=True,
    )

    cache_in_memory: BoolProperty(
        name="Load In Memory",
        description="Hold the whole cache in RAM for faster scrub / playback",
        default=True,
    )
    # ===== CACHE ===== #

    gravity: FloatProperty(
        name="Gravity",
        description="Make stuff fall, like off a cliff to its death. Pulls "
                    "along world -Z unless a gravity object aims it",
        default=0.0,
        precision=6,
    )

    velocity: FloatProperty(
        name="Velocity",
        description="Make stuff fall, like off a cliff to its death.",
        default=0.98,
        precision=6,
    )

    stretch: FloatProperty(
        name="Stretch",
        description="Value for linear stretch.",
        default=1.25,
        min=0.0,
        precision=6,
    )

    shrink_grow: FloatProperty(
        name="Shrink Grow",
        description=("Multiplier on the rest length of every linear spring. "
                     "Below 1 gathers the cloth in, above 1 lets it out"),
        default=1.0,
        min=0.0,            # a negative rest length turns the springs inside out
        soft_max=2.0,       # no hard ceiling: type in whatever you want
        precision=6,
    )


    e_bend_force: FloatProperty(
        name="!= e_bend Debug =!",
        description="Test Value.",
        default=1.0,
        precision=6,
    )

    bend_stabilize: FloatProperty(
        name="Stabilize",
        description="Higher to increase stiffness, lower for stability.",
        default=1.0,
        precision=6,
    )

    target_sew_length: FloatProperty(
        name="Target Sew Length",
        description="Sew springs at this distance.",
        default=0.0,
        precision=6,
    )

    butt_sew_force: FloatProperty(
        name="Merge Range",
        description="Force that holds sewn points to the mean location.",
        default=0.0,
        soft_min=0.0,
        precision=6,
    )

    sew_force: FloatProperty(
        name="Sew Force",
        description="Force sews points together.",
        default=0.0,
        soft_min=0.0,
        soft_max=1.0,
        precision=6,
    )

    sew_bend: BoolProperty(
        name="Sew Bend",
        description="Bend force including sew edges",
        default=True,
        update=update_sew_bend,
    )

    sew_bend_from_start: BoolProperty(
        name="Sew Bend From Target",
        description="Use Target Shape to set Sew Bend Angles",
        default=True,
        update=update_sew_bend_from_start,
    )

    sew_bend_force: FloatProperty(
        name="Sew Bend Force",
        description="Bend force for sew edges",
        default=1.0,
        precision=6,
        update=update_sew_bend_force,
    )

    sew_target_angle: FloatProperty(
        name="Sew Bend Target Angle",
        description="Bend force for sew edges",
        default=0.0,
        precision=6,
        update=update_sew_target_angle,
    )

    bend_force: FloatProperty(
        name="Bend Force",
        description="Bend force including sew edges",
        soft_min=0.0,
        #soft_max=1.24,
        default=0.0,
        precision=6,
    )

    # ===== SELF COLLISION ===== #

    sc_radius: FloatProperty(
        name="SC Radius",
        description="Target margin for self collisions",
        soft_min=0.0001,
        soft_max=2.5,
        default=0.05,
        precision=6,
    )

    auto_radius: BoolProperty(
        name="SC Auto Radius",
        description="Set the radius once then set this to False",
        default=True,
    )

    sc_damping: FloatProperty(
        name="SC Damping",
        description="Calm the forces of the self collisions.",
        soft_min=0.0001,
        soft_max=2.5,
        default=1.0,
        precision=6,
    )

    sc_vel_damping: FloatProperty(
        name="SC Velocity Damping",
        description="Reduce velocity when self collisions hapen.",
        soft_min=0.0,
        soft_max=1,
        default=0.5,
    )

    self_collision: BoolProperty(
        name="Self Collision",
        description="Do the self collision thang!.",
        default=False,
    )

    sc_edges: BoolProperty(
        name="Self Collision Edge Collisions",
        description="Detect edge to edge self collisions.",
        default=True,
    )

    
    # Split depth, not an absolute pair count: the broad phase stops splitting
    # once a box holds root_pairs / 2**depth.  The fastest setting turned out
    # to be a fixed number of splits at every mesh size tried, so an absolute
    # count got slower the bigger the mesh.  Depth only changes speed -- the
    # collision result is identical at every value.  See split_tuner.py.
    sc_box_depth: IntProperty(
        name="SC Split Depth",
        description="How many times self collision splits space before "
                    "testing pairs directly. Changes speed only, never the "
                    "result. 5 is close to the fastest for most meshes",
        default=5,
        min=0,
        max=16,
    )

    sc_box_depth_auto: BoolProperty(
        name="SC Auto Split Depth",
        description="Tune the split depth while the sim runs, by timing it. "
                    "Averages several steps and only moves when the gain is "
                    "real, then settles",
        default=False,
    )
    
    # ===== COLLISION SUBSTEPS ===== #
    # A collision test compares where a point was with where it is now.  If it
    # travelled further than the collision margin in one step it can start on
    # one side of a surface and finish on the other, so nothing is ever inside
    # the margin and the contact is missed entirely -- measured on a cloth
    # dropped onto a sphere, contact holds while movement stays under about
    # twice the margin and is lost completely beyond that.  Subdividing the
    # movement and testing each slice keeps every step inside the margin.
    #
    # Defaults are deliberately inert: 1 step and auto off reproduce the
    # previous behaviour exactly.
    collision_substeps: IntProperty(
        name="Collision Substeps",
        description="Split each step's movement into this many collision "
                    "tests. 1 is the original behaviour",
        default=1,
        min=1,
        soft_max=16,
    )

    collision_auto_substeps: BoolProperty(
        name="Auto Substeps",
        description="Choose the substep count automatically by comparing how "
                    "far things move against the collision margin",
        default=False,
    )

    collision_max_substeps: IntProperty(
        name="Max Substeps",
        description="Upper limit when Auto Substeps is on, so a fast frame "
                    "cannot stall the sim",
        default=8,
        min=1,
        soft_max=64,
    )

    collision_substep_margin: FloatProperty(
        name="Substep Safety",
        description="Fraction of the collision margin a single substep is "
                    "allowed to move. Lower is safer and slower",
        default=0.5,
        min=0.05,
        soft_max=1.0,
        precision=6,
    )
    # ===== COLLISION SUBSTEPS ===== #

    # ===== WIND ===== #
    wind: FloatVectorProperty(
        name="Wind",
        description="Wind direction and strength. Force on each vertex is "
                    "scaled by how squarely it faces the wind: nothing when "
                    "the surface is edge-on, full when it is square to it",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
        precision=6,
    )

    gravity_object: PointerProperty(
        name="Gravity Object",
        description="Aim gravity by rotating this object. Its +Z axis is the "
                    "direction things fall, so rotate it to point the way you "
                    "want. Only the size of the Gravity value is used, not "
                    "its sign. Leave empty to fall along world Z",
        type=bpy.types.Object,
    )

    # Used instead of the vector when an object is aiming the wind: the object
    # supplies the direction, so all that is left to say is how hard it blows.
    # Mirrored between the cloth and that object by cb_wind_strength.
    wind_strength: FloatProperty(
        name="Wind Strength",
        description="How hard the wind blows. Direction comes from the wind "
                    "object. Shared with that object, so changing it in "
                    "either place changes both",
        default=1.0,
        soft_min=-10.0,
        soft_max=10.0,
        precision=6,
        update=WIND.cb_wind_strength,
    )

    wind_object: PointerProperty(
        name="Wind Object",
        description="Aim the wind by rotating this object instead of typing a "
                    "vector. Its local +Z is the direction the wind blows. "
                    "The Wind vector above still sets the strength",
        type=bpy.types.Object,
    )

    # Deliberately uncapped.  A floor of 0 lets the wind die away between
    # gusts, a negative floor lets it blow back, and there is no ceiling, so a
    # range like -0.5 to 4 gusts, drops out, and occasionally reverses.  The
    # soft range only sets how far the slider drags.
    wind_turbulence_min: FloatProperty(
        name="Turbulence Min",
        description="Low end of the gust range, as a multiple of the wind "
                    "vector. 0 lets the wind die away, negative lets it "
                    "reverse",
        default=1.0,
        soft_min=-2.0,
        soft_max=4.0,
        precision=6,
    )

    wind_turbulence_max: FloatProperty(
        name="Turbulence Max",
        description="High end of the gust range, as a multiple of the wind "
                    "vector. Not capped",
        default=1.0,
        soft_min=-2.0,
        soft_max=8.0,
        precision=6,
    )

    wind_transition: IntProperty(
        name="Gust Length",
        description="Steps taken to travel from one gust to the next. Low is "
                    "squally, high is a slow swell",
        default=30,
        min=1,
        soft_max=300,
    )

    wind_smoothness: FloatProperty(
        name="Smoothness",
        description="How gently the wind moves between values. 0 changes at a "
                    "constant rate with a corner at each gust, 1 eases in and "
                    "out of every one",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=6,
    )

    wind_direction_variation: FloatProperty(
        name="Direction Variation",
        description="How far the wind swings off its aim, in degrees. Real "
                    "wind shifts direction as well as strength",
        default=0.0,
        min=0.0,
        soft_max=89.0,
        precision=6,
    )

    wind_spatial_amount: FloatProperty(
        name="Variation Across Cloth",
        description="How much the wind differs from place to place, so gusts "
                    "travel across the surface instead of the whole sheet "
                    "breathing as one. 0 applies the same wind everywhere",
        default=0.0,
        min=0.0,
        soft_max=1.0,
        precision=6,
    )

    wind_spatial_scale: FloatProperty(
        name="Variation Size",
        description="Size of the patches of stronger and weaker wind, in "
                    "scene units",
        default=1.0,
        min=0.0001,
        soft_max=20.0,
        precision=6,
    )

    wind_seed: IntProperty(
        name="Seed",
        description="Change for a different but equally repeatable gust "
                    "pattern",
        default=1,
        min=0,
    )
    # ===== WIND ===== #

    # ===== SELF COLLISION ===== #

    # ===== OBJECT COLLISION ===== #
    ob_collision: BoolProperty(
        name="Object Collision",
        description="Detect collisions with this object even if it's a lobster (but not a 17mm box end wrench).",
        update=cb_ob_collision_data,
        default=False,
    )

    ob_edges: BoolProperty(
        name="Object Edge Collision",
        description="Detect edge to edge collisions.",
        default=True,
    )

    ob_point_tris: BoolProperty(
        name="Object Point to Cloth Tris",
        description="Detect object points against cloth tris.",
        default=True,
    )

    cl_point_tris: BoolProperty(
        name="Cloth Point to Object Tris",
        description="Detect cloth points against object tris.",
        default=True,
    )

    # See sc_box_depth.  The old absolute default of 37 was 1.2-4.9x slower
    # than a depth of 5 across the meshes measured.
    ob_box_depth: IntProperty(
        name="Object Split Depth",
        description="How many times object collision splits space before "
                    "testing pairs directly. Changes speed only, never the "
                    "result. 5 is close to the fastest for most meshes",
        default=5,
        min=0,
        max=16,
    )

    ob_box_depth_auto: BoolProperty(
        name="Object Auto Split Depth",
        description="Tune the split depth while the sim runs, by timing it. "
                    "Averages several steps and only moves when the gain is "
                    "real, then settles",
        default=False,
    )
    
    ob_collision_id: IntProperty(
        name="Object Collision ID",
        description="ID for collsion objects",
        default=-1,
    )

    
    ob_recollide: BoolProperty(
        name="Recheck Collisions",
        description="Re-resolve collider contacts between the bend and "
                    "stretch iterations, so stiff cloth cannot be dragged "
                    "through sharp points. Slower",
        default=False,
    )

    ob_recollide_every: IntProperty(
        name="Recollide Every",
        description="Recheck contacts after every Nth bend or stretch "
                    "iteration. 1 is every iteration, higher is cheaper and "
                    "less accurate",
        default=1,
        min=1,
        soft_max=8,
    )
    
    ob_collision_radius: FloatProperty(
        name="Collision Radius",
        description="Margin for object collisions",
        soft_min=0.0001,
        default=0.05,
        precision=6,
    )    
    
    
    ob_collision_tri_damping: FloatProperty(
        name="Collision Tri Dramping",
        description="Ob Point Cloth Tri Force",
        min=0.0,
        default=1.0,
        precision=6,
    )
    
    ob_collision_edge_damping: FloatProperty(
        name="Collision Edge Dramping",
        description="Ob edge to edge force",
        min=0.0,
        default=1.0,
        precision=6,
    )

    ob_friction: FloatProperty(
        name="Cloth Friction",
        description=("CLOTH property: master cloth-to-object friction.  0 = "
                     "frictionless, 1 = max grip.  The per-contact value is this "
                     "x the cloth MC_friction group x the collider's Collider "
                     "Friction x its MC_ob_friction group"),
        min=0.0,
        max=1.0,
        default=0.0,
        precision=4,
    )

    ob_collider_friction: FloatProperty(
        name="Collider Friction",
        description=("COLLIDER property: this collider's friction, multiplied by "
                     "its MC_ob_friction vertex group.  Set to 1 and paint the "
                     "group to 0 for slick spots"),
        min=0.0,
        max=1.0,
        default=1.0,
        precision=4,
        update=cb_collider_friction,
    )

    ob_static_threshold: FloatProperty(
        name="Static Friction Threshold",
        description=("CLOTH property: slip distance (world units) below which a "
                     "collided vertex stays stuck.  0 = derive the threshold "
                     "from the contact force instead (friction x normal push)"),
        min=0.0,
        soft_max=1.0,
        default=0.0,
        precision=6,
    )

    # ===== OBJECT COLLISION ===== #

    # ===== MAGNETIC ===== #
    magnetic_id: IntProperty(
        name="Magnetic ID",
        description="Key into the magnetic data dictionary.",
        default=-1
    )

    magnetic_force: FloatProperty(
        name="Magnetic Force",
        description="Force to move towards point clouds",
        soft_min=0.0,
        soft_max=1.0,
        default=0.0,
        precision=6,
        update=cb_magnetic_data,
    )

    # Lives on the hook Empty, not on the cloth: each hook pulls by its own
    # amount.  1.0 makes the bound verts follow the Empty rigidly, like the
    # hook modifier; lower values let them lag behind it.
    hook_force: FloatProperty(
        name="Hook Force",
        description="How hard this hook pulls its vertices. 1.0 follows the "
                    "empty exactly, 0.0 does nothing",
        soft_min=0.0,
        soft_max=1.0,
        min=0.0,
        default=1.0,
        precision=6,
    )
    
    magnetic_range: FloatProperty(
        name="Magnetic Range",
        description="Range to look for points",
        min=0.0,
        #soft_max=1.0,
        default=0.01,
        precision=6,
    )    

    magnetic_target: BoolProperty(
        name="Magnetic Target",
        description="Check this object for magnetic force",
        default=False,
        update=cb_magnetic_data,
    )

    magnetic_target_high: BoolProperty(
        name="Magnetic Target High",
        description="Pick forces highest on the normal.",
        default=False,
        update=cb_magnetic_mode_high,
    )

    magnetic_target_near: BoolProperty(
        name="Magnetic Target Near",
        description="Pick forces from nearest neighbor.",
        default=False,
        update=cb_magnetic_mode_near,
    )

    magnetic_target_low: BoolProperty(
        name="Magnetic Target Low",
        description="Pick forces lowest on the normal.",
        default=False,
        update=cb_magnetic_mode_low,
    )

    magnetic_neightbor_count: IntProperty(
        name="Magnetic Neighbor Count",
        description="How many neighbors when using high and low.",
        default=1,
    )

    preserve_boundary_edges: BoolProperty(
        name="Preserve Boundary Edges",
        description="Finds noundary edges and treats them as pinned.",
        default=False,
        update=cb_boundary_edges,
    )
    
    lock_axis_x: BoolProperty(
        name="Lock movement on this axis",
        description="Locks the movement of the cloth on its local X axis.",
        default=False,
    )    
    
    lock_axis_y: BoolProperty(
        name="Lock movement on this axis",
        description="Locks the movement of the cloth on its local Y axis.",
        default=False,
    )    

    lock_axis_z: BoolProperty(
        name="Lock movement on this axis",
        description="Locks the movement of the cloth on its local Z axis.",
        default=False,
    )        
    # ===== MAGNETIC ===== #


    # ===== AIR EFFECTS ===== #
    inflate: FloatProperty(
        name="Inflate",
        description="Key into the magnetic data dictionary.",
        default=0.0,
        precision=6,
    )

    air_drag: FloatProperty(
        name="Air Drag",
        description="Change velocity relative to vertex normals",
        default=0.0,
        soft_max=1.0,
        soft_min=0.0,
        precision=6,
    )
    
    inverted_air_drag: FloatProperty(
        name="Inverted Air Drag",
        description="Change velocity relative to vertex normals",
        default=0.0,
        soft_max=1.0,
        soft_min=0.0,
        precision=6,
    )    
    # ===== AIR EFFECTS ===== #


        
    
    
    

    # ===== GRID FILL ===== #
    gf_fill_type: bpy.props.EnumProperty(
        name="Fill",
        description="What to fill the loop with",
        items=[
            ('QUADS', "Quads", "Lattice of quads, triangulated band at the border"),
            ('TRIS_GRID', "Triangles (Grid)",
             "Staggered lattice of uniform triangles"),
            ('TRIS_POISSON', "Triangles (Even)",
             "Poisson-sampled triangles of similar area. No grid direction, so "
             "cloth made from it stretches the same way on every axis instead "
             "of being stiffer along the grid lines"),
        ],
        default='QUADS', update=GFILL.cb_live)

    gf_spacing: FloatProperty(
        name="Spacing",
        description="Grid cell size / sample spacing. Point count grows with "
                    "the square of 1/Spacing, so small values get expensive "
                    "fast - the build refuses rather than locking up, and says "
                    "what value to use instead",
        min=0.00001, soft_min=0.001, soft_max=10.0,
        default=0.1, precision=6, update=GFILL.cb_live)

    gf_flatten: bpy.props.EnumProperty(
        name="Flatten",
        description="How to lay a non-planar loop out flat before filling",
        items=[
            ('AUTO', "Auto", "Project when the loop is near planar, unroll when not"),
            ('PROJECT', "Project", "Drop onto the best-fit plane. Exact when flat"),
            ('UNROLL', "Unroll",
             "Rebuild in 2D from the 3D edge lengths and turning angles, so "
             "edge lengths survive. Better for domed or saddle loops"),
        ],
        default='PROJECT', update=GFILL.cb_live)

    gf_planarity_limit: FloatProperty(
        name="Planarity Limit",
        description="Auto switches to Unroll above this (plane distance over "
                    "loop diameter)",
        min=0.0, soft_max=0.2, default=0.02, precision=4, update=GFILL.cb_live)

    gf_angle_limit: FloatProperty(
        name="Corner Angle",
        description="Border turns sharper than this are kept as vertices",
        min=0.0, max=180.0, default=20.0, update=GFILL.cb_live)

    gf_rotate: FloatProperty(
        name="Rotate", description="Rotate the lattice (degrees)",
        default=0.0, update=GFILL.cb_live)

    gf_scale: FloatProperty(
        name="Scale",
        description="Multiplier on Spacing. Affects the border sampling and the "
                    "fill together, so they stay the same density",
        min=0.001, soft_min=0.05, soft_max=10.0,
        default=1.0, precision=6, update=GFILL.cb_live)

    gf_offset: bpy.props.FloatVectorProperty(
        name="Grid Offset",
        description="Slide the lattice by this fraction of a cell, for nudging "
                    "the grid lines until they sit well against the shape. Does "
                    "not change density",
        size=2, default=(0.0, 0.0), soft_min=-1.0, soft_max=1.0,
        update=GFILL.cb_live)

    gf_inset: FloatProperty(
        name="Border Inset",
        description="Drop lattice points this close to the border, as a "
                    "fraction of Spacing, so the stitched band has room",
        min=0.0, soft_max=2.0, default=0.6, update=GFILL.cb_live)

    gf_seed: IntProperty(
        name="Seed", description="Poisson sampling seed",
        default=0, update=GFILL.cb_live)

    gf_relax_iters: IntProperty(
        name="Relax",
        description="Laplacian passes over the Poisson samples. Measured not to "
                    "improve area uniformity, so 0 by default -- a visual choice",
        min=0, soft_max=20, default=0, update=GFILL.cb_live)

    gf_smooth_iters: IntProperty(
        name="Smooth",
        description="Relax the interior with the border locked. This is what "
                    "conforms the patch to a non-planar loop. Always recomputed "
                    "from the unsmoothed shape, so it never accumulates",
        min=0, soft_max=1000, default=10, update=GFILL.cb_smooth_live)

    gf_max_edge: FloatProperty(
        name="Max Edge",
        description="Drop triangles with a side longer than this times Spacing. "
                    "0 = off. Only meaningful when the interior really is "
                    "sampled at Spacing",
        min=0.0, soft_max=10.0, default=0.0, update=GFILL.cb_live)

    gf_qhull: bpy.props.StringProperty(
        name="Qhull",
        description="Options passed to scipy's Delaunay, e.g. QJ to joggle "
                    "degenerate input. Blank for the default",
        default="", update=GFILL.cb_live)

    gf_holes: BoolProperty(
        name="Holes",
        description="Treat a loop nested inside another as a hole rather than "
                    "its own patch",
        default=True, update=GFILL.cb_live)

    gf_selected_only: BoolProperty(
        name="Selected",
        description="Only use selected boundary edges",
        default=False, update=GFILL.cb_live)

    gf_live: BoolProperty(
        name="Live",
        description="Rebuild whenever a setting changes. Rebuilds are queued "
                    "and coalesced, so dragging a slider triggers one rebuild "
                    "rather than one per frame. Turn off if a heavy loop makes "
                    "the sliders chug",
        default=True)
    # ===== GRID FILL ===== #

    # ===== UV SHAPE ===== #
    uvs_map: bpy.props.EnumProperty(
        name="UV Map",
        description="Which UV map to read the flat layout from",
        items=UVS.uv_map_items)

    uvs_unwrap: bpy.props.EnumProperty(
        name="Unwrap",
        description="Run an unwrap before reading the UVs",
        items=[
            ('NONE', "Use Existing", "Read the UV map as it is"),
            ('ANGLE_BASED', "Angle Based", "Blender's default. Best angles"),
            ('CONFORMAL', "Conformal", "LSCM. Faster, more stretch"),
            ('SMART', "Smart Project",
             "Auto-seams by angle. The one that works on a closed solid with "
             "no seams marked, at the cost of many small islands"),
        ],
        default='ANGLE_BASED')

    uvs_seam_source: bpy.props.EnumProperty(
        name="Seams From",
        description="Where to get seams when the mesh needs cutting open",
        items=[
            ('AUTO', "Auto (Topological)",
             "Work out the cuts from the topology: handle loops, an opening arc "
             "for a closed shell, and a path joining extra boundary loops"),
            ('SHARP', "Sharp Edges", "Where faces meet above an angle"),
            ('PLANE', "Plane", "Edges lying on a plane - the symmetry cut"),
            ('SELECTION', "Selection", "Currently selected edges"),
            ('MC_SEAMS', "MC Seams",
             "Where sew_tools joined panels, from the MC_seam_id attribute"),
        ],
        default='AUTO')

    uvs_sharp_angle: FloatProperty(
        name="Sharp Angle", min=0.0, max=180.0, default=40.0)

    uvs_plane_axis: bpy.props.EnumProperty(
        name="Axis", items=[('0', "X", "X"), ('1', "Y", "Y"), ('2', "Z", "Z")],
        default='0')

    uvs_plane_offset: FloatProperty(
        name="Offset", default=0.0, precision=6)

    uvs_plane_tol: FloatProperty(
        name="Tolerance", min=0.0, default=0.0001, precision=6)

    uvs_shape_name: bpy.props.StringProperty(
        name="Shape Key",
        description="Shape key to write the flat pattern into. Basis is created "
                    "if the mesh has no keys yet",
        default="uv_shape")

    uvs_scale_mode: bpy.props.EnumProperty(
        name="Scale",
        description="How to size the flat layout against the 3D mesh",
        items=[
            ('LEAST_SQUARES', "Least Squares",
             "Minimise squared edge-length error. Best when islands stretch "
             "unevenly"),
            ('EDGE_MEAN', "Edge Mean", "Match the average edge length"),
            ('AREA', "Area", "Match total area"),
        ],
        default='LEAST_SQUARES')

    uvs_flat_axis: bpy.props.EnumProperty(
        name="Flat Axis",
        description="Axis the flat pattern is flattened along",
        items=[('0', "X", "Lay out in YZ"), ('1', "Y", "Lay out in XZ"),
               ('2', "Z", "Lay out in XY")],
        default='2')

    uvs_warn_closed: BoolProperty(
        name="Warn If Not Unwrappable",
        description="Check the topology before building and say so when pieces "
                    "will overlap",
        default=True)
    # ===== UV SHAPE ===== #

    # ===== TARGET ===== #
    target_object: bpy.props.PointerProperty(
        name="Target Object",
        description="Get Data From This Object",
        type=bpy.types.Object,
        update=cb_target_update)    
    # ===== TARGET ===== #


def export_internal_texts(context):
    """Save every internal text datablock as a .py file into the folder
    set in scene.mc_props.text_export_path.
    Returns (success_count, error_list).
    """
    folder = bpy.path.abspath(context.scene.MC_props.text_export_path)

    if not folder:
        return 0, ["No export folder set."]

    if not os.path.isdir(folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            return 0, [f"Could not create folder '{folder}': {e}"]

    success_count = 0
    errors = []

    for text in bpy.data.texts:
        # Build a safe filename, forcing a .py extension
        name = text.name
        base, ext = os.path.splitext(name)
        filename = f"{base}.py" if ext.lower() != ".py" else name

        filepath = os.path.join(folder, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text.as_string())
            success_count += 1
        except OSError as e:
            errors.append(f"Failed to write '{filename}': {e}")

    return success_count, errors


class MC_OT_export_internal_texts(bpy.types.Operator):
    bl_idname = "mc.export_internal_texts"
    bl_label = "Export Internal Texts"
    bl_description = "Save all internal text files as .py files to the chosen folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        count, errors = export_internal_texts(context)

        if errors:
            for err in errors:
                self.report({'WARNING'}, err)

        if count:
            self.report({'INFO'}, f"Exported {count} text file(s).")
        elif not errors:
            self.report({'WARNING'}, "No internal text files found to export.")

        return {'FINISHED'}


# Operator to add a new preset
class MC_AddPresetOperator(bpy.types.Operator):
    bl_idname = "mc.add_preset"
    bl_label = "Add Preset"
    bl_description = "Save the active object's cloth settings as a new preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: bpy.props.StringProperty(name="Preset Name", default="New Preset")

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        preset = scene.MC_props.mc_presets.add()
        preset.name = self.preset_name or "New Preset"
        preset_store(preset, context.object.MC_props)
        self.report({'INFO'}, "Saved preset '%s' (%d settings)"
                    % (preset.name, len(PRESET_KEYS)))
        return {'FINISHED'}

# Operator to remove a preset
class MC_RemovePresetOperator(bpy.types.Operator):
    bl_idname = "mc.remove_preset"
    bl_label = "Remove Preset"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.MC_props.mc_presets):
            scene.MC_props.mc_presets.remove(self.index)
        return {'FINISHED'}

# Operator to load a preset
class MC_LoadPresetOperator(bpy.types.Operator):
    bl_idname = "mc.load_preset"
    bl_label = "Load Preset"
    bl_description = "Apply this preset to every selected cloth (or the active object)"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        if not (0 <= self.index < len(scene.MC_props.mc_presets)):
            return {'CANCELLED'}
        preset = scene.MC_props.mc_presets[self.index]

        obs = [o for o in context.selected_objects
               if o.type == 'MESH' and o.MC_props.cloth]
        if not obs and context.object is not None:
            obs = [context.object]
        if not obs:
            self.report({'WARNING'}, "No cloth object to apply the preset to.")
            return {'CANCELLED'}

        for ob in obs:
            preset_apply(preset, ob.MC_props)
        self.report({'INFO'}, "Applied preset '%s' to %d object(s)"
                    % (preset.name, len(obs)))
        return {'FINISHED'}


class SetActiveObjectOperator(bpy.types.Operator):
    """Set as active"""
    bl_idname = "mc.set_active_object"
    bl_label = "Set Active Object"
    bl_options = {'REGISTER', 'UNDO'}

    obj_name: bpy.props.StringProperty()

    def execute(self, context):
        if bpy.context.object:
            if bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set()
        if self.obj_name in bpy.data.objects:
            obj = bpy.data.objects[self.obj_name]
            # Deselect all for cleaner activation (optional)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {'FINISHED'}


class ApplyAsTarget(bpy.types.Operator):
    """Applys the MC_current shape key to Basis and MC_target shapes."""
    bl_idname = "mc.apply_as_target"
    bl_label = "Apply As Target"
    bl_options = {'REGISTER', 'UNDO'}

    obj_name: bpy.props.StringProperty()

    def execute(self, context):
        ob = context.object
        if ob.name in context.scene.objects:
            MC5.apply_as_target(ob)
            MC5.cloth_refresh(ob)

        return {'FINISHED'}


class ResetCloth(bpy.types.Operator):
    """Resets the cloth simulation to the Basis shape."""
    bl_idname = "mc.reset_cloth"
    bl_label = "Reset Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    obj_name: bpy.props.StringProperty()

    def execute(self, context):
        if self.obj_name in bpy.data.objects:
            ob = bpy.data.objects[self.obj_name]
            MC5.reset_cloth_ob(ob)

        return {'FINISHED'}


class DisableAll(bpy.types.Operator):
    """Disables cloth settings for all objects."""
    bl_idname = "mc.disable_all"
    bl_label = "Disable All Cloth Settings"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for ob in bpy.data.objects:
            prop = ob.MC_props
            prop["cache_playback"] = False
            prop.cloth = False
            prop.continuous = False
            prop.animated = False
        return {'FINISHED'}


def get_max_radius(ob):
    """See MC5.get_max_radius.  A vertex in no triangle cannot be measured and
    must stay out of the minimum the caller takes, or one loose vertex sets the
    self collision radius to zero.  Returns (radii, how many were measured)."""

    vc = len(ob.data.vertices)
    sco = np.empty((vc, 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks['MC_target'].data.foreach_get('co', sco.ravel())

    tridex = U.get_tridex_3(ob)


    max_radius = np.zeros(sco.shape[0], dtype=np.float32)
    found = np.zeros(sco.shape[0], dtype=bool)

    for v in range(vc):
        in_tris = tridex == v
        where = np.any(in_tris, axis=1)
        checks = tridex[where]
        in_tris.shape = tridex.shape
        #if v == 36:
        edges = checks[~in_tris[where]]
        eco = sco[edges]
        evecs = eco[1::2] - eco[::2]
        orivecs = sco[v] - eco[::2]
        uevecs = U.u_vecs(evecs)
        dots = U.compare_vecs(orivecs, uevecs)
        cpoe = uevecs * dots[:, None]
        rad_vecs = orivecs - cpoe
        rads = U.measure_vecs(rad_vecs)
        if rads.shape[0] == 0:
            continue                # no opposing edge, so nothing to measure
        max_radius[v] = np.min(rads)
        found[v] = True


        #vis = U.visualize_forces(eco[::2] + cpoe, end=None, name="forces Mesh", offset=0.0, offset_axis=0)
        #vis.matrix_world = ob.matrix_world
        #print(rads, "how long is a Chinaman?")
    if np.any(found):
        max_radius[~found] = max_radius[found].max()
    max_radius = max_radius[:, None] * 0.98
    return max_radius, int(np.count_nonzero(found))
    

class AutoSCRadius(bpy.types.Operator):
    """Set the self collision radius based on geometry."""
    bl_idname = "mc.auto_radius"
    bl_label = "Auto Set SC Radius"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        ob = bpy.context.object
        max_rad, found = get_max_radius(ob)
        if not found:
            U.popup_error("'%s' has no faces, so there is nothing to measure a "
                          "self collision radius from." % ob.name, icon='INFO')
            return {'CANCELLED'}
        ob.MC_props.sc_radius = np.min(max_rad)

        print("setting radius")
        return {'FINISHED'}


class MC_UvShape(bpy.types.Operator):
    """Creates a shape key based on the active UV map."""
    bl_idname = "object.uv_shape"
    bl_label = "UV Shape"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = bpy.context.object
        
        
        if ob.type == "MESH":
            
            no_faces = len(ob.data.polygons) == 0
            
            if len(ob.data.uv_layers) == 0:
                U.check_faces(ob)
                U.unwrap_object(ob)

            idx = ob.data.uv_layers.active_index
            U.uv_shape(ob, uvm=idx)
            
            if no_faces:
                U.delete_faces(ob, obm=None, face_idx=[0], type=0)
                        
        else:
            msg = "Object must be a mesh you silly ninny."
            bpy.context.window_manager.popup_menu(U.oops, title=msg, icon='ERROR')
            
        return {'FINISHED'}


# MC_PT_panel_grid is retired -- grid_fill.py provides the Grid Fill panel, and
# uv_shape_tools.py now owns the UV Shape panel.


class MC_PT_panel_cloth(Panel):
    bl_label = "Main"
    bl_idname = "MC_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["main"]

    def draw(self, context):
        layout = self.layout
        obj = context.object
        sc = context.scene
        box = layout.box()
        box.scale_y = 1.5
        if obj:
            box.operator("mc.disable_all", text="Disable All", icon="CANCEL")#, emboss=alert)
            box.prop(obj.MC_props, "cloth", toggle=True, icon="OUTLINER_OB_SURFACE")
            box.operator("mc.apply_as_target", text="Apply")#, emboss=alert)
            row = box.row()    
            row.prop(sc.MC_props, "reset_selected", text="")
            re_text = "Reset"
            if sc.MC_props.reset_selected:
                re_text = "Re Sel"
            op = row.operator("mc.reset_cloth", text=re_text)#, emboss=alert)
            row.prop(obj.MC_props, "reset_frame", text="")
            if obj.MC_props.cloth:
                box.prop(obj.MC_props, "animated", toggle=True, icon="TRIA_RIGHT")
                box.prop(obj.MC_props, "continuous", toggle=True, icon="TIME")
                box.prop(obj.MC_props, "target_object")
                op.obj_name = obj.name
        else:
            box.label(text="No object selected")


class MC_PT_panel_forces(Panel):
    bl_label = "Forces"
    bl_idname = "MC_PT_panel_forces"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["forces"]

    def draw(self, context):
        layout = self.layout
        if U.needs_cloth(layout, context):
            return
        obj = context.object
        box = layout.box()
        box.scale_y = 1.5
        if obj:

            #box = layout.box()
            #box.scale_y = 1.0
            box.prop(obj.MC_props, "gravity")
            row = box.row(align=True)
            row.prop(obj.MC_props, "gravity_object", text="Aim")
            row.operator("mc.gravity_add_object", text="", icon='ADD')
            if obj.MC_props.gravity_object is not None:
                col = box.column()
                col.scale_y = 0.7
                col.label(text="Its +Z is the way gravity pulls.", icon='INFO')
            box.prop(obj.MC_props, "velocity")
            box = layout.box()
            box.prop(obj.MC_props, "stretch")
            box.prop(obj.MC_props, "shrink_grow")
            box = layout.box()
            #box.label(text="Bend Force -> Stabilize")
            #row = box.row()
            box.prop(obj.MC_props, "bend_force")
            box.prop(obj.MC_props, "bend_stabilize")
            box = layout.box()
            box.prop(obj.MC_props, "inflate")
            box.prop(obj.MC_props, "air_drag")
            box.prop(obj.MC_props, "inverted_air_drag")
            # a development dial, not a setting: out of sight unless asked for
            if context.scene.MC_props.debug_mode:
                box = layout.box()
                box.prop(obj.MC_props, "e_bend_force")
        else:
            box.label(text="No object selected")


# The old "Stitch" panel lived here.  Its force properties now sit in the
# Sewing panel in sew_tools.py, alongside the tools that build the seams.


class MC_PT_panel_collision(Panel):
    """Holds the collision sub-panels."""
    bl_label = "Collision"
    bl_idname = "MC_PT_panel_collision"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["collision"]

    def draw(self, context):
        pass            # the sub-panels carry it


class MC_PT_panel_object_collision(Panel):
    bl_label = "Object"
    bl_idname = "MC_PT_panel_object_collision"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_collision"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        if U.needs_mesh(layout, context):
            return
        obj = context.object

        # Any mesh can be a collider, and it is marked on the collider itself,
        # so this much shows for a plain mesh too.
        box = layout.box()
        box.prop(obj.MC_props, "ob_collision", text="Collider")

        # the rest tunes how *this* cloth answers a collision
        if U.active_cloth(context) is None:
            U.panel_note(layout, "Cloth settings: turn on Cloth")
            return

        box = layout.box()
        box.label(text="Cloth Settings")
        box.prop(obj.MC_props, "ob_collision_radius", text="Radius")
        # speed only -- the result is identical at every depth
        if context.scene.MC_props.debug_mode:
            box.prop(obj.MC_props, "ob_box_depth")
            box.prop(obj.MC_props, "ob_box_depth_auto")

        box.separator()
        box.label(text="Friction")
        box.prop(obj.MC_props, "ob_friction", text="Cloth Friction")
        box.prop(obj.MC_props, "ob_static_threshold", text="Static Threshold")
        box.prop(obj.MC_props, "ob_collider_friction", text="Collider Friction")


class MC_PT_panel_settings(Panel):
    """Scene-wide settings, so no poll: they are worth reaching whatever is
    selected (and the engine status lives here)."""
    bl_label = "Settings"
    bl_idname = "MC_PT_panel_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["settings"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        ob = context.object
        sc = context.scene
        layout = self.layout
        if ob:
            box = layout.box()
            box.scale_y = 0.8
            box.label(text="Object Collision")
            row = box.row()
            row.scale_y = 1.0
            row.prop(ob.MC_props, "cl_point_tris", text="Tris")
            row.prop(ob.MC_props, "ob_point_tris", text="Points")
            row.prop(ob.MC_props, "ob_edges", text="Edges")
            if sc.MC_props.debug_mode:
                row = box.row()
                row.prop(ob.MC_props, "ob_collision_tri_damping", text="Tri Force of Wisdom")
                row.prop(ob.MC_props, "ob_collision_edge_damping", text="Edge Force of Wisdom")
            row = box.row()
            row.prop(ob.MC_props, "ob_recollide", text="Recollide")
            if ob.MC_props.ob_recollide:
                row.prop(ob.MC_props, "ob_recollide_every", text="Every")

            box = layout.box()
            box.scale_y = 0.8
            box.label(text="Self Collision")
            box.prop(ob.MC_props, "sc_edges", text="Edges")

            # one setting for both object and self collision -- see
            # MC5.collision_stage
            box = layout.box()
            box.label(text="Collision Substeps")
            p = ob.MC_props
            box.prop(p, "collision_auto_substeps")
            if p.collision_auto_substeps:
                box.prop(p, "collision_max_substeps")
                box.prop(p, "collision_substep_margin")
            else:
                box.prop(p, "collision_substeps")
            col = box.column()
            col.scale_y = 0.7
            if p.collision_auto_substeps or p.collision_substeps > 1:
                col.label(text="Catches fast movement that would", icon='INFO')
                col.label(text="otherwise pass through a collider.")
            else:
                col.label(text="Off: one collision test per step.", icon='INFO')

        # which implementation runs collision, and whether the C++ one loaded
        box = layout.box()
        box.label(text="Collision Engine")
        box.row().prop(sc.MC_props, "collision_backend", expand=True)
        if sc.MC_props.collision_backend == 'CPP':
            col = box.column()
            col.scale_y = 0.7
            if MC5.NATIVE.get() is not None:
                col.label(text="C++: object and self collision", icon='CHECKMARK')
            else:
                col.label(text="C++ unavailable, running Python", icon='ERROR')
                if MC5.NATIVE.LAST_ERROR:
                    col.label(text=MC5.NATIVE.LAST_ERROR[:60])

        # optional python packages: only magnetic targets need one
        box = layout.box()
        box.label(text="Dependencies")
        col = box.column()
        col.scale_y = 0.7
        if U.have_module("scipy"):
            col.label(text="scipy: installed (magnetic targets)",
                      icon='CHECKMARK')
        else:
            col.label(text="scipy missing: magnetic targets are off",
                      icon='ERROR')
            box.operator("mc.install_dependency",
                         text="Install scipy", icon='IMPORT').module = "scipy"
            col = box.column()
            col.scale_y = 0.7
            col.label(text="Downloads from PyPI into your Blender", icon='INFO')
            col.label(text="scripts/modules folder. Needs internet.")

        box = layout.box()
        box.prop(sc.MC_props, "debug_mode", icon='CONSOLE')
        if sc.MC_props.debug_mode:
            box.label(text="Developer")
            box.prop(sc.MC_props, "text_export_path", text="Python Path")
            box.operator("mc.export_internal_texts", icon='EXPORT')
            box.prop(sc.MC_props, "dll_path")
            found = U.find_dll(required=False)
            col = box.column()
            col.scale_y = 0.7
            col.label(text=("Solver: %s" % found) if found
                      else "Solver DLL not found", icon='INFO')


class MC_PT_panel_collision_objects(Panel):
    """Every collider in the scene.  Hidden until there is one."""
    bl_label = "Scene Colliders"
    bl_idname = "MC_PT_panel_collision_objects"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_collision"
    bl_order = 20
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        COLLISION_DATA = MC5.COLLISION_DATA
        if not COLLISION_DATA:
            U.panel_note(layout, "No colliders in this scene yet")
        if len(COLLISION_DATA.keys()) > 0:
            box = layout.box()
            box.scale_y = 0.8
            deletables = []
            for k, v in COLLISION_DATA.items():
                ob = v.ob
                try:    
                    alert = ob == bpy.context.object
                    row = box.row()
                    row.scale_y = 1.0
                    op = row.operator("mc.set_active_object", text=ob.name, emboss=alert)
                    op.obj_name = ob.name
                    row.prop(ob.MC_props, "ob_collision", text="Collide")
                except:
                    deletables += [k]
            if deletables:
                for k in deletables:
                    del(COLLISION_DATA[k])


class MC_PT_panel_self_collision(Panel):
    bl_label = "Self"
    bl_idname = "MC_PT_panel_self_collision"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_collision"
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        if U.needs_cloth(layout, context):
            return
        obj = context.object
        box = layout.box()
        box.label(text="Collision")
        if obj:
            box.prop(obj.MC_props, "self_collision")
            # speed only -- the result is identical at every depth
            if context.scene.MC_props.debug_mode:
                box.prop(obj.MC_props, "sc_box_depth")
                box.prop(obj.MC_props, "sc_box_depth_auto")
            row = box.row()
            row.prop(obj.MC_props, "sc_radius")
            row.operator("mc.auto_radius", text="Calculate")
            box.prop(obj.MC_props, "sc_damping")
            box.prop(obj.MC_props, "sc_vel_damping")

        else:
            box.label(text="No object selected")
            

class MC_PT_panel_magnetic(Panel):
    """Holds the magnetic sub-panels."""
    bl_label = "Magnetic"
    bl_idname = "MC_PT_panel_magnetic"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["magnetic"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        pass            # the sub-panels carry it


class MC_PT_panel_magnetics(Panel):
    bl_label = "Settings"
    bl_idname = "MC_PT_panel_magnetics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_magnetic"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        if U.needs_mesh(layout, context):
            return
        obj = context.object
        box = layout.box()
        box.scale_y = 1.5
        sce = context.scene
        if obj:
            box.prop(obj.MC_props, "magnetic_target", toggle=True)
            box = layout.box()
            box.scale_y = 1.0    
            box.prop(obj.MC_props, "magnetic_range")
            box.prop(obj.MC_props, "magnetic_force")
            box.prop(obj.MC_props, "preserve_boundary_edges")
            box = layout.box()
            box.prop(obj.MC_props, "magnetic_target_near")
            box.prop(obj.MC_props, "magnetic_target_high")
            box.prop(obj.MC_props, "magnetic_target_low")
            box.prop(obj.MC_props, "magnetic_neightbor_count")
            box = layout.box()
            box.label(text="Lock Axis:")
            row = box.row()
            row.prop(obj.MC_props, "lock_axis_x", text='X', toggle=True)
            row.prop(obj.MC_props, "lock_axis_y", text='Y', toggle=True)
            row.prop(obj.MC_props, "lock_axis_z", text='Z', toggle=True)
        else:
            box.label(text="No object selected")
        
        box.prop(sce.MC_props, "all_magnetic_targets", text='Apply All')
        

class MC_PT_panel_objs(Panel):
    """Every cloth in the scene.  Hidden until there is one."""
    bl_label = "Cloth Objects"
    bl_idname = "MC_PT_panel_objs"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["objects"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        DATA = MC5.DATA
        if not any(v is not None for v in DATA.values()):
            U.panel_note(layout, "No cloth objects in this scene yet")
        if len(DATA.keys()) > 0:
            box = layout.box()
            box.scale_y = 0.8
            deletables = []
            for k, v in DATA.items():
                if v is None:
                    continue    
                ob = v.ob
                try:    
                    alert = ob == bpy.context.object
                    row = box.row()
                    row.scale_y = 1.0
                    op = row.operator("mc.set_active_object", text=ob.name, emboss=alert)
                    op.obj_name = ob.name
                    op = row.operator("mc.reset_cloth", text="Reset")#, emboss=alert)
                    op.obj_name = ob.name
                    #op.alert = alert
                    row.prop(ob.MC_props, "cloth", text="CL")    
                    row.prop(ob.MC_props, "animated", text="An")    
                    row.prop(ob.MC_props, "continuous", text="Co")
                except:
                    deletables += [k]
            if deletables:
                for k in deletables:
                    del(DATA[k])
            

class MC_PT_panel_magnetic_targets(Panel):
    """Every magnetic target in the scene.  Hidden until there is one."""
    bl_label = "Scene Targets"
    bl_idname = "MC_PT_panel_magnetic_targets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_parent_id = "MC_PT_panel_magnetic"
    bl_order = 10
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        MAGNETIC_DATA = MC5.MAGNETIC_DATA
        if not MAGNETIC_DATA:
            U.panel_note(layout, "No magnetic targets in this scene yet")
        if len(MAGNETIC_DATA.keys()) > 0:
            box = layout.box()
            box.scale_y = 0.8
            deletables = []
            for k, v in MAGNETIC_DATA.items():
                ob = v.ob
                try:    
                    alert = ob == bpy.context.object
                    row = box.row()
                    row.scale_y = 1.0
                    op = row.operator("mc.set_active_object", text=ob.name, emboss=alert)
                    op.obj_name = ob.name
                    row.prop(ob.MC_props, "magnetic_target", text="MT")
                except:
                    deletables += [k]
            if deletables:
                for k in deletables:
                    del(MAGNETIC_DATA[k])


class MC_PresetPanel(bpy.types.Panel):
    bl_label = "Presets"
    bl_idname = "MC_PT_presets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["presets"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        # a preset is a set of cloth settings: it needs a cloth to land on
        if U.needs_cloth(layout, context):
            return
        scene = context.scene

        # Button to add new preset
        layout.operator("mc.add_preset", text="Save Current as Preset")

        # List of presets with load and remove buttons
        for i, preset in enumerate(scene.MC_props.mc_presets):
            row = layout.row()
            row.label(text=preset.name)
            op = row.operator("mc.load_preset", text="Load")
            op.index = i
            op = row.operator("mc.remove_preset", text="Remove")
            op.index = i
        

class MC_OT_install_dependency(bpy.types.Operator):
    """Download and install an optional python package into Blender's user
    scripts/modules folder.  Needs an internet connection and takes a few
    seconds; Blender is unresponsive while it runs"""
    bl_idname = "mc.install_dependency"
    bl_label = "Install Dependency"
    bl_options = {'REGISTER'}

    module: bpy.props.StringProperty(default="scipy")

    def execute(self, context):
        if U.have_module(self.module):
            self.report({'INFO'}, "%s is already installed." % self.module)
            return {'FINISHED'}

        win = getattr(context, "window", None)
        if win:
            win.cursor_set('WAIT')
        try:
            ok = U.external_lib(module=self.module)
        finally:
            if win:
                win.cursor_set('DEFAULT')

        if not ok:
            self.report({'ERROR'}, "Could not install %s. The console has "
                                   "what pip said." % self.module)
            return {'CANCELLED'}

        # let the features that gave up on it this session try again
        U._REQUIRE_TRIED.discard(self.module)
        MC5._SCIPY_WARNED = False
        self.report({'INFO'}, "%s installed." % self.module)
        for area in getattr(context.screen, "areas", []):
            area.tag_redraw()
        return {'FINISHED'}


# ===================================================================== CACHE
class MC_OT_cache_bake(bpy.types.Operator):
    """Bake the cloth simulation to disk, one .npy per frame"""
    bl_idname = "mc.cache_bake"
    bl_label = "Bake Cache"
    bl_options = {'REGISTER'}

    from_current: bpy.props.BoolProperty(
        name="From Current Frame",
        description=("Start baking at the current frame using the live sim "
                     "state (seeded from the two frames before it if they are "
                     "already cached).  Leaves earlier cached frames alone"),
        default=False,
    )

    def _range(self, context):
        p = context.object.MC_props
        if p.cache_use_scene_range:
            return context.scene.frame_start, context.scene.frame_end
        return p.cache_start, p.cache_end

    def invoke(self, context, event):
        ob = context.object
        if ob is None or ob.type != 'MESH' or not ob.MC_props.cloth:
            self.report({'ERROR'}, "Active object is not a cloth.")
            return {'CANCELLED'}
        if MC5.CACHE.bake_running():
            self.report({'ERROR'}, "A bake is already running.")
            return {'CANCELLED'}
        # anything a dead bake left behind (flags, overridden Animated)
        MC5.CACHE.clear_stale_bake()

        self.ob = ob
        self.start, self.end = self._range(context)
        if self.end < self.start:
            self.report({'ERROR'}, "Cache end is before cache start.")
            return {'CANCELLED'}

        p = ob.MC_props
        self._saved = (p.animated, p.continuous)
        # kept on the object too, so a bake that dies can still be undone
        ob[MC5.CACHE.RESTORE_KEY] = [int(p.animated), int(p.continuous)]
        # bracket-assign: don't fire cb_cloth (it would rebuild C from rest and
        # drop the live state we need for "from current")
        p["cache_playback"] = False
        p["continuous"] = False
        p["animated"] = True
        MC5.install_handler()

        scene = context.scene
        if self.from_current:
            self.cur = min(max(scene.frame_current, self.start), self.end)
        else:
            self.cur = self.start
            if p.cache_reset_on_bake:
                MC5.CACHE.wipe(ob)
            MC5.reset_cloth_ob(ob)        # bake a normal run from the rest state

        C = MC5.get_cloth(ob)
        if C is None:
            p["animated"], p["continuous"] = self._saved
            del ob[MC5.CACHE.RESTORE_KEY]
            self.report({'ERROR'}, "No cloth data.")
            return {'CANCELLED'}
        if self.from_current and MC5.CACHE.load_frame_disk(ob, self.cur - 1) is not None:
            MC5.CACHE.seed_resume(C, self.cur - 1)

        sc = context.scene.MC_props
        sc.cache_baking = True
        sc.cache_bake_paused = False
        self._paused = False
        MC5.CACHE.bake_ended()              # drop any old state (a stop request)
        MC5.CACHE.bake_heartbeat(ob.name)

        self._timer = context.window_manager.event_timer_add(0.001, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "MC cache: baking... (Esc to stop, Space or P to pause)")
        return {'RUNNING_MODAL'}

    def _apply_pause(self, context, paused):
        """Take the bake in or out of its held state.

        While held, `animated` is switched off so that scrubbing the timeline
        to look at what has been baked so far cannot quietly run the solver
        another step and desync the cache from the sim.  Bracket-assigned so
        cb_cloth does not fire and rebuild the cloth from rest.
        """
        self._paused = paused
        self.ob.MC_props["animated"] = not paused
        done = self.cur - self.start + 1
        total = self.end - self.start + 1
        if paused:
            context.workspace.status_text_set(
                "MC cache: PAUSED at %d / %d (Space or P to resume, Esc to "
                "stop)" % (done, total))
        else:
            context.workspace.status_text_set(
                "MC cache: baking %d / %d" % (done, total))

    def modal(self, context, event):
        # any error ends the bake properly; an exception escaping here used to
        # kill the operator and leave the scene's baking flag stuck on
        try:
            return self._modal(context, event)
        except Exception:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, "Bake stopped by an error (see the console).")
            try:
                return self._finish(context, cancelled=True)
            except Exception:
                traceback.print_exc()
                MC5.CACHE.bake_ended()
                MC5.CACHE.clear_stale_bake()
                return {'CANCELLED'}

    def _modal(self, context, event):
        sc = context.scene.MC_props

        # asked to stop from elsewhere (its cache was deleted)
        if MC5.CACHE.bake_stop_requested():
            return self._finish(context, cancelled=True)

        if event.type == 'ESC':
            return self._finish(context, cancelled=True)

        if event.type in {'SPACE', 'P'} and event.value == 'PRESS':
            sc.cache_bake_paused = not sc.cache_bake_paused
            self._apply_pause(context, sc.cache_bake_paused)
            return {'RUNNING_MODAL'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        MC5.CACHE.bake_heartbeat()       # paused or not: this bake is alive

        # the panel button sets the flag directly, so the state is reconciled
        # here rather than in the key handler -- one path for both
        if sc.cache_bake_paused != self._paused:
            self._apply_pause(context, sc.cache_bake_paused)
        if self._paused:
            return {'RUNNING_MODAL'}

        scene = context.scene
        scene.frame_set(self.cur)          # triggers mc_handler -> physics -> 1 step
        C = MC5.get_cloth(self.ob)
        if C is None:
            self.report({'WARNING'}, "Cloth data vanished mid-bake.")
            return self._finish(context, cancelled=True)

        MC5.CACHE.save_frame(self.ob, self.cur, C.co)
        done = self.cur - self.start + 1
        total = self.end - self.start + 1
        context.workspace.status_text_set("MC cache: baking %d / %d" % (done, total))

        if self.cur >= self.end:
            return self._finish(context, cancelled=False)
        self.cur += 1
        MC5.CACHE.bake_heartbeat()       # a slow step shouldn't look like a dead bake
        return {'RUNNING_MODAL'}

    def _finish(self, context, cancelled):
        MC5.CACHE.bake_ended()
        wm = context.window_manager
        if getattr(self, "_timer", None):
            wm.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)

        sc = context.scene.MC_props
        sc.cache_baking = False
        sc.cache_bake_paused = False

        p = self.ob.MC_props
        p.animated, p.continuous = self._saved
        if MC5.CACHE.RESTORE_KEY in self.ob:
            del self.ob[MC5.CACHE.RESTORE_KEY]

        frames = MC5.CACHE.cached_frames(self.ob)
        if frames:
            MC5.CACHE.write_manifest(self.ob, len(self.ob.data.vertices),
                                     frames[0], frames[-1])
        MC5.CACHE.free(self.ob)           # drop stale RAM copy; next playback reloads

        last = self.cur if cancelled else self.end
        self.report({'INFO'}, "Cache %s: %d-%d"
                    % ("stopped at" if cancelled else "baked", self.start, last))
        return {'CANCELLED'} if cancelled else {'FINISHED'}


class MC_OT_cache_bake_pause(bpy.types.Operator):
    """Hold the running bake, or let it carry on"""
    bl_idname = "mc.cache_bake_pause"
    bl_label = "Pause Bake"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.MC_props.cache_baking and MC5.CACHE.bake_running()

    def execute(self, context):
        sc = context.scene.MC_props
        # the modal picks this up on its next tick and does the rest
        sc.cache_bake_paused = not sc.cache_bake_paused
        self.report({'INFO'}, "Bake %s"
                    % ("paused" if sc.cache_bake_paused else "resumed"))
        return {'FINISHED'}


class MC_OT_cache_update_frame(bpy.types.Operator):
    """Overwrite the cached vertex positions for the current frame with the
    mesh's current positions (keeps the old data as a .bak)"""
    bl_idname = "mc.cache_update_frame"
    bl_label = "Update Cached Frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = context.object
        if ob is None or ob.type != 'MESH':
            return {'CANCELLED'}
        if ob.data.is_editmode:
            ob.update_from_editmode()

        vc = len(ob.data.vertices)
        co = np.empty((vc, 3), dtype=np.float32)
        keys = ob.data.shape_keys.key_blocks if ob.data.shape_keys else None
        if keys and 'MC_current' in keys:
            keys['MC_current'].data.foreach_get('co', co.ravel())
        else:
            ob.data.vertices.foreach_get('co', co.ravel())

        frame = context.scene.frame_current
        MC5.CACHE.overwrite_frame(ob, frame, co, backup=True)
        self.report({'INFO'}, "Updated cache frame %d" % frame)
        return {'FINISHED'}


class MC_OT_cache_delete_frame(bpy.types.Operator):
    """Delete the cached frame at the current frame"""
    bl_idname = "mc.cache_delete_frame"
    bl_label = "Delete Cached Frame"
    bl_options = {'REGISTER', 'UNDO'}

    and_forward: bpy.props.BoolProperty(
        name="And Everything After",
        description=("Also delete every cached frame after this one, so you can "
                     "re-bake from here with Bake From Current"),
        default=False,
    )

    def execute(self, context):
        ob = context.object
        if ob is None:
            return {'CANCELLED'}
        frame = context.scene.frame_current
        n = MC5.CACHE.delete_frame(ob, frame, forward=self.and_forward)
        self.report({'INFO'}, "Deleted %d cached frame(s) from %d" % (n, frame))
        return {'FINISHED'}


class MC_OT_cache_free(bpy.types.Operator):
    """Release the in-memory copy of this cache (files on disk are untouched)"""
    bl_idname = "mc.cache_free"
    bl_label = "Free Cache RAM"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if context.object is not None:
            MC5.CACHE.free(context.object)
        self.report({'INFO'}, "Released in-memory cache")
        return {'FINISHED'}


class MC_OT_cache_delete(bpy.types.Operator):
    """Delete this object's entire cache folder from disk"""
    bl_idname = "mc.cache_delete"
    bl_label = "Delete Cache"
    bl_options = {'REGISTER', 'INTERNAL'}

    use_confirm: bpy.props.BoolProperty(default=True)

    def invoke(self, context, event):
        if self.use_confirm:
            return context.window_manager.invoke_props_dialog(self, width=360)
        return self.execute(context)

    def draw(self, context):
        ob = context.object
        col = self.layout.column()
        name = ob.name if ob else "?"
        col.label(text="Are you sure you want to delete cache for '%s'?" % name,
                  icon='TRASH')
        if ob is not None:
            n, mb = MC5.CACHE.cache_stats(ob)
            col.label(text="%d frames - %.1f MB - this cannot be undone."
                      % (n, mb))

    def execute(self, context):
        ob = context.object
        if ob is None:
            return {'CANCELLED'}
        ob.MC_props["cache_playback"] = False
        # a bake of this object (running or paused) has nothing left to add to
        stopped = MC5.CACHE.stop_bake(ob)
        MC5.CACHE.wipe_all(ob)
        self.report({'INFO'}, "Deleted cache for '%s'%s"
                    % (ob.name, " and stopped its bake" if stopped else ""))
        return {'FINISHED'}


class MC_PT_panel_cache(Panel):
    bl_label = "Cache"
    bl_idname = "MC_PT_panel_cache"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = U.MC_TAB
    bl_order = U.ORDER["cache"]
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        if U.needs_cloth(layout, context):
            return
        ob = context.object
        sc = context.scene
        if ob is None:
            layout.box().label(text="No object selected")
            return
        p = ob.MC_props

        box = layout.box()
        box.prop(sc.MC_props, "cache_dir", text="Folder")
        box.label(text=MC5.CACHE.info_string(ob), icon='FILE_CACHE')
        if MC5.CACHE.is_temporary(sc):
            col = box.column()
            col.scale_y = 0.7
            col.label(text="File not saved: cache is temporary", icon='ERROR')
            col.label(text="(moved beside the .blend when you save)")

        box = layout.box()
        box.label(text="Bake")
        box.prop(p, "cache_use_scene_range")
        if not p.cache_use_scene_range:
            row = box.row(align=True)
            row.prop(p, "cache_start", text="Start")
            row.prop(p, "cache_end", text="End")
        box.prop(p, "cache_reset_on_bake")
        box.prop(p, "cache_in_memory")
        # the flag alone can be left over from a bake that no longer exists
        if sc.MC_props.cache_baking and MC5.CACHE.bake_running():
            paused = sc.MC_props.cache_bake_paused
            row = box.row()
            row.alert = paused
            row.scale_y = 1.4
            row.operator("mc.cache_bake_pause",
                         text="Resume Bake" if paused else "Pause Bake",
                         icon='PLAY' if paused else 'PAUSE')
            col = box.column()
            col.scale_y = 0.7
            col.label(text="Paused - resume to carry on" if paused
                      else "Baking - Space or P also pauses", icon='INFO')
        else:
            row = box.row(align=True)
            row.operator("mc.cache_bake", text="Bake",
                         icon='REC').from_current = False
            row.operator("mc.cache_bake", text="From Current",
                         icon='PLAY').from_current = True

        box = layout.box()
        box.prop(p, "cache_playback", text="Playback", toggle=True, icon='PLAY')

        box = layout.box()
        box.label(text="Edit Frame")
        box.operator("mc.cache_update_frame", text="Update This Frame",
                     icon='FILE_TICK')
        row = box.row(align=True)
        row.operator("mc.cache_delete_frame", text="Delete Frame",
                     icon='X').and_forward = False
        row.operator("mc.cache_delete_frame", text="Truncate Here",
                     icon='TRACKING_CLEAR_FORWARDS').and_forward = True

        box = layout.box()
        box.operator("mc.cache_free", text="Free RAM", icon='MEMORY')
        row = box.row()
        row.alert = True
        row.operator("mc.cache_delete", text="Delete Cache", icon='TRASH')
# ===================================================================== CACHE


CLASSES = [
    MC_Preset,
    MC_props,
    MC_props_scene,
    MC_AddPresetOperator,
    MC_RemovePresetOperator,
    MC_LoadPresetOperator,
    AutoSCRadius,
    MC_UvShape,
    # parents before their children: a sub-panel needs its parent registered
    MC_PT_panel_cloth,
    MC_PT_panel_forces,
    MC_PT_panel_collision,
    MC_PT_panel_object_collision,
    MC_PT_panel_self_collision,
    MC_PT_panel_collision_objects,
    MC_PT_panel_objs,
    MC_PT_panel_magnetic,
    MC_PT_panel_magnetics,
    MC_PT_panel_magnetic_targets,
    MC_PresetPanel,
    MC_PT_panel_settings,
    MC_PT_panel_cache,
    MC_OT_export_internal_texts,
    MC_OT_install_dependency,
    MC_OT_cache_bake,
    MC_OT_cache_bake_pause,
    MC_OT_cache_update_frame,
    MC_OT_cache_delete_frame,
    MC_OT_cache_free,
    MC_OT_cache_delete,
    SetActiveObjectOperator,
    ResetCloth,
    ApplyAsTarget,
    DisableAll,
] + list(SEW.CLASSES) + list(GFILL.CLASSES) + list(UVS.CLASSES) + list(HOOK.CLASSES) + list(WIND.CLASSES)

# Register the classes
def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    
    bpy.types.Object.MC_props = bpy.props.PointerProperty(type=MC_props)
    bpy.types.Scene.MC_props = bpy.props.PointerProperty(type=MC_props_scene)
    MC5.install_handler()
    MC5.install_file_handlers()     # reopen files ready to run; keep unsaved caches

    # Rebuild the sim data of whatever is already in the file.  When the addon
    # is enabled during startup bpy.data is still restricted, so this runs from
    # a timer the moment Blender is ready instead.
    try:
        _rebuild_existing()
    except AttributeError:
        bpy.app.timers.register(_rebuild_existing, first_interval=0.0)


def _rebuild_existing():
    for ob in bpy.data.objects:
        if ob.MC_props.cloth:
            ob.MC_props.cloth = True # force the update
        if ob.MC_props.magnetic_target:
            ob.MC_props.magnetic_target = True # force the update
        if ob.MC_props.ob_collision:
            ob.MC_props.ob_collision = True
    return None                     # timer: run once


def unregister():
    # Unregister in reverse order to handle dependencies properly
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Object.MC_props
    del bpy.types.Scene.MC_props
    MC5.install_handler(clear=True)
    MC5.install_file_handlers(clear=True)

if __name__ == "__main__":
    register()