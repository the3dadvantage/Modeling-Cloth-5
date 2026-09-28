"""How the sidebar is arranged.

Sixteen panels used to sit side by side in one tab, all expanded.  They are
grouped now, but nothing hides: both tabs and every panel are always in the
sidebar whatever is selected, so they can be found.  A panel with nothing to
act on says what it is waiting for instead of disappearing.

  * two tabs: the simulation (MC5) and the mesh/pattern tools (MC5 Tools)
  * collision and magnetic panels are nested under one parent each
  * no panel has a poll: nothing vanishes with the selection
  * every panel draws, with a note, for no object / a plain mesh / a cloth
  * sew forces stay with the simulation; the seam tools go to the tools tab
  * the order in each tab is stated, not inherited from registration
"""
import bpy, os, sys, glob
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
UI.U.popup_error = lambda *a, **k: None
MC5 = UI.MC5
U = MC5.U

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


PANELS = [c for c in UI.CLASSES
          if isinstance(c, type) and issubclass(c, bpy.types.Panel)]
BY_ID = {p.bl_idname: p for p in PANELS}


def visible(tab=None):
    """Every panel in the tab, in the order Blender would show them:
    sub-panels indented under their parent.  Nothing polls, so nothing is
    filtered -- that is the point of this suite."""
    out = []
    for p in PANELS:
        if tab is not None and getattr(p, "bl_category", None) != tab:
            continue
        try:
            if hasattr(p, "poll") and not p.poll(bpy.context):
                fails.append("%s has a poll that hides it" % p.__name__)
                continue
        except Exception as e:
            fails.append("%s.poll raised %r" % (p.__name__, e))
            continue
        out.append(p)

    def where(p):
        """Sort as Blender stacks them: each child under its own parent."""
        parent = getattr(p, "bl_parent_id", "")
        if parent and parent in BY_ID:
            return (getattr(BY_ID[parent], "bl_order", 0), 1,
                    getattr(p, "bl_order", 0))
        return (getattr(p, "bl_order", 0), 0, 0)

    out.sort(key=where)
    return out


def labels(tab=None):
    return [("  " + p.bl_label) if getattr(p, "bl_parent_id", "") else p.bl_label
            for p in visible(tab)]


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.COLLISION_DATA.clear()
    MC5.MAGNETIC_DATA.clear()
    MC5.OC_DATA["obs"] = []


def grid(name, loc=(0, 0, 0)):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5,
                                    size=2.0, location=loc)
    ob = bpy.context.object
    ob.name = name
    ob.data.vertices.foreach_set('select', np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    return ob


print("== the tabs ==", flush=True)
tabs = {}
for p in PANELS:
    tabs.setdefault(getattr(p, "bl_category", "?"), []).append(p.bl_label)
check(set(tabs) == {U.MC_TAB, U.MC_TOOLS_TAB},
      "two tabs: %s" % sorted(tabs))
check(sorted(tabs.get(U.MC_TOOLS_TAB, [])) == ["Grid Fill", "Sewing", "UV Shape"],
      "the tools tab holds the mesh tools: %s" % sorted(tabs.get(U.MC_TOOLS_TAB, [])))

EXPECTED_MC5 = ["Main", "Forces", "  Sewing", "Collision", "  Object",
                "  Self", "  Scene Colliders", "Magnetic", "  Settings",
                "  Scene Targets", "Wind", "Hooks", "Cache", "Presets",
                "Cloth Objects", "Settings"]
EXPECTED_TOOLS = ["Grid Fill", "UV Shape", "Sewing"]

print("\n== nothing selected: the sidebar is all still there ==", flush=True)
clear()
check(labels(U.MC_TAB) == EXPECTED_MC5,
      "every MC5 panel is present: %s" % labels(U.MC_TAB))
check(labels(U.MC_TOOLS_TAB) == EXPECTED_TOOLS,
      "and the tools tab too: %s" % labels(U.MC_TOOLS_TAB))
check(not any(hasattr(p, "poll") for p in PANELS),
      "no panel has a poll, so none can vanish: %s"
      % [p.__name__ for p in PANELS if hasattr(p, "poll")])

print("\n== a plain mesh, not a cloth ==", flush=True)
clear()
plain = grid("Plain")
bpy.context.view_layer.objects.active = plain
check(labels(U.MC_TAB) == EXPECTED_MC5, "the same panels, whatever is selected")
check(labels(U.MC_TOOLS_TAB) == EXPECTED_TOOLS, "and the same tools")

print("\n== a cloth ==", flush=True)
cloth = grid("Cloth", (0, 0, 1))
bpy.context.view_layer.objects.active = cloth
cloth.MC_props.cloth = True
shown = labels(U.MC_TAB)
check(shown == EXPECTED_MC5, "still the same panels")
check(shown.index("Main") == 0, "Main is first (%s)" % shown[:3])
check(shown[-1] == "Settings", "Settings is last")

collider = grid("Collider", (0, 0, -2))
collider.MC_props.ob_collision = True
target = grid("Target", (0, 3, 0))
target.MC_props.magnetic_target = True
bpy.context.view_layer.objects.active = cloth

print("\n== sewing is split the way it was asked for ==", flush=True)
check(BY_ID["MC_PT_panel_sewing"].bl_category == U.MC_TOOLS_TAB,
      "the seam tools are on the tools tab")
sew_forces = BY_ID["MC_PT_panel_sew_forces"]
check(sew_forces.bl_category == U.MC_TAB
      and sew_forces.bl_parent_id == "MC_PT_panel_forces",
      "the sew forces are a sub-panel of Forces on the sim tab")

print("\n== closed by default ==", flush=True)
closed = {p.bl_label for p in PANELS
          if 'DEFAULT_CLOSED' in getattr(p, "bl_options", set())}
for want in ("Cache", "Presets", "Settings", "Cloth Objects"):
    check(want in closed, "%s starts collapsed" % want)
for want in ("Main", "Forces"):
    check(want not in closed, "%s starts open" % want)

print("\n== every panel still draws ==", flush=True)
import types


class Rec:
    """Records what a draw put on screen."""
    def __init__(self, seen=None):
        self.seen = {"props": 0, "ops": 0, "labels": []} if seen is None else seen
        self.scale_y = self.scale_x = 1.0
        self.alert = False
        self.enabled = self.active = True
        self.use_property_split = False

    def _chain(self, *a, **k):
        return Rec(self.seen)
    box = row = column = split = column_flow = grid_flow = _chain

    def prop(self, *a, **k):
        self.seen["props"] += 1
        return self

    def operator(self, *a, **k):
        self.seen["ops"] += 1
        return types.SimpleNamespace()

    def label(self, *a, **k):
        if "text" in k:
            self.seen["labels"].append(k["text"])
        elif a:
            self.seen["labels"].append(a[0])
        return None
    separator = template_list = menu = prop_search = lambda self, *a, **k: None


class Self:
    def __init__(self, panel):
        self.layout = Rec()
        for k in dir(panel):
            v = panel.__dict__.get(k)
            if callable(v) and not k.startswith("__") and k not in ("draw", "poll"):
                setattr(self, k, v.__get__(self))


def draw_all(label):
    """Draw every panel and report which ones came out completely blank."""
    problems, blank = [], []
    for p in visible():
        me = Self(p)
        try:
            p.draw(me, bpy.context)
        except Exception as e:
            problems.append("%s: %s: %s" % (p.__name__, type(e).__name__, e))
            continue
        seen = me.layout.seen
        if p.bl_idname in PARENTS:
            continue            # a parent panel is just a header
        if not (seen["props"] or seen["ops"] or seen["labels"]):
            blank.append(p.bl_label)
    check(not problems, "%s: every panel drew %s" % (label, problems))
    check(not blank, "%s: none came out blank %s" % (label, blank))


PARENTS = {"MC_PT_panel_collision", "MC_PT_panel_magnetic"}

clear()
draw_all("nothing selected")

plain = grid("Plain")
bpy.context.view_layer.objects.active = plain
draw_all("a plain mesh")

cloth = grid("Cloth", (0, 0, 1))
bpy.context.view_layer.objects.active = cloth
cloth.MC_props.cloth = True
draw_all("a cloth")

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.empty_add()
draw_all("an empty")

print("\n== the notes say what is missing ==", flush=True)
clear()
notes = {}
for p in visible():
    me = Self(p)
    p.draw(me, bpy.context)
    notes[p.bl_label] = me.layout.seen["labels"]
check(any("Select a mesh" in t for t in notes.get("Forces", [])),
      "Forces asks for a mesh: %s" % notes.get("Forces"))
check(any("mesh" in t.lower() for t in notes.get("Grid Fill", [])),
      "Grid Fill asks for a mesh: %s" % notes.get("Grid Fill"))

plain = grid("Plain2")
bpy.context.view_layer.objects.active = plain
notes = {}
for p in visible():
    me = Self(p)
    p.draw(me, bpy.context)
    notes[p.bl_label] = me.layout.seen["labels"]
check(any("Cloth" in t for t in notes.get("Forces", [])),
      "on a plain mesh, Forces points at the Cloth toggle: %s" % notes.get("Forces"))
check(any("colliders" in t.lower() for t in notes.get("Scene Colliders", [])),
      "Scene Colliders says it is empty: %s" % notes.get("Scene Colliders"))

print("\n================ %s ================"
      % ("ALL PASS" if ok and not fails else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
sys.stdout.flush()
os._exit(0 if (ok and not fails) else 1)
