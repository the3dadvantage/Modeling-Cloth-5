"""Stale continuous timers must not survive a module reload.

install_handler clears frame handlers by __name__, which catches every stale
one, but a timer can only be unregistered by the exact function object that
was registered.  Reloading the text builds new function objects, so the
previous module's timer used to stay alive -- one extra physics() call per
tick for every reload, under continuous only.  That is why continuous advanced
the sim faster than animated.
"""
import bpy, os, glob
import numpy as np

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

ok = True
fails = []


def check(c, m):
    global ok
    ok = ok and bool(c)
    if not c:
        fails.append(m)
    print(("  PASS " if c else "  FAIL ") + m, flush=True)


def frame_handlers():
    return sum(1 for h in bpy.app.handlers.frame_change_post
               if h.__name__ == 'mc_handler')


print("== five reloads must leave one timer and one handler ==", flush=True)
instances = []
for i in range(5):
    m = bpy.data.texts["MC5.py"].as_module()   # what reloading the text does
    m.install_handler()
    instances.append(m)

live = [i for i, m in enumerate(instances)
        if bpy.app.timers.is_registered(m.mc_handler_continuous)]
print("   live timers from instances %s, frame handlers %d"
      % (live, frame_handlers()), flush=True)
check(frame_handlers() == 1, "exactly one frame handler")
check(len(live) == 1, "exactly one continuous timer (got %d)" % len(live))
check(live == [len(instances) - 1], "and it is the newest one %s" % live)

print("\n== a stale timer retires itself if it ever does fire ==", flush=True)
stale = instances[0]
check(stale.mc_handler_continuous() is None,
      "an old instance's timer returns None (unregisters itself)")
check(bpy.app.driver_namespace.get(instances[-1].TIMER_KEY)
      is instances[-1].mc_handler_continuous,
      "the newest instance is the one parked as live")

print("\n== animated and continuous advance the sim identically ==", flush=True)
UI = bpy.data.texts["MC_ui.py"].as_module()
UI.register()
UI.U.popup_error = lambda *a, **k: None
# register() runs install_handler on UI's OWN MC5 instance, so that is the one
# the registered handlers call -- patching any other instance counts nothing.
MC5 = UI.MC5
scene = bpy.context.scene

CALLS = {"n": 0}
_real = MC5.physics


def counting(C):
    CALLS["n"] += 1
    return _real(C)


MC5.physics = counting


def build():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    MC5.DATA.clear()
    MC5.OC_DATA["obs"] = []
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=2.0, location=(0, 0, 0))
    ob = bpy.context.object
    ob.data.vertices.foreach_set('select',
                                 np.zeros(len(ob.data.vertices), dtype=bool))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.MC_props.cloth = True
    p = ob.MC_props
    p.sc_box_depth_auto = False
    p.gravity = -9.8
    p.velocity = 0.98
    p.stretch = 1.25
    p.bend_force = 0.5
    scene.frame_set(p.reset_frame + 1)
    return ob


def run(mode, ticks=60):
    ob = build()
    reset = ob.MC_props.reset_frame
    if mode == "animated":
        ob.MC_props.animated = True
    else:
        ob.MC_props.continuous = True
    CALLS["n"] = 0
    for f in range(ticks):
        if mode == "animated":
            scene.frame_set(reset + 2 + f)      # fires the handler itself
        else:
            MC5.mc_handler_continuous()
    return np.array(MC5.get_cloth(ob).co).copy(), CALLS["n"]


a, na = run("animated")
c, nc = run("continuous")
print("   physics() calls over 60 ticks: animated %d, continuous %d"
      % (na, nc), flush=True)
check(na == 60 and nc == 60,
      "one physics step per tick in both modes (%d / %d)" % (na, nc))
d = float(np.abs(a - c).max())
check(d == 0.0, "the two modes land in exactly the same place (%.3e)" % d)

for m in instances:
    m.install_handler(clear=True)

print("\n================ %s ================"
      % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
for f in fails:
    print("  FAILED: %s" % f, flush=True)
