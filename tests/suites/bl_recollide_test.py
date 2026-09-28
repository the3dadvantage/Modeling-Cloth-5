"""Recollide between bend / stretch iterations.

  * Handing the DLL one iteration per call must give exactly the same result
    as one call with the whole list (checked with recollide on but set to fire
    every 1000th iteration, i.e. split but idle).
  * On the stiff-grid-on-a-cone scene the tip goes through the cloth with
    recollide off, and stays clear of it with recollide on.
  * The lookahead caches the pairs a sheet is about to land on.
  * An off-centre, tilted grid on the tip stays calm, keeps its edges, and
    is held clear (the jumping this was reported with).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collide_bench as B    # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("TEST %-4s %s %s" % ("ok" if cond else "FAIL", name, detail), flush=True)


B.load()
off = B.run_scene("oc_cone", {"ob_recollide": False})
idle = B.run_scene("oc_cone", {"ob_recollide": True, "ob_recollide_every": 1000})
on = B.run_scene("oc_cone", {"ob_recollide": True, "ob_recollide_every": 1})

d = float(abs(idle["co"] - off["co"]).max())
check("split DLL calls identical to one call", d == 0.0, "(max difference %.3g)" % d)
check("recollide off: tip goes through", off["tip_clear_worst"] < 0.0,
      "(worst clearance %+.4f)" % off["tip_clear_worst"])
check("recollide on: tip never through", on["tip_clear_worst"] > 0.0,
      "(worst clearance %+.4f)" % on["tip_clear_worst"])
check("recollide on: less distortion at the tip",
      on["stretch_max"] < off["stretch_max"],
      "(max edge stretch %.2f vs %.2f)" % (on["stretch_max"], off["stretch_max"]))


# ---- first contact: the lookahead.  A sheet 0.22 above a floor that came
# down 0.15 this step.  The broad phase's normal reach (2 x thickness = 0.1
# around this step's motion) does not get to the floor, so without the
# lookahead the cached pairs would be empty -- and next step, when the sheet
# lands, recollide would have nothing to work with.  With the lookahead the
# boxes also cover next step's motion, so the pairs are already cached.
MC5 = B.mc5()
OC = MC5.OC
B.clear()
B.collider(B.grid(4, 6.0))
ob = B.make_cloth(B.grid(8, 1.0, (0, 0, 0.22)))
MC5.update_ob_colliders(MC5.OC_DATA)
C = MC5.get_cloth(ob)
MC5.refresh_colliders(C)
co_now = C.co.copy()
co_start = co_now.copy()
co_start[:, 2] += 0.15


def cached_pairs(lookahead):
    C.co[:] = co_now
    C.start_co[:] = co_start
    OC.begin_velocity(C)
    pt, ee, tp = OC.collision_force(MC5.OC_DATA, C, C.ob, None, None,
                                    co_start, C.co, lookahead=lookahead)
    return 0 if pt is None else len(pt[0])


without = cached_pairs(None)
with_la = cached_pairs(1.0)
check("sheet about to land, no lookahead: nothing cached", without == 0,
      "(%d pairs)" % without)
check("sheet about to land, lookahead: its pairs are cached", with_la >= C.vc,
      "(%d pairs for %d verts)" % (with_la, C.vc))
check("the collision pass itself did nothing either way (nothing touching yet)",
      np.array_equal(C.co, co_now))


# ---- the reported bug: an off-centre, tilted grid on the tip went jumpy.
# Causes, all in the contact geometry: face and edge contacts measured along
# planes that run past a sharp tip (points pushed back across it, neighbours
# collapsed onto each other, cloth pumped up to a false peak and dropped on
# alternate frames), the tip contact's triangle chosen by a projection that
# tilted with the push, and the tip's push reaching only sum(w^2) of the way.
def tilted_run(frames=80):
    B.clear()
    ob = B.grid(B.CONE_SUBS, B.CONE_SIZE, (0.25, 0.15, B.CONE_APEX + 0.08),
                rot=(np.radians(15.0), 0.0, 0.0))
    B.collider(B.cone(0.5, B.CONE_APEX, 32))
    B.make_cloth(ob, gravity=B.CONE_GRAVITY, bend_force=B.CONE_BEND,
                 stretch=B.CONE_BEND + 1.25, ob_recollide=True, ob_recollide_every=1,
                 **B.ALL_PHASES)
    MC5.update_ob_colliders(MC5.OC_DATA)
    C = MC5.get_cloth(ob)
    edges, _ = B._mesh_topology(ob)
    rest = np.array(C.co, dtype=np.float64)
    rl = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    max_rise, crushed, prev = 0.0, 0, None
    for fr in range(frames):
        B.bpy.context.scene.frame_set(ob.MC_props.reset_frame + 1 + fr)
        MC5.physics(C)
        co = np.array(C.co, dtype=np.float64)
        wc = B.world(ob, co)
        mid = 0.5 * (wc[edges[:, 0]] + wc[edges[:, 1]])
        at_tip = (np.hypot(mid[:, 0], mid[:, 1]) < 0.15) & (mid[:, 2] > 0.6)
        ratio = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1) / rl
        crushed += bool(at_tip.any() and ratio[at_tip].min() < 0.6)
        if prev is not None and fr >= 5:
            near = (np.hypot(wc[:, 0], wc[:, 1]) < 0.15) & (wc[:, 2] > 0.6)
            if near.any():
                max_rise = max(max_rise, float((wc[near, 2] - prev[near, 2]).max()))
        prev = wc
    tip = B._tip_clearance(prev, np.asarray(C.tridex), (0.0, 0.0, B.CONE_APEX))
    return max_rise, crushed, tip, frames


# A sheet sliding over a frictionless point lifts each vertex that comes
# over the tip up to the peak -- a few cm on this 6.7 cm grid, and physical.
# The bug was different: 5-11 cm kicks on alternate frames, edges at the tip
# crushed to 6%, and the tip settling to 1.5 cm under a 5 cm thickness.
max_rise, crushed, tip, frames = tilted_run()
check("off-centre tilted grid: no kicks (largest tip-region rise in a frame)",
      max_rise < 0.05, "(%.3f; the jumping reached 0.05-0.11)" % max_rise)
check("off-centre tilted grid: edges at the tip not crushed",
      crushed <= 3, "(under 60%% of rest in %d of %d frames)" % (crushed, frames))
check("off-centre tilted grid: tip held a full thickness clear",
      tip > 0.8 * 0.05, "(clearance %+.4f, thickness 0.05)" % tip)

print("================ %s ================"
      % ("ALL PASS" if all(results) else "FAILED"), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
