# Modeling Cloth 5

**0.9.0 — beta.** Cloth simulation for Blender that you can keep modelling with:
edit the pattern while the cloth is running and the simulation carries on, with
new vertices placed into the moving cloth rather than snapped back to a rest
shape.

By Rich Colburn. Licensed under the GNU General Public License, version 3 or
later.

Windows only for now, Blender 5.0 or newer. See **Known limits** at the end.

---

## Install

Preferences ▸ Add-ons ▸ Install from Disk, and pick
`modeling_cloth-0.9.0-windows-x64.zip`.

Everything it needs is inside the zip, including scipy (used by magnetic
targets), so there is nothing to install by hand and no network needed.

Two tabs appear in the 3D viewport sidebar (**N**):

* **MC5** — the simulation
* **MC5 Tools** — building the mesh: Grid Fill, UV Shape, Sewing

---

## Quick start

1. Select a mesh. In **MC5 ▸ Main**, turn on **Cloth**.
2. Turn on **Animated** and play the timeline, or **Continuous** to let it run
   without playback.
3. **Forces** sets gravity, stretch and bend.
4. To have it land on something, select that object and tick
   **Collision ▸ Object ▸ Collider**. Collider is set *on the collider*, not on
   the cloth.

**Reset** returns the cloth to its target shape. **Apply** (Apply As Target)
makes the current simulated shape the new rest shape — useful once a garment has
settled and you want to keep going from there.

---

## The target workflow

This is the part that is unlike other cloth systems, and it is worth
understanding.

Set **Main ▸ Target Object** to a second mesh — the pattern. From then on:

* The target's edge lengths and angles are what the springs aim for, so
  reshaping the target reshapes what the cloth is pulling towards, live.
* **Edit the target's topology while the simulation runs** — knife cut, extrude,
  subdivide, join in another mesh, delete — and the cloth follows. Vertices you
  already had keep their simulated positions *and* their velocity. New ones are
  placed by mapping them through the surrounding faces, so a vertex cut into the
  middle of an edge appears in the middle of that edge as the cloth currently
  is, not where the pattern's rest shape puts it.
* It works the other way too: edit the **cloth** mesh and the target is brought
  onto the new topology, keeping its own shape.

Vertices are followed by an id stored in a mesh attribute called
`Kitten Mana`. You will see it in the attribute list. Leave it alone — deleting
it means the next edit cannot tell old vertices from new ones.

---

## Things that surprise people

**Selected vertices are pinned.** Selection is how you grab and hold part of a
cloth, so anything selected stops moving. Blender leaves new geometry selected
after an edit, so **deselect after editing** or that area will sit frozen.

**The active shape key must be `MC_current`.** The addon keeps `Basis`,
`MC_target` and `MC_current` on a cloth and simulates into `MC_current`. Switch
the active key and physics stops until you switch back.

**Reset ▸ Re Sel** resets only the selected vertices (the toggle beside the
button).

---

## Collision

**Object** — tick **Collider** on whatever the cloth should hit. Radius,
friction and static threshold are set on the cloth.

**Self** — self collision, with a radius. **Calculate** measures a sensible
radius from the mesh.

**Scene Colliders** lists everything marked as a collider, so you can find and
toggle them without hunting in the outliner.

If contacts look jittery on a dense mesh, raise **Settings ▸ Collision
Substeps**, or turn on **Recollide** (Settings ▸ Object Collision), which
re-resolves contacts between solver iterations — slower, but much steadier for
cloth over sharp shapes.

---

## Cache

**Cache** bakes the simulation to disk, one file per frame, next to the .blend.

* **Bake** runs from the start frame; **From Current** carries on from where you
  are, leaving earlier frames alone.
* **Space** or **P** pauses a running bake; **Esc** stops it.
* **Playback** replays the cache instead of simulating. It works in edit mode
  too, so you can scrub frames while looking at the mesh up close.
* **Update This Frame** overwrites one cached frame with the mesh as it is now,
  keeping the old data as a `.bak` — for fixing a single frame by hand.

If you change the vertex count, the cache no longer fits and playback switches
off with a message. Re-bake.

Before the file has been saved the cache goes to a temporary folder and is moved
next to the .blend when you save. The panel says so while that is the case.

---

## Other features

**Sewing** (Tools tab) — select two edge chains in edit mode and sew them.
Offset, flip and face-fill for adjusting seams. How hard seams pull is under
**Forces ▸ Sewing** on the MC5 tab.

**Hooks** — hook selected vertices to an empty and move them while the cloth
runs.

**Wind** — turbulent wind with an optional object to aim it: rotate that object
and its +Z is the direction.

**Magnetic** — pull a cloth towards another mesh's surface. This is the one
feature that uses scipy; it is bundled, so it just works.

**Grid Fill / UV Shape** (Tools tab) — build a grid mesh inside a boundary, or
build a flat pattern piece from a UV layout.

**Presets** store the cloth settings so you can reuse them on another garment.

---

## Settings

**Collision Engine** — C++ or Python. C++ is the default and much faster; the
Python implementation is the reference and is there to fall back on if you
suspect the compiled one. The panel says which is actually loaded.

**Collision Substeps** — more substeps for fast-moving cloth, with an automatic
mode.

**Dependencies** — says whether scipy is present, with a button to install it if
you are running from source rather than the packaged addon.

**Debug Mode** — extra diagnostics in the console plus a few developer dials.
Turn this on before reporting a problem: solver timings and setup traces only
print with it enabled.

---

## Known limits in 0.9.0

* **Windows only.** The solver and collision libraries are compiled, and only
  Windows builds exist. The addon will not install on macOS or Linux.
* **Interactive use is the least tested part.** The automated tests drive Blender
  headlessly, so anything involving live dragging, modal operators or viewport
  redraw has had the least exercise. This is where problems are most likely.
* A simulation with no solver library cannot run; if the library is missing you
  get a message pointing at the setting that overrides its location
  (Settings ▸ Debug Mode ▸ DLL path).

## Reporting a problem

<https://github.com/the3dadvantage/modeling-cloth/issues> — newer versions are
on the [releases page](https://github.com/the3dadvantage/modeling-cloth/releases).

Turn on **Settings ▸ Debug Mode**, reproduce it, and include the console output
(Window ▸ Toggle System Console). Saying what was selected and whether you were
in edit mode at the time helps more than anything else.
