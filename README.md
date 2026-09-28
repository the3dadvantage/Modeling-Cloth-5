# Modeling Cloth 5

Cloth simulation for Blender that you can keep modelling with. Edit the pattern
while the cloth is running and the simulation carries on — new vertices are
placed into the moving cloth instead of snapping back to a rest shape.

**0.9.0 — beta.** Blender 5.0 or newer.

---

## Download

**[➜ Get the latest release](https://github.com/the3dadvantage/modeling-cloth/releases/latest)**

Pick the one for your machine:

| Your machine | File |
|---|---|
| Windows | `modeling_cloth-0.9.0-windows-x64.zip` |
| Mac, Apple Silicon (M1 and later) | `modeling_cloth-0.9.0-macos-arm64.zip` |
| Mac, Intel | `modeling_cloth-0.9.0-macos-x64.zip` |
| Linux | `modeling_cloth-0.9.0-linux-x64.zip` |

Then in Blender: **Preferences ▸ Add-ons ▸ Install from Disk**, pick the zip, and
two tabs appear in the 3D viewport sidebar (**N**) — **MC5** and **MC5 Tools**.

Everything it needs is inside the zip, including scipy, so there is nothing else
to install and no network needed.

> Downloading the repository with the green **Code** button gives you the source,
> not an installable addon — the compiled parts are built per platform and are
> not in the source tree. Use a release zip.

## What it does

- A solver with stretch, bend, gravity, wind, inflate and air drag
- Object and self collision, with friction and substeps, in C++
- **Target workflow** — a second mesh drives the rest shape, and editing it
  (knife, extrude, subdivide, join, delete) reshapes the running cloth without
  restarting it
- Sewing: sew edge chains and let seams pull together
- Hooks, magnetic targets, vertex-group weighting for most forces
- Bake to disk, scrub, edit a single cached frame, play back in edit mode
- Pattern tools: grid fill, and building a flat piece from a UV layout

**[Full user guide](addon/README.md)** — the workflow, the settings, and the
handful of behaviours that surprise people (notably: selected vertices are
pinned, so deselect after an edit).

## Status

Beta. It is used daily on Windows and has an automated test suite (22 suites,
plus an install-and-run test of the packaged addon), but:

- **Interactive use is the least tested part.** The tests drive Blender
  headlessly, so live dragging, modal operators and viewport redraw have had the
  least exercise. That is where problems are most likely.
- macOS and Linux builds are produced by CI but have had less real use than the
  Windows one.

Bug reports are welcome. Turn on **MC5 ▸ Settings ▸ Debug Mode**, reproduce the
problem, and include the console output (**Window ▸ Toggle System Console**).

## Building from source

You only need this to develop the addon; users should take a release zip.

```
cpp\build_msvc.bat                      # Windows: both native libraries
./cpp/build_unix.sh                     # macOS / Linux: same
python tools/build_addon.py --platform windows-x64 --fetch-wheels
```

Every platform is also built by CI on a `v*` tag. See
**[PACKAGING.md](PACKAGING.md)** for the layout, the per-platform builds, how
scipy is bundled, and how to run the tests.

## License

GPL-3.0-or-later. © Rich Colburn.
