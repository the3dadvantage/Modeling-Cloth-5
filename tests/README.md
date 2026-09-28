# Tests

Run everything headless with Blender 5.2, from the root of the repository (the
suites find the addon's modules relative to their own location, so a clone works
anywhere; `MC_SRC` overrides it):

```
blender-5.2.0-windows-x64\blender.exe -b --factory-startup --python tests\run_bench.py -- check
```

## Collision benchmark (`run_bench.py`, `collide_bench.py`)

This is the reference for the C++ collision port. Each scene is built from
explicit vertex and face lists, stepped with the real solver, and its final
vertex positions are compared with a stored baseline in `baselines/`.

| command | what it does |
|---|---|
| `list` | list the scenes and what each one covers |
| `check [scenes] [--tol X]` | compare with the baselines (default tolerance 0, bit-identical); exits with code 1 on any mismatch |
| `capture [scenes]` | store new baselines, refusing any scene whose collision didn't engage |
| `determinism [scenes]` | run each scene twice in one process; the runs must be identical |
| `time [scenes]` | timings only |

`MC_SRC=<folder>` points the harness at another copy of the addon (a backup
or a fork) without touching the baselines.

`--set prop=value` (repeatable) overrides a cloth property for `check`,
`time` or `determinism`, to compare variants against a scene:

```
blender-5.2.0-windows-x64\blender.exe -b --factory-startup --python tests\run_bench.py -- time oc_cone --set ob_recollide=False
```

`capture` refuses `--set`, so a variant can never become a baseline.

`oc_cone` is the only scene with recollide on; all the others keep it off.
Its pass test is that the cone tip never gets through the cloth.

Every scene checks that collision really took part: it counts actual contacts
and runs a physical test (resting height, penetration, gap between layers,
distance from the slope). A cloth that falls straight through its collider is
deterministic too, so without this check it would match its baseline forever.

For the C++ port, `check --tol` with a small tolerance is the positional test.
The physical metrics that `check` prints (rest height, worst penetration,
layer gap, slide) are the real acceptance test.

## C++ collision (`cpp/mc_collide`)

`cpp\mc_collide\build.bat` builds `Python\mc_collide.dll` (Visual Studio with
the C++ workload). It's safe to run while Blender is open: the addon loads a
copy of the DLL, not the file itself. Restart Blender to pick up a rebuild.

The Settings panel has a **Collision Engine** switch: C++ or Python. It falls
back to Python automatically, and says why, if the DLL is missing or was built
for a different interface version.

- Baselines are always captured from the Python reference.
- `run_bench.py -- check --backend cpp --tol 1.0` runs the C++ version against
  them.
- `shadow_compare.py` runs both implementations on identical inputs at every
  call and reports the first real difference:

```
blender-5.2.0-windows-x64\blender.exe -b --factory-startup --python tests\shadow_compare.py
```

Expected results:
- Broad-phase pair lists and self-collision candidates are identical.
- Contacts (object and self) agree to float32 round-off.
- The one known exception is `oc_ridge`: a vertex exactly over the ridge is a
  true tie between the two faces, and each version breaks it the other way.

Chaotic scenes (a sheet balanced on a ridge, a collider pushing through) then
amplify round-off, so a full-run `check` shows larger deviations there, while
the physical metrics still match.

## Suites (`suites/`)

These are standalone regression scripts. Each one prints `ALL PASS` or `FAILED`:

```
blender-5.2.0-windows-x64\blender.exe -b --factory-startup --python tests\suites\<name>.py
```

`bl_bench_selftest.py` checks that the benchmark can fail. It must catch a 2%
change to contact damping, refuse a fall-through run, and report a frame-count
mismatch. It needs baselines captured first.

## Known gaps (not baselined)

- A collider that moves faster than the collision margin per frame punches
  through, whatever the substep count. Substeps only subdivide the cloth's
  motion; the collider still jumps its whole frame in the first substep.
