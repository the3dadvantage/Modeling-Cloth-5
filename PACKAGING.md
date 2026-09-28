# Packaging Modeling Cloth as an addon

The sources stay where they are: `Python\*.py`, loaded as text datablocks while
developing.  Packaging copies them into an extension alongside a manifest.

    Python\*.py                 the modules (unchanged, 19 files)
    addon\blender_manifest.toml the extension manifest -- edit metadata here
    addon\__init__.py           entry point: register() / unregister()
    addon\wheels\*.whl          bundled python packages (scipy)
    tools\build_addon.py        builds dist\modeling_cloth-<version>.zip
    tools\test_addon.py         builds, installs into a temp config, uses it
    tests\addon_smoke.py        the checks that test runs

## Build

    blender.exe --factory-startup -b --python tools\build_addon.py

or with Blender's own python (`blender-5.2.0-windows-x64\5.2\python\bin\python.exe`):

    py tools\build_addon.py [--platform windows-x64|macos-arm64|macos-x64|linux-x64|all]
                            [--version 5.0.1] [--no-zip] [--legacy]
                            [--install] [--solver-dll <path>]
                            [--fetch-wheels] [--no-wheels]

Two things land in `dist\`:

    dist\modeling_cloth\           the folder -- everything the addon needs
    dist\modeling_cloth-0.9.0-windows-x64.zip  that folder, zipped and validated

The folder is what gets zipped, and it is kept, so it can also be zipped by
hand or copied into Blender's extensions folder.  Blender's own extension
builder writes the zip, so the manifest is validated on the way.  Two compiled
libraries are packed in beside the modules:

  * `mc_collide.dll` -- from `Python\`, built by `cpp\mc_collide\build.bat`
  * `mc_cloth_solver.dll` -- the bend / stretch solver, taken from the first of
    `SOLVER_DLL_CANDIDATES` in `tools\build_addon.py` that exists (currently the
    Release build of "Cloth Test 1"), renamed to the name `utils.find_dll`
    looks for first

Rebuild the DLLs before building the addon if they have changed; the zip is a
snapshot of whatever is on disk at that moment.

## Platforms

One build per platform, because three of the pieces are compiled or
platform-tagged: the cloth solver, the collision library, and the scipy wheel.
`--platform all` builds every platform it can and says why it skipped the rest.

**Only windows-x64 can be built here.**  The other three need their native
libraries built on (or for) that platform first:

| Platform | Needs, in `addon\lib\<platform>\` |
|---|---|
| windows-x64 | nothing -- picked up where they are built |
| macos-arm64 | `mc_cloth_solver.dylib`, `mc_collide.dylib` |
| macos-x64 | `mc_cloth_solver.dylib`, `mc_collide.dylib` |
| linux-x64 | `mc_cloth_solver.so`, `mc_collide.so` |

Drop the libraries in and the same command builds that platform.  A build with
no solver is **refused**, not shipped: the solver is what simulates, so without
it the addon would install and quietly do nothing.  A missing `mc_collide` is
only a warning -- collision falls back to the python implementation.

### Building every platform with GitHub Actions

`.github\workflows\build.yml` builds all four platforms and produces the zips,
which is the only way to get macOS builds without a Mac.  It runs on a `v*` tag,
on a pull request that touches the code, or on demand (Actions ▸ build ▸ Run
workflow, with a switch for whether to bundle scipy).

    libs      one runner per platform: windows-latest, macos-14 (arm64),
              macos-13 (intel), ubuntu-22.04.  Each builds the two libraries
              and checks they export the symbols the bridges call.
    package   collects those libraries into addon\lib\<platform>\ and runs
              tools\build_addon.py --platform all
    verify    downloads Blender on each platform, installs that platform's zip
              into a throwaway config and uses it -- tools\test_addon.py --zip
    release   attaches the zips to a draft release, only on a v* tag and only
              if verify passed everywhere

`verify` is what makes a macOS or Linux build more than "it compiled": the same
install-and-run pass the Windows build gets, on the platform itself.  It is also
why `release` lists both `package` and `verify` in `needs` -- nothing ships that
has not been installed and used somewhere.

The Blender it tests against is set by `BLENDER_VERSION` / `BLENDER_SERIES` in
the workflow.  Keep its python version in step with `WHEEL_PYTHON` in
`tools\build_addon.py`: the bundled wheels are built for one python version, and
testing against a Blender with a different one would fail for a reason that has
nothing to do with the addon.

The zips come out as run artifacts, and on a tag they are also attached to a
**draft** release so a human decides when it goes public.

ubuntu-22.04 is deliberate: the oldest glibc we support, because a binary built
on a newer one will not load on an older distro.

The `package` job has no Blender, so the zips are assembled directly rather than
by Blender's builder -- the same layout, tested both ways.  The manifest is
therefore not validated there; validate locally with

    blender.exe --command extension validate dist\modeling_cloth-0.9.0-windows-x64.zip

The 22 headless suites in `tests\suites` do not run in CI: they load the modules
as text datablocks, which needs the working tree rather than an installed addon.
`verify` covers the installed-addon path instead.

### Building the libraries by hand

`cpp\build_msvc.bat` (Windows) and `cpp\build_unix.sh` (macOS / Linux) build both
libraries into `addon\lib\<platform>\`, which is what the CI jobs call and what
`tools\build_addon.py` reads.  `cpp\mc_collide\build.bat` still exists for
development: it drops `mc_collide.dll` straight into `Python\` where the text
datablock version of the addon finds it.

`cpp\solver\cloth_solver.cpp` is the solver source, in the repo so CI can build
it.  If you also keep a Visual Studio project for it elsewhere, point that
project at this file rather than its own copy -- two copies drift, and the drift
is invisible until a build somewhere behaves differently.

`MC_LIB_DIR` adds another folder to the library search, for a library built
somewhere outside the tree.

### Building the libraries for mac / linux

Both sources are already portable: the export macro is `#if defined(_WIN32)` on
each side, and the solver's MSVC precompiled header (which pulls in windows.h)
is behind `#ifdef _MSC_VER`.  Nothing else in either file is Windows-specific --
no windows.h of their own, no MSVC intrinsics, no OpenMP.  The Windows build was
rebuilt and the whole suite re-run after those changes.

`cpp\build_unix.sh` is the counterpart of `cpp\mc_collide\build.bat`.  Run it on
the machine you are building for -- nothing cross-compiles, and macOS cannot be
built from Windows at all:

    ./cpp/build_unix.sh                       # both libraries
    ./cpp/build_unix.sh --solver-src ~/cloth_solver.cpp

It works out the platform from `uname`, builds with clang++ (or g++), and writes
straight into `addon/lib/<platform>/`.  `cloth_solver.cpp` lives in the Visual
Studio project rather than this repo, so copy it to `cpp/solver/` on that
machine or pass `--solver-src`.

Then, back on any machine:

    py tools\build_addon.py --platform macos-arm64 --fetch-wheels

The python side needs no changes -- `U.lib_suffix()` picks `.dll`, `.dylib` or
`.so` and the library search uses it.

To check a build before packaging it, on that machine:

    blender --background --factory-startup --python tests/suites/bl_backend_test.py

which is the suite that exercises both native libraries against the python
reference.

## Bundled scipy

Magnetic targets need scipy; nothing else does.  The wheel in
`addon\wheels\<platform>\` is packed into the zip and installed by Blender when
the extension is enabled, so it works on a machine that cannot reach PyPI.
That is most of the download: about 35 MB against 0.3 MB without it.

    py tools\build_addon.py --fetch-wheels    # refresh the wheel, then build
    py tools\build_addon.py --no-wheels       # leave it out (0.3 MB)

Wheels are per python version *and* per platform, so each platform has its own
folder and `--fetch-wheels` downloads the right one (it tries the known tags for
that platform in turn, newest first).  They are fetched for **python 3.13**,
which is Blender 5.x; a Blender built against another python cannot use them, so
change `WHEEL_PYTHON` in `tools\build_addon.py` if that moves.  Only scipy is
bundled: its one dependency is numpy, which Blender ships.

The build rewrites the manifest's `wheels = [...]` from whatever is actually in
`addon\wheels`, because the manifest names each wheel by filename and Blender
refuses to install one that names a file the zip does not have.

Without the wheel, scipy is pip-installed into `scripts/modules` on demand
(about 5 s), at most once per session, and MC5 settings > Dependencies has a
button for installing it up front.  `--no-deps` is used there so pip cannot
drop a second numpy beside Blender's own.

## Install

Any of these work, and all three were tested end to end on Blender 5.2:

**1. The built zip.**  Preferences > Add-ons > Install from Disk, pick
`dist\modeling_cloth-0.9.0-windows-x64.zip`.  Or:

    blender.exe --command extension install-file -r user_default --enable dist\modeling_cloth-0.9.0-windows-x64.zip

`py tools\build_addon.py --install` does the same straight after building.

**2. A zip made by hand.**  Right-click `dist\modeling_cloth` > Send to >
Compressed folder.  That puts the package *inside* a folder in the zip, which
Blender accepts as well as the flat shape its own builder produces.  Install it
the same way as above.

**3. The folder, no zip.**  Copy `dist\modeling_cloth` into

    %APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\

Blender finds it at the next start, listed under Add-ons like any other
extension.  Handy for iterating: replace the folder and restart.

`--legacy` also writes a pre-4.2 style zip (the package inside a folder,
`bl_info` in `__init__.py`) for older Blender versions.  It is not needed for
Blender 4.2+.

## Test the packaged addon

    py tools\test_addon.py

Builds, installs into a throwaway `BLENDER_USER_RESOURCES` folder, then enables
and uses the installed copy: both DLLs found inside the addon folder, cloth
falls, object and self collision run on the C++ backend, the cache writes,
every panel draws, disable / re-enable is clean.  The real preferences are not
touched.

## Metadata to edit

In `addon\blender_manifest.toml`:

  * `version` -- bump for each release (the build takes it from here)
  * `maintainer` -- "Rich Colburn <the3dadvantage@gmail.com>"
  * `license` -- `SPDX:GPL-3.0-or-later` (Blender add-ons must be GPL-compatible)
  * `blender_version_min` -- 5.0.0
  * `platforms` -- `windows-x64` only, because the DLLs are Windows builds
  * `id` -- `modeling_cloth`; changing it makes Blender treat it as a different
    addon, so installed copies of the old id stay behind

Keep the `version` in `addon\__init__.py`'s `bl_info` in step with the manifest
if the legacy zip matters; extensions ignore `bl_info`.

## Notes

  * scipy is only needed by magnetic targets, and is imported when that feature
    is first used, so a machine with no network can still enable the addon.
  * `register()` defers rebuilding existing cloth objects to a timer when
    Blender is still starting up (`bpy.data` is restricted at that point).
  * Every module finds its neighbours through `bpy.data.texts` first and falls
    back to a relative import, so the same files work both ways.  When adding a
    module import, add it to *both* branches -- a missing name in the fallback
    only shows up in the installed addon.
