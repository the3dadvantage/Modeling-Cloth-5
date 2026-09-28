"""Build the Modeling Cloth addon.

The sources stay where they are (Python\\*.py, loaded as text datablocks while
developing).  This gathers a copy with the manifest and __init__.py from
addon\\, drops the compiled libraries in beside them, and hands the folder to
Blender's own extension builder, which validates the manifest as it goes.

    blender.exe --factory-startup -b --python tools\\build_addon.py
    py tools\\build_addon.py                     # any python 3.8+ works too

Two things come out of dist\\:

    dist\\modeling_cloth\\          the folder: everything the addon needs
    dist\\modeling_cloth-5.0.0.zip  the same folder, zipped and validated

The folder is kept, so it can be zipped by hand (right-click > Send to >
Compressed folder) or copied straight into Blender's extensions folder.
Blender installs either shape: the package inside a folder, or the manifest at
the top level.

Options:
    --version 5.0.1        override the manifest version
    --no-zip               just the folder
    --legacy               also write a pre-4.2 style zip (folder inside)
    --install              install the built zip into this machine's Blender
    --solver-dll <path>    use this cloth solver DLL instead of searching

Install by hand with Preferences > Add-ons > Install from Disk, or

    blender.exe --command extension install-file -r user_default <zip>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Python")
ADDON = os.path.join(ROOT, "addon")
WHEELS = os.path.join(ADDON, "wheels")
DIST = os.path.join(ROOT, "dist")

# Bundled python packages.  Blender installs an extension's wheels itself, so
# magnetic targets work on a machine that cannot reach PyPI.  scipy's only
# dependency is numpy, which Blender ships, so nothing else is needed.
WHEEL_PACKAGES = ["scipy"]
WHEEL_PYTHON = "3.13"           # Blender 5.x
BLENDER = os.environ.get(
    "MC_BLENDER", os.path.join(ROOT, "blender-5.2.0-windows-x64", "blender.exe"))

PKG = "modeling_cloth"          # must match `id` in blender_manifest.toml

# One build per platform, because the solver and collision libraries are
# compiled and the scipy wheel is built for a specific python + platform.
#
#   libs   : where this platform's native libraries are kept, and what the
#            shared-library suffix is
#   wheels : the wheel platform tags to try, best first
PLATFORMS = {
    "windows-x64": {"suffix": ".dll", "wheels": ["win_amd64"]},
    "macos-arm64": {"suffix": ".dylib",
                    "wheels": ["macosx_12_0_arm64", "macosx_11_0_arm64"]},
    "macos-x64": {"suffix": ".dylib",
                  "wheels": ["macosx_12_0_x86_64", "macosx_10_13_x86_64"]},
    "linux-x64": {"suffix": ".so",
                  "wheels": ["manylinux_2_28_x86_64", "manylinux2014_x86_64"]},
}
DEFAULT_PLATFORM = "windows-x64"

# Compiled libraries.  For windows-x64 they are picked up where they are built;
# for any other platform they are expected in addon\lib\<platform>\ , named
# mc_cloth_solver.<suffix> and mc_collide.<suffix>.
LIBS = os.path.join(ADDON, "lib")


def native_libs(platform):
    """(solver, collide) paths for this platform.  Either may be None.

    The solver is what actually simulates -- without it the addon installs and
    does nothing -- so a build refuses to package a platform it has no solver
    for rather than shipping something that cannot work."""
    suffix = PLATFORMS[platform]["suffix"]
    per_platform = os.path.join(LIBS, platform)
    solver_names = ["mc_cloth_solver" + suffix, "cloth_solver" + suffix]
    collide_names = ["mc_collide" + suffix]

    # addon\lib\<platform>\ is where cpp\build_msvc.bat, cpp\build_unix.sh and
    # the CI workflow all put them.  Python\ is also checked on Windows because
    # cpp\mc_collide\build.bat drops mc_collide.dll there for development.
    # MC_LIB_DIR adds a folder of your own, for a library built elsewhere.
    roots = [per_platform]
    if platform == DEFAULT_PLATFORM:
        roots.append(SRC)
    extra = os.environ.get("MC_LIB_DIR")
    if extra:
        roots.insert(0, extra)

    def first(names):
        for root in roots:
            for n in names:
                p = os.path.join(root, n)
                if os.path.isfile(p):
                    return p
        return None

    return first(solver_names), first(collide_names)


def say(*a):
    print(*a, flush=True)


def manifest_version(text):
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "0.0.0"


def wheel_dir(platform):
    return os.path.join(WHEELS, platform)


def fetch_wheels(platform):
    """Download the bundled packages for Blender's python on `platform`.

    A wheel is per python version *and* per platform, so each build needs its
    own.  pip needs an exact platform tag, and which tags a project publishes
    changes over time, so the known tags are tried in turn."""
    dest = wheel_dir(platform)
    os.makedirs(dest, exist_ok=True)
    for name in WHEEL_PACKAGES:
        for old in os.listdir(dest):
            if old.startswith(name + "-") and old.endswith(".whl"):
                os.remove(os.path.join(dest, old))
        done = False
        for tag in PLATFORMS[platform]["wheels"]:
            say("fetching %s for python %s / %s..." % (name, WHEEL_PYTHON, tag))
            cmd = [sys.executable, "-m", "pip", "download", name,
                   "--no-deps", "--only-binary=:all:",
                   "--python-version", WHEEL_PYTHON,
                   "--platform", tag, "--dest", dest]
            if subprocess.run(cmd).returncode == 0:
                done = True
                break
            say("  no wheel for %s, trying the next tag" % tag)
        if not done:
            raise SystemExit("could not download %s for %s" % (name, platform))


def stage_wheels(out, man, platform):
    """Copy the wheels in and make the manifest say exactly what is there.

    The manifest has to name each wheel by filename, so a version bump on disk
    would otherwise leave it pointing at a file that is no longer in the zip --
    Blender refuses to install that, and the build would not have told anyone.
    """
    src_dir = wheel_dir(platform)
    wheels = sorted(f for f in os.listdir(src_dir)
                    if f.endswith(".whl")) if os.path.isdir(src_dir) else []
    if wheels:
        os.makedirs(os.path.join(out, "wheels"), exist_ok=True)
        total = 0
        for w in wheels:
            shutil.copy2(os.path.join(src_dir, w), os.path.join(out, "wheels", w))
            total += os.path.getsize(os.path.join(src_dir, w))
        say("bundled %d wheel(s), %.0f MB: %s"
            % (len(wheels), total / 1e6, ", ".join(wheels)))
        listing = "wheels = [%s]" % ", ".join('"./wheels/%s"' % w for w in wheels)
    else:
        say("no wheels bundled -- scipy will be pip-installed on demand")
        listing = ""

    line = re.compile(r'^wheels\s*=\s*\[[^\]]*\]', re.M | re.S)
    if listing:
        if line.search(man):
            return line.sub(listing, man, count=1)
        return re.sub(r'^(license\s*=\s*\[[^\]]*\])', r'\1\n\n' + listing,
                      man, count=1, flags=re.M | re.S)

    # nothing to list: take the comment that introduces it out with the line,
    # rather than leaving a note about wheels that are not there
    return re.sub(r'\n*(?:^#[^\n]*\n)*^wheels\s*=\s*\[[^\]]*\]', "", man,
                  count=1, flags=re.M)


def stage(version, platform, solver_override=None):
    """The folder holding exactly what the addon needs, and nothing else.

    Kept after the build: zip it by hand, or copy it into Blender's
    extensions folder as an unzipped install."""
    out = os.path.join(DIST, "%s-%s" % (PKG, platform))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    mods = sorted(f for f in os.listdir(SRC) if f.endswith(".py"))
    for f in mods:
        shutil.copy2(os.path.join(SRC, f), os.path.join(out, f))
    shutil.copy2(os.path.join(ADDON, "__init__.py"), os.path.join(out, "__init__.py"))
    readme = os.path.join(ADDON, "README.md")
    if os.path.isfile(readme):
        shutil.copy2(readme, os.path.join(out, "README.md"))

    man = open(os.path.join(ADDON, "blender_manifest.toml"), encoding="utf-8").read()
    if version:
        man = re.sub(r'^version\s*=\s*"[^"]+"', 'version = "%s"' % version, man, count=1, flags=re.M)
    man = re.sub(r'^platforms\s*=\s*\[[^\]]*\]', 'platforms = ["%s"]' % platform,
                 man, count=1, flags=re.M | re.S)
    man = stage_wheels(out, man, platform)
    open(os.path.join(out, "blender_manifest.toml"), "w", encoding="utf-8").write(man)

    suffix = PLATFORMS[platform]["suffix"]
    solver, collide = native_libs(platform)
    if solver_override:
        solver = solver_override

    libs = []
    if collide:
        shutil.copy2(collide, os.path.join(out, "mc_collide" + suffix))
        libs.append("mc_collide" + suffix)
    else:
        say("WARNING: no mc_collide%s for %s -- collision falls back to python"
            % (suffix, platform))
    if solver:
        shutil.copy2(solver, os.path.join(out, "mc_cloth_solver" + suffix))
        libs.append("mc_cloth_solver%s  (from %s)" % (suffix, solver))
    else:
        # leave nothing half-built behind to be mistaken for a distribution
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(
            "no cloth solver library for %s.\n"
            "  The solver is what simulates: without it the addon installs and\n"
            "  does nothing, so this build is refused rather than shipped.\n"
            "  Build it for %s and put it in %s as mc_cloth_solver%s."
            % (platform, platform, os.path.join(LIBS, platform), suffix))

    say("staged %d modules + %s" % (len(mods) + 1, ", ".join(libs)))
    return out, manifest_version(man)


def build_extension(src):
    """Blender's builder: validates the manifest and writes the zip."""
    if not os.path.isfile(BLENDER):
        say("blender.exe not found at %s (set MC_BLENDER) -- zipping by hand"
            % BLENDER)
        return None
    os.makedirs(DIST, exist_ok=True)
    cmd = [BLENDER, "--command", "extension", "build",
           "--source-dir", src, "--output-dir", DIST]
    # Blender's output is not always decodable as the console codepage, and an
    # undecodable byte used to blow up in subprocess's reader thread
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.returncode and (r.stderr or "") or "")
    say(out.strip())
    if r.returncode != 0:
        raise SystemExit("extension build failed")
    m = re.search(r'created:\s*"?([^"\n]+\.zip)', out)
    return m.group(1).strip() if m else None


def zip_manually(src, version, legacy, platform=""):
    """Fallback / legacy zip.  A legacy addon zip holds the package folder;
    an extension zip holds the manifest at the top level."""
    os.makedirs(DIST, exist_ok=True)
    name = "%s-%s%s%s.zip" % (PKG, version, ("-" + platform) if platform else "",
                              "-legacy" if legacy else "")
    path = os.path.join(DIST, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(src)):
            arc = os.path.join(PKG, f) if legacy else f
            z.write(os.path.join(src, f), arc)
    return path


def install(zip_path):
    """Install into this machine's Blender (replaces any copy of the same id)."""
    r = subprocess.run([BLENDER, "--command", "extension", "install-file",
                        "-r", "user_default", "--enable", zip_path],
                       capture_output=True, text=True)
    say(((r.stdout or "") + (r.stderr or "")).strip())
    if r.returncode != 0:
        raise SystemExit("install failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--no-zip", action="store_true",
                    help="build the folder only")
    ap.add_argument("--legacy", action="store_true",
                    help="also write a pre-4.2 style addon zip")
    ap.add_argument("--install", action="store_true",
                    help="install the built zip into this machine's Blender")
    ap.add_argument("--fetch-wheels", action="store_true",
                    help="(re)download the bundled packages first")
    ap.add_argument("--no-wheels", action="store_true",
                    help="leave the wheels out; scipy is pip-installed on demand")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM,
                    choices=sorted(PLATFORMS) + ["all"],
                    help="which platform to build for (default %s)" % DEFAULT_PLATFORM)
    ap.add_argument("--solver-dll", default=None)
    args = ap.parse_args([a for a in sys.argv[1:] if a != "--"]
                         if "--" not in sys.argv else
                         sys.argv[sys.argv.index("--") + 1:])

    if args.solver_dll and not os.path.isfile(args.solver_dll):
        raise SystemExit("no such file: %s" % args.solver_dll)

    wanted = sorted(PLATFORMS) if args.platform == "all" else [args.platform]
    built, refused = [], []
    for platform in wanted:
        say("\n===== %s =====" % platform)
        if args.fetch_wheels and not args.no_wheels:
            fetch_wheels(platform)
        try:
            zip_path = build_one(args, platform)
        except SystemExit as e:
            if args.platform != "all":
                raise
            refused.append((platform, str(e)))
            say(str(e))
            continue
        built.append((platform, zip_path))

    if len(wanted) > 1:
        say("\n===== summary =====")
        for platform, path in built:
            say("  built    %-12s %s" % (platform, path))
        for platform, why in refused:
            say("  skipped  %-12s %s" % (platform, why.splitlines()[0]))


def build_one(args, platform):
    if args.no_wheels:
        globals()["WHEELS"] = os.path.join(ADDON, "no-such-folder")

    src, version = stage(args.version, platform, args.solver_dll)
    say("folder:     %s" % src)
    if args.no_zip:
        return src

    zip_path = (build_extension(src)
                or zip_manually(src, version, legacy=False, platform=platform))
    # Blender's builder names the zip after the id and version only, so every
    # platform would write the same file and the last one would win
    if zip_path and platform not in os.path.basename(zip_path):
        stem, ext = os.path.splitext(zip_path)
        named = "%s-%s%s" % (stem, platform, ext)
        shutil.move(zip_path, named)
        zip_path = named
    say("extension:  %s" % zip_path)
    if args.legacy:
        say("legacy:     %s" % zip_manually(src, version, legacy=True,
                                           platform=platform))
    if args.install:
        install(zip_path)
    return zip_path


if __name__ == "__main__":
    main()
