"""Build the addon, install it into a throwaway Blender config, and use it.

Nothing here touches the real Blender preferences: BLENDER_USER_RESOURCES
points at a temp folder for the whole run, so the installed copy, and the
addon being enabled, disappear with it.

    blender.exe --factory-startup -b --python tools\\test_addon.py
    py tools\\test_addon.py
    py tools\\test_addon.py --zip dist\\modeling_cloth-0.9.0-linux-x64.zip

It proves the packaged addon works with no text datablocks anywhere: both
libraries are found inside the installed folder, cloth falls, collision runs on
the C++ backend, the cache writes, every panel draws, and it disables and
re-enables cleanly.  tests\\addon_smoke.py holds the checks.

With `--zip` it tests a zip that already exists instead of building one, which
is how CI checks a macOS or Linux build: the same install-and-use pass, run on
that platform against the binary that was just compiled there.  Set MC_BLENDER
to the Blender to test with.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get(
    "MC_BLENDER", os.path.join(ROOT, "blender-5.2.0-windows-x64", "blender.exe"))
SMOKE = os.path.join(ROOT, "tests", "addon_smoke.py")


def run(cmd, env):
    # utf-8 with replacement: Blender's output is not always decodable as the
    # console codepage, and that used to fail inside subprocess's reader thread
    p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None,
                    help="test this zip instead of building one")
    args = ap.parse_args([a for a in sys.argv[1:] if a != "--"]
                         if "--" not in sys.argv else
                         sys.argv[sys.argv.index("--") + 1:])

    if not os.path.isfile(BLENDER):
        raise SystemExit("Blender not found at %s (set MC_BLENDER)" % BLENDER)

    if args.zip:
        zip_path = os.path.abspath(args.zip)
        if not os.path.isfile(zip_path):
            raise SystemExit("no such zip: %s" % zip_path)
        print("testing %s" % zip_path, flush=True)
    else:
        # build (into dist\, same zip the user installs)
        code, out = run([sys.executable, os.path.join(ROOT, "tools", "build_addon.py")],
                        dict(os.environ))
        print(out.strip(), flush=True)
        if code != 0:
            raise SystemExit("build failed")
        zips = sorted((os.path.join(ROOT, "dist", f)
                       for f in os.listdir(os.path.join(ROOT, "dist"))
                       if f.endswith(".zip") and "legacy" not in f),
                      key=os.path.getmtime)
        zip_path = zips[-1]

    tmp = tempfile.mkdtemp(prefix="mc_addon_test_")
    env = dict(os.environ, BLENDER_USER_RESOURCES=tmp)
    try:
        code, out = run([BLENDER, "--command", "extension", "install-file",
                         "-r", "user_default", "--enable", zip_path], env)
        print(out.strip(), flush=True)
        if code != 0:
            raise SystemExit("install failed")

        code, out = run([BLENDER, "-b", "--python-exit-code", "1",
                         "--python", SMOKE], env)
        keep = [ln for ln in out.splitlines()
                if ("PASS" in ln or "FAIL" in ln or "SKIP" in ln
                    or ln.startswith("==") or "Error" in ln or ", in " in ln)]
        print("\n".join(keep), flush=True)
        raise SystemExit(code)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
