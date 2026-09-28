"""Gather everything the repository needs into one folder.

    py tools\\stage_repo.py            # -> repo_root\\

The *contents* of that folder are the repository root: Python\\, addon\\, cpp\\,
tests\\, tools\\, .github\\ and the top-level files.  Upload the contents, not
the folder itself, or everything ends up one level too deep.

Left out on purpose:
  * addon\\lib and addon\\wheels -- built per platform, 29-37 MB each, fetched
    by the build and by CI
  * Python\\*.dll -- built by cpp\\build_msvc.bat
  * __pycache__, cpp\\obj, dist -- build output
  * everything else in the working directory, which holds unrelated projects
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "repo_root")

# (source, destination) -- destination None means the same name
TREES = [
    ("Python", None),
    ("addon", None),
    ("cpp", None),
    ("tests", None),
    ("tools", None),
    (".github", None),
]
FILES = ["README.md", "PACKAGING.md"]

# The repository root holds only this project, so a plain ignore list belongs
# there -- not the allowlist this working directory needs.
GITIGNORE = """\
# build output
/dist/
/build/
/cpp/obj/

# native libraries: built per platform by cpp/build_msvc.bat,
# cpp/build_unix.sh or the CI workflow
/addon/lib/
/Python/*.dll

# bundled wheels, 29-37 MB each: build_addon.py --fetch-wheels gets them,
# and CI fetches them on every run
/addon/wheels/

# python
__pycache__/
*.py[cod]

# blender
*.blend1
*.blend2
*.blend3
*.blend4
"""

SKIP_DIRS = {"__pycache__", "obj", "lib", "wheels", ".vs", "sandbox"}
SKIP_EXTS = {".dll", ".dylib", ".so", ".whl", ".pyc", ".lib", ".exp", ".obj",
             ".blend", ".blend1", ".blend2", ".blend3", ".blend4"}


def keep(path):
    parts = set(path.replace("\\", "/").split("/"))
    if parts & SKIP_DIRS:
        return False
    return os.path.splitext(path)[1].lower() not in SKIP_EXTS


def copy_tree(src, dst):
    n = 0
    for base, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, src)
            if not keep(rel):
                continue
            target = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(full, target)
            n += 1
    return n


def main():
    # Clear it out rather than removing the folder itself: a shell sitting in
    # that directory (or an explorer window) keeps it from being deleted, and
    # the rebuild should not fail for that.
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        p = os.path.join(OUT, name)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

    total = 0
    for src, dst in TREES:
        s = os.path.join(ROOT, src)
        if not os.path.isdir(s):
            print("  missing, skipped: %s" % src)
            continue
        n = copy_tree(s, os.path.join(OUT, dst or src))
        print("  %-10s %d file(s)" % (src, n))
        total += n

    for f in FILES:
        s = os.path.join(ROOT, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, f))
            total += 1
            print("  %s" % f)

    with open(os.path.join(OUT, ".gitignore"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(GITIGNORE)
    total += 1
    print("  .gitignore (written for a repository root, not this folder)")

    size = sum(os.path.getsize(os.path.join(b, f))
               for b, _d, fs in os.walk(OUT) for f in fs)
    print("\n%d files, %.1f MB in %s" % (total, size / 1e6, OUT))
    print("Upload the CONTENTS of that folder as the repository root.")


if __name__ == "__main__":
    main()
