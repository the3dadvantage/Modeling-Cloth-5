#!/usr/bin/env bash
# Builds the two native libraries on macOS or Linux, the counterpart of
# cpp/mc_collide/build.bat.
#
# Run it on the machine you are building for -- a mac binary needs a mac, and a
# linux binary needs linux (or WSL).  Nothing here cross-compiles.
#
#   ./cpp/build_unix.sh                      # both libraries
#   ./cpp/build_unix.sh --solver-src ~/cloth_solver.cpp
#
# The results land in addon/lib/<platform>/ , which is where
# tools/build_addon.py looks:
#
#   addon/lib/macos-arm64/mc_cloth_solver.dylib
#   addon/lib/macos-arm64/mc_collide.dylib
#
# Then, back on any machine:
#
#   py tools/build_addon.py --platform macos-arm64 --fetch-wheels
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# ---------------------------------------------------------------- platform
case "$(uname -s)" in
    Darwin)
        SUFFIX=".dylib"
        case "$(uname -m)" in
            arm64) PLATFORM="macos-arm64" ;;
            *)     PLATFORM="macos-x64" ;;
        esac
        # -install_name keeps the library loadable from wherever the addon is
        # unzipped, rather than the path it was built in
        LINK_FLAGS=(-dynamiclib -install_name "@rpath/LIBNAME")
        ;;
    Linux)
        SUFFIX=".so"
        PLATFORM="linux-x64"
        LINK_FLAGS=(-shared)
        ;;
    *)
        echo "unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac

CXX="${CXX:-clang++}"
command -v "$CXX" >/dev/null 2>&1 || CXX=g++
command -v "$CXX" >/dev/null 2>&1 || { echo "no clang++ or g++ found" >&2; exit 1; }

OUT="$ROOT/addon/lib/$PLATFORM"
mkdir -p "$OUT"
echo "building for $PLATFORM with $CXX into $OUT"

# The same switches build.bat uses: optimised, no fast-math (the solver relies
# on predictable float behaviour), C++17, hidden visibility so only the
# functions marked API / MC_API are exported.
FLAGS=(-O2 -std=c++17 -fPIC -fvisibility=hidden -Wall)

build() {
    local src="$1" name="$2"
    local lib="$name$SUFFIX"
    local flags=("${LINK_FLAGS[@]}")
    local i
    for i in "${!flags[@]}"; do
        flags[$i]="${flags[$i]//LIBNAME/$lib}"
    done
    echo "  $src -> $lib"
    "$CXX" "${FLAGS[@]}" "${flags[@]}" "$src" -o "$OUT/$lib"
}

# ---------------------------------------------------------------- collision
build "$HERE/mc_collide/mc_collide.cpp" "mc_collide"

# ------------------------------------------------------------------- solver
# cloth_solver.cpp lives in the Visual Studio project on the Windows machine,
# so copy it next to this script (cpp/solver/) or point --solver-src at it.
SOLVER_SRC="$HERE/solver/cloth_solver.cpp"
while [ $# -gt 0 ]; do
    case "$1" in
        --solver-src) SOLVER_SRC="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -f "$SOLVER_SRC" ]; then
    build "$SOLVER_SRC" "mc_cloth_solver"
else
    echo
    echo "NOTE: no cloth solver source at $SOLVER_SRC"
    echo "      Copy cloth_solver.cpp there (or pass --solver-src <path>) and"
    echo "      run this again.  Without it the addon installs but cannot"
    echo "      simulate, and tools/build_addon.py will refuse to package it."
fi

echo
echo "done.  Contents of $OUT:"
ls -l "$OUT"
