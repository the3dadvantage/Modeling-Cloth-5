"""Collision benchmark runner.

    blender -b --factory-startup --python tests/run_bench.py -- <command> [scenes] [--tol X]

commands
    list          the scenes and what each one exercises
    capture       run scenes and store them as baselines (refuses any scene
                  whose collision did not engage)
    check         run scenes and compare with the baselines; exit code 1 on
                  any mismatch
    determinism   run every scene twice in this process; they must be
                  bit-identical
    time          run scenes and report timings only

With no scene names every scene is used.  --tol is the allowed per-vertex
distance for `check` (default 0: bit-identical).  Capture in one Blender
process and check in another to prove results survive a restart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collide_bench as B    # noqa: E402


def out(msg=""):
    print(msg, flush=True)


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    tol = 0.0
    if "--tol" in argv:
        i = argv.index("--tol")
        tol = float(argv[i + 1])
        del argv[i:i + 2]
    if "--backend" in argv:
        i = argv.index("--backend")
        B.BACKEND = argv[i + 1].upper()
        del argv[i:i + 2]
    overrides = {}
    while "--set" in argv:
        i = argv.index("--set")
        k, v = argv[i + 1].split("=", 1)
        try:
            overrides[k] = eval(v, {})          # 1, 2.5, True, (0, 0, 1)
        except Exception:
            overrides[k] = v
        del argv[i:i + 2]
    cmd = argv[0] if argv else "list"
    names = argv[1:] or list(B.SCENES)
    bad = [n for n in names if n not in B.SCENES]
    if bad:
        out("unknown scene(s): %s" % ", ".join(bad))
        sys.exit(2)
    if overrides and cmd == "capture":
        out("--set is for comparing variants; baselines are captured as pinned")
        sys.exit(2)
    if cmd == "capture" and B.BACKEND != "PYTHON":
        out("baselines are captured from the Python reference only")
        sys.exit(2)
    return cmd, names, tol, overrides


def main():
    cmd, names, tol, overrides = parse()
    if cmd == "list":
        for s in B.SCENES.values():
            out("%-20s %3d frames  %s" % (s.name, s.frames, s.doc))
        return 0

    B.load()
    meta = B.fingerprint()
    out("BENCH code %s  blender %s  numpy %s  dll %s  backend %s%s"
        % (meta["code"], meta["blender"], meta["numpy"], meta["dll"], meta["backend"],
           ("  overrides %s" % overrides) if overrides else ""))
    failures = 0

    for name in names:
        res = B.run_scene(name, overrides)
        tag = "engaged" if res["engaged"] else "NOT ENGAGED"
        head = "%-20s %6.2fs  %-11s %s" % (name, res["time_s"], tag,
                                           res["engaged_msg"])
        if cmd == "capture":
            try:
                B.save_baseline(name, res)
                out("CAPTURED " + head)
            except RuntimeError as e:
                failures += 1
                out("REFUSED  " + head + "  -- " + str(e))

        elif cmd == "check":
            c = B.compare(name, res, tol)
            if "max_dev" not in c:
                failures += 1
                out("FAIL     %-20s %s" % (name, c["why"]))
                continue
            state = "PASS    " if c["ok"] else "FAIL    "
            failures += not c["ok"]
            out("%s %s" % (state, head))
            out("         deviation max %.3g mean %.3g   time %.2fs (baseline %.2fs)%s"
                % (c["max_dev"], c["mean_dev"], c["time"], c["base_time"],
                   "" if c["same_code"] else "   [code changed since baseline]"))
            if not c["ok"] or c["max_dev"] > 0.0:
                for m, (b, r) in c["metrics"].items():
                    out("           %-18s baseline %.5f  now %.5f" % (m, b, r))

        elif cmd == "determinism":
            again = B.run_scene(name, overrides)
            d = abs(again["co"] - res["co"]).max()
            ok = d == 0.0 and res["engaged"]
            failures += not ok
            out("%s %s  second run differs by %.3g"
                % ("PASS    " if ok else "FAIL    ", head, d))

        elif cmd == "time":
            out("TIME     " + head)

        else:
            out("unknown command %r" % cmd)
            return 2

    out("BENCH %s: %s" % (cmd, "ALL PASS" if not failures
                           else "%d FAILED" % failures))
    return 1 if failures else 0


code = 1
try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
sys.stdout.flush()
os._exit(code)
