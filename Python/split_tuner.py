"""How finely the collision broad phase splits space.

Both collision passes find candidate pairs by recursively halving space until a
box holds few enough pairs to brute-force.  That threshold used to be an
absolute pair count (sc_box_count, ob_box_count).  Measured across mesh sizes,
the fastest threshold is not a fixed count at all -- it is a fixed number of
SPLITS.  So the threshold is now expressed as a depth:

    box_threshold = root_pairs / 2**depth

Measurements (collision time vs depth, crumpled cloth):

    self collision   441..3721 verts  best depth 4 at every size
                                      depths 3-7 all within ~5% of best
    object collision 441..8281 verts  best depth 5, 6, 8, 8
                                      depths 4-9 all within ~10% of best

A fixed depth of 5 was within 12% of the best at all eight cases.  The old
absolute defaults were up to 4.9x slower (object collision's 37), and self
collision's 73737 drifted toward the slow end as meshes grew, because an
absolute count means a deeper and deeper tree on a bigger mesh.

Changing the depth does NOT change the physics: positions came out
bit-identical across every depth tested.  It only moves time around, which is
what makes tuning it on the fly safe.

The tuner
---------
The curve has a wide flat bottom, so a tuner that reacts to single-frame
timings just wanders inside it chasing noise.  This one:

  * searches depth, not count -- a step of 1 is a 2x jump in box size, and the
    whole space is about a dozen integers
  * averages `samples` timings before judging any depth
  * only moves when a neighbour beats home by more than `threshold`
    (hysteresis), which is what stops it wandering
  * settles, then re-checks every `settle_calls` calls in case the cloth has
    changed shape enough to move the optimum

It keeps its state on the cloth, never in a user-facing property, so it does
not make the UI value jitter or fill the undo stack every frame.
"""

import math

DEFAULT_DEPTH = 5
DEPTH_MIN = 0
DEPTH_MAX = 16          # matches SC_MAX_DEPTH / OC_MAX_DEPTH


def box_threshold(root_pairs, depth):
    """Pair count at which a box stops splitting, for a target depth."""
    depth = min(max(int(depth), DEPTH_MIN), DEPTH_MAX)
    root_pairs = max(0, int(root_pairs))
    return max(1, root_pairs >> depth)


class DepthTuner:
    """Hill-climb the split depth on averaged timings, with hysteresis."""

    def __init__(self, depth=DEFAULT_DEPTH, lo=1, hi=12, samples=6,
                 threshold=0.06, settle_calls=240):
        self.lo = int(lo)
        self.hi = int(hi)
        self.depth = min(max(int(depth), self.lo), self.hi)
        self.samples = max(1, int(samples))
        self.threshold = float(threshold)
        self.settle_calls = max(1, int(settle_calls))
        self.state = "searching"
        self.moves = 0
        self._sums = {}
        self._queue = []
        self._active = self.depth
        self._countdown = 0
        self._start_round(keep_home=False)

    # -------------------------------------------------------------- queries
    def current(self):
        """Depth to use for the next collision call."""
        return self._active

    def mean(self, depth):
        s = self._sums.get(depth)
        return s[0] / s[1] if s and s[1] else float("inf")

    # -------------------------------------------------------------- feeding
    def record(self, seconds):
        """Report how long the collision call at current() took."""
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return
        if not math.isfinite(seconds) or seconds < 0.0:
            return

        s = self._sums.setdefault(self._active, [0.0, 0])
        s[0] += seconds
        s[1] += 1

        if self.state == "settled":
            self._countdown -= 1
            if self._countdown <= 0:
                # re-check: the cloth may have changed enough to move the best
                self._start_round(keep_home=False)
            return

        if s[1] < self.samples:
            return
        if self._queue:
            self._active = self._queue.pop(0)
            return
        self._decide()

    # -------------------------------------------------------------- internals
    def _start_round(self, keep_home):
        home = self._sums.get(self.depth) if keep_home else None
        self._sums = {self.depth: home} if home else {}
        self.state = "searching"
        self._queue = [d for d in (self.depth - 1, self.depth + 1)
                       if self.lo <= d <= self.hi]
        # home is already measured when we have just moved onto it, so go
        # straight to the neighbours rather than timing it all over again
        if home and home[1] >= self.samples and self._queue:
            self._active = self._queue.pop(0)
        else:
            self._active = self.depth

    def _decide(self):
        home = self.mean(self.depth)
        nbrs = [d for d in (self.depth - 1, self.depth + 1) if d in self._sums]
        best = min(nbrs, key=self.mean) if nbrs else None
        if best is not None and self.mean(best) < home * (1.0 - self.threshold):
            self.depth = best
            self.moves += 1
            self._start_round(keep_home=True)
        else:
            self.state = "settled"
            self._active = self.depth
            self._countdown = self.settle_calls
