"""What a worker actually used, so a declaration can be checked against it.

**The one input to planning with no receipt.** Every other claim this project
makes is verified by reading an artifact: a perturbation records the values
either side of it, a comparison records what fired, a physics surface records
the digest of what was loaded. Resource declarations were trusted.

The cost of trusting them is on record. ``vram_per_env_mb: 0`` -- a comment
asserting "CPU-only; the model servers are the GPU consumers" against a worker
whose EGL context takes 727 MiB -- did not make the planner's packing slightly
wrong. It removed the constraint: with zero declared, any number of workers
fits any GPU and the planner will never refuse. Six containers got contexts,
four did not, and EGL busy-waits rather than failing, so four workers spun at
100% CPU for fifteen hours with no error, no exit, and no log line.

A wrong number produces a wrong plan, which someone notices. A zero produces no
plan constraint at all, which nobody does.

Recording the observed figures makes the declaration checkable the same way
everything else here is checkable -- and the correction machinery already
exists. ``--expect-from`` already re-fits duration from a prior run because
duration is recorded; nothing can re-fit memory because memory was not.

What is measurable, and what is not
-----------------------------------

**RSS: everywhere.** ``getrusage`` reports the high-water mark of this process,
which needs no sampling and cannot be missed between polls.

**VRAM: only when the platform will attribute it.** There is no per-process GPU
accounting API available here -- no pynvml, and torch is a CPU build -- so this
falls back to ``nvidia-smi``, which reports per-PID usage. That works for a
worker running on the host and does NOT work inside a container, whose PID
lives in a different namespace and never appears in the list.

So the VRAM figure is null for containerised workers, and **null means "could
not attribute", not "used none"**. Conflating those two is the exact mistake
that started this: a zero that meant "nobody measured" was read by the planner
as "costs nothing".

That still catches the failure it needs to. The local backend runs on the host
and is what anyone reaches for first, so a false declaration shows up there
before it can waste a night in containers.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess


def peak_rss_mb() -> int:
    """This process's high-water resident set, in MiB.

    A high-water mark rather than a sample: the kernel keeps it, so a peak
    between two polls cannot be missed. ``ru_maxrss`` is KiB on Linux and bytes
    on macOS; only Linux is supported here and the unit is asserted by a test
    rather than assumed from memory.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def current_vram_mb(pid: int | None = None) -> int | None:
    """GPU memory attributed to ``pid`` right now, or None if it cannot be.

    None is a real answer and not a failure: a containerised worker's PID is
    not the PID the driver sees, so it will never appear. Returning 0 there
    would manufacture the very claim -- "this worker uses no VRAM" -- that this
    module exists to stop anyone making by accident.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


class PeakWatcher:
    """Running maxima, sampled where the caller already has a natural moment.

    Deliberately not a background thread. A sampler racing the episode loop
    would contend for the one core the worker is pinned to, and the quantity
    being measured is the cost of that pinning -- an observer that changes the
    observation is worse than a coarse observation.

    Episode boundaries are the natural moment: an EGL context is allocated at
    the first reset and flat thereafter, which was measured rather than assumed
    (727 MiB after reset, 727 MiB after five steps).
    """

    def __init__(self) -> None:
        self.rss_mb = 0
        #: None until something is attributable. Stays None for a container,
        #: and that absence is reported rather than turned into a zero.
        self.vram_mb: int | None = None

    def sample(self) -> None:
        self.rss_mb = max(self.rss_mb, peak_rss_mb())
        now = current_vram_mb()
        if now is not None:
            self.vram_mb = now if self.vram_mb is None else max(self.vram_mb, now)


__all__ = ["PeakWatcher", "current_vram_mb", "peak_rss_mb"]
