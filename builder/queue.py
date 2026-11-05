"""Run builder jobs one at a time, paced to the plan's weekly budget.

    .venv/bin/python -m builder.queue jobs.txt --ceiling 99 --pool claude

Each line of jobs.txt is the arguments to builder.run_builder
("--scenario basic --mode pixels --budget 3 --tag opus"). Before each job it reads
`~/.agents/bin/usage`; it starts the job only while the pool's weekly use is under
the ceiling, and otherwise waits (checking every 5 minutes) until the pool resets or
the deadline passes. A started job always finishes. Done lines are recorded in
jobs.txt.done so a restart picks up where it stopped.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

USAGE = Path.home() / ".agents" / "bin" / "usage"
REPO = Path(__file__).resolve().parent.parent


def weekly_used(pool):
    """Percent of the pool's weekly limit used, or None if it cannot be read."""
    out = subprocess.run([str(USAGE)], capture_output=True, text=True).stdout
    if pool == "fable":
        m = re.search(r"Fable\s+([\d.]+)% used", out)
    else:
        m = re.search(r"^claude.*?\n\s+weekly\s+([\d.]+)% used", out, re.S | re.M)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--ceiling", type=float, default=99.0)
    ap.add_argument("--pool", default="claude", choices=["claude", "fable"])
    ap.add_argument("--hours", type=float, default=3.0, help="stop waiting after this long")
    a = ap.parse_args()
    jobs = [l.strip() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    done_file = Path(a.jobs + ".done")
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
    deadline = time.time() + a.hours * 3600
    for job in jobs:
        if job in done:
            continue
        while True:
            used = weekly_used(a.pool)
            if used is not None and used < a.ceiling:
                break
            if time.time() > deadline:
                print(f"deadline reached at {used}% used; stopping", flush=True)
                return 0
            print(f"{a.pool} at {used}% (ceiling {a.ceiling}); waiting", flush=True)
            time.sleep(300)
        print(f"[{time.strftime('%H:%M')}] {a.pool} {used}% -> {job}", flush=True)
        p = subprocess.run([str(REPO / ".venv" / "bin" / "python"), "-m", "builder.run_builder", *job.split()],
                           cwd=REPO)
        if p.returncode != 0:  # the builder never started (a model that will not load): stop, keep the job
            print(f"builder exited {p.returncode} before running; stopping", flush=True)
            return 1
        with done_file.open("a") as f:
            f.write(job + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
