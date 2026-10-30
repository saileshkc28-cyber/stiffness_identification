"""Phase 1 quality check: flag corrupted samples in phase1_samples.csv.

    python flag_corrupted_phase1.py                       (phase1_distribution_runs)
    python flag_corrupted_phase1.py some_other_run_folder

With one global E and a fixed load, u = u_ref / alpha exactly, so u_true * alpha_true
must be the same number for every sample. A sample that breaks this was not solved
correctly (wrong linear solve), even if Kratos reported no error.

Such rows get status "failed: homogeneity check ..." so collapse_phase1.py (which only
keeps status "ok") leaves them out. Nothing is deleted: the original file is first
copied to phase1_samples_before_qa.csv, and the numbers in every row stay untouched.

This belongs to Phase 1 (it uses alpha_true, which Phase 1 knows). Phase 2 still sees
only the collapsed displacement statistics.
"""
import csv
import os
import shutil
import sys

import numpy as np

TOLERANCE = 1e-6   # relative deviation of u*alpha from the median; clean samples are ~1e-11

root = sys.argv[1] if len(sys.argv) > 1 else "phase1_distribution_runs"
path = os.path.join(root, "phase1_samples.csv")
backup = os.path.join(root, "phase1_samples_before_qa.csv")

with open(path, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

i_alpha = header.index("alpha_true")
i_status = header.index("status")
u_cols = [i for i, h in enumerate(header) if h.startswith("u_true_")]

ok = [r for r in rows if r[i_status].strip() == "ok"]
products = np.array([[float(r[i]) * float(r[i_alpha]) for i in u_cols] for r in ok])
median = np.median(products, axis=0)

flagged = []
for r in ok:
    p = np.array([float(r[i]) * float(r[i_alpha]) for i in u_cols])
    dev = float(np.max(np.abs(p / median - 1.0)))
    if not np.isfinite(dev) or dev > TOLERANCE:
        r[i_status] = f"failed: homogeneity check (u*alpha off by {dev:.2e})"
        flagged.append((r[0], dev))

print(f"{len(ok)} samples marked ok, {len(flagged)} fail the u*alpha check:")
for sid, dev in flagged:
    print(f"  sample {sid}: relative deviation {dev:.3e}")

if not flagged:
    print("nothing to change")
    sys.exit(0)

if not os.path.exists(backup):
    shutil.copyfile(path, backup)
    print(f"original saved as {backup}")


def quote(text):
    return '"' + text.replace('"', '""') + '"'


# same layout as the original: numbers as written, status in quotes
with open(path, "w", newline="") as f:
    f.write(",".join(header) + "\n")
    for r in rows:
        f.write(",".join(r[:i_status] + [quote(r[i_status])] + r[i_status + 1:]) + "\n")

print(f"updated {path}: {len(ok) - len(flagged)} samples remain ok")
print("now run:  python collapse_phase1.py")
