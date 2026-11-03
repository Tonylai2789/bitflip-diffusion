"""Compare two baseline runs bit-for-bit (the document's determinism check).

Usage:
    python src/compare_runs.py logs/seed0_a logs/seed0_b

Exit code 0 = settings match and outputs are bitwise identical; 1 = otherwise.
Prints a JSON report; redirect it to a file to keep it with the runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ra, rb = (json.loads((d / "run.json").read_text()) for d in (a, b))

    settings_diff = {k: [ra["settings"].get(k), rb["settings"].get(k)]
                     for k in ra["settings"].keys() | rb["settings"].keys()
                     if ra["settings"].get(k) != rb["settings"].get(k)}
    env_diff = {k: [ra["environment"].get(k), rb["environment"].get(k)]
                for k in ra["environment"].keys() | rb["environment"].keys()
                if ra["environment"].get(k) != rb["environment"].get(k)}

    sa, sb = np.load(a / "samples.npy"), np.load(b / "samples.npy")
    same_shape = sa.shape == sb.shape
    bitwise = same_shape and sa.tobytes() == sb.tobytes()
    report = {
        "run_a": str(a), "run_b": str(b),
        "settings_identical": not settings_diff, "settings_diff": settings_diff,
        "environment_diff": env_diff,
        "samples_bitwise_identical": bitwise,
        "sha256": [ra["results"]["samples_sha256"], rb["results"]["samples_sha256"]],
    }
    if same_shape:
        diff = np.abs(sa.astype(np.float64) - sb.astype(np.float64))
        report["num_differing_values"] = int((sa.view(np.uint32) != sb.view(np.uint32)).sum())
        report["max_abs_diff"] = float(diff.max())
        pngs = sorted((a / "images").glob("*.png"))
        report["png_pixel_mismatches"] = [p.name for p in pngs
                                          if not np.array_equal(np.asarray(Image.open(p)),
                                                                np.asarray(Image.open(b / "images" / p.name)))]

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["settings_identical"] and bitwise else 1)


if __name__ == "__main__":
    main()
