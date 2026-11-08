"""Check value for the step 6 Wilson interval (plan-mnist-week1.md): k = 0, n = 300 -> [0, 1.26%]."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mnist_fault import wilson  # noqa: E402


def test_wilson_zero_of_300():
    lo, hi = wilson(0, 300)
    assert lo == 0
    assert hi == pytest.approx(0.0126, abs=5e-5)
