"""Hand tests for src/bitflip.py, run on CPU and on the Mac GPU (MPS).

Step 0 of plan-mnist-week1.md: the MPS cases must pass, not be skipped.
Run from mnist/: pytest tests/ -v -rs  (or from the repo root: pytest mnist/tests -v -rs)
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bitflip import field, fingerprint, flip, restore  # noqa: E402

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
]


def one_weight(value, device):
    """A 1-input, 1-output FP16 linear layer whose only weight is `value`."""
    m = torch.nn.Linear(1, 1, bias=False).half().to(device)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "value, bit, expected",
    [
        (0.5, 14, 32768.0),  # doc
        (1.0, 14, math.inf),  # doc
        (0.5, 0, 0.50048828125),  # doc
        (0.5, 15, -0.5),  # added: bit 15 needs the -32768 mask
    ],
)
def test_hand_values(device, value, bit, expected):
    m = one_weight(value, device)
    flip(m, "weight", 0, bit)
    assert m.weight.item() == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bit", range(16))
def test_flip_then_restore_keeps_fingerprint(device, bit):
    m = one_weight(0.3, device)
    before = fingerprint(m.weight)
    pattern = flip(m, "weight", 0, bit)
    assert fingerprint(m.weight) != before
    restore(m, "weight", 0, pattern)
    assert fingerprint(m.weight) == before


@pytest.mark.parametrize("device", DEVICES)
def test_int16_view_shares_storage(device):
    p = one_weight(0.5, device).weight
    assert p.data.view(-1).view(torch.int16).data_ptr() == p.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_flip_changes_what_inference_sees(device):
    m = one_weight(0.5, device)
    x = torch.ones(1, 1, dtype=torch.float16, device=device)
    assert m(x).item() == 0.5
    flip(m, "weight", 0, 14)
    assert m(x).item() == 32768.0


@pytest.mark.parametrize("device", DEVICES)
def test_fp16_matmul_runs(device):
    # Fixed, non-random data: the study allows no seeds outside the master-seed role table.
    a = torch.linspace(-1, 1, 64 * 784).reshape(64, 784)
    b = torch.linspace(-1, 1, 784 * 128).reshape(784, 128)
    out = a.half().to(device) @ b.half().to(device)
    assert out.dtype == torch.float16 and out.device.type == device
    assert torch.allclose(out.float().cpu(), a @ b, rtol=0.02, atol=0.5)


def test_field():
    assert [field(b) for b in (15, 14, 10, 9, 0)] == ["sign", "exponent", "exponent", "mantissa", "mantissa"]
    with pytest.raises(ValueError):
        field(16)
