"""Checks for the ResNet-20 model, sampling groups and augmentation in src/resnet20_fault.py (plan section 2)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resnet20_fault import augment, build_model, groups, locate  # noqa: E402


@pytest.mark.parametrize("channels, size, stem, weights", [(1, 28, 144, 268_048), (3, 32, 432, 268_336)])
def test_counts_and_shapes(channels, size, stem, weights):
    m = build_model(channels).eval()  # also checks total weights and parameters
    assert m.conv1.weight.numel() == stem
    g = groups(m)
    sizes = {k: sum(m.get_parameter(n).numel() for n in v) for k, v in g.items()}
    assert sizes == {"stage1": stem + 13_824, "stage2": 50_688, "stage3": 202_752, "fc": 640}
    assert sum(sizes.values()) == weights
    assert sum(len(v) for v in g.values()) == 20  # 19 convolutions + 1 linear
    assert all(n.endswith(".weight") and "bn" not in n for v in g.values() for n in v)
    x = torch.zeros(2, channels, size, size)
    assert m(x).shape == (2, 10)
    # spatial sizes after each stage: size, size/2, size/4
    feats = m.layer1(m.bn1(m.conv1(x)))
    assert feats.shape[1:] == (16, size, size)
    assert m.layer3(m.layer2(feats)).shape[1:] == (64, size // 4, size // 4)


def test_locate():
    m = build_model(1)
    names = groups(m)["stage1"]
    assert locate(m, names, 0) == ("conv1.weight", 0)
    assert locate(m, names, 143) == ("conv1.weight", 143)
    assert locate(m, names, 144) == ("layer1.0.conv1.weight", 0)
    with pytest.raises(IndexError):
        locate(m, names, 144 + 13_824)


def test_augment_off_is_identity_and_crop_stays_in_padded_image():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
    assert torch.equal(augment(x, 0, False), x)
    torch.manual_seed(0)
    y = augment(x, 4, True)
    assert y.shape == x.shape
    # every output value is either 0 (padding) or a value from the same image and channel
    for i in range(2):
        for c in range(3):
            vals = set(x[i, c].flatten().tolist()) | {0.0}
            assert set(y[i, c].flatten().tolist()) <= vals
