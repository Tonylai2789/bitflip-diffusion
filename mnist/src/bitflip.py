"""Single-bit faults in stored FP16 weights: flip, restore, fingerprint.

An element is addressed by (parameter name, flat index in storage order, from 0).
flip and restore write through an int16 view of the live parameter, so the change
lands in the same tensor inference uses, on whatever device it lives on.

FP16 layout: bit 15 sign, bits 14-10 exponent, bits 9-0 mantissa (bit 0 = rightmost).
"""
import hashlib

import torch


def field(bit):
    """FP16 field of a bit position: 'sign', 'exponent' or 'mantissa'."""
    if not 0 <= bit <= 15:
        raise ValueError(f"bit must be 0..15, got {bit}")
    return "sign" if bit == 15 else "exponent" if bit >= 10 else "mantissa"


def _bits(model, name):
    """int16 view sharing storage with the named FP16 parameter (never a copy)."""
    p = dict(model.named_parameters())[name]
    assert p.dtype == torch.float16, f"{name} is {p.dtype}, expected torch.float16"
    assert p.is_contiguous(), f"{name} is not contiguous"
    return p.data.view(-1).view(torch.int16)


@torch.no_grad()
def flip(model, name, index, bit):
    """Invert one bit of one element in place. Returns the original 16-bit pattern."""
    field(bit)  # validates the bit position
    bits = _bits(model, name)
    pattern = int(bits[index])
    mask = -32768 if bit == 15 else 1 << bit  # 0x8000 = 32768 doesn't fit in int16
    bits[index] = pattern ^ mask
    return pattern


@torch.no_grad()
def restore(model, name, index, pattern):
    """Write a saved pattern back. Never restore by flipping again."""
    _bits(model, name)[index] = pattern


def fingerprint(tensor):
    """SHA-256 of the tensor's raw bytes (hashed from a CPU copy)."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
