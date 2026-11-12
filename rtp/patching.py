"""Activation patching with forward hooks (used by `localize` now and by the M4 transplant later).

Hidden-state index l (as in ``output_hidden_states``) is the residual stream after decoder block l-1, so
patching index l >= 1 overwrites the OUTPUT of block l-1 at chosen (row, position) pairs. Index 0 overwrites
the output of the input-embedding module; that equals ``hidden_states[0]`` for Llama and Qwen (models that
rescale embeddings after the module, e.g. Gemma, would differ). Positions are indices into the padded batch.
Forward passes shorter than the largest patched position (cached extension steps) are left untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from rtp.models import get_blocks

Patch = tuple[int, int, torch.Tensor]  # (batch row, padded position, replacement vector [d])


def _module_for(model: Any, layer: int, n_blocks: int):
    if layer == 0:
        return model.get_input_embeddings()
    if 1 <= layer <= n_blocks:
        return get_blocks(model)[layer - 1]
    raise ValueError(f"hidden-state index {layer} is not patchable (valid: 0..{n_blocks})")


@contextmanager
def patched(model: Any, patches: dict[int, list[Patch]] | None) -> Iterator[None]:
    """Overwrite hidden-state index ``l`` at the given (row, position) with the given vectors, for every
    forward pass inside the context whose sequence covers those positions."""
    if not patches:
        yield
        return
    n_blocks = len(get_blocks(model))
    handles = []
    try:
        for layer, plist in patches.items():
            if not plist:
                continue
            module = _module_for(model, layer, n_blocks)
            rows = torch.tensor([p[0] for p in plist], dtype=torch.long)
            pos = torch.tensor([p[1] for p in plist], dtype=torch.long)
            vecs = torch.stack([p[2] for p in plist])
            max_pos = int(pos.max())

            def hook(module, args, output, rows=rows, pos=pos, vecs=vecs, max_pos=max_pos):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[1] <= max_pos:  # a cached extension step: the patched positions are not in it
                    return output
                hs[rows.to(hs.device), pos.to(hs.device), :] = vecs.to(device=hs.device, dtype=hs.dtype)
                return output

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()
