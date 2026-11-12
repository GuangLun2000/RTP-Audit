"""Sealed dev/test split access (hard constraint 4).

Data are split by category. Any read of the ``test`` split must pass ``unseal=True``
(``--unseal`` on the CLI); otherwise ``SealedSplitError`` is raised.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SPLITS = ("dev", "test")


class SealedSplitError(PermissionError):
    pass


def check_split_access(split: str, unseal: bool) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}', expected one of {SPLITS}")
    if split == "test" and not unseal:
        raise SealedSplitError(
            "the 'test' split is sealed: method development uses 'dev' only. Pass --unseal "
            "(unseal=True) to read it deliberately."
        )
    if split == "test":
        log.warning("UNSEALED: reading the test split")


def split_of(category: str, test_categories: list[str] | tuple[str, ...]) -> str:
    return "test" if category in set(test_categories) else "dev"
