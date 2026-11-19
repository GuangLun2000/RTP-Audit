"""Development and confirmation phases.

The instrument (items, readout, estimators, thresholds) is developed on the models listed under ``models``
in the run config. Confirmation models stay unmeasured until the instrument is frozen: the project's own
set (``rtp.registry.CONFIRMATION_MODELS``, valid under every config) plus whatever a config lists under
``confirm_models``. The freeze writes ``runs/<config>/FREEZE.json``. Before that file exists, EVERY command
that loads a model refuses a confirmation model or a variant of one (the ladder and open-set stages, and
the legacy and frozen stages too), and the ladder family refuses the sealed test split. A result on those
models or categories can then never have guided a change of the instrument.
"""

from __future__ import annotations

from rtp import registry
from rtp.config import Config
from rtp.paths import run_dir

FREEZE_FILE = "FREEZE.json"


class PhaseError(RuntimeError):
    pass


def is_frozen(cfg: Config) -> bool:
    return (run_dir(cfg.name) / FREEZE_FILE).exists()


def is_confirm_model(cfg: Config, model_id: str) -> bool:
    base = registry.base_model_id(model_id)
    return (model_id in cfg.confirm_models or base in cfg.confirm_models or base in registry.CONFIRMATION_MODELS)


def check_model_allowed(cfg: Config, model_id: str) -> None:
    """Raise unless ``model_id`` may be measured now (a development model, or any model after the freeze)."""
    if is_confirm_model(cfg, model_id) and not is_frozen(cfg):
        raise PhaseError(
            f"'{model_id}' is a confirmation model and the instrument of config '{cfg.name}' is not frozen yet "
            f"(no {FREEZE_FILE} in the run directory). No stage loads a confirmation model before the freeze; "
            f"develop on: {cfg.models}.")


def check_test_split_allowed(cfg: Config, split: str) -> None:
    """The sealed test categories are part of the confirmation phase: with ``--unseal`` alone they stay
    closed to the ladder family until the instrument is frozen."""
    if split == "test" and not is_frozen(cfg):
        raise PhaseError(f"the test split of config '{cfg.name}' is opened only after the instrument is frozen "
                         f"(no {FREEZE_FILE} in the run directory); develop on the dev split")
