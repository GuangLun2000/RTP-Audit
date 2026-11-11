"""YAML run configuration: loading, defaults and validation.

Every run is driven by one YAML file. Unknown keys are rejected so that typos do not
silently fall back to defaults. ``Config.data_hash`` covers only what determines the
generated data (seed + data section), so measurement settings never invalidate data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_LEVELS = (0, 1, 2, 3, 4)
HANDICAPS = (0, 1, 2, 3, 4)
HANDICAP_MIN_EVIDENCE = 3  # spec 7.1: handicap only applies when ratings are shown (evidence >= 3)
OPEN_ENDED_PROMPT_TYPES = ("name", "choose", "recommend", "pick")


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    categories_file: str = "data/seeds/categories.yaml"
    placebo_file: str = "data/seeds/placebo_names.yaml"
    natural_negatives_file: str = "data/seeds/natural_negatives.csv"
    categories: list[str] | None = None  # None -> every category in categories_file
    n_real_per_category: int = 8
    n_placebo_per_category: int = 6
    evidence_levels: list[int] = field(default_factory=lambda: list(EVIDENCE_LEVELS))
    handicaps: list[int] = field(default_factory=lambda: list(HANDICAPS))
    handicap_rating_deltas: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 1.0])
    cards_per_cell: int = 4
    test_categories: list[str] = field(default_factory=list)
    probe_entities_per_category: int = 30
    probe_templates: int = 10


@dataclass
class MeasureConfig:
    shard_size: int = 2000
    gen_check_fraction: float = 0.02
    gen_max_new_tokens: int = 4
    consistency_warn_below: float = 0.90
    max_batch_size: int = 64
    max_batch_tokens: int = 32768


@dataclass
class OpenEndedConfig:
    prompt_types: list[str] = field(default_factory=lambda: list(OPEN_ENDED_PROMPT_TYPES))
    n_samples: int = 20
    temperature: float = 1.0
    max_new_tokens: int = 16


@dataclass
class Config:
    name: str
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    open_ended: OpenEndedConfig = field(default_factory=OpenEndedConfig)
    models: list[str] = field(default_factory=list)
    source_path: str | None = None  # not part of the hash

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("source_path", None)
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def config_hash(self) -> str:
        return _hash(self.to_dict())

    @property
    def data_hash(self) -> str:
        return _hash({"seed": self.seed, "data": dataclasses.asdict(self.data)})

    def resolve(self, rel: str) -> Path:
        """Resolve a seed-file path: absolute paths as-is, relative ones against the repo root."""
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def _hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _build(cls, raw: dict[str, Any] | None, where: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{where}': {unknown}; allowed: {sorted(known)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any], source_path: str | None = None) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    allowed = {"name", "seed", "data", "measure", "open_ended", "models"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {unknown}; allowed: {sorted(allowed)}")
    if "name" not in raw:
        raise ConfigError("config needs a 'name'")
    cfg = Config(
        name=raw["name"],
        seed=raw.get("seed", 0),
        data=_build(DataConfig, raw.get("data"), "data"),
        measure=_build(MeasureConfig, raw.get("measure"), "measure"),
        open_ended=_build(OpenEndedConfig, raw.get("open_ended"), "open_ended"),
        models=list(raw.get("models") or []),
        source_path=source_path,
    )
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        alt = REPO_ROOT / path
        if alt.exists():
            path = alt
        else:
            raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw, source_path=str(path))


def validate(cfg: Config) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    need(isinstance(cfg.name, str) and cfg.name.strip() != "" and "/" not in cfg.name,
         "name must be a non-empty string without '/'")
    need(isinstance(cfg.seed, int), "seed must be an integer")
    d = cfg.data
    need(d.n_real_per_category >= 3, "n_real_per_category must be >= 3 (target + 2 real competitors)")
    need(d.n_placebo_per_category >= 2, "n_placebo_per_category must be >= 2 (target substitute + distractor)")
    need(len(d.evidence_levels) > 0 and set(d.evidence_levels) <= set(EVIDENCE_LEVELS),
         f"evidence_levels must be a non-empty subset of {list(EVIDENCE_LEVELS)}")
    need(len(set(d.evidence_levels)) == len(d.evidence_levels), "evidence_levels has duplicates")
    need(len(d.handicaps) > 0 and set(d.handicaps) <= set(HANDICAPS),
         f"handicaps must be a non-empty subset of {list(HANDICAPS)}")
    need(0 in d.handicaps, "handicaps must contain 0 (used for evidence levels < 3)")
    need(len(set(d.handicaps)) == len(d.handicaps), "handicaps has duplicates")
    need(len(d.handicap_rating_deltas) == len(HANDICAPS), "handicap_rating_deltas needs one value per handicap 0-4")
    need(all(x >= 0 for x in d.handicap_rating_deltas) and d.handicap_rating_deltas[0] == 0,
         "handicap_rating_deltas must be >= 0 and start at 0")
    need(d.cards_per_cell >= 1, "cards_per_cell must be >= 1")
    need(d.probe_entities_per_category >= 1 and d.probe_templates >= 1, "probe sizes must be >= 1")
    if d.categories is not None:
        need(len(d.categories) == len(set(d.categories)) and len(d.categories) > 0,
             "data.categories must be non-empty and unique")
        missing = sorted(set(d.test_categories) - set(d.categories))
        need(not missing, f"test_categories not in data.categories: {missing}")
    m = cfg.measure
    need(m.shard_size >= 1, "shard_size must be >= 1")
    need(0.0 <= m.gen_check_fraction <= 1.0, "gen_check_fraction must be in [0, 1]")
    need(m.max_batch_size >= 1 and m.max_batch_tokens >= 1, "batch limits must be >= 1")
    o = cfg.open_ended
    need(set(o.prompt_types) <= set(OPEN_ENDED_PROMPT_TYPES) and len(o.prompt_types) > 0,
         f"open_ended.prompt_types must be a non-empty subset of {list(OPEN_ENDED_PROMPT_TYPES)}")
    need(o.n_samples >= 1 and o.max_new_tokens >= 1 and o.temperature > 0, "bad open_ended sampling settings")
    need(all(isinstance(x, str) and x for x in cfg.models), "models must be a list of non-empty strings")
