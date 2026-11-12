"""Schema checks for generated records (no external schema library).

Beyond field types, the checks enforce the design invariants of spec §7.1:
target placed at its rotation, brand name first on each option line, and pairs whose two
versions differ only in the target name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rtp.config import HANDICAP_MIN_EVIDENCE


class SchemaError(ValueError):
    pass


_NUM = (int, float)
CATALOGUE_FIELDS = {"category": str, "entity": str, "is_placebo": bool, "tier": (str, type(None)), "aliases": list,
                    "split": str, "natural_negative": (str, type(None)), "match_case": bool}
CHOICE_FIELDS = {"item_id": str, "pair_id": str, "card_set": str, "split": str, "category": str, "evidence_level": int,
                 "handicap": int, "target_entity": str, "target_is_placebo": bool, "rotation": int,
                 "target_label": str, "pair_entity": str, "placebo_substitute": str, "options": list, "prompt": str}
OPTION_FIELDS = {"label": str, "name": str, "role": str, "is_placebo": bool, "description": (str, type(None)),
                 "price": (*_NUM, type(None)), "features": (list, type(None)), "rating": (*_NUM, type(None)),
                 "n_reviews": (int, type(None)), "lab_note": (str, type(None))}
PROBE_FIELDS = {"probe_id": str, "split": str, "category": str, "entity": str, "attributes": dict,
                "template_id": int, "frame": str, "control": (str, type(None)), "other_entity": (str, type(None)),
                "other_attributes": (dict, type(None)), "other_first": (bool, type(None)), "text": str,
                "name_char_span": list}
ROLE_COUNTS = {"target": 1, "competitor": 2, "placebo_distractor": 1}


def _check_fields(rec: dict[str, Any], spec: dict[str, Any], where: str) -> None:
    missing = sorted(set(spec) - set(rec))
    extra = sorted(set(rec) - set(spec))
    if missing or extra:
        raise SchemaError(f"{where}: missing fields {missing}, unexpected fields {extra}")
    for k, t in spec.items():
        v = rec[k]
        if isinstance(v, bool) and t in (int, _NUM) and t is not bool:
            raise SchemaError(f"{where}: field '{k}' must be {t}, got bool")
        if not isinstance(v, t):
            raise SchemaError(f"{where}: field '{k}' has type {type(v).__name__}, expected {t}")


def validate_catalogue(records: Iterable[dict[str, Any]]) -> int:
    n = 0
    seen = set()
    for r in records:
        _check_fields(r, CATALOGUE_FIELDS, f"catalogue[{n}]")
        key = (r["category"], r["entity"])
        if key in seen:
            raise SchemaError(f"catalogue: duplicate entity {key}")
        seen.add(key)
        if r["split"] not in ("dev", "test"):
            raise SchemaError(f"catalogue: bad split {r['split']}")
        n += 1
    return n


def validate_choice_items(items: list[dict[str, Any]]) -> int:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        where = f"choice item {it.get('item_id')}"
        _check_fields(it, CHOICE_FIELDS, where)
        e, h = it["evidence_level"], it["handicap"]
        if not 0 <= e <= 4 or not 0 <= h <= 4:
            raise SchemaError(f"{where}: evidence/handicap out of range")
        if e < HANDICAP_MIN_EVIDENCE and h != 0:
            raise SchemaError(f"{where}: handicap {h} at evidence {e} < {HANDICAP_MIN_EVIDENCE}")
        opts = it["options"]
        if [o.get("label") for o in opts] != ["A", "B", "C", "D"]:
            raise SchemaError(f"{where}: options must be labelled A-D in order")
        for o in opts:
            _check_fields(o, OPTION_FIELDS, f"{where} option {o.get('label')}")
        roles = defaultdict(int)
        for o in opts:
            roles[o["role"]] += 1
        if dict(roles) != ROLE_COUNTS:
            raise SchemaError(f"{where}: role counts {dict(roles)} != {ROLE_COUNTS}")
        tgt = opts[it["rotation"]]
        if tgt["role"] != "target" or it["target_label"] != tgt["label"] or tgt["name"] != it["target_entity"]:
            raise SchemaError(f"{where}: target not at rotation {it['rotation']}")
        if tgt["is_placebo"] != it["target_is_placebo"]:
            raise SchemaError(f"{where}: target_is_placebo inconsistent")
        expected_name = it["placebo_substitute"] if it["target_is_placebo"] else it["pair_entity"]
        if it["target_entity"] != expected_name:
            raise SchemaError(f"{where}: target name does not match its condition")
        if not it["item_id"].startswith(it["pair_id"] + "-"):
            raise SchemaError(f"{where}: item_id must extend pair_id")
        for o in opts:  # brand name first on the option line
            if f"\n{o['label']}. {o['name']}" not in "\n" + it["prompt"].split("\n\n", 1)[1]:
                raise SchemaError(f"{where}: option {o['label']} line does not start with the brand name")
        pairs[it["pair_id"]].append(it)
    for pid, group in pairs.items():
        _check_pair(pid, group)
    _check_card_sets(items)
    return len(items)


def _check_card_sets(items: list[dict[str, Any]]) -> None:
    """Matched cards: within a card set, every cell shows the same options in the same order (names, roles,
    non-target cards); cells differ only in visible evidence and the target's handicapped rating."""
    ref: dict[tuple[str, bool, int], tuple[Any, ...]] = {}
    for it in items:
        key = (it["card_set"], it["target_is_placebo"], it["rotation"])
        sig = (it["pair_entity"], it["placebo_substitute"], tuple((o["name"], o["role"]) for o in it["options"]))
        if ref.setdefault(key, sig) != sig:
            raise SchemaError(f"card set {it['card_set']}: options differ between cells (item {it['item_id']})")


def _check_pair(pair_id: str, group: list[dict[str, Any]]) -> None:
    keys = sorted((g["target_is_placebo"], g["rotation"]) for g in group)
    if keys != [(False, r) for r in range(4)] + [(True, r) for r in range(4)]:
        raise SchemaError(f"pair {pair_id}: needs real+placebo x 4 rotations, got {keys}")
    shared = ("card_set", "split", "category", "evidence_level", "handicap", "pair_entity", "placebo_substitute")
    for k in shared:
        if len({g[k] for g in group}) != 1:
            raise SchemaError(f"pair {pair_id}: field '{k}' differs within the pair")
    by_rot: dict[int, dict[bool, dict[str, Any]]] = defaultdict(dict)
    for g in group:
        by_rot[g["rotation"]][g["target_is_placebo"]] = g
    for rot, versions in by_rot.items():
        real, plac = versions[False], versions[True]
        for a, b in zip(real["options"], plac["options"]):
            if a["role"] == "target":
                strip = lambda o: {k: v for k, v in o.items() if k not in ("name", "is_placebo")}  # noqa: E731
                if strip(a) != strip(b):
                    raise SchemaError(f"pair {pair_id} rot {rot}: target card differs between versions")
            elif a != b:
                raise SchemaError(f"pair {pair_id} rot {rot}: non-target option {a['label']} differs between versions")
        # the prompts may differ only at the target name
        rn, pn = real["target_entity"], plac["target_entity"]
        if real["prompt"].replace(f"\n{real['target_label']}. {rn}", f"\n{real['target_label']}. {pn}") != plac["prompt"]:
            raise SchemaError(f"pair {pair_id} rot {rot}: prompts differ beyond the target name")


def validate_probe_items(items: list[dict[str, Any]]) -> int:
    from rtp.data.probe_items import ATTRIBUTES, N_LEVELS

    by_entity: dict[str, dict[str, int]] = {}
    for it in items:
        where = f"probe {it.get('probe_id')}"
        _check_fields(it, PROBE_FIELDS, where)
        attrs = it["attributes"]
        if set(attrs) != set(ATTRIBUTES):
            raise SchemaError(f"{where}: attributes {sorted(attrs)} != {sorted(ATTRIBUTES)}")
        for a, v in attrs.items():
            if not (isinstance(v, int) and 0 <= v < N_LEVELS[a]):
                raise SchemaError(f"{where}: attribute {a}={v} out of range")
        s, t = it["name_char_span"]
        if it["text"][s:t] != it["entity"] or it["text"].rfind(it["entity"]) != s:
            raise SchemaError(f"{where}: name_char_span does not point at the last mention of the entity")
        if it["frame"] not in ("background", "choice") or it["control"] not in (None, "two_brand", "undescribed"):
            raise SchemaError(f"{where}: bad frame/control {it['frame']}/{it['control']}")
        if (it["control"] is None) != (it["other_entity"] is None):
            raise SchemaError(f"{where}: other_entity must be set exactly for control items")
        if it["control"] == "undescribed" and it["text"].count(it["entity"]) != 1:
            raise SchemaError(f"{where}: an undescribed read entity must appear exactly once")
        for ent, a in ((it["entity"], attrs), (it["other_entity"], it["other_attributes"])):
            if ent is None:
                continue
            prev = by_entity.setdefault((it["category"], ent), a)
            if prev != a:
                raise SchemaError(f"{where}: entity {ent} has inconsistent attributes")
    return len(items)
