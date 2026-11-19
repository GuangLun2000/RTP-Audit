"""Schema and design checks for ladder items (rtp.data.ladder_items).

Beyond field types the checks enforce what the estimators rely on: every unit appears once at each
option position; the two compared options differ only in the manipulated attribute (and, on full cards,
in their card text, which the swap counterbalances); each ladder step makes the disadvantaged option
strictly worse than the step before; the control is always fictional, and an anchor's target too.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rtp.data.constants import LADDER_STRATA, LADDER_TIERS, LADDER_TYPES
from rtp.data.schema import SchemaError, _check_fields

_NUM = (int, float)
LADDER_FIELDS = {"item_id": str, "unit_id": str, "family": str, "tier": str, "split": str, "category": str,
                 "stratum": str, "evidence_type": str, "d": int, "target_entity": str, "target_is_fictional": bool,
                 "control_entity": str, "base": int, "swap": int, "order": int, "target_label": str,
                 "control_label": str, "requirement": (str, type(None)), "options": list, "prompt": str}
LADDER_OPTION_FIELDS = {"label": str, "name": str, "role": str, "is_fictional": bool,
                        "description": (str, type(None)), "price": (*_NUM, type(None)),
                        "features": (list, type(None)), "rating": (*_NUM, type(None)),
                        "n_reviews": (int, type(None))}
MANIPULATED = {"rating": ("rating",), "price": ("price",), "reviews": ("n_reviews",),
               "composite": ("rating", "price", "n_reviews", "features"), "none": ()}
SHOWN = {"name": (), "full": ("description", "price", "features", "rating", "n_reviews"),
         "comp": ("description", "price", "features", "rating", "n_reviews")}
SHOWN_ATTR = {"rating": ("rating",), "price": ("price",), "reviews": ("rating", "n_reviews")}
LETTERS = ("A", "B", "C", "D")


def _worse(etype: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if option ``a`` is strictly worse than ``b`` on the attribute of ``etype``."""
    if etype == "rating":
        return a["rating"] < b["rating"]
    if etype == "price":
        return a["price"] > b["price"]
    if etype == "reviews":
        return a["n_reviews"] < b["n_reviews"]
    raise ValueError(etype)


def validate_ladder_items(items: list[dict[str, Any]], fictional: dict[str, set[str]],
                          real: dict[str, set[str]]) -> int:
    """``fictional`` / ``real``: the catalogue's names per category."""
    seen_ids: set[str] = set()
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    composite_top = max((abs(it["d"]) for it in items if it.get("evidence_type") == "composite"), default=0)
    for it in items:
        where = f"ladder item {it.get('item_id')}"
        _check_fields(it, LADDER_FIELDS, where)
        if it["item_id"] in seen_ids:
            raise SchemaError(f"{where}: duplicate item_id")
        seen_ids.add(it["item_id"])
        if it["family"] not in ("anchor", "real") or it["tier"] not in LADDER_TIERS:
            raise SchemaError(f"{where}: bad family/tier {it['family']}/{it['tier']}")
        stratum, etype, d = it["stratum"], it["evidence_type"], it["d"]
        if stratum not in LADDER_STRATA or etype not in (*LADDER_TYPES, "composite", "none"):
            raise SchemaError(f"{where}: bad stratum/evidence type {stratum}/{etype}")
        if (etype == "composite") != (stratum == "comp") or (stratum == "attr" and etype == "none"):
            raise SchemaError(f"{where}: evidence type {etype} does not belong to stratum {stratum}")
        if etype == "none" and d != 0:
            raise SchemaError(f"{where}: d = {d} without an evidence type")
        if (it["requirement"] is None) != (stratum != "comp"):
            raise SchemaError(f"{where}: a requirement is stated exactly in the 'comp' stratum")
        opts = it["options"]
        n = len(opts)
        if n != (4 if it["tier"] == "ctx" else 2) or [o.get("label") for o in opts] != list(LETTERS[:n]):
            raise SchemaError(f"{where}: {n} options / labels {[o.get('label') for o in opts]}")
        for o in opts:
            _check_fields(o, LADDER_OPTION_FIELDS, f"{where} option {o.get('label')}")
        roles = sorted(o["role"] for o in opts)
        if roles != sorted(["target", "control"] + ["context"] * (n - 2)):
            raise SchemaError(f"{where}: roles {roles}")
        by_label = {o["label"]: o for o in opts}
        tgt, ctl = by_label[it["target_label"]], by_label[it["control_label"]]
        if tgt["role"] != "target" or ctl["role"] != "control" or opts[it["order"]] is not tgt:
            raise SchemaError(f"{where}: target/control labels do not match the options")
        if tgt["name"] != it["target_entity"] or ctl["name"] != it["control_entity"]:
            raise SchemaError(f"{where}: target/control names do not match the options")
        cat = it["category"]
        if not ctl["is_fictional"] or ctl["name"] not in fictional[cat]:
            raise SchemaError(f"{where}: the control must be a fictional name of the category")
        if tgt["is_fictional"] != it["target_is_fictional"] or (it["family"] == "anchor") != tgt["is_fictional"]:
            raise SchemaError(f"{where}: family {it['family']} with target_is_fictional={tgt['is_fictional']}")
        if tgt["name"] not in (fictional[cat] if tgt["is_fictional"] else real[cat]) or tgt["name"] == ctl["name"]:
            raise SchemaError(f"{where}: target '{tgt['name']}' is not a valid target of the category")
        names = [o["name"] for o in opts]
        if len(set(names)) != n or any(o["role"] == "context" and (o["is_fictional"] or o["name"] not in real[cat])
                                       for o in opts):
            raise SchemaError(f"{where}: option names must be distinct and context options real brands")
        shown = SHOWN_ATTR[etype] if stratum == "attr" else SHOWN[stratum]
        for o in opts:
            vis = tuple(f for f in ("description", "price", "features", "rating", "n_reviews") if o[f] is not None)
            if set(vis) != set(shown):
                raise SchemaError(f"{where}: option {o['label']} shows {vis}, expected {shown}")
            if f"\n{o['label']}. {o['name']}" not in "\n" + it["prompt"].split("\n\n", 1)[1]:
                raise SchemaError(f"{where}: option {o['label']} line does not start with the name")
        # the compared options differ only in the manipulated attribute (and in their card text)
        free = set(MANIPULATED[etype]) | {"description", "features"}
        for f in ("price", "rating", "n_reviews"):
            if f not in free and tgt[f] != ctl[f]:
                raise SchemaError(f"{where}: field '{f}' differs between target and control")
        worse, better = (tgt, ctl) if d > 0 else (ctl, tgt)
        if d == 0:
            if any(tgt[f] != ctl[f] for f in ("price", "rating", "n_reviews")):
                raise SchemaError(f"{where}: d = 0 but target and control numbers differ")
        elif etype in LADDER_TYPES:
            if not _worse(etype, worse, better):
                raise SchemaError(f"{where}: the disadvantaged option is not worse on {etype}")
        else:  # composite: worse on every type
            if not all(_worse(t, worse, better) for t in LADDER_TYPES):
                raise SchemaError(f"{where}: composite step is not worse on every type")
        if it["requirement"] is not None:
            has = [it["requirement"] in (o["features"] or []) for o in (tgt, ctl)]
            fails_expected = etype == "composite" and d != 0 and abs(d) == composite_top
            want = [not (fails_expected and o is worse) for o in (tgt, ctl)]
            if has != want:
                raise SchemaError(f"{where}: requirement met by target/control = {has}, expected {want}")
            if f"this feature: {it['requirement']}." not in it["prompt"]:
                raise SchemaError(f"{where}: the requirement sentence is missing from the prompt")
        units[it["unit_id"]].append(it)
    for uid, group in units.items():
        n = len(group[0]["options"])
        if sorted(g["order"] for g in group) != list(range(n)):
            raise SchemaError(f"unit {uid}: needs one item per option position, got {[g['order'] for g in group]}")
        keys = ("family", "tier", "split", "category", "stratum", "evidence_type", "d", "target_entity",
                "control_entity", "base", "swap", "requirement")
        for k in keys:
            if len({g[k] for g in group}) != 1:
                raise SchemaError(f"unit {uid}: field '{k}' differs between option orders")
        ref = {o["name"]: {f: v for f, v in o.items() if f != "label"} for o in group[0]["options"]}
        for g in group[1:]:
            if {o["name"]: {f: v for f, v in o.items() if f != "label"} for o in g["options"]} != ref:
                raise SchemaError(f"unit {uid}: option content differs between option orders")
    _check_monotone(items)
    return len(items)


def _check_monotone(items: list[dict[str, Any]]) -> None:
    """Within (category, stratum, type, base): the disadvantaged option gets strictly worse with |d|."""
    cells: dict[tuple, dict[int, dict[str, Any]]] = defaultdict(dict)
    for it in items:
        if it["d"] == 0 or it["evidence_type"] not in LADDER_TYPES:
            continue
        by_label = {o["label"]: o for o in it["options"]}
        worse = by_label[it["target_label"] if it["d"] > 0 else it["control_label"]]
        cells[(it["category"], it["stratum"], it["evidence_type"], it["base"])][abs(it["d"])] = worse
    for key, by_level in cells.items():
        levels = sorted(by_level)
        for lo, hi in zip(levels, levels[1:]):
            if not _worse(key[2], by_level[hi], by_level[lo]):
                raise SchemaError(f"ladder {key}: step {hi} is not worse than step {lo} "
                                  f"(rounding collapsed two steps? widen the step list)")
