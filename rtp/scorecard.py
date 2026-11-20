"""What a cycle shows, in a form a person can act on.

A scorecard has three layers:

1. FINDINGS: plain statements derived from the numbers by the fixed rules below, each with its kind and
   with what to do next. This is the part that feeds the next version of the program.
2. TABLES: the ruler (anchor curve per step), the brands on it, the scale-dependent metrics of prior work,
   the stage pairs. Compact enough to be read in full inside a feedback bundle.
3. The full numbers stay in ``scorecard.json`` and ``analysis/*/summary.json`` / ``checks.json``.

Everything here is a pure function of the JSON summaries written by rtp.analyze and of the cycle's ledger:
no model, no score file, no pandas. It is outside every GPU stage's import closure.
(``rtp/report.py`` is something else: the legacy cross-model report of the four-option family.)

Kinds of findings, and what each one means for the program:

    bug         the pipeline is wrong (a wiring identity fails, an analysis crashed). Fix it, with a test.
    instrument  the ruler needs work. Develop phase: change the config or code, bump ``ladder.version``, add
                a DECISIONS.md line, re-measure (``--redo-stale``). Confirm phase: report it, change nothing.
    research    a result about the models. Report it with its interval; never change methods, thresholds or
                data because of it.
    pending     something this cycle could not evaluate, and which job would.
    budget      time: deferred or failed jobs, how far off the cost model was.

The rules only SAY what a number means. The pass flags are computed in rtp.scale.checks from its own
constants; the advisory constants below are not thresholds of any check.
"""

from __future__ import annotations

import math
from typing import Any

BELOW_FIRST_STEP_ADVISE = 0.5
"""Advice "add a finer first step" when more than this share of the brands sits inside the first step."""
FLAT_CURVES_ADVISE = 0.5
"""Advice "this evidence does not move the model" when more than this share of a cell's category curves
never rises, even if the pooled curve does."""
FLAT_SE_ADVISE = 2.0
"""A curve whose pooled top response is not above this many of its standard errors is reported as not
rising: a monotone fit of pure noise is positive in most draws, and its brands then look "beyond the ladder"."""
BRAND_FREE = ("option position", "fictional names", "evidence, first step", "evidence, top step")
"""Contrast families without a real brand (rtp/scale/contrast.py): if THEY do not share one multiplier, the
scale of a stage pair is a choice of reference."""
IDENTITY_TOL = 1e-6
"""Pure-scale wiring check: relative tolerance of the multipliers, absolute tolerance of a preference level."""
KINDS = ("bug", "instrument", "research", "pending", "budget")
KIND_TITLES = {"bug": "Fix first (bugs)", "instrument": "Instrument (the ruler)",
               "research": "Results (report with their intervals; never tune them away)",
               "pending": "Not evaluated yet", "budget": "Budget"}
UNIT_NOTE = ("x units of a preference level: rating = stars; price = relative price increase (0.15 = +15%); "
             "reviews = natural log of the review-count ratio (1.61 = 5 times fewer reviews); composite = level.")
INSTRUMENT_NOTE = ("An instrument change is legitimate only in the develop phase and only as a whole: the PI "
                   "agrees, `ladder.version` is bumped, DECISIONS.md gets a line, and the development models are "
                   "re-measured (`AFTER_INSTRUMENT_CHANGE = True` in the notebook). Several findings are collected "
                   "into ONE change.")
STEP_KEYS = {"rating": "ladder.rating_deltas", "price": "ladder.price_deltas", "reviews": "ladder.review_ratios"}


# --------------------------------------------------------------------------- formatting
def num(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.{nd}g}"
    return str(x)


def pct(x: Any) -> str:
    return "n/a" if x is None or x != x else f"{100.0 * float(x):.0f}%"


def interval(lo: Any, hi: Any, nd: int = 3) -> str:
    return "" if lo is None or hi is None else f" [{num(lo, nd)}, {num(hi, nd)}]"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def step_label(etype: str, x: Any) -> str:
    """A ladder step in words: how much worse the target is than the control."""
    if x is None:
        return "n/a"
    if etype == "rating":
        return f"{float(x):g} stars lower"
    if etype == "price":
        return f"{100.0 * float(x):g}% dearer"
    if etype == "reviews":
        return f"{float(f'{math.exp(float(x)):.3g}'):g}x fewer reviews"
    return f"level {float(x):g}"


def table(rows: list[dict[str, Any]], columns: list[tuple[str, Any]]) -> list[str]:
    """Markdown table; a column is (header, key or function of the row). Columns that are empty in every
    row are dropped."""
    if not rows:
        return ["(nothing to show)"]
    body = [[str(f(row)) if callable(f) else num(row.get(f)) for _, f in columns] for row in rows]
    keep = [j for j in range(len(columns)) if any(r[j] not in ("", "n/a") for r in body)]
    out = ["| " + " | ".join(columns[j][0] for j in keep) + " |", "|" + "---|" * len(keep)]
    out += ["| " + " | ".join(r[j] for j in keep) + " |" for r in body]
    return out


def short(model: str) -> str:
    """Model id without its organisation: the tables stay narrow."""
    return model.split("/", 1)[-1]


def finding(kind: str, scope: str, text: str, action: str = "") -> dict[str, str]:
    return {"kind": kind, "scope": scope, "text": text, "action": action}


# --------------------------------------------------------------------------- rules: one model
def _cells(res: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for key, cell in (res.get("headline") or {}).items():
        stratum, _, etype = key.partition("/")
        out.append((stratum, etype, cell))
    return out


def _share(num_: Any, den: Any) -> str:
    return pct(num_ / den) if num_ is not None and den else "n/a"


def _ladder_findings(scope: str, stratum: str, etype: str, cell: dict[str, Any], max_censored: Any,
                     change: str, jump_max: Any = None) -> list[dict[str, str]]:
    """One (stratum, evidence type) ladder: flat, patchy, not monotone, too short, a jump at the first step,
    too coarse."""
    out: list[dict[str, str]] = []
    where = f"{stratum}/{etype}"
    key = STEP_KEYS.get(etype)
    top = f"{step_label(etype, cell.get('x_top'))} = {num(cell.get('A_top'))} logits"
    n_cat, n_flat = cell.get("n_category_curves") or 0, cell.get("n_flat_category_curves") or 0
    a_top, se_top = cell.get("A_top"), cell.get("se_top")
    in_noise = a_top is not None and se_top is not None and a_top <= FLAT_SE_ADVISE * se_top
    if cell.get("curve_rises") is False or in_noise:
        return [finding("instrument", scope,
                        f"{where}: the anchor curve does not rise beyond its noise (response at the top step, {top}, "
                        f"standard error {num(se_top, 2)}): this evidence does not move the model here, so no "
                        f"preference level on it can be trusted.",
                        "A longer ladder does not cure a flat curve. Check how this evidence is rendered (wording, "
                        "position on the card, size of the steps) on a few prompts; otherwise report the type as "
                        "not usable for this model.")]
    if n_cat and n_flat / n_cat > FLAT_CURVES_ADVISE:
        out.append(finding("instrument", scope,
                           f"{where}: the pooled curve rises, but {n_flat} of {n_cat} category curves do not; the "
                           f"brands of those categories have no level ({pct(cell.get('share_on_flat_curve'))} of all "
                           f"brands).",
                           "The evidence works only in some categories: compare the base values of the flat "
                           "categories (rtp/data/ladder_items.py `base_values`) before changing a step."))
    if cell.get("pooled_monotone") is False:
        out.append(finding("instrument", scope,
                           f"{where}: the pooled raw responses are not monotone along the ladder; the monotone fit "
                           f"merges the violating steps ({cell.get('n_monotone_category_curves')} of {n_cat} category "
                           f"curves are monotone).",
                           "Steps closer together than the noise cannot be told apart: see the ruler table; merge "
                           "or widen the steps that the fit merges."))
    cen = cell.get("share_censored")
    if cen is not None and max_censored is not None and cen > max_censored:
        q95, a_top = cell.get("abs_delta_q95"), cell.get("A_top")
        times = f", {num(q95 / a_top)} times the top step" if q95 and a_top and a_top > 0 else ""
        out.append(finding("instrument", scope,
                           f"{where}: {pct(cen)} of the brands with a level are worth more than the whole ladder "
                           f"(top step: {top}; "
                           f"the 95th percentile of |name effect| is {num(q95)} logits{times}).",
                           (f"Add larger steps to `{key}`" if key else "Extend this ladder")
                           + f" until the top response exceeds that percentile ({change})."))
    first = f"{step_label(etype, cell.get('x_first'))} = {num(cell.get('A_first'))} logits"
    low = cell.get("share_in_first_step")
    if key and cell.get("first_step_unresolved") and low:
        second = f"{step_label(etype, cell.get('x_second'))} = {num(cell.get('A_second'))} logits"
        out.append(finding("instrument", scope,
                           f"{where}: the ladder does not resolve its first step. The response at the first step "
                           f"({first}) is already {_share(cell.get('A_first'), cell.get('A_second'))} of the response "
                           f"at the second ({second}); a response proportional to the difference would give "
                           f"{_share(cell.get('x_first'), cell.get('x_second'))} (jump index "
                           f"{num(cell.get('jump_index'))}, limit {num(jump_max)}). {pct(low)} of the brands sit "
                           f"inside that step: their level is the ratio |name effect| / first-step response (median "
                           f"{num(cell.get('ratio_to_first_step_median'))}) times the step size, not a reading in x "
                           f"units.",
                           f"A finer step in `{key}` helps only if the card can display it and the response then "
                           f"shrinks with it. Otherwise this evidence acts as a tie-breaker: use a stratum whose "
                           f"options differ in more than one attribute, or trade the evidence off against a second "
                           f"attribute so that the response crosses 0 inside the ladder ({change}). Probe on "
                           f"anchors first; they need no real brand."))
    elif key and cell.get("first_step_resolved") is False:
        out.append(finding("instrument", scope,
                           f"{where}: the first step is not resolved ({first}): the model does not register it, and "
                           f"{pct(cell.get('share_below_floor'))} of the brands with a level lie below the first step "
                           f"it does register.",
                           f"The first step of `{key}` is too fine, not too coarse: replace it by a step between it "
                           f"and the next one ({change})."))
    elif key and low is not None and low > BELOW_FIRST_STEP_ADVISE:  # typed ladders only: the composite tier is coarse
        out.append(finding("instrument", scope,
                           f"{where}: {pct(low)} of the brands sit inside the first step ({first}; median |name "
                           f"effect| {num(cell.get('abs_delta_median'))} logits): their levels are interpolated "
                           f"between 0 and one point.",
                           f"Add a finer first step to `{key}`, about the size of the median |name effect| "
                           f"({change})."))
    return out


def _anchor_findings(scope: str, ac: dict[str, Any], names: dict[str, Any], limit: Any,
                     change: str) -> list[dict[str, str]]:
    n_bound = ac.get("n_rows_bound_only")
    if ac.get("pass") is None and n_bound:
        return [finding("pending", scope,
                        f"fictional anchors: not evaluated, because in all {n_bound} cells with a ratio too many real "
                        f"brands are beyond the ladder (their spread is squeezed; the SD ratio "
                        f"{num(ac.get('max_ratio_bound_only'))} is only an upper bound). In logits the fictional "
                        f"pulls spread at most {num(ac.get('max_ratio_logit'))} times as much as the real name "
                        f"effects.",
                        "Extend the ladder first (see the ladder findings); do not replace names on this evidence.")]
    if ac.get("pass") is not False:
        return []
    worst = ac.get("worst") or {}
    cell = f"{worst.get('stratum')}/{worst.get('evidence_type')}" + (" (bare names on the attribute-only curve)"
                                                                     if worst.get("cross_stratum") else "")
    text = (f"fictional anchors: the SD of the made-up names' preference levels reaches {num(ac.get('max_ratio'))} "
            f"of the real brands' (limit {num(limit)})")
    if worst:
        text += (f", in {cell}. In logits there: SD of the made-up names' pulls {num(worst.get('sd_pull'))}, SD of "
                 f"the real name effects {num(worst.get('sd_delta'))} (ratio {num(worst.get('ratio_logit'))})")
    if names:
        top = ", ".join(f"{r['name']} ({r['category']}: {num(r['pull'])} logits, {num(r['ratio'])}x a typical real "
                        f"brand)" for r in (names.get("top") or [])[:3])
        text += (f". In the NAME-ONLY stratum (not necessarily the cell above) {names.get('n_ratio_at_least_1')} of "
                 f"{names.get('n_names')} names pull at least as hard as a typical real brand of their category; "
                 f"largest there: {top}")
    if n_bound:
        text += f". {n_bound} further cells give only an upper bound (real brands censored) and were not used"
    return [finding("instrument", scope, text + ".",
                    "A ratio of two spreads fails where real names barely matter as well as where made-up names are "
                    "loud: read the two SDs of every cell (table 'Fictional anchors per cell') before touching a "
                    "name. Names are replaced only by a rule written in DECISIONS.md beforehand, in "
                    f"data/seeds/placebo_names.yaml ({change}).")]


def _tied_gap(rd: dict[str, Any], four: dict[str, Any] | None = None) -> str:
    """Letter mass on tied items against items with an evidence difference (two-option units), and in the
    four-option arm when it was measured."""
    def mean(tied: bool) -> float | None:
        rows = [r for r in rd.get("by_cell") or [] if r.get("arm") == "pair" and bool(r.get("tied")) == tied]
        n = sum(r["n_units"] for r in rows)
        return sum(r["mean_c_forced"] * r["n_units"] for r in rows) / n if n else None

    tied, differs = mean(True), mean(False)
    text = "" if tied is None or differs is None else (
        f" On tied items (both options show the same evidence) it is {num(tied)}, on items with an evidence "
        f"difference {num(differs)}.")
    if four and four.get("mean_c_forced") is not None:
        text += f" With four options and the same question it is {num(four['mean_c_forced'])}."
    return text


def model_findings(model: str, res: dict[str, Any], phase: str = "develop",
                   scaled: dict[float, Any] | None = None) -> list[dict[str, str]]:
    """``scaled``: mean letter mass of this model's pure-scale variants in the same card, by factor."""
    scope = short(model)
    if res.get("error"):
        return [finding("bug", scope, f"the analysis crashed: {res['error']}",
                        "Reproduce with `python -m rtp.cli analyze`; fix with a regression test. No GPU stage is "
                        "affected: the scores are stored.")]
    out: list[dict[str, str]] = []
    checks = res.get("checks") or {}
    limits = checks.get("thresholds") or {}
    change = "an instrument change" if phase == "develop" else "confirm phase: report it as it is, change nothing"
    rd = checks.get("readout") or {}
    if rd.get("pass") is False:
        moves = "".join(f" The number moves with scale: with every logit x {c:g} the same model reads {num(v)}."
                        for c, v in sorted((scaled or {}).items()) if v is not None)
        out.append(finding("instrument", scope,
                           f"readout limited to the forced letter choice: the option letters carry "
                           f"{num(rd.get('mean_c_forced'))} of the probability after the prefill (flag at "
                           f"{num(limits.get('COVERAGE_MIN'))}).{_tied_gap(rd, res.get('readout_ctx'))}{moves} The letter-logit difference "
                           f"is defined whatever the rest of the distribution does, so levels and multipliers are "
                           f"reported; they describe the choice between the option letters, and which tokens take "
                           f"the remaining mass is not stored by the ladder stage.",
                           "Run the token probe (`cycle probe`, notebook cell 7): it lists the tokens that take the "
                           "mass and the option the greedy continuation ends on. Decide a change of the prompt or of "
                           f"the prefill on that, not on this mean ({change})."))
    max_censored = limits.get("CENSORED_SHARE_MAX")
    for stratum, etype, cell in _cells(res):
        if "n_category_curves" in cell:  # the name stratum has no ladder of its own
            out += _ladder_findings(scope, stratum, etype, cell, max_censored, change,
                                    limits.get("FIRST_STEP_JUMP_MAX"))
    for stratum, _, cell in _cells(res):
        beyond = {t: v["share_censored"] for t, v in (cell.get("on_attr_curve") or {}).items()
                  if stratum == "name" and max_censored is not None and (v.get("share_censored") or 0.0) > max_censored}
        if beyond:
            out.append(finding("instrument", scope,
                               "name-only stratum: a bare name is worth more than the whole attribute-only ladder for "
                               + ", ".join(f"{pct(v)} of the brands on the {t} ladder" for t, v in beyond.items())
                               + f" (median |name effect| {num(cell.get('abs_delta_median'))} logits): for those "
                               f"brands the name-only level is a bound.",
                               "Matters only if a bare name's level in evidence units is to be reported: then add "
                               "larger steps to " + ", ".join(f"`{STEP_KEYS[t]}`" for t in beyond)
                               + f" ({change}). The scale statistics (c, m, T1) do not need it."))
    out += _anchor_findings(scope, checks.get("anchor_clean") or {}, res.get("names") or {},
                            limits.get("ANCHOR_SD_RATIO_MAX"), change)
    cc = checks.get("common_channel") or {}
    if cc.get("pass") is False:
        out.append(finding("research", scope,
                           f"direct and converted preference levels disagree by a median of "
                           f"{num(cc.get('median_abs_diff_level'))} ladder levels over {cc.get('n_cells')} brand cells "
                           f"(correlation {num(cc.get('corr'))}): real names respond to evidence differently from "
                           f"fictional ones.",
                           "A property of the model (a name-by-evidence interaction), not a defect: report both "
                           "levels; do not pick the one that looks better."))
    elif cc.get("pass") is None:
        out.append(finding("pending", scope, "common channel (direct vs converted level): no direct readings yet.",
                           "They are in the `ext` tier (cycle C1)."))
    for key, cell in (res.get("choice_set") or {}).items():
        if cell.get("n_theta"):
            out.append(finding("research", scope,
                               f"two options vs four, {key}: median level differs by "
                               f"{num(cell.get('theta_median_diff_4_minus_2'))} x units (four minus two) over "
                               f"{cell.get('n_theta')} brands, correlation {num(cell.get('theta_corr'))}.",
                               "Robustness of the level to the choice set; a large difference is a finding."))
    return out


# --------------------------------------------------------------------------- rules: one stage pair
def scale_variant(a: str, b: str) -> float | None:
    """c if ``b`` is the pure-scale variant ``a@scale=c``, else None."""
    prefix = a + "@scale="
    if not b.startswith(prefix):
        return None
    try:
        return float(b[len(prefix):])
    except ValueError:
        return None


def variant_of(model: str, models: dict[str, Any]) -> str | None:
    """The model of ``models`` that ``model`` is a pure-scale variant of, if any."""
    base = model.split("@scale=", 1)[0]
    return base if "@scale=" in model and base in models and base != model else None


def pair_models(name: str) -> tuple[str, str]:
    a, _, b = name.replace(" [test]", "").partition(" -> ")
    return a, b


def pair_cells(res: dict[str, Any], cross: bool | None = None) -> list[dict[str, Any]]:
    cells = [c for c in ((res.get("summary") or {}).get("cells") or []) if c.get("arm", "pair") == "pair"]
    return cells if cross is None else [c for c in cells if bool(c.get("cross_stratum")) == cross]


def _c_interval(cell: dict[str, Any]) -> str:
    """Interval of a scale multiplier: the crossed bootstrap when it was run, else the category bootstrap."""
    return interval(cell.get("c_lo_crossed", cell.get("c_lo")), cell.get("c_hi_crossed", cell.get("c_hi")))


def identity_text(ident: dict[str, Any]) -> str:
    text = (f"multipliers {num(ident.get('c_min'), 6)}..{num(ident.get('c_max'), 6)} on {ident.get('n_curves')} "
            f"curves; largest change of a preference level {num(ident.get('max_abs_theta_diff'), 2)} over "
            f"{ident.get('n_brand_cells')} brand cells; {ident.get('n_flag_changes')} flag changes")
    if ident.get("n_families"):
        text += (f"; {ident['n_families']} contrast families (position bias, made-up names, evidence steps, real "
                 f"names) grew {num(ident.get('family_min'), 6)}..{num(ident.get('family_max'), 6)}, largest |T1| "
                 f"against any reference {num(ident.get('max_abs_T1'), 2)}")
    return text


def _identity_findings(scope: str, ident: dict[str, Any], c_true: float) -> list[dict[str, str]]:
    """``B = A@scale=c``: a wiring check of the pipeline, not a result about a model."""
    if not ident.get("n_curves") or not ident.get("n_brand_cells"):
        return [finding("pending", scope, "pure-scale check: no rising curve or no brand with a level in both models, "
                                          "so the identity was not exercised.",
                        "It needs at least one evidence type the model responds to.")]
    tol = IDENTITY_TOL * c_true
    ok = (abs(ident["c_min"] - c_true) <= tol and abs(ident["c_max"] - c_true) <= tol
          and (ident.get("max_abs_theta_diff") or 0.0) <= IDENTITY_TOL and not ident.get("n_flag_changes"))
    if ident.get("n_families"):  # every logit contrast, not only the anchor curves, must grow by c
        ok = (ok and abs(ident["family_min"] - c_true) <= tol and abs(ident["family_max"] - c_true) <= tol
              and (ident.get("max_abs_T1") or 0.0) <= IDENTITY_TOL)
    text = f"pure-scale check (every logit x {c_true:g}): {identity_text(ident)}."
    if ok:
        return [finding("research", scope, text + " The level is invariant to scale on this model, as designed (H2, "
                                                  "scale part).")]
    return [finding("bug", scope, text + " A pure scale change MUST leave every level unchanged.",
                    "The readout or the scale hook is not homogeneous: check `head_dev` of both ladder stages and "
                    "rtp/models.py `scale_logits` before any other result is read.")]


def _cell(r: dict[str, Any]) -> str:
    return f"{r.get('stratum')}/{r.get('evidence_type')}"


def _ti(r: dict[str, Any], value: str = "T1") -> str:
    return f"{num(r.get(value))}{interval(r.get('lo'), r.get('hi'))}"


def _contrast_findings(scope: str, res: dict[str, Any], caveat: str) -> list[dict[str, str]]:
    """Is the stage change one multiplier? (rtp/scale/contrast.py)"""
    out: list[dict[str, str]] = []
    fam = [r for r in res.get("families") or [] if r.get("multiplier") is not None and (r.get("n_categories") or 0) >= 3
           and r.get("lo") is not None]
    if fam:
        by: dict[str, list[float]] = {}
        for r in fam:
            by.setdefault(r["family"], []).append(r["multiplier"])
        ranges = "; ".join(f"{k} {num(min(v))}" + ("" if len(v) == 1 else f" to {num(max(v))}") for k, v in by.items())
        free = [r for r in fam if r["family"] in BRAND_FREE]
        verdict = ""
        if len(free) > 1:
            low, high = min(free, key=lambda r: r["multiplier"]), max(free, key=lambda r: r["multiplier"])
            verdict = (f" Among the contrasts without a real brand, one multiplier does not describe this stage pair: "
                       f"{high['family']} ({_cell(high)}) grew {_ti(high, 'multiplier')}, {low['family']} "
                       f"({_cell(low)}) {_ti(low, 'multiplier')}, and the two intervals do not overlap."
                       if high["lo"] > low["hi"] else
                       " The contrasts without a real brand are compatible with one multiplier (the intervals of "
                       "the largest and the smallest overlap).")
        out.append(finding("research", scope,
                           f"multipliers by contrast family (later stage / earlier stage, geometric mean over "
                           f"categories; uniform sharpening predicts ONE number): {ranges}.{verdict}",
                           "If the brand-free contrasts disagree, 'the scale' is a choice of reference: report the "
                           "table, not one c. " + caveat))
    moved = [r for r in res.get("counterfactual") or [] if r.get("explained_best") is not None]
    bare = [r for r in moved if r.get("stratum") == "name"] or sorted(
        moved, key=lambda r: -abs(r["share_b"] - r["share_a"]))[:1]
    if bare:
        parts = "; ".join(
            f"{_cell(r)} {num(r['share_a'])} -> {num(r['share_b'])}"
            + (f", the earlier stage sharpened by its measured c = {num(r['c'])} gives {num(r['share_cf'])} "
               f"({pct(r['explained'])} of the change)" if r.get("explained") is not None else "")
            + f", the uniform sharpening that comes closest (x {num(r['c_best'])}) {num(r['share_best'])} "
              f"({pct(r['explained_best'])})" for r in bare)
        out.append(finding("research", scope,
                           "raw choice share of real brands with no evidence against them, and what sharpening "
                           "alone reproduces: " + parts + ".",
                           "100% = the raw-metric change is scale only. The rest is the name effect growing more "
                           "(or less) than the other contrasts, the position bias among them. All cells: table 'Raw "
                           "choice share and uniform sharpening'. " + caveat))
    return out


def _single_reference_findings(scope: str, res: dict[str, Any], caveat: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    moved, quiet = [], []
    for c in pair_cells(res):
        if c.get("T1_cat_mean") is None or c.get("T1_t_lo") is None:
            continue
        label = f"{c.get('stratum')}/{c.get('evidence_type')}" + (" (name on the attr curve)" if c.get("cross_stratum")
                                                                  else "")
        text = f"{label} {num(c.get('T1_cat_mean'))}{interval(c.get('T1_t_lo'), c.get('T1_t_hi'))}"
        (moved if c["T1_t_lo"] > 0 or c["T1_t_hi"] < 0 else quiet).append(text)
    if moved:
        out.append(finding("research", scope,
                           "name effects did NOT grow by the scale factor (T1 = log m - log c, t interval over "
                           "categories, excludes 0): " + "; ".join(moved) + ". Positive = real names gained more than "
                           "the scale factor explains (a content change), negative = less.", caveat))
    if quiet:
        out.append(finding("research", scope,
                           "no departure from the scale factor detected (T1 interval includes 0, as uniform sharpening "
                           "predicts): " + "; ".join(quiet) + ". The width of an interval says how large a content "
                           "change could have been missed.", caveat))
    return out


def _reference_findings(scope: str, references: list[dict[str, Any]], caveat: str) -> list[dict[str, str]]:
    """Did the real names grow like the brand-free contrasts? A cell counts as a departure only when its T1
    interval excludes 0, with one sign, against EVERY reference the design offers."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in references:
        if r.get("lo") is not None and r.get("T1") is not None:
            cells.setdefault(_cell(r), []).append(r)

    def ref(r: dict[str, Any]) -> str:
        return f"the {r['reference']} of {r['ref_stratum']}/{r['ref_evidence_type']}"

    robust, mixed, quiet = [], [], []
    for cell, rows in cells.items():
        lo, hi = min(rows, key=lambda r: r["T1"]), max(rows, key=lambda r: r["T1"])
        span = f"{cell}: from {_ti(lo)} against {ref(lo)} to {_ti(hi)} against {ref(hi)}"
        n_out = sum(r["lo"] > 0 or r["hi"] < 0 for r in rows)
        if all(r["lo"] > 0 for r in rows) or all(r["hi"] < 0 for r in rows):
            robust.append(span)
        elif n_out:
            mixed.append(f"{span} (excludes 0 against {n_out} of {len(rows)} references)")
        else:
            quiet.append(span)
    head = ("T1 = log multiplier of the real names - log multiplier of a reference (the whole anchor curve = the "
            "project's c, its first step, its top step, the made-up names of the same cell); mean and t interval "
            "over categories. ")
    out = []
    if robust:
        out.append(finding("research", scope,
                           "real names did NOT grow like the brand-free contrasts, whichever reference is taken (T1 "
                           "excludes 0 against every reference; positive = the names gained more): "
                           + "; ".join(robust) + ".", head + caveat))
    if mixed:
        out.append(finding("research", scope,
                           "real names against the brand-free contrasts, verdict NOT robust to the reference (T1 "
                           "excludes 0 against some references only): " + "; ".join(mixed) + ".",
                           head + "No content change is claimed for a cell while the verdict depends on the "
                           "reference; with many references one interval can exclude 0 by chance. " + caveat))
    if quiet:
        out.append(finding("research", scope,
                           "no departure from uniform sharpening against any reference (every T1 interval includes "
                           "0): " + "; ".join(quiet) + ". The widths say how large a change could have been missed.",
                           head + caveat))
    return out


def pair_findings(name: str, res: dict[str, Any], phase: str = "develop",
                  readout_note: str = "") -> list[dict[str, str]]:
    """``readout_note``: a sentence added to every result of the pair when a model's readout flag is False."""
    a, b = pair_models(name)
    scope = f"{short(a)} -> {short(b)}" + (" [test]" if name.endswith(" [test]") else "")
    if res.get("error"):
        return [finding("bug", scope, f"the comparison crashed: {res['error']}",
                        "Reproduce with `python -m rtp.cli analyze --pair`; fix with a regression test.")]
    c_true = scale_variant(a, b)
    if c_true is not None:
        return _identity_findings(scope, res.get("identity") or {}, c_true)
    out: list[dict[str, str]] = []
    summary = res.get("summary") or {}
    h1 = (summary.get("H1") or {}).get("pair") or {}
    scale = [r for r in summary.get("scale") or [] if r.get("arm", "pair") == "pair" and r.get("stratum") == "attr"]
    own = {(c.get("stratum"), c.get("evidence_type")): c for c in pair_cells(res, cross=False)}
    parts = [f"{r.get('evidence_type')} {num(r.get('c'))}{_c_interval(own.get(('attr', r.get('evidence_type')), {}))}"
             if r.get("valid") else f"{r.get('evidence_type')} n/a (curve flat in one model)" for r in scale]
    if parts:
        out.append(finding("research", scope,
                           "scale multiplier c of the anchor curves (fictional names only), per evidence type: "
                           + "; ".join(parts) + ".",
                           "c > 1 = the later stage is more decisive on that evidence. Not a change of preference."
                           + readout_note))
    if h1.get("range") is not None:
        lo, hi = h1.get("lo_crossed", h1.get("lo")), h1.get("hi_crossed", h1.get("hi"))
        out.append(finding("research", scope,
                           f"H1 (one common scale): log c spreads by {num(h1['range'])} over the three evidence "
                           f"types (a factor of {num(math.exp(h1['range']))} between the most and the least rescaled "
                           f"type). Under resampling the spread ranges from {num(lo)} to {num(hi)}: noise inflates "
                           f"every resampled spread, so this is not an interval around the estimate.",
                           "A spread is positive by construction: it is judged against the seed replicates of "
                           "cycles C2/C3, not against 0. If it stays outside their band, the split is reported per "
                           "evidence type."))
    elif scale:
        out.append(finding("pending", scope, "H1 (one common scale) needs a rising curve for all three evidence types "
                                             "in both models.", "See the instrument findings of the two models."))
    caveat = ("The intervals cover the choice of categories only; run-to-run training noise is unknown until the seed "
              "replicates of cycles C2/C3." + (" Exploratory (develop phase)." if phase == "develop" else "")
              + readout_note)
    if res.get("references"):
        out += _reference_findings(scope, res["references"], caveat)
    else:  # a summary written before the reference table existed: the whole-curve reference alone
        out += _single_reference_findings(scope, res, caveat)
    out += _contrast_findings(scope, res, caveat)
    left = res.get("tiers_left_out") or {}
    if left:
        out.append(finding("pending", scope, f"compared on tiers {res.get('tiers_compared')} only; left out because "
                                             f"one model lacks them: {left}.", "Finish the deferred ladder job."))
    return out


# --------------------------------------------------------------------------- rules: the session
def budget_findings(card: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    jobs = card.get("jobs") or []
    undone = [e for e in jobs if e.get("state") != "done"]
    if undone:
        out.append(finding("budget", "cycle", f"{len(undone)} of {len(jobs)} jobs are not done: "
                           + "; ".join(f"{e['job']} ({e['state']})" for e in undone) + ".",
                           "Re-run `cycle run` (cell 7): finished jobs are reused. A failed job has its traceback "
                           "in the feedback bundle."))
    ran = [e for e in jobs if e.get("state") == "done" and e.get("actual_s") and e.get("predicted_s")]
    if ran:
        actual, predicted = sum(e["actual_s"] for e in ran), sum(e["predicted_s"] for e in ran)
        out.append(finding("budget", "cycle",
                           f"the {len(ran)} jobs that ran took {fmt_time(actual)}; the plan predicted "
                           f"{fmt_time(predicted)} (actual / predicted = {num(actual / predicted, 2)}).",
                           "The cost model recalibrates itself on finished stages: the next `cycle plan` already "
                           "uses the measured rate."))
    return out


def _readout(res: dict[str, Any]) -> dict[str, Any]:
    return ((res or {}).get("checks") or {}).get("readout") or {}


def scaled_readout(model: str, models: dict[str, Any]) -> dict[float, Any]:
    """Mean letter mass of the pure-scale variants of ``model`` in the card, by factor."""
    out = {}
    for other, res in models.items():
        c = scale_variant(model, other)
        if c is not None and _readout(res).get("mean_c_forced") is not None:
            out[c] = _readout(res)["mean_c_forced"]
    return out


def readout_note(pair: str, models: dict[str, Any]) -> str:
    """What a pair's results are conditional on when a model's readout flag is False."""
    suffix = " [test]" if pair.endswith(" [test]") else ""
    low = [(m, _readout(models.get(m + suffix))) for m in pair_models(pair)]
    low = [(m, rd) for m, rd in low if rd.get("pass") is False]
    if not low:
        return ""
    return (" Conditional on the forced letter readout: letter mass "
            + ", ".join(f"{num(rd.get('mean_c_forced'))} ({short(m)})" for m, rd in low) + ".")


def findings(card: dict[str, Any]) -> list[dict[str, str]]:
    """Every finding of a scorecard, ordered by kind (bugs first)."""
    phase = card.get("phase", "develop")
    out: list[dict[str, str]] = []
    models = card.get("models") or {}
    for model, res in models.items():
        if variant_of(model, models) is None:  # a pure-scale variant repeats every finding of its base model
            out += model_findings(model, res, phase, scaled_readout(model, models))
    for name, res in (card.get("pairs") or {}).items():
        out += pair_findings(name, res, phase, readout_note(name, models))
    for model, info in (card.get("open_set") or {}).items():
        if info.get("coverage_warning"):
            out.append(finding("instrument", short(model),
                               f"open question: only {pct(info.get('coverage'))} of the answer mass is on the "
                               f"catalogue brands, so the concentration numbers describe a minority of the answers.",
                               "Read `samples.parquet` (text) for preambles or refusals before using any N_eff."))
    out += budget_findings(card)
    return sorted(out, key=lambda f: KINDS.index(f["kind"]))


def findings_markdown(items: list[dict[str, str]]) -> list[str]:
    if not items:
        return ["(no finding: nothing has been analysed yet)", ""]
    lines: list[str] = []
    for kind in KINDS:
        part = [f for f in items if f["kind"] == kind]
        if not part:
            continue
        lines += [f"### {KIND_TITLES[kind]}", ""]
        if kind == "instrument":
            lines += [INSTRUMENT_NOTE, ""]
        for i, f in enumerate(part, 1):
            lines.append(f"{i}. **{f['scope']}**: {f['text']}" + (f"\n   - Next: {f['action']}" if f["action"] else ""))
        lines.append("")
    return lines


# --------------------------------------------------------------------------- tables: one model
def _verdict(block: dict[str, Any] | None, detail: str, failed: str = "FAIL") -> str:
    state = (block or {}).get("pass")
    word = "pass" if state is True else failed if state is False else "not evaluated"
    return word + (f" ({detail})" if detail else "")


def verdict_rows(models: dict[str, Any]) -> list[str]:
    rows = []
    for model, res in models.items():
        ch = res.get("checks") or {}
        rd, ld, ac, cc = (ch.get(k) or {} for k in ("readout", "ladder", "anchor_clean", "common_channel"))
        anchors = (f"SD ratio up to {num(ac.get('max_ratio'))}" if ac.get("max_ratio") is not None
                   else f"real brands censored: SD ratio at most {num(ac.get('max_ratio_bound_only'))}"
                   if ac.get("n_rows_bound_only") else "")
        rows.append({
            "model": short(model), "form": res.get("letter_form"), "tiers": ", ".join(res.get("tiers") or []),
            "readout": _verdict(rd, f"letter mass {num(rd.get('mean_c_forced'))}" if rd else "", "LIMITED"),
            "ladder": _verdict(ld, f"{pct(ld.get('share_censored'))} beyond top; "
                                   f"{pct(ld.get('share_first_step_unresolved'))} inside an unresolved first step; "
                                   f"{ld.get('n_curves_flat')} of {ld.get('n_curves')} curves flat" if ld else ""),
            "anchors": _verdict(ac, anchors),
            "channel": _verdict(cc, f"median {num(cc.get('median_abs_diff_level'))} levels"
                                if cc.get("median_abs_diff_level") is not None else "")})
    return table(rows, [("model", "model"), ("tiers", "tiers"), ("letter form", "form"), ("readout", "readout"),
                        ("ladder", "ladder"), ("fictional anchors", "anchors"), ("direct vs converted", "channel")])


def _level(r: dict[str, Any]) -> str:
    return num(r.get("theta_median")) + interval(r.get("theta_q25"), r.get("theta_q75"))


def _rising(r: dict[str, Any]) -> str:
    if "n_category_curves" not in r:
        return ""
    return f"{r['n_category_curves'] - r.get('n_flat_category_curves', 0)} of {r['n_category_curves']}"


def model_section(res: dict[str, Any]) -> list[str]:
    """The tables of one model (no heading)."""
    if res.get("error"):
        return [f"analysis failed: {res['error']}", ""]
    lines: list[str] = []
    rd = (res.get("checks") or {}).get("readout") or {}
    if rd:
        lines += [f"Readout: letter mass at the forced position {num(rd.get('mean_c_forced'))} (free position "
                  f"{num(rd.get('mean_c_free'))}); {pct(rd.get('share_low_coverage'))} of units below the coverage "
                  f"limit; agreement with the letters+names readout on {rd.get('n_mass_units')} sampled units: "
                  f"correlation {num(rd.get('corr_L_mass'))}, median gap {num(rd.get('median_abs_diff_L_mass'))} "
                  f"logits; median |position bias| {num(rd.get('median_abs_order_bias'))} logits (averaged out over the "
                  f"two orders); {pct(rd.get('share_letter_collision'))} of units with a letter-colliding name.", ""]
    by_cell = (rd.get("by_cell") or []) + ((res.get("readout_ctx") or {}).get("by_cell") or [])
    if by_cell:
        lines += ["**Letter mass by kind of item** (tied = both options show the same evidence; it also moves with "
                  "scale, so compare cells within a model, not models):", ""]
        lines += table(by_cell, [
            ("options", lambda r: "two" if r.get("arm") == "pair" else "four"), ("tier", "tier"),
            ("target", lambda r: "fictional" if r.get("family") == "anchor" else "real brand"), ("stratum", "stratum"),
            ("evidence", lambda r: "tied" if r.get("tied") else "differs"), ("units", "n_units"),
            ("letter mass", "mean_c_forced"), ("below the flag", lambda r: pct(r.get("share_low_coverage"))),
            ("free position", "mean_c_free")]) + [""]
    ladders = [r for r in res.get("ladders") or [] if r.get("arm", "pair") == "pair"]
    lines += ["**The ruler** (pooled anchor curve between two fictional names; logits per step):", ""]
    lines += table(ladders, [("stratum", "stratum"), ("evidence", "evidence_type"),
                             ("step", lambda r: step_label(r["evidence_type"], r.get("x"))),
                             ("response A", lambda r: f"{num(r.get('A'))} ± {num(r.get('se'), 2)}"),
                             ("monotone fit", "A_fit"),
                             ("brands in this step", lambda r: pct(r.get("share_of_brands")))])
    cells = [{"stratum": s, "evidence_type": t, **c} for s, t, c in _cells(res)]
    lines += ["", "**Brands on the ruler** (name effect in logits; preference level in x units; jump index: 0 = the "
                  "response grows in proportion to the difference, 1 = any difference gives the full response. Where "
                  "the first step is unresolved, read the level in ladder levels: inside the first step it is |name "
                  "effect| / first-step response):", ""]
    lines += table(cells, [
        ("stratum", "stratum"), ("evidence", "evidence_type"), ("brands", "n_brands"),
        ("median name effect", "delta_median"), ("negative", lambda r: pct(r.get("share_delta_negative"))),
        ("median level [q25, q75]", _level), ("beyond top", lambda r: pct(r.get("share_censored"))),
        ("inside first step", lambda r: pct(r.get("share_in_first_step"))),
        ("jump index of the first step", lambda r: "" if r.get("jump_index") is None else
         num(r["jump_index"]) + (" (unresolved)" if r.get("first_step_unresolved") else "")),
        ("median |level| in ladder levels", "abs_level_median"),
        ("no level (flat curve)", lambda r: pct(r.get("share_on_flat_curve"))), ("category curves rising", _rising)])
    bare = next((c for s, _, c in _cells(res) if s == "name" and c.get("on_attr_curve")), None)
    if bare:
        lines += ["", "A bare name (no evidence shown) on the attribute-only curves: " + "; ".join(
            f"{t} median {num(v.get('theta_median'))} x units ({pct(v.get('share_censored'))} beyond top)"
            for t, v in bare["on_attr_curve"].items()) + "."]
    if res.get("raw"):
        lines += ["", "**Scale-dependent metrics of prior work, same items** (they move under a pure scale change; "
                      "shown for contrast):", ""]
        lines += table(res["raw"], [
            ("stratum", "stratum"), ("evidence", "evidence_type"), ("choice share (prob)", "choice_share_prob"),
            ("choice share (argmax)", "choice_share_argmax"), ("N_eff", "n_eff"),
            ("flip rate at top step (prob)", "flip_rate_prob"), ("flip rate at top step (argmax)", "flip_rate_argmax")])
    names = res.get("names") or {}
    if names.get("top"):
        lines += ["", f"**Fictional names with the largest pull** (name-only stratum; "
                      f"{names.get('n_ratio_at_least_1')} of {names.get('n_names')} pull at least as hard as a typical "
                      f"real brand of their category; median ratio {num(names.get('median_ratio'))}):", ""]
        lines += table(names["top"][:8], [
            ("category", "category"), ("name", "name"),
            ("pull (logits)", lambda r: f"{num(r.get('pull'))} ± {num(r.get('pull_se'), 2)}"),
            ("median real brand", "typical_real"), ("ratio", "ratio")])
    anchors = [r for r in ((res.get("checks") or {}).get("anchor_clean") or {}).get("rows") or []
               if r.get("arm", "pair") == "pair"]
    if anchors:
        lines += ["", "**Fictional anchors per cell** (spread of the made-up names against the spread of the real "
                      "brands; the flag is decided on the largest ratio of levels that is not a bound):", ""]
        lines += table(anchors, [
            ("stratum", lambda r: f"{r.get('stratum')}" + (" (on attr curve)" if r.get("cross_stratum") else "")),
            ("evidence", "evidence_type"), ("SD ratio of levels", "ratio"),
            ("SD of pulls (logits)", "sd_pull"), ("SD of real name effects (logits)", "sd_delta"),
            ("ratio in logits", "ratio_logit"), ("note", lambda r: "bound only" if r.get("bound_only") else "")])
    if res.get("choice_set"):
        rows = [{"cell": k, **v} for k, v in res["choice_set"].items()]
        lines += ["", "**Two options vs four** (same brands; four-option effects re-centred on the same zero):", ""]
        lines += table(rows, [("cell", "cell"), ("brands", "n_brands"), ("name effect corr", "delta_corr"),
                              ("median name effect, 4 - 2", "delta_median_diff_4_minus_2"),
                              ("level corr", "theta_corr"), ("median level, 4 - 2", "theta_median_diff_4_minus_2")])
    return lines + [""]


# --------------------------------------------------------------------------- tables: one stage pair
def _before_after(metric: str):
    def cell(r: dict[str, Any]) -> str:
        return "" if f"{metric}_a" not in r else f"{num(r[f'{metric}_a'])} -> {num(r[f'{metric}_b'])}"

    return cell


def _level_change(r: dict[str, Any]) -> str:
    return num(r.get("median_d_theta")) + (f" (n={r.get('n_d_theta')})" if r.get("n_d_theta") else "")


def pair_section(name: str, res: dict[str, Any], note: str = "") -> list[str]:
    """``note``: what the results are conditional on (``readout_note``)."""
    if res.get("error"):
        return [f"comparison failed: {res['error']}", ""]
    a, b = pair_models(name)
    lines = [f"Tiers compared: {', '.join(res.get('tiers_compared') or [])}; letter form `{res.get('letter_form')}`."
             + note]
    if scale_variant(a, b) is not None:
        lines.append(f"Pure-scale identity: {identity_text(res.get('identity') or {})}.")
    lines += ["", "c = ratio of the anchor curves summed over steps and categories (scale); m = the same ratio of the "
                  "summed name effects of real brands; T1 = log m - log c (0 under uniform sharpening). 'T1 pooled' "
                  "is log m - log c of the two printed numbers; 'T1 by category' is the mean of the same statistic "
                  "computed per category, with its t interval (the primary interval). They differ when categories "
                  "differ in size. Intervals of c, m and the pooled T1: crossed bootstrap (categories and fictional "
                  "names).", ""]
    lines += table(pair_cells(res), [
        ("stratum", lambda r: f"{r.get('stratum')}" + (" (on attr curve)" if r.get("cross_stratum") else "")),
        ("evidence", "evidence_type"), ("brands", "n_entities"),
        ("c", lambda r: num(r.get("c")) + _c_interval(r)),
        ("m", lambda r: num(r.get("m")) + interval(r.get("m_lo_crossed", r.get("m_lo")),
                                                   r.get("m_hi_crossed", r.get("m_hi")))),
        ("T1 pooled", lambda r: num(r.get("T1")) + interval(r.get("T1_lo_crossed", r.get("T1_lo")),
                                                           r.get("T1_hi_crossed", r.get("T1_hi")))),
        ("T1 by category", lambda r: num(r.get("T1_cat_mean")) + interval(r.get("T1_t_lo"), r.get("T1_t_hi"))),
        ("categories", "n_cat_T1"), ("median change of level", _level_change),
        ("name effects negative in A", "n_negative_a")])
    if res.get("shape"):
        lines += ["", "**Shape of the ruler** (response of B / response of A, divided by the curve's c; 1 at every "
                      "step = the curve was only stretched):", ""]
        lines += table(res["shape"], [("stratum", "stratum"), ("evidence", "evidence_type"),
                                      ("step", lambda r: step_label(r["evidence_type"], r.get("x"))),
                                      ("A", "A_a"), ("B", "A_b"), ("ratio", "ratio"), ("ratio / c", "ratio_over_c")])
    variant = scale_variant(a, b) is not None
    if res.get("families") and not variant:
        lines += ["", "**Multipliers by contrast family** (B / A; geometric mean over categories with its t interval; "
                      "uniform sharpening predicts the same number in every row):", ""]
        lines += table(res["families"], [
            ("stratum", "stratum"), ("family", "family"), ("evidence", "evidence_type"),
            ("multiplier", lambda r: _ti(r, "multiplier")), ("categories", "n_categories")])
    if res.get("references") and not variant:
        lines += ["", "**T1 against every reference** (log multiplier of the real names - log multiplier of the "
                      "reference; mean over categories with its t interval; 0 = the names grew like the reference):", ""]
        lines += table(res["references"], [
            ("real names of", _cell), ("reference", "reference"),
            ("of", lambda r: f"{r.get('ref_stratum')}/{r.get('ref_evidence_type')}"), ("T1", _ti),
            ("categories", "n_categories")])
    if res.get("counterfactual"):
        lines += ["", "**Raw choice share and uniform sharpening** (real brand vs fictional control, no evidence "
                      "against the brand; single letter form; 'sharpened' = A with every logit multiplied by c):", ""]
        lines += table(res["counterfactual"], [
            ("stratum", "stratum"), ("evidence", "evidence_type"), ("A", "share_a"), ("B", "share_b"),
            ("measured c", "c"), ("A sharpened by c", "share_cf"), ("explained", lambda r: pct(r.get("explained"))),
            ("best factor", "c_best"), ("A sharpened by it", "share_best"),
            ("explained at best", lambda r: pct(r.get("explained_best")))])
    if res.get("raw"):
        lines += ["", "**Scale-dependent metrics, A -> B** (what a raw-metric study would report as the change):", ""]
        lines += table(res["raw"], [("stratum", "stratum"), ("evidence", "evidence_type"),
                                    ("choice share (prob)", _before_after("choice_share_prob")),
                                    ("N_eff", _before_after("n_eff")),
                                    ("flip rate at top step (prob)", _before_after("flip_rate_prob")),
                                    ("flip rate at top step (argmax)", _before_after("flip_rate_argmax"))])
    return lines + [""]


def _by_key(field: str, prefix: str = ""):
    def cell(r: dict[str, Any]) -> str:
        return ", ".join(f"{prefix}{k} {num(v)}" for k, v in (r.get(field) or {}).items())

    return cell


def open_set_rows(open_set: dict[str, Any]) -> list[str]:
    rows = [{"model": short(m), **info} for m, info in open_set.items()]
    return table(rows, [("model", "model"), ("mass on catalogue brands", lambda r: pct(r.get("coverage"))),
                        ("N_eff (closed set, by prompt)", _by_key("n_eff")), ("top-1 share", _by_key("top1")),
                        ("sampled N_eff by temperature", _by_key("sampled_n_eff_by_temperature", "T="))])


# --------------------------------------------------------------------------- the digest
def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def _vals(values: list[Any], nd: int = 3) -> str:
    return "[" + ", ".join(num(v, nd) for v in values) + "]"


def _model_digest(model: str, res: dict[str, Any]) -> list[str]:
    """One model in compact lines: every number the scorecard's tables carry, one line per cell."""
    if res.get("error"):
        return [f"### {model}", f"analysis failed: {res['error']}"]
    lines = [f"### {model} (tiers {', '.join(res.get('tiers') or [])}; letter form {res.get('letter_form')})"]
    rd = (res.get("checks") or {}).get("readout") or {}
    if rd:
        lines.append(f"readout: letter mass {num(rd.get('mean_c_forced'))} (free position {num(rd.get('mean_c_free'))}), "
                     f"{pct(rd.get('share_low_coverage'))} of units below the flag; L vs letters+names on "
                     f"{rd.get('n_mass_units')} units: corr {num(rd.get('corr_L_mass'))}, median gap "
                     f"{num(rd.get('median_abs_diff_L_mass'))}; median |position bias| "
                     f"{num(rd.get('median_abs_order_bias'))}; letter-colliding names "
                     f"{pct(rd.get('share_letter_collision'))}")
    by_cell = (rd.get("by_cell") or []) + ((res.get("readout_ctx") or {}).get("by_cell") or [])
    if by_cell:
        lines.append("letter mass by kind of item (options tier target stratum evidence: mass, share below the flag, units):")
        lines += [f"  {'two' if r.get('arm') == 'pair' else 'four'} {r.get('tier')} "
                  f"{'fictional' if r.get('family') == 'anchor' else 'real'} {r.get('stratum')} "
                  f"{'tied' if r.get('tied') else 'differs'}: {num(r.get('mean_c_forced'))}, "
                  f"{pct(r.get('share_low_coverage'))}, n={r.get('n_units')}" for r in by_cell]
    head = res.get("headline") or {}
    ladders: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in res.get("ladders") or []:
        if r.get("arm", "pair") == "pair":
            ladders.setdefault((r["stratum"], r["evidence_type"]), []).append(r)
    if ladders:
        lines.append("ruler (x = step in natural units, A = logits between two fictional names, brands% = share of the "
                     "brands with a level inside each step):")
    for (stratum, etype), rows in ladders.items():
        rows = sorted(rows, key=lambda r: r["level"])
        cell = head.get(f"{stratum}/{etype}") or {}
        lines.append(f"  {stratum}/{etype} x={_vals([r.get('x') for r in rows])} A={_vals([r.get('A') for r in rows])} "
                     f"se_top={num(cell.get('se_top'), 2)} jump={num(cell.get('jump_index'))} brands%="
                     + "[" + ", ".join(pct(r.get("share_of_brands")).rstrip("%") for r in rows) + "] "
                     f"beyond_top={pct(cell.get('share_censored'))} rising_category_curves={_rising(cell) or 'n/a'}")
    if head:
        lines.append("brands (real vs fictional; D = name effect in logits; level in x units):")
    for key, c in head.items():
        if "n_brands" not in c:
            continue
        line = (f"  {key} n={c.get('n_brands')} median_D={num(c.get('delta_median'))} negative="
                f"{pct(c.get('share_delta_negative'))} median|D|={num(c.get('abs_delta_median'))} q95|D|="
                f"{num(c.get('abs_delta_q95'))}")
        if c.get("theta_median") is not None:
            line += (f" | level median={num(c.get('theta_median'))}{interval(c.get('theta_q25'), c.get('theta_q75'))} "
                     f"in_first_step={pct(c.get('share_in_first_step'))} D/first_step median="
                     f"{num(c.get('ratio_to_first_step_median'))} beyond_top={pct(c.get('share_censored'))} "
                     f"no_level={pct(c.get('share_on_flat_curve'))}")
        if c.get("on_attr_curve"):
            line += " | on attr curves: " + "; ".join(
                f"{t} {num(v.get('theta_median'))} (beyond {pct(v.get('share_censored'))})"
                for t, v in c["on_attr_curve"].items())
        lines.append(line)
    anchors = [r for r in ((res.get("checks") or {}).get("anchor_clean") or {}).get("rows") or []
               if r.get("arm", "pair") == "pair"]
    if anchors:
        lines.append("fictional anchors per cell (SD ratio of levels | SD of pulls / SD of real name effects, logits):")
        lines += [f"  {r.get('stratum')}/{r.get('evidence_type')}{' (bare names on attr curve)' if r.get('cross_stratum') else ''}"
                  f" {num(r.get('ratio'))} | {num(r.get('sd_pull'))} / {num(r.get('sd_delta'))} = "
                  f"{num(r.get('ratio_logit'))}{' [bound only]' if r.get('bound_only') else ''}" for r in anchors]
    names = res.get("names") or {}
    if names.get("top"):
        lines.append(f"fictional names (name-only): {names.get('n_ratio_at_least_1')} of {names.get('n_names')} pull at "
                     f"least as hard as a typical real brand; median ratio {num(names.get('median_ratio'))}; largest: "
                     + "; ".join(f"{r['name']} ({r['category']}) {num(r['pull'])}±{num(r.get('pull_se'), 2)} "
                                 f"ratio {num(r['ratio'])}" for r in names["top"][:8]))
    if res.get("raw"):
        lines.append("raw metrics: " + "; ".join(
            f"{r['stratum']}/{r['evidence_type']} share {num(r.get('choice_share_prob'))} (argmax "
            f"{num(r.get('choice_share_argmax'))}) N_eff {num(r.get('n_eff'))}"
            + (f" flip@top {num(r.get('flip_rate_prob'))} (argmax {num(r.get('flip_rate_argmax'))})"
               if r.get("flip_rate_prob") is not None else "") for r in res["raw"]))
    if res.get("choice_set"):
        lines.append("two vs four options: " + "; ".join(
            f"{k} n={v.get('n_brands')} D corr {num(v.get('delta_corr'))} diff "
            f"{num(v.get('delta_median_diff_4_minus_2'))}, level corr {num(v.get('theta_corr'))} diff "
            f"{num(v.get('theta_median_diff_4_minus_2'))}" for k, v in res["choice_set"].items()))
    return lines


def _pair_digest(name: str, res: dict[str, Any], note: str) -> list[str]:
    if res.get("error"):
        return [f"### {name}", f"comparison failed: {res['error']}"]
    a, b = pair_models(name)
    lines = [f"### {name} (tiers {', '.join(res.get('tiers_compared') or [])}){note}"]
    if scale_variant(a, b) is not None:
        return lines + [f"pure-scale identity: {identity_text(res.get('identity') or {})}"]
    own = pair_cells(res, cross=False)
    lines.append("c (anchor curves): " + "; ".join(f"{_cell(c)} {num(c.get('c'))}{_c_interval(c)}" for c in own))
    h1 = ((res.get("summary") or {}).get("H1") or {}).get("pair") or {}
    if h1.get("range") is not None:
        lines.append(f"H1: log c spreads by {num(h1['range'])} over the evidence types (resampled "
                     f"{num(h1.get('lo_crossed', h1.get('lo')))}..{num(h1.get('hi_crossed', h1.get('hi')))})")
    lines.append("real names (m | T1 pooled | T1 by category | median change of level | negative in A):")
    for c in pair_cells(res):
        cell = _cell(c) + (" on attr curve" if c.get("cross_stratum") else "")
        lines.append(f"  {cell}: m {num(c.get('m'))}{interval(c.get('m_lo_crossed', c.get('m_lo')), c.get('m_hi_crossed', c.get('m_hi')))}"
                     f" | {num(c.get('T1'))}{interval(c.get('T1_lo_crossed', c.get('T1_lo')), c.get('T1_hi_crossed', c.get('T1_hi')))}"
                     f" | {num(c.get('T1_cat_mean'))}{interval(c.get('T1_t_lo'), c.get('T1_t_hi'))} (n={c.get('n_cat_T1')})"
                     f" | {_level_change(c)} | {c.get('n_negative_a')}")
    shape: dict[str, list[dict[str, Any]]] = {}
    for r in res.get("shape") or []:
        shape.setdefault(_cell(r), []).append(r)
    if shape:
        lines.append("shape (B / A at each step): " + "; ".join(
            f"{k} {_vals([r.get('ratio') for r in sorted(rows, key=lambda r: r['level'])])}" for k, rows in shape.items()))
    fam: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in res.get("families") or []:
        fam.setdefault((r["stratum"], r["family"]), []).append(r)
    if fam:
        lines.append("multipliers by contrast family (B / A, t interval over categories, n categories):")
        lines += [f"  {stratum} | {family}: " + "; ".join(
            f"{r['evidence_type']} {_ti(r, 'multiplier')} n={r.get('n_categories')}" for r in rows)
            for (stratum, family), rows in fam.items()]
    refs: dict[str, list[dict[str, Any]]] = {}
    for r in res.get("references") or []:
        refs.setdefault(_cell(r), []).append(r)
    if refs:
        lines.append("T1 of the real names against every reference (0 = grew like the reference):")
        lines += [f"  {cell}: " + "; ".join(f"{r['reference']} of {r['ref_stratum']}/{r['ref_evidence_type']} {_ti(r)}"
                                          for r in rows) for cell, rows in refs.items()]
    if res.get("counterfactual"):
        lines.append("raw choice share and uniform sharpening (A -> B | A sharpened by the measured c | by the best factor):")
        lines += [f"  {_cell(r)}: {num(r.get('share_a'))} -> {num(r.get('share_b'))} | x{num(r.get('c'))}: "
                  f"{num(r.get('share_cf'))} ({pct(r.get('explained'))}) | x{num(r.get('c_best'))}: "
                  f"{num(r.get('share_best'))} ({pct(r.get('explained_best'))})" for r in res["counterfactual"]]
    if res.get("raw"):
        lines.append("raw metrics A -> B: " + "; ".join(
            f"{_cell(r)} share {_before_after('choice_share_prob')(r)}, N_eff {_before_after('n_eff')(r)}"
            + (f", flip@top {_before_after('flip_rate_prob')(r)}" if "flip_rate_prob_a" in r else "")
            for r in res["raw"]))
    return lines


def digest_lines(card: dict[str, Any]) -> list[str]:
    """The scorecard as compact text for copy and paste: the findings (statement only; the fixed next steps
    are in ``scorecard.md``) and every number of the tables, one line per cell. Pure function of the card."""
    models, pairs = card.get("models") or {}, card.get("pairs") or {}
    ins = card.get("instrument") or {}
    lines = [f"CYCLE {card.get('cycle')} ({card.get('phase')}, config {card.get('config')}) written {card.get('time')}"
             f"{'' if card.get('final') else ' (session still running)'}; ladder version {ins.get('ladder_version')}, "
             f"data {ins.get('data_hash')}, readout code {ins.get('ladder')}, commit {str(ins.get('git_commit'))[:10]}",
             f"QUESTION {card.get('question')}", "",
             "JOBS (state, predicted -> actual): " + "; ".join(
                 f"{e['job']}: {e['state']}, {fmt_time(e.get('predicted_s') or 0)} -> "
                 f"{'-' if e.get('actual_s') is None else fmt_time(e['actual_s'])}" for e in card.get("jobs") or []),
             f"elapsed {fmt_time(card.get('elapsed_s') or 0)} of a {fmt_time(card.get('budget_s') or 0)} budget", "",
             "## Findings (statements; kinds: bug, instrument, research, pending, budget)"]
    items = card.get("findings")
    if items is None:
        items = findings(card)
    lines += [f"- [{f['kind']}] {f['scope']}: {_cut(f['text'], 560)}" for f in items] or ["(none)"]
    if models:
        lines += ["", "## Checks"] + verdict_rows(models) + ["", UNIT_NOTE]
    for m, res in models.items():
        base = variant_of(m, models)
        if base is None:
            lines += [""] + _model_digest(m, res)
        else:
            lines += ["", f"### {m}: pure-scale variant of {base}; letter mass "
                          f"{num(_readout(res).get('mean_c_forced'))}; see its stage pair"]
    for name, res in pairs.items():
        lines += [""] + _pair_digest(name, res, readout_note(name, models))
    if card.get("open_set"):
        lines += ["", "## Open question (which brand is named unprompted)"] + open_set_rows(card["open_set"])
    if card.get("changes"):
        lines += ["", f"## Changed since cycle {card.get('compared_with')} (largest first)"]
        lines += [f"- {c['key']}: {num(c['before'])} -> {num(c['now'])}" for c in card["changes"][:12]]
    if card.get("decisions"):
        lines += ["", "## Decisions this cycle informs"] + [f"- {d}" for d in card["decisions"]]
    return lines


# --------------------------------------------------------------------------- the scorecard
def scorecard_markdown(card: dict[str, Any]) -> str:
    ins = card.get("instrument") or {}
    lines = [f"# Scorecard: cycle {card['cycle']} ({card['phase']}, config {card['config']})", "",
             f"**Question.** {card['question']}", "",
             f"- written {card['time']}{'' if card['final'] else ' (session still running)'}; elapsed "
             f"{fmt_time(card['elapsed_s'])} of a {fmt_time(card['budget_s'])} budget",
             f"- instrument: ladder version {ins.get('ladder_version')}, data hash `{ins.get('data_hash')}`, readout "
             f"code `{ins.get('ladder')}`, commit `{str(ins.get('git_commit'))[:10]}`"]
    if card["phase"] == "develop":
        lines.append("- phase **develop**: every number below is exploratory; the instrument may still change")
    items = card.get("findings")
    if items is None:
        items = findings(card)
    lines += ["", "## Findings", ""] + findings_markdown(items)
    models, pairs = card.get("models") or {}, card.get("pairs") or {}
    lines += ["## Instrument checks per model", ""]
    if not models:
        lines += ["(no model has a finished ladder stage yet)", ""]
    else:
        lines += verdict_rows(models) + ["", UNIT_NOTE, ""]
        for m, res in models.items():
            base = variant_of(m, models)
            if base is None:
                lines += [f"### {m}", ""] + model_section(res)
            else:
                lines += [f"### {m}", "", f"Pure-scale variant of {base}: every logit is multiplied by a constant, so "
                          f"its tables repeat the ones above with the logit columns rescaled. What must hold is in "
                          f"the stage pair below.", ""]
    lines += ["## Stage pairs", ""]
    if not pairs:
        lines += ["(no pair has both models analysed yet)", ""]
    for name, res in pairs.items():
        lines += [f"### {name}", ""] + pair_section(name, res, readout_note(name, models))
    if card.get("open_set"):
        lines += ["## Open question (which brand does the model name unprompted)", ""]
        lines += open_set_rows(card["open_set"]) + [""]
    if card.get("compared_with"):
        lines += [f"## What changed since cycle {card['compared_with']}", "",
                  "Numbers of the checks, the headline cells and the stage pairs that moved by more than 5% (largest "
                  "relative change first).", ""]
        lines += [f"- {c['key']}: {num(c['before'])} -> {num(c['now'])}" for c in card.get("changes") or []] or \
            ["(no headline number moved by more than 5%)"]
        lines.append("")
    if card.get("decisions"):
        lines += ["## Decisions this cycle informs", ""] + [f"- {d}" for d in card["decisions"]] + [""]
    lines += ["## Jobs (budget ledger)", "", "| job | state | predicted | actual |", "|---|---|---|---|"]
    for e in card.get("jobs") or []:
        act = "—" if e.get("actual_s") is None else fmt_time(e["actual_s"])
        lines.append(f"| {e['job']} | {e['state']} | {fmt_time(e['predicted_s'])} | {act} |")
    lines.append("")
    if card.get("cut_order"):
        lines += ["## Cut first if the session runs long", ""]
        lines += [f"{i}. {c}" for i, c in enumerate(card["cut_order"], 1)] + [""]
    lines += ["Full numbers: `scorecard.json` next to this file; per model and pair `analysis/*/summary.json`, "
              "`checks.json` and the `.parquet` tables.", ""]
    return "\n".join(lines) + "\n"
