"""Choose which factor families to ship — decided on pre-2018 folds only.

The ablation showed that adding every family makes the model *worse*, so a choice has to
be made. Making it on the full walk-forward would be selecting on the same window the
README reports, which is the subtler cousin of tuning on the test set.

So selection runs on walk-forward folds strictly before `config.TEST_START`, and the
chosen set is then reported on 2018-onward, which the selection never saw. The scoring
metric is skill versus persistence rather than R^2, because each fold's R^2 denominator is
that year's own variance and is not comparable across years; QLIKE is scored alongside it.

**The rule.** A family set is adopted only if it beats core-only on a *paired* fold-by-fold
t-test at `t > 2`, on both metrics, at both fold granularities. That is the same bar
`methods.py` and `settle.py` use, and `metrics.paired_test` is the only implementation of
it. This file used to apply a one-standard-error rule instead — see `choose` for why that
changed.
"""

from __future__ import annotations


import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import feature_families, fit_and_eval_frames
from volradar.models import dataset as D
from volradar.models import estimators as E

SELECTION_YEARS = [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]

# Reported at both granularities; a family must clear the bar at both to be adopted.
GRANULARITIES = [("annual", 1), ("semi-annual", 2)]


def candidates(panel: str) -> list[tuple[str, ...]]:
    """A focused shortlist rather than all 2^k subsets.

    The exhaustive sweep is both far too slow on the sector panel and statistically
    self-defeating: with only six selection folds, picking the best of hundreds of
    combinations mostly selects noise. So this tests a nested sequence from smallest to
    largest, plus the specific hypotheses worth checking on their own.

    The families split into two kinds. `jump`, `semivar` and `range_ext` are *own-history*
    blocks — more ways of reading the same price series, and the standard HAR-J / HAR-RS
    specifications from the literature. `vix`, `breadth`, `macro` and `sector_xs` are
    *external* series, which carry an operational cost the own-history blocks do not: a
    provider can revise, delist or silently truncate them.
    """
    seq: list[tuple[str, ...]] = [
        (),                                   # core only — the bar every set must clear
        # --- own-history blocks (no new data dependency) ---------------------------------
        ("semivar",),                         # HAR-RS: up/down variance split
        ("jump",),                            # HAR-J: continuous/jump split
        ("range_ext",),                       # Rogers-Satchell + Yang-Zhang estimators
        ("semivar", "jump"),
        ("semivar", "jump", "range_ext"),
        # --- external series -------------------------------------------------------------
        ("vix",),                             # implied vol: the literature's #2 predictor
        ("vix", "vix_short"),                 # is the near end of the curve worth its cost?
        ("semivar", "jump", "range_ext", "vix"),
        ("vix", "breadth"),                   # + cross-sectional correlation
        ("vix", "breadth", "sector_xs"),      # + sector dispersion
        ("vix", "breadth", "sector_xs", "macro"),   # everything external
        ("breadth",),                         # is breadth useful without VIX?
    ]
    if panel == "sector":
        seq += [("vix", "sector_rel"), ("semivar", "jump", "range_ext", "vix", "sector_rel")]
    return seq


# Tuning a candidate is 12 fits on the full training frame, and each candidate is scored at
# two fold granularities on identical features — so without this the tuning runs twice for
# no reason. Keyed on everything that can change the answer.
_TUNED: dict[tuple, dict] = {}


def _params_for(panel: str, features: list[str], h: int, tr_sub, sub) -> dict:
    key = (panel, h, tuple(features))
    if key not in _TUNED:
        fit = D.embargoed_train(tr_sub, h, pd.Timestamp(config.VAL_START))
        val = D.time_slice(sub, config.VAL_START, config.VAL_END)
        _TUNED[key] = (E.tune_gbm(fit, val, features, h, quiet=True)
                       if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)
    return _TUNED[key]


def score_set(panel: str, features: list[str], h: int,
              folds: list[tuple[pd.Timestamp, pd.Timestamp]], per_year: int = 1) -> dict:
    """Walk-forward skill for one feature set, on the SAME protocol production uses.

    Specifically: fitted on `training_frame` and scored on `load_panel`, which differ when
    the panel trains pooled. Scoring a candidate on a sector-only fit and then shipping a
    pooled one would be measuring a model nobody runs — and the sector panel trains pooled.
    """
    tr_sub, sub = fit_and_eval_frames(panel, features, h)
    params = _params_for(panel, features, h, tr_sub, sub)

    min_test = max(60 // per_year, 30)
    skills, qgains = [], []
    for cutoff, end in folds:
        train = D.embargoed_train(tr_sub, h, cutoff)
        test = D.time_slice(sub, cutoff, end)
        if len(train) < 400 or len(test) < min_test:
            continue
        pred = E.fit_gbm(train, features, h, params).predict(test)
        y, per = test[config.fwd_col(h)], test[config.persist_col(h)]
        skills.append(M.regression_metrics(y, pred, per)["skill_vs_persistence"])
        qgains.append(M.qlike(y, per) - M.qlike(y, pred))

    s, q = pd.Series(skills, dtype=float), pd.Series(qgains, dtype=float)
    return {"n_folds": len(s), "skill_mean": s.mean(), "skill_std": s.std(ddof=0),
            "skill_min": s.min(), "folds_positive": int((s > 0).sum()),
            "qlike_gain": q.mean(),
            # Retained for the paired test: every candidate sees identical folds.
            "_skill_folds": s.to_numpy(), "_qlike_folds": q.to_numpy()}


def select(panel: str, h: int | None = None) -> pd.DataFrame:
    """Score every candidate family set, at both fold granularities."""
    h = h or config.HEADLINE_HORIZON
    fams = feature_families(panel)
    rows = []

    for label, per_year in GRANULARITIES:
        folds = D.fold_edges(SELECTION_YEARS, per_year)
        print(f"\n=== Family selection: {panel}, h={h}, {label} folds "
              f"{SELECTION_YEARS[0]}-{SELECTION_YEARS[-1]} (pre-test only) ===")
        for combo in candidates(panel):
            feats = list(fams["core"]) + [c for k in combo for c in fams[k]]
            s = score_set(panel, feats, h, folds, per_year)
            rows.append({"panel": panel, "granularity": label,
                         "families": "core+" + ("+".join(combo) or "only"),
                         "combo": list(combo), "n_features": len(feats), **s})
            print(f"  {rows[-1]['families']:<44} skill {s['skill_mean']:+.3f} "
                  f"+/- {s['skill_std']:.3f}  QLIKE gain {s['qlike_gain']:+.4f}  "
                  f"({s['folds_positive']}/{s['n_folds']} folds +)")
    return pd.DataFrame(rows)


def choose(out: pd.DataFrame, panel: str) -> tuple[list[str], pd.DataFrame]:
    """Adopt the largest family set that beats core-only by the project-wide bar.

    **Why this replaced the one-standard-error rule.** The 1-SE rule (Breiman et al.) is a
    perfectly defensible way to break ties: treat everything within one standard error of
    the best as equivalent, then take the simplest. But it was the only place in this
    project that used it — `methods.py` and `settle.py` both ask for a paired fold-by-fold
    t-test at t > 2 — and the two rules disagree in the region that mattered. The VIX family
    was worth about +0.075 R^2 on the market panel and was rejected by the 1-SE rule while
    never being put to the paired test the rest of the repo requires.

    Having two bars in one repo means the answer depends on which file asked the question,
    which is not a property a selection rule may have. The paired test is the one kept
    because it is the more appropriate test: every candidate is scored on identical folds,
    so the comparison is paired and the shared year-to-year variation cancels, whereas the
    1-SE rule compares two unpaired marginal spreads and throws that cancellation away.

    Note this is not simply the looser rule. It is stricter in one direction — a family must
    now win on *both* skill and QLIKE at *both* fold granularities — and it is applied
    against core-only rather than against whichever candidate happened to top the table.
    """
    base_key = "core+only"
    verdicts = []
    for label, _ in GRANULARITIES:
        g = out[(out["granularity"] == label)]
        base = g[g["families"] == base_key].iloc[0]
        for _, r in g.iterrows():
            if r["families"] == base_key:
                continue
            vs = M.paired_test(r["_skill_folds"], base["_skill_folds"])
            vq = M.paired_test(r["_qlike_folds"], base["_qlike_folds"])
            verdicts.append({
                "panel": panel, "granularity": label, "families": r["families"],
                "combo": r["combo"], "n_features": r["n_features"],
                "skill_delta": vs["delta"], "skill_t": vs["t"], "skill_wins": vs["wins"],
                "qlike_delta": vq["delta"], "qlike_t": vq["t"],
                "n_folds": vs["n"],
                "adopts": M.adopts(vs) and M.adopts(vq),
            })
    v = pd.DataFrame(verdicts)

    # A set ships only if it clears the bar at every granularity tested.
    passed = (v.groupby("families")["adopts"].all())
    winners = [f for f, ok in passed.items() if ok]

    print(f"\n  Paired vs core-only (t > {M.ADOPT_T} on skill AND QLIKE, both granularities):")
    for fam in sorted(v["families"].unique()):
        sub = v[v["families"] == fam]
        mark = "ADOPT" if fam in winners else "hold"
        detail = "  ".join(f"{r['granularity'][:4]} t_skill={r['skill_t']:+.2f} "
                           f"t_qlike={r['qlike_t']:+.2f}" for _, r in sub.iterrows())
        print(f"    {mark:<6} {fam:<44} {detail}")

    if not winners:
        print("    -> nothing clears the bar; shipping core only")
        return [], v
    # Among the sets that clear it, take the one with the largest mean skill delta.
    best = (v[v["families"].isin(winners)]
            .groupby("families")["skill_delta"].mean().idxmax())
    combo = list(v[v["families"] == best]["combo"].iloc[0])
    print(f"    -> chosen for {panel}: core + {combo}")
    return combo, v


def main() -> None:
    config.ensure_dirs()
    all_rows, all_verdicts, chosen = [], [], {}
    for panel in ("market", "sector"):
        out = select(panel)
        combo, verdicts = choose(out, panel)
        chosen[panel] = combo
        all_rows.append(out.drop(columns=[c for c in out.columns if c.startswith("_")]))
        all_verdicts.append(verdicts)

    pd.concat(all_rows, ignore_index=True).to_csv(
        config.REPORT_DIR / "family_selection.csv", index=False)
    pd.concat(all_verdicts, ignore_index=True).to_csv(
        config.REPORT_DIR / "family_verdicts.csv", index=False)
    print("\nSet these in config.SELECTED_FAMILIES:")
    for k, v in chosen.items():
        print(f'    "{k}": {v},')


if __name__ == "__main__":
    main()
