import React from "react";
import { Card } from "./ui";
import { foldsPhrase, num, pct01, signed } from "../format";

// Which bake-off candidates were actually adopted into the shipped pipeline. Kept explicit
// so the table cannot drift out of step with config.TRAINING_SOURCE.
const ADOPTED = { "sector:pooled": true };

const Row = ({ k, v, good }) => (
  <div className="kv">
    <span className="k">{k}</span>
    <span className={`v ${good === true ? "good-txt" : good === false ? "bad-txt" : ""}`}>{v}</span>
  </div>
);

/**
 * Everything that qualifies the forecast, in one place.
 *
 * A dashboard that shows a prediction without its error rate is marketing. This tab is
 * deliberately as prominent as the forecast itself, and it leads with the results that are
 * least flattering — the horizon that was withheld, the events that were missed.
 */
export function TrackRecord({ perf }) {
  const pooled = (perf.pooled || []).filter((r) => r.served);
  const stability = perf.stability || [];
  const events = perf.events || [];
  const sweep = perf.events_sweep || [];
  const decision = perf.decision || [];
  const ablation = perf.ablation || [];

  const mkt = pooled.find((r) => r.panel === "market" && r.horizon === 10);
  const sec = pooled.find((r) => r.panel === "sector" && r.horizon === 10);
  const model = decision.find((d) => d.strategy === "model_sized");
  const persist = decision.find((d) => d.strategy === "persistence_sized");
  const sig = (perf.decision_significance || [])[0];
  const detected = events.filter((e) => e.detected).length;
  const bakeoff = perf.method_bakeoff || [];
  const alarm = perf.alarm || [];
  const settled = perf.settled || [];
  const intraday = perf.intraday_study || [];
  const conformal = [
    ...(perf.conformal_market || []),
    ...(perf.conformal_sector || []),
  ];
  const sim = perf.simulation || [];
  const byExp = (name) => sim.find((s) => s.experiment === name);
  const ceiling = byExp("ceiling");
  const mvt = byExp("model_vs_truth");
  const nullExp = byExp("null_no_predictability");
  const shuffle = byExp("shuffled_target");
  const scenarios = perf.simulation_scenarios || [];

  return (
    <>
      <h3 className="section-title">How accurate is it, really?</h3>
      <p className="section-note">
        Every number below comes from walk-forward testing: a model is retrained at the
        start of each year on data available up to that point, then scored on the year that
        follows. No model was ever shown the data it is being graded on.
      </p>

      <div className="grid three">
        {[["Market (SPY)", mkt], ["Sectors", sec]].map(([label, r]) =>
          r ? (
            <Card key={label} title={label} sub={`${r.n_oos.toLocaleString()} out-of-sample days · ${r.oos_start} to ${r.oos_end}`}>
              <Row k="Variance explained (R²)" v={num(r.r2)} good={r.r2 > 0.3} />
              <Row k="Persistence baseline R²" v={num(r.r2_persistence)} />
              <Row k="Error removed vs baseline" v={pct01(r.skill_vs_persistence)} good={r.skill_vs_persistence > 0} />
              <Row k="Rise/fall accuracy" v={pct01(r.dir_accuracy)} good={r.dir_accuracy > r.dir_baseline} />
              <Row k="  vs always-guessing" v={pct01(r.dir_baseline)} />
              <Row k="Rise/fall MCC" v={num(r.dir_mcc)} good={r.dir_mcc > 0.2} />
              <Row k="Regime accuracy (3-class)" v={pct01(r.regime_accuracy)} good={r.regime_accuracy > r.regime_baseline} />
              <Row k="  vs always-guessing" v={pct01(r.regime_baseline)} />
            </Card>
          ) : null
        )}

        <Card title="Year-by-year stability" sub="Error removed vs the persistence baseline, per year">
          {stability
            .filter((s) => perf.horizons_served.includes(s.horizon))
            .map((s) => (
              <Row
                key={`${s.panel}-${s.horizon}`}
                k={`${s.panel} · ${s.horizon}d`}
                v={`${signed(s.mean * 100, 0)}% · ${s.years_positive}/${s.n} yrs`}
                good={s.years_positive === s.n}
              />
            ))}
          <p className="sub" style={{ margin: "14px 0 0" }}>
            This is the metric that decides whether an edge is real. Both the model and the
            baseline are scored on the same window, so it cannot be flattered by a calm year.
          </p>
        </Card>
      </div>

      {perf.horizons_withheld?.length > 0 && (
        <div className="callout" style={{ marginTop: 16 }}>
          <strong>What we withheld.</strong> A {perf.horizons_withheld.join(", ")}-day horizon
          was built and tested but is <strong>not shown</strong> in the dashboard. In
          walk-forward it beat the persistence baseline in only{" "}
          {foldsPhrase(perf, "market", perf.horizons_withheld[0]) || "a minority of"} years on
          the market panel, with a large miss in an ultra-calm year, and the simpler HAR model
          was no more reliable. Every model tested extrapolates a volatility floor that very
          quiet years fall through. The sector panel at that horizon is stable
          {foldsPhrase(perf, "sector", perf.horizons_withheld[0])
            ? ` (${foldsPhrase(perf, "sector", perf.horizons_withheld[0])} years)`
            : ""}
          , but the two are shown together at one horizon and the market half is the half that
          fails. The measurement is kept; the number is not shipped.
        </div>
      )}

      <h3 className="section-title">Did it flag real volatility spikes?</h3>
      <p className="section-note">
        Stress episodes were located automatically in the out-of-sample period, then we
        checked whether the model had raised a stressed-regime alarm in the 60 sessions
        before each one began. Named events are attached afterwards for readability — the
        detection itself is computed from the data.
      </p>

      <div className="grid two">
        <Card title="Event detection" sub={`${detected} of ${events.length} episodes flagged in advance`}>
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Episode</th>
                  <th className="num">Peak vol</th>
                  <th>Warned?</th>
                  <th className="num">Lead</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e, i) => (
                  <tr key={i}>
                    <td>
                      {e.event}
                      <span className="muted mono" style={{ fontSize: 11, marginLeft: 6 }}>
                        {e.onset}
                      </span>
                    </td>
                    <td className="num">{num(e.peak_vol_ann, 0)}%</td>
                    <td className={e.detected ? "good-txt" : "bad-txt"}>
                      {e.detected ? "yes" : "missed"}
                    </td>
                    <td className="num muted">{e.lead_days != null ? `${e.lead_days}d` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="callout" style={{ marginTop: 14 }}>
            <strong>This is the weakest part of the product, and the honest framing
            matters.</strong> At the pre-set alarm threshold the model warned ahead of only{" "}
            {events.length ? Math.round((detected / events.length) * 100) : 0}% of stress
            episodes. It forecasts the volatility <em>level</em> accurately, but it does not
            anticipate exogenous shocks — nothing in market data foreshadowed COVID-19 or the
            2024 yen carry unwind. There is also a structural reason it warns late: a
            squared-error model shrinks its predictions toward the mean, so the forecast
            rarely reaches the stressed band until a spike is already forming. Treat this as a
            gauge of current risk conditions, <strong>not a crash predictor</strong>.
          </div>
        </Card>

        <Card title="Earlier warnings cost more false alarms" sub="The full tradeoff, at every possible alarm threshold">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th className="num">Alarm at</th>
                  <th className="num">Detected</th>
                  <th className="num">Median lead</th>
                  <th className="num">Days on alarm</th>
                  <th className="num">False alarms</th>
                </tr>
              </thead>
              <tbody>
                {sweep.map((s, i) => (
                  <tr key={i}>
                    <td className="num">{pct01(s.alarm_percentile)}</td>
                    <td className="num good-txt">{pct01(s.detection_rate)}</td>
                    <td className="num muted">{num(s.median_lead_days, 0)}d</td>
                    <td className="num muted">{pct01(s.alarm_share_of_days)}</td>
                    <td className="num bad-txt">{pct01(s.false_alarm_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ margin: "14px 0 0" }}>
            Showing every threshold rather than one chosen point. A model set to warn
            constantly catches everything and is worthless — the cost is in the last column.
          </p>
        </Card>
      </div>

      <h3 className="section-title">Testing the test: simulated markets with a known answer</h3>
      <p className="section-note">
        On real data, true volatility is never observed — every metric above is scored
        against <em>realized</em> volatility, which is only a noisy estimate of it. So the
        whole pipeline is also run on synthetic markets generated from a written-down
        process, where the true volatility is known exactly. That answers questions real
        data cannot.
      </p>

      <div className="grid three">
        <Card title="The accuracy ceiling" sub="How well could a PERFECT forecaster score?">
          {ceiling && (
            <>
              <Row k="Perfect forecast, R² vs proxy" v={num(ceiling.perfect_r2_vs_proxy)} />
              <Row k="Persistence baseline" v={num(ceiling.persistence_r2)} />
              <p className="sub" style={{ margin: "14px 0 0" }}>
                Even perfect knowledge of future volatility only scores{" "}
                <strong>{num(ceiling.perfect_r2_vs_proxy)}</strong> against the noisy
                yardstick — not 1.0. That is the number our real-data R² should be read
                against, and it reframes 0.42 considerably.
              </p>
            </>
          )}
        </Card>

        <Card title="Our error vs the yardstick's" sub="Same model, scored both ways">
          {mvt && (
            <>
              <Row k="Scored vs the noisy proxy" v={num(mvt.r2_vs_proxy)} />
              <Row k="Scored vs TRUE volatility" v={num(mvt.r2_vs_truth)} />
              {ceiling && (
                <Row
                  k="Share of attainable captured"
                  v={pct01(mvt.r2_vs_proxy / ceiling.perfect_r2_vs_proxy)}
                  good
                />
              )}
              <p className="sub" style={{ margin: "14px 0 0" }}>
                About half the apparent error is noise in the measuring stick rather than a
                failure of the model.
              </p>
            </>
          )}
        </Card>

        <Card title="Does the harness lie?" sub="Two placebo tests it has to fail to pass">
          {nullExp && (
            <>
              <Row k="No-signal market, R²" v={num(nullExp.r2_mean)} good={nullExp.r2_mean <= 0.02} />
              <Row k="  same market, 'skill'" v={num(nullExp.skill_mean)} good={false} />
            </>
          )}
          {shuffle && (
            <Row k="Shuffled target, R²" v={num(shuffle.r2_mean)} good={shuffle.r2_mean <= 0.02} />
          )}
          <div className="callout" style={{ marginTop: 14 }}>
            <strong>This changed how we read our own metrics.</strong> In a market built to
            have <em>no</em> forecastable volatility, R² correctly reads negative — but
            "skill vs persistence" still reads about <strong>+0.43</strong>. Both are right:
            with constant volatility the persistence baseline is pure measurement noise, so
            predicting the average beats it without forecasting anything. So skill vs
            persistence <strong>cannot establish predictability on its own</strong> — it has
            to be read next to R². The real results clear both bars.
          </div>
        </Card>
      </div>

      {scenarios.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <Card
            title="Monte Carlo: would volatility targeting have paid, across many markets?"
            sub="The real backtest is one sample; this runs the same strategy over simulated markets from the same process"
          >
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Sizing rule</th>
                    <th className="num">Mean Sharpe</th>
                    <th className="num">Gap vs measurement</th>
                    <th className="num">95% range</th>
                    <th className="num">Win rate</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarios.map((s) => (
                    <tr key={s.arm} className={s.arm === "oracle_sized" ? "on" : ""}>
                      <td>
                        {{
                          buy_and_hold: "Buy & hold",
                          persistence_sized: "Sized on current vol",
                          model_sized: "Sized on forecast",
                          oracle_sized: "Sized on PERFECT foresight",
                        }[s.arm] || s.arm}
                      </td>
                      <td className="num">{num(s.sharpe_mean)}</td>
                      <td className={`num ${s.sharpe_gap_vs_persistence > 0 ? "good-txt" : "muted"}`}>
                        {signed(s.sharpe_gap_vs_persistence, 3)}
                      </td>
                      <td className="num muted">
                        [{signed(s.gap_ci_low, 2)}, {signed(s.gap_ci_high, 2)}]
                      </td>
                      <td className="num">{pct01(s.win_rate_vs_persistence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout" style={{ marginTop: 14 }}>
              <strong>The ceiling is the forecast, not the strategy.</strong> Perfect
              foresight would have earned about <strong>+0.97</strong> Sharpe over sizing on
              current volatility, and won in 100% of simulated markets — so volatility
              targeting itself works very well when the input is good. Our forecast captured
              only a few percent of that, winning 67% of markets with a range that still
              spans zero — which matches the real backtest almost exactly. The honest reading:
              the model is statistically accurate but not yet <em>sharp</em> enough for the
              gain to show up reliably in money. That is a headroom result, not a dead end.
            </div>
          </Card>
        </div>
      )}

      <h3 className="section-title">Methods that were tried</h3>
      <p className="section-note">
        Each candidate is a different <em>method</em> — a different training set, loss
        function, feature block or combination rule — scored on identical pre-test folds and
        compared to the incumbent fold by fold. Paired comparison matters here: judging two
        separate averages lets an <em>unstable</em> incumbent raise its own replacement bar,
        which is backwards. A method is adopted only if it clears t&nbsp;&gt;&nbsp;2 on{" "}
        <em>both</em> skill and QLIKE, at <em>both</em> an annual and a semi-annual fold
        slicing — a result that appears at one calendar cut and vanishes at another is not a
        result.
      </p>
      <p className="section-note">
        The <strong>In MCS</strong> column is the more careful statement. Rather than crowning
        the lowest loss in a table whose top entries sit within noise of each other, the Model
        Confidence Set reports which methods cannot be <em>statistically separated</em> from
        the best one. Everything marked “ruled out” is genuinely ruled out; everything else is
        a tie the data cannot break.
      </p>

      <div className="grid two">
        <Card title="Method bake-off" sub="Skill vs persistence on pre-test folds; adopted only if it beat the incumbent consistently">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th className="num">Panel</th>
                  <th className="num">Skill</th>
                  <th className="num">QLIKE gain</th>
                  <th className="num" title="Hansen-Lunde-Nason Model Confidence Set: methods that cannot be statistically separated from the best one, under QLIKE">
                    In MCS
                  </th>
                  <th className="num">Adopted</th>
                </tr>
              </thead>
              <tbody>
                {bakeoff.filter((b) => b.horizon === 10 && (b.granularity ?? "annual") === "annual").map((b, i) => (
                  <tr key={i} className={ADOPTED[`${b.panel}:${b.method}`] ? "on" : ""}>
                    <td>
                      {b.method}
                      <small style={{ display: "block", color: "var(--dim)", fontSize: 11.5 }}>
                        {b.note}
                      </small>
                    </td>
                    <td className="num muted">{b.panel}</td>
                    <td className="num">{num(b.skill_mean, 3)}</td>
                    <td className="num muted">{num(b.qlike_gain, 3)}</td>
                    <td className="num">
                      {b.in_mcs === undefined || b.in_mcs === null
                        ? <span className="muted">—</span>
                        : b.in_mcs
                        ? <span className="good-txt">yes</span>
                        : <span className="muted">ruled out</span>}
                    </td>
                    <td className="num">
                      {b.method === "incumbent"
                        ? "—"
                        : ADOPTED[`${b.panel}:${b.method}`]
                        ? <span className="good-txt">yes</span>
                        : <span className="muted">no</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ margin: "14px 0 0" }}>
            Only <strong>pooled training</strong> on the sector panel was adopted — fitting on
            all 91 series (471k rows instead of 37k) rather than each sector alone. Jump
            features looked significant on annual folds and vanished on semi-annual ones, so
            they were rejected: a result that depends on how you slice the calendar is not a
            result.
          </p>
        </Card>

        <Card title="A dedicated alarm head" sub="Can a model built for stress beat thresholding the mean forecast?">
          <p className="sub" style={{ marginTop: 0 }}>
            The mean forecast is accurate about the volatility <em>level</em> and warns late
            about spikes, for a structural reason: squared error estimates a conditional mean,
            and the mean of a right-skewed variable sits below the outcomes that matter. So a
            separate head was built with its own binary target and its own metric — MCC on
            “will the next 10 days be stressed?”, never R², because a quantile forecast does
            not estimate a mean and recall alone is gamed by alarming every day.
          </p>
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Rule</th>
                  <th className="num">Panel</th>
                  <th className="num">MCC</th>
                  <th className="num">Stress recall</th>
                  <th className="num">Precision</th>
                  <th className="num">t</th>
                </tr>
              </thead>
              <tbody>
                {alarm.filter((a) => a.granularity === "annual").map((a, i) => (
                  <tr key={i}>
                    <td>{a.rule}</td>
                    <td className="num muted">{a.panel}</td>
                    <td className="num">{num(a.alarm_mcc, 3)}</td>
                    <td className="num">{pct01(a.stressed_recall)}</td>
                    <td className="num muted">{pct01(a.precision)}</td>
                    <td className="num muted">
                      {a.testable === false ? "untested" : num(a.t, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="callout" style={{ marginTop: 14 }}>
            <strong>It is not shipped, and “not established” is not the same as “does not
            work”.</strong> On sectors the high-quantile rule catches far more stress than the
            rule in production, and the paired advantage is stable — but it clears the bar at
            only the finest of three fold slicings, and the market panel contradicts it
            outright. Rows marked <em>untested</em> are folds with too little stress in them to
            rank anything; that is a missing measurement, not a failed one.
          </div>
        </Card>

        <Card title="Open questions, settled" sub="Decided on pre-test folds, at two fold granularities">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th className="num">Panel</th>
                  <th className="num">Metric</th>
                  <th className="num">Folds</th>
                  <th className="num">t</th>
                </tr>
              </thead>
              <tbody>
                {settled.map((r, i) => (
                  <tr key={i}>
                    <td>{r.model}</td>
                    <td className="num muted">{r.panel}</td>
                    <td className="num muted">{r.metric}</td>
                    <td className="num muted">{r.folds}</td>
                    <td className={`num ${r.t > 2 ? "good-txt" : ""}`}>{num(r.t, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Does intraday data pay?" sub="Scoped study on the Oxford-Man realized library, 2000-2016">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Variant</th>
                  <th className="num">R²</th>
                  <th className="num">Skill</th>
                  <th className="num">Direction</th>
                </tr>
              </thead>
              <tbody>
                {intraday.map((r, i) => (
                  <tr key={i}>
                    <td>{r.variant}</td>
                    <td className="num">{num(r.r2, 3)}</td>
                    <td className="num muted">{num(r.skill, 3)}</td>
                    <td className="num muted">{pct01(r.dir_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ margin: "14px 0 0" }}>
            Intraday data helps as an <strong>input</strong> and hurts as a <strong>target</strong>.
            A 10-day average of daily variances has already removed most of the measurement
            noise a 5-minute estimator would remove, so the precision buys little at this
            horizon. Buying it is not justified for this product.
          </p>
        </Card>

        <Card title="Are the uncertainty bands honest?" sub="Nominal 80% range vs the coverage actually achieved out-of-sample">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Panel</th>
                  <th className="num">Method</th>
                  <th className="num">Nominal</th>
                  <th className="num">Realized</th>
                  <th className="num">Median width</th>
                </tr>
              </thead>
              <tbody>
                {conformal.map((c, i) => (
                  <React.Fragment key={i}>
                    <tr>
                      <td>{c.panel}</td>
                      <td className="num muted">static</td>
                      <td className="num muted">{pct01(c.nominal_coverage)}</td>
                      <td className="num">{pct01(c.static_realized_coverage)}</td>
                      <td className="num muted">{num(c.static_median_width_vol_pts, 1)} pts</td>
                    </tr>
                    <tr className="on">
                      <td />
                      <td className="num">adaptive</td>
                      <td className="num muted">{pct01(c.nominal_coverage)}</td>
                      <td className="num good-txt">{pct01(c.adaptive_realized_coverage)}</td>
                      <td className="num muted">{num(c.adaptive_median_width_vol_pts, 1)} pts</td>
                    </tr>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ margin: "14px 0 0" }}>
            A standard conformal interval assumes the data is exchangeable. Markets are not,
            and it showed: the static band promised 80% and delivered 73% on the market panel.
            The shipped band steers its own width from realized misses, which brings coverage
            back near target. The bands you see are the ones that were measured — not the
            ones that were promised.
          </p>
        </Card>
      </div>

      <h3 className="section-title">Was it useful?</h3>
      <p className="section-note">
        Volatility targeting: scale exposure so the portfolio runs at a constant 15% risk
        level rather than a constant dollar amount. The real test is beating the amber arm,
        which already uses a volatility estimate — it just measures instead of forecasting.
      </p>

      <div className="grid two">
        <Card title="Risk-adjusted outcome" sub="2012–2024 out-of-sample, 1bp cost per trade">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Sizing rule</th>
                  <th className="num">Return</th>
                  <th className="num">Volatility</th>
                  <th className="num">Sharpe</th>
                  <th className="num">Max drawdown</th>
                  <th className="num" title="Distance from the 15% risk target">Target miss</th>
                </tr>
              </thead>
              <tbody>
                {decision.map((d) => (
                  <tr key={d.strategy} className={d.strategy === "model_sized" ? "on" : ""}>
                    <td>
                      {d.strategy === "model_sized" ? "Sized on forecast" :
                        d.strategy === "persistence_sized" ? "Sized on current vol" : "Buy & hold"}
                    </td>
                    <td className="num">{pct01(d.ann_return)}</td>
                    <td className="num">{pct01(d.ann_vol)}</td>
                    <td className="num">{num(d.sharpe)}</td>
                    <td className="num">{pct01(d.max_drawdown)}</td>
                    <td className="num">{num(d.vol_miss)} pts</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {model && persist && sig && (
            <div className={`callout ${sig.significant ? "good" : ""}`} style={{ marginTop: 14 }}>
              <strong>
                {sig.significant
                  ? "The trading benefit is statistically established."
                  : "The trading benefit is not statistically established."}
              </strong>{" "}
              Sizing on the forecast produced a Sharpe of {num(model.sharpe)} against{" "}
              {num(persist.sharpe)} for sizing on current volatility — a gap of{" "}
              {signed(sig.sharpe_gap, 3)}. Resampling the paired daily returns in 21-day
              blocks (which preserves volatility clustering) puts the 95% confidence
              interval at{" "}
              <strong>
                [{signed(sig.ci_low, 3)}, {signed(sig.ci_high, 3)}]
              </strong>
              . {sig.significant
                ? "The interval excludes zero."
                : "The interval spans zero, so on this evidence the forecast did not reliably beat simply measuring current volatility for position sizing — despite forecasting the volatility level itself far more accurately."}
            </div>
          )}
        </Card>

        <Card title="What each factor family was worth" sub="Leave-one-out walk-forward, headline horizon">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Panel</th>
                  <th>Family</th>
                  <th className="num">Contribution to R²</th>
                </tr>
              </thead>
              <tbody>
                {ablation
                  .filter((a) => a.mode === "leave_one_out")
                  .map((a, i) => (
                    <tr key={i}>
                      <td className="muted">{a.panel}</td>
                      <td>{a.family}</td>
                      <td className={`num ${a.delta_r2 > 0 ? "good-txt" : "bad-txt"}`}>
                        {signed(a.delta_r2, 3)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ margin: "14px 0 0" }}>
            Negative means the model was <em>better</em> without that family. Across
            selection folds <strong>no external family reliably beat price history alone</strong> —
            every candidate landed within one standard error of it, so the simplest set was
            kept. The VIX, cross-sectional correlation, rates and credit are each plausible
            predictors, and each turns out to be information the realized volatility already
            carries.
          </p>
        </Card>
      </div>
    </>
  );
}
