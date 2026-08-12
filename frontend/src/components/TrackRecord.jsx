import React from "react";
import { Card } from "./ui";
import { num, pct01, signed } from "../format";

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
  const conformal = [
    ...(perf.conformal_market || []),
    ...(perf.conformal_sector || []),
  ];

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
          walk-forward it beat the persistence baseline in only 8 of 13 years, with a
          catastrophic miss in the ultra-calm 2023 market, and the simpler HAR model was no
          more reliable. Every model tested extrapolates a volatility floor that very quiet
          years fall through. The measurement is kept; the number is not shipped.
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

      <h3 className="section-title">Methods that were tried</h3>
      <p className="section-note">
        Each candidate is a different <em>method</em> — a different training set, loss
        function, feature block or combination rule — scored on identical pre-test folds and
        compared to the incumbent fold by fold. Paired comparison matters here: judging two
        separate averages lets an <em>unstable</em> incumbent raise its own replacement bar,
        which is backwards.
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
                  <th className="num">Adopted</th>
                </tr>
              </thead>
              <tbody>
                {bakeoff.filter((b) => b.horizon === 10).map((b, i) => (
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
