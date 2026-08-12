import React from "react";
import { Card, Chip, RegimeMeter } from "./ui";
import { featureBlurb, featureLabel, pct, REGIME_COPY, signed } from "../format";

/** The headline: where market volatility is forecast to go over the next h sessions. */
export function MarketCard({ f, horizon }) {
  const rising = f.direction === "rising";
  return (
    <Card
      title="Market volatility forecast"
      sub={`S&P 500 (SPY) · next ${horizon} trading days · as of ${f.as_of}`}
    >
      <div className="headline">
        <span className="big">{f.forecast_vol_ann.toFixed(1)}</span>
        <span className="unit">% annualized</span>
        <span className={`delta ${rising ? "up" : "down"}`}>
          {rising ? "▲" : "▼"} {signed(f.change_pct)}%
        </span>
        <Chip regime={f.regime} />
      </div>

      {f.forecast_lo_ann != null && (
        <p className="from-line" style={{ marginTop: 10 }}>
          Likely range{" "}
          <strong className="mono">
            {pct(f.forecast_lo_ann)} – {pct(f.forecast_hi_ann)}
          </strong>
          {f.measured_coverage != null && (
            <span className="muted">
              {" "}
              · this band has contained the outcome{" "}
              <strong>{(f.measured_coverage * 100).toFixed(0)}%</strong> of the time
              out-of-sample
            </span>
          )}
        </p>
      )}

      <p className="from-line">
        Currently running at <strong className="mono">{pct(f.current_vol_ann)}</strong> (
        {f.current_regime}) — the forecast is for volatility to{" "}
        <strong>{rising ? "rise" : "fall"}</strong> over the next {horizon} sessions.
      </p>
      <p className="from-line" style={{ marginTop: 4 }}>
        {REGIME_COPY[f.regime]}
      </p>

      <RegimeMeter
        bands={f.regime_bands_ann}
        current={f.current_vol_ann}
        forecast={f.forecast_vol_ann}
      />
    </Card>
  );
}

/**
 * What is pushing today's forecast. Contributions come from the tree ensemble's exact
 * per-feature decomposition and are shown in annualized-vol points, so "the VIX term
 * structure is adding 1.4 points" is literally true rather than a rough attribution.
 */
export function DriversCard({ drivers }) {
  if (!drivers || drivers.length === 0) {
    return (
      <Card title="What's driving the forecast" sub="Per-feature contribution">
        <p className="muted">Driver decomposition unavailable for this model.</p>
      </Card>
    );
  }
  const max = Math.max(...drivers.map((d) => Math.abs(d.contribution_vol_pts)), 0.01);

  return (
    <Card
      title="What's driving the forecast"
      sub="Each input's exact contribution, in annualized volatility points"
    >
      {drivers.map((d) => {
        const v = d.contribution_vol_pts;
        const up = v >= 0;
        return (
          <div className="driver" key={d.feature}>
            <div className="name">
              {featureLabel(d.feature)}
              <small>{featureBlurb(d.feature)}</small>
            </div>
            <div className={`amt ${up ? "bad-txt" : "good-txt"}`}>
              {signed(v, 2)} pts
            </div>
            <div className="track">
              <div
                className="fill"
                style={{
                  width: `${(Math.abs(v) / max) * 100}%`,
                  right: up ? "auto" : 0,
                  left: up ? 0 : "auto",
                  background: up ? "var(--stressed)" : "var(--calm)",
                }}
              />
            </div>
          </div>
        );
      })}
      <p className="sub" style={{ margin: "14px 0 0" }}>
        Positive pushes the forecast up (more volatility expected), negative pushes it down.
      </p>
    </Card>
  );
}
