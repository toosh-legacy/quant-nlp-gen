import React from "react";
import { Card, Chip } from "./ui";
import { Gauge } from "./Gauge";
import { Direction } from "./icons";
import { featureBlurb, featureLabel, pct, REGIME_COPY, signed } from "../format";

/**
 * The headline: where market volatility is forecast to go over the next h sessions.
 * The number sits next to its own recent record, so it is never read without context.
 */
export function MarketCard({ f, horizon, history }) {
  const rising = f.direction === "rising";
  const recent = (history?.series || []).slice(-125);
  return (
    <Card
      className="feature"
      title="Market volatility forecast"
      sub={`S&P 500 (SPY) · next ${horizon} trading days · as of ${f.as_of}`}
    >
      <div className="headline">
        <span className="big">{f.forecast_vol_ann.toFixed(1)}</span>
        <span className="unit">% annualized</span>
        <span className={`delta ${rising ? "up" : "down"}`}>
          <Direction rising={rising} size={13} /> {signed(f.change_pct)}%
        </span>
        <Chip regime={f.regime} />
      </div>

      <Gauge bands={f.regime_bands_ann} current={f.current_vol_ann} forecast={f.forecast_vol_ann}
             lo={f.forecast_lo_ann} hi={f.forecast_hi_ann} />

      <div className="kv"><span className="k">Realized now</span>
        <span className="v">{pct(f.current_vol_ann)} <span className="dim">{f.current_regime}</span></span></div>
      <div className="kv"><span className="k">Likely range (80%)</span>
        <span className="v">{f.forecast_lo_ann != null ? `${f.forecast_lo_ann.toFixed(1)} – ${f.forecast_hi_ann.toFixed(1)}%` : "—"}</span></div>
      <div className="kv"><span className="k">That band's measured hit rate</span>
        <span className="v">{f.measured_coverage != null ? `${(f.measured_coverage * 100).toFixed(0)}% out-of-sample` : "—"}</span></div>

      {recent.length > 1 && <RecentRecord series={recent} />}

      <p className="from-line" style={{ marginTop: 12 }}>
        Volatility is forecast to <strong>{rising ? "rise" : "fall"}</strong> over the next{" "}
        {horizon} sessions. {REGIME_COPY[f.regime]}
      </p>
    </Card>
  );
}

/** The last ~6 months of out-of-sample forecasts against outcomes, drawn small. */
function RecentRecord({ series, width = 520, height = 64 }) {
  const vals = series.flatMap((d) => [d.forecast, d.realized]).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i) => (i / (series.length - 1)) * width;
  const y = (v) => height - 4 - ((v - lo) / (hi - lo || 1)) * (height - 8);
  const path = (k) => series.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(d[k]).toFixed(1)}`).join("");
  return (
    <figure className="recent">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" width="100%" height={height}
           role="img" aria-label="Recent out-of-sample forecasts versus realized volatility">
        <path d={path("realized")} fill="none" stroke="var(--muted)" strokeOpacity=".7" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
        <path d={path("forecast")} fill="none" stroke="var(--accent)" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption className="dim">
        Last {series.length} sessions out-of-sample: <span style={{ color: "var(--accent)" }}>forecast</span>{" "}
        vs <span className="muted">realized</span>, {series[0].date} to {series.at(-1).date}
      </figcaption>
    </figure>
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
