import React from "react";
import { pct } from "../format";

export function Chip({ regime }) {
  return (
    <span className={`chip ${regime}`}>
      <span className="dot" />
      {regime}
    </span>
  );
}

export function Card({ title, sub, children, ...rest }) {
  return (
    <div className="card" {...rest}>
      {title && <h2>{title}</h2>}
      {sub && <p className="sub">{sub}</p>}
      {children}
    </div>
  );
}

/**
 * The regime meter. Bands come from the API (frozen on training data), not from the
 * frontend, so what the user sees is exactly the rule the backtest scored.
 */
export function RegimeMeter({ bands, current, forecast }) {
  if (!bands) return null;
  const { calm_max, normal_max } = bands;
  const max = Math.max(normal_max * 1.9, forecast * 1.15, current * 1.15);
  const at = (v) => `${Math.min(100, Math.max(0, (v / max) * 100))}%`;

  return (
    <div className="meter">
      <div className="meter-track">
        <div className="meter-seg calm" style={{ width: at(calm_max) }} />
        <div
          className="meter-seg normal"
          style={{ width: `${((normal_max - calm_max) / max) * 100}%` }}
        />
        <div className="meter-seg stressed" style={{ flex: 1 }} />
      </div>
      <div className="meter-marks">
        <span className="meter-pin" style={{ left: at(forecast) }} title="forecast" />
        <span className="meter-mark" style={{ left: at(calm_max) }}>
          {pct(calm_max, 0)}
        </span>
        <span className="meter-mark" style={{ left: at(normal_max) }}>
          {pct(normal_max, 0)}
        </span>
      </div>
      <div className="meter-legend">
        <span style={{ color: "var(--calm)" }}>calm</span>
        <span style={{ color: "var(--normal)" }}>normal</span>
        <span style={{ color: "var(--stressed)" }}>stressed</span>
        <span style={{ marginLeft: "auto" }}>▎forecast</span>
      </div>
    </div>
  );
}

export function State({ children }) {
  return <div className="state">{children}</div>;
}
