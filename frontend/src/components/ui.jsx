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

export function Card({ title, sub, action, className = "", children, ...rest }) {
  const cls = `card ${className.replace(/card/, "")}`.trim();
  return (
    <div className={cls} {...rest}>
      {(title || action) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <p className="sub">{sub}</p>}
          </div>
          {action}
        </div>
      )}
      {!title && !action && sub && <p className="sub">{sub}</p>}
      {children}
    </div>
  );
}

export function Logo({ size = 34 }) {
  return (
    <svg className="logo" viewBox="0 0 34 34" width={size} height={size} aria-hidden="true">
      <defs>
        <linearGradient id="lg-sweep" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#5eb1ff" stopOpacity="0" />
          <stop offset="1" stopColor="#5eb1ff" stopOpacity=".55" />
        </linearGradient>
      </defs>
      <circle cx="17" cy="17" r="15" fill="none" stroke="#5eb1ff" strokeOpacity=".55" />
      <circle cx="17" cy="17" r="9.5" fill="none" stroke="#5eb1ff" strokeOpacity=".3" />
      <circle cx="17" cy="17" r="4" fill="none" stroke="#5eb1ff" strokeOpacity=".25" />
      <g className="sweep">
        <path d="M17 17 L32 17 A15 15 0 0 0 28.5 7.4 Z" fill="url(#lg-sweep)" />
        <line x1="17" y1="17" x2="32" y2="17" stroke="#8b7cff" strokeWidth="1.6" strokeLinecap="round" />
      </g>
      <circle cx="24" cy="11" r="1.8" fill="#f2b84b" />
      <circle cx="10" cy="21" r="1.5" fill="#34d399" />
    </svg>
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

export function State({ children, loading = false }) {
  return (
    <div className="state">
      {loading && <div className="loader"><Logo size={56} /></div>}
      {children}
    </div>
  );
}
