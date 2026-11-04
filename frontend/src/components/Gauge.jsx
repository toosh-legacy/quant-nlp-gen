import React from "react";
import { C } from "../theme";

const W = 300, H = 178, CX = 150, CY = 160, R = 124, T = 16;

const polar = (frac, r = R) => {
  const a = Math.PI * (1 - frac);           // 0 -> left, 1 -> right
  return [CX + r * Math.cos(a), CY - r * Math.sin(a)];
};
const arc = (f0, f1, r = R) => {
  const [x0, y0] = polar(f0, r), [x1, y1] = polar(f1, r);
  return `M${x0} ${y0} A${r} ${r} 0 ${f1 - f0 > 0.5 ? 1 : 0} 1 ${x1} ${y1}`;
};

/**
 * Semicircular gauge of market volatility on the frozen regime bands.
 * Bright needle = forecast, thin marker = realized vol today, faint arc = 80% interval.
 */
export function Gauge({ bands, current, forecast, lo, hi }) {
  if (!bands) return null;
  const max = Math.max(bands.normal_max * 1.8, forecast * 1.15, current * 1.15, hi ? hi * 1.05 : 0);
  const f = (v) => Math.min(1, Math.max(0, v / max));
  const angle = (v) => -90 + 180 * f(v);
  const [cx, cy] = polar(f(current), R + T / 2 + 5);
  const [cx2, cy2] = polar(f(current), R - T / 2 - 5);

  return (
    <div className="gauge-wrap">
      <svg className="gauge" viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 360 }}
           role="img" aria-label={`Forecast ${forecast.toFixed(1)} percent`}>
        <path d={arc(0, 1)} stroke={C.panel2} strokeWidth={T} fill="none" strokeLinecap="round" />
        <path d={arc(0, f(bands.calm_max))} stroke={C.calm} strokeOpacity=".75" strokeWidth={T} fill="none" />
        <path d={arc(f(bands.calm_max), f(bands.normal_max))} stroke={C.normal} strokeOpacity=".75" strokeWidth={T} fill="none" />
        <path d={arc(f(bands.normal_max), 1)} stroke={C.stressed} strokeOpacity=".75" strokeWidth={T} fill="none" />
        {lo != null && hi != null && (
          <path d={arc(f(lo), f(hi), R - T - 8)} stroke={C.accent} strokeOpacity=".55"
                strokeWidth="5" fill="none" strokeLinecap="round" />
        )}
        <line x1={cx} y1={cy} x2={cx2} y2={cy2} stroke={C.text} strokeWidth="2.5" strokeLinecap="round" />

        <g transform={`rotate(${angle(forecast).toFixed(2)} ${CX} ${CY})`}>
          {/* No SVG filter here: a vertical line has a zero-width bounding box, so an
              objectBoundingBox filter region collapses and the needle vanishes. */}
          <line x1={CX} y1={CY} x2={CX} y2={CY - R + 14} stroke={C.accent} strokeOpacity=".35"
                strokeWidth="8" strokeLinecap="round" />
          <line x1={CX} y1={CY} x2={CX} y2={CY - R + 14} stroke="#fff" strokeWidth="3"
                strokeLinecap="round" />
        </g>
        <circle cx={CX} cy={CY} r="7" fill={C.panel2} stroke="#fff" strokeWidth="2" />

        {[bands.calm_max, bands.normal_max].map((v) => {
          const [x, y] = polar(f(v), R + T + 4);
          return <text key={v} x={x} y={y} fill={C.dim} fontSize="10" textAnchor="middle">{v.toFixed(0)}%</text>;
        })}
        <text x={CX - R} y={CY + 16} fill={C.dim} fontSize="10" textAnchor="middle">0</text>
        <text x={CX + R} y={CY + 16} fill={C.dim} fontSize="10" textAnchor="middle">{max.toFixed(0)}%</text>
      </svg>
    </div>
  );
}
