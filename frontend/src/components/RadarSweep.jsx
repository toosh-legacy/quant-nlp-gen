import React, { useMemo, useState } from "react";
import { C, pressureColor, regimeColor } from "../theme";
import { pct, signed } from "../format";

const R = 172;          // outer radius in SVG units
const VIEW = 230;       // half-width of the viewBox

/**
 * The volatility radar. One spoke per sector.
 *
 *   distance from centre  forecast volatility (further out = more turbulent)
 *   faint trail           from today's realized vol to the forecast — the expected move
 *   colour                the forecast regime (calm / normal / stressed)
 *   blip size             share of news coverage the sector is getting (if the overlay ran)
 *   pulsing ring          stressed regime, or news and forecast disagree sharply
 *
 * Purely a view of /api/sectors and /api/news — it computes no forecast of its own.
 */
export function RadarSweep({ sectors, news, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const newsBy = useMemo(
    () => Object.fromEntries((news?.sectors || []).map((s) => [s.ticker, s])),
    [news]
  );
  const rows = sectors.ranking;
  const n = rows.length;
  // Scale to the forecasts, not to today's realized vol: the forecasts are what the radar
  // is for, and a single spiking sector would otherwise squash every blip into the middle.
  // A realized value beyond the edge is drawn on the rim.
  const vmax = Math.max(...rows.map((r) => r.forecast_vol_ann)) * 1.2;
  const step = vmax > 40 ? 10 : 5;
  const scale = Math.ceil(vmax / step) * step;
  const rings = [];
  for (let v = step; v <= scale; v += step) rings.push(v);
  const rOf = (v) => (Math.min(Math.max(0, v), scale) / scale) * R;
  const labelAngle = -Math.PI / 2 + Math.PI / n;     // halfway between the first two spokes
  const maxAtt = Math.max(0.01, ...Object.values(newsBy).map((s) => s.attention || 0));

  // Fixed spoke order (alphabetical by ticker) so a sector never jumps around between days.
  const order = [...rows].sort((a, b) => a.ticker.localeCompare(b.ticker));
  const pts = order.map((s, i) => {
    const a = (i / n) * 2 * Math.PI - Math.PI / 2;
    const nw = newsBy[s.ticker];
    const size = nw ? 5 + 9 * Math.sqrt((nw.attention || 0) / maxAtt) : 7;
    const pulse = s.regime === "stressed" || (nw && nw.divergence);
    return {
      s, a, nw, size, pulse,
      x: Math.cos(a) * rOf(s.forecast_vol_ann),
      y: Math.sin(a) * rOf(s.forecast_vol_ann),
      cx: Math.cos(a) * rOf(s.current_vol_ann),
      cy: Math.sin(a) * rOf(s.current_vol_ann),
      lx: Math.cos(a) * (R + 24),
      ly: Math.sin(a) * (R + 24),
    };
  });
  const poly = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const tip = hover != null ? pts[hover] : null;

  return (
    <div className="radar-wrap radar">
      <svg viewBox={`${-VIEW} ${-VIEW} ${VIEW * 2} ${VIEW * 2}`} role="img"
           aria-label="Radar of forecast volatility by sector">
        <defs>
          <radialGradient id="rg-bg">
            <stop offset="0%" stopColor={C.accent} stopOpacity="0.10" />
            <stop offset="70%" stopColor={C.accent} stopOpacity="0.03" />
            <stop offset="100%" stopColor={C.accent} stopOpacity="0" />
          </radialGradient>
          <linearGradient id="rg-sweep" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={C.accent} stopOpacity="0" />
            <stop offset="100%" stopColor={C.accent} stopOpacity="0.32" />
          </linearGradient>
          <filter id="rg-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <circle r={R} fill="url(#rg-bg)" stroke={C.line} />
        {rings.map((v) => (
          <g key={v}>
            <circle r={rOf(v)} fill="none" stroke={C.line} strokeDasharray="2 4" />
            <text x={Math.cos(labelAngle) * rOf(v) + 3} y={Math.sin(labelAngle) * rOf(v) - 2}
                  fill={C.dim} fontSize="9.5">{v}%</text>
          </g>
        ))}
        {pts.map((p) => (
          <line key={p.s.ticker} x1={0} y1={0} x2={Math.cos(p.a) * R} y2={Math.sin(p.a) * R}
                stroke={C.line} strokeOpacity="0.7" />
        ))}

        {/* sweep wedge: 40 degrees, rotating */}
        <g className="sweep">
          <path d={`M0 0 L${R} 0 A${R} ${R} 0 0 0 ${(R * Math.cos(-0.7)).toFixed(1)} ${(R * Math.sin(-0.7)).toFixed(1)} Z`}
                fill="url(#rg-sweep)" />
          <line x1="0" y1="0" x2={R} y2="0" stroke={C.accent} strokeOpacity="0.6" strokeWidth="1.2" />
        </g>

        <polygon points={poly} fill={C.accent} fillOpacity="0.07" stroke={C.accent}
                 strokeOpacity="0.45" strokeWidth="1.2" />

        {pts.map((p) => (
          <line key={`t${p.s.ticker}`} x1={p.cx} y1={p.cy} x2={p.x} y2={p.y}
                stroke={regimeColor(p.s.regime)} strokeOpacity="0.45" strokeWidth="2"
                strokeLinecap="round" strokeDasharray="1 3" />
        ))}
        {pts.map((p) => (
          <circle key={`c${p.s.ticker}`} cx={p.cx} cy={p.cy} r="2.6" fill="none"
                  stroke={C.muted} strokeWidth="1" />
        ))}

        {pts.map((p, i) => {
          const col = regimeColor(p.s.regime);
          const on = selected === p.s.ticker;
          return (
            <g key={`b${p.s.ticker}`} transform={`translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`}>
              {p.pulse && <circle className="blip-ring" r={p.size} fill="none" stroke={col} />}
              <g className={`blip ${on ? "on" : ""}`} role="button" tabIndex={0}
                 aria-label={`${p.s.sector}: forecast ${p.s.forecast_vol_ann.toFixed(1)} percent, ${p.s.regime}`}
                 onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                 onFocus={() => setHover(i)} onBlur={() => setHover(null)}
                 onClick={() => onSelect?.(p.s.ticker)}
                 onKeyDown={(e) => {
                   if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect?.(p.s.ticker); }
                 }}>
                <circle r={p.size + 6} fill="transparent" />
                <circle r={p.size} fill={col} fillOpacity="0.85" filter="url(#rg-glow)"
                        stroke={on ? "#fff" : "none"} strokeWidth="1.5" />
              </g>
            </g>
          );
        })}

        {pts.map((p) => (
          <text key={`l${p.s.ticker}`} x={p.lx} y={p.ly} textAnchor="middle"
                dominantBaseline="middle" fontSize="11" fontWeight="600"
                fill={selected === p.s.ticker ? C.text : C.muted}>
            {p.s.ticker}
          </text>
        ))}
        <circle r="3" fill={C.accent} />
      </svg>

      {tip && (
        <div className="radar-tip"
             style={{ left: `${50 + (tip.x / (VIEW * 2)) * 100}%`,
                      top: `${50 + (tip.y / (VIEW * 2)) * 100}%` }}>
          <b>{tip.s.sector} <span className="dim mono">{tip.s.ticker}</span></b>
          <div className="row"><span>Forecast</span><span>{pct(tip.s.forecast_vol_ann)}</span></div>
          <div className="row"><span>Now</span><span>{pct(tip.s.current_vol_ann)}</span></div>
          <div className="row"><span>Change</span><span>{signed(tip.s.change_pct)}%</span></div>
          <div className="row"><span>Regime</span>
            <span style={{ color: regimeColor(tip.s.regime) }}>{tip.s.regime}</span></div>
          {tip.nw && tip.nw.pressure != null && (
            <>
              <div className="row"><span>News pressure</span>
                <span style={{ color: pressureColor(tip.nw.pressure) }}>
                  {tip.nw.pressure.toFixed(0)}</span></div>
              <div className="row"><span>Stories (72h)</span><span>{tip.nw.n}</span></div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function RadarLegend({ hasNews }) {
  return (
    <div className="legend-row">
      <span><i style={{ background: C.calm }} />calm</span>
      <span><i style={{ background: C.normal }} />normal</span>
      <span><i style={{ background: C.stressed }} />stressed</span>
      <span><i style={{ border: `1px solid ${C.muted}`, background: "transparent" }} />today</span>
      <span>· distance = forecast vol</span>
      {hasNews && <span>· size = news attention</span>}
    </div>
  );
}
