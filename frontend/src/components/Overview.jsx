import React, { useState } from "react";
import { Card, Chip } from "./ui";
import { C, pressureColor, regimeColor } from "../theme";
import { pct, signed, SUPER_SECTORS } from "../format";

const REGIMES = ["calm", "normal", "stressed"];
const open = (t) => { window.location.hash = `#sectors/${t}`; };
const keyOpen = (t) => (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(t); }
};

/* ------------------------------------------------------------------ KPI strip */

/** Four headline numbers, each with a small picture of what it summarises. */
export function KpiStrip({ forecast, sectors, news }) {
  const rows = sectors.ranking;
  const counts = Object.fromEntries(REGIMES.map((r) => [r, rows.filter((s) => s.regime === r).length]));
  const falling = rows.filter((s) => s.change_pct < 0).length;
  const mover = [...rows].sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct))[0];
  const lo = forecast.forecast_lo_ann, hi = forecast.forecast_hi_ann;
  const span = Math.max(hi * 1.15, 30);

  return (
    <section className="kpis" aria-label="Headline numbers">
      <div className="kpi">
        <span className="kpi-label">S&amp;P 500 · next 10 days</span>
        <span className="kpi-value">{forecast.forecast_vol_ann.toFixed(1)}<small>%</small></span>
        <svg className="kpi-viz" viewBox="0 0 200 22" preserveAspectRatio="none" aria-hidden="true">
          <rect x="0" y="9" width="200" height="4" rx="2" fill={C.panel2} />
          <rect x={(lo / span) * 200} y="6" width={((hi - lo) / span) * 200} height="10" rx="5"
                fill={C.accent} fillOpacity=".28" />
          <rect x={(forecast.forecast_vol_ann / span) * 200 - 1.5} y="3" width="3" height="16" rx="1.5"
                fill={C.accent} />
          <rect x={(forecast.current_vol_ann / span) * 200 - 1} y="5" width="2" height="12" rx="1" fill={C.muted} />
        </svg>
        <span className="kpi-foot">80% range {pct(lo)}–{pct(hi)} · now {pct(forecast.current_vol_ann)}</span>
      </div>

      <div className="kpi">
        <span className="kpi-label">Sector regimes</span>
        <span className="kpi-value">{counts.calm}<small> calm</small> · {counts.stressed}<small> stressed</small></span>
        <div className="kpi-stack" aria-hidden="true">
          {REGIMES.map((r) => counts[r] > 0 && (
            <span key={r} style={{ flex: counts[r], background: regimeColor(r) }} />
          ))}
        </div>
        <span className="kpi-foot">
          {REGIMES.map((r) => `${counts[r]} ${r}`).join(" · ")} of {rows.length}
        </span>
      </div>

      <div className="kpi">
        <span className="kpi-label">Direction of travel</span>
        <span className="kpi-value">{falling}<small> of {rows.length} easing</small></span>
        <div className="kpi-split" aria-hidden="true">
          <span style={{ flex: falling || 0.001, background: C.calm }} />
          <span style={{ flex: rows.length - falling || 0.001, background: C.stressed }} />
        </div>
        <span className="kpi-foot">
          Biggest move: <a href={`#sectors/${mover.ticker}`}>{mover.sector}</a>{" "}
          <span className="mono">{signed(mover.change_pct)}%</span>
        </span>
      </div>

      <div className="kpi">
        <span className="kpi-label">News pressure</span>
        {news?.available ? (
          <>
            <span className="kpi-value" style={{ color: pressureColor(news.market.pressure) }}>
              {news.market.pressure.toFixed(0)}<small style={{ color: "var(--muted)" }}> / 100</small>
            </span>
            <div className="kpi-split" aria-hidden="true">
              <span style={{ flex: news.market.bearish || 0.001, background: C.stressed }} />
              <span style={{ flex: news.market.neutral || 0.001, background: "#3a4660" }} />
              <span style={{ flex: news.market.bullish || 0.001, background: C.calm }} />
            </div>
            <span className="kpi-foot">
              {news.market.bearish} bearish · {news.market.neutral} neutral · {news.market.bullish} bullish
            </span>
          </>
        ) : (
          <>
            <span className="kpi-value">—</span>
            <span className="kpi-foot">News overlay not built</span>
          </>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------- dumbbell chart */

/**
 * Where each sector is heading: realized volatility now (hollow) to the forecast (filled),
 * one row per sector, most stable first. The connecting line is the expected move.
 */
export function SectorOutlook({ sectors, horizon }) {
  const rows = sectors.ranking;
  const [hover, setHover] = useState(null);
  const max = Math.ceil(Math.max(...rows.flatMap((s) => [s.current_vol_ann, s.forecast_vol_ann])) / 5) * 5 + 5;
  const W = 1000, L = 170, Rpad = 92, ROW = 36, TOP = 30;
  const H = TOP + rows.length * ROW + 8;
  const x = (v) => L + (v / max) * (W - L - Rpad);
  const ticks = Array.from({ length: max / 5 + 1 }, (_, i) => i * 5).filter((t) => t % (max > 40 ? 10 : 5) === 0);

  return (
    <Card
      title="Where each sector is heading"
      sub={`Realized volatility now → forecast for the next ${horizon} trading days · annualized % · most stable first`}
      className="feature"
    >
      <div className="dumbbell-legend">
        <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="none" stroke={C.muted} strokeWidth="2" /></svg> now</span>
        <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="6" fill={C.calm} /></svg> forecast, coloured by regime</span>
        <span className="dim">select a row to open the sector</span>
      </div>
      <svg className="dumbbell" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Dumbbell chart of current and forecast volatility per sector">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={TOP - 8} y2={H - 4} stroke={C.line} strokeDasharray={t ? "3 4" : ""} />
            <text x={x(t)} y={TOP - 14} textAnchor="middle" className="axis-t">{t}%</text>
          </g>
        ))}
        {rows.map((s, i) => {
          const y = TOP + i * ROW + ROW / 2;
          const col = regimeColor(s.regime);
          const on = hover === i;
          const a = x(s.current_vol_ann), b = x(s.forecast_vol_ann);
          return (
            <g key={s.ticker} className={`db-row ${on ? "on" : ""}`} tabIndex={0} role="link"
               aria-label={`${s.sector}: now ${s.current_vol_ann.toFixed(1)}%, forecast ${s.forecast_vol_ann.toFixed(1)}%, ${s.regime}`}
               onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
               onFocus={() => setHover(i)} onBlur={() => setHover(null)}
               onClick={() => open(s.ticker)} onKeyDown={keyOpen(s.ticker)}>
              <rect x="0" y={y - ROW / 2 + 2} width={W} height={ROW - 4} rx="8" className="db-hit" />
              <text x="12" y={y + 4.5} className="db-name">{s.sector}</text>
              <text x={L - 14} y={y + 4.5} textAnchor="end" className="db-tk">{s.ticker}</text>
              <line x1={a} x2={b} y1={y} y2={y} stroke={col} strokeOpacity=".55" strokeWidth="4" strokeLinecap="round" />
              <circle cx={a} cy={y} r="6" fill="var(--bg-2)" stroke={C.muted} strokeWidth="2" />
              <circle cx={b} cy={y} r="8.5" fill={col} stroke="var(--bg-2)" strokeWidth="2" />
              <text x={W - 8} y={y + 4.5} textAnchor="end" className="db-val">
                {s.forecast_vol_ann.toFixed(1)}%
                <tspan className={s.change_pct < 0 ? "db-down" : "db-up"} dx="8">{signed(s.change_pct, 0)}%</tspan>
              </text>
            </g>
          );
        })}
      </svg>
    </Card>
  );
}

/* ------------------------------------------------------- super-sector summary */

/** Defensive / cyclical / sensitive: does the usual stability ordering hold today? */
export function SuperSectors({ sectors }) {
  const by = Object.fromEntries(sectors.ranking.map((s) => [s.ticker, s]));
  const groups = SUPER_SECTORS.map((g) => {
    const members = g.tickers.map((t) => by[t]).filter(Boolean);
    const avg = (k) => members.reduce((a, s) => a + s[k], 0) / members.length;
    return { key: g.id, label: g.name, members, now: avg("current_vol_ann"), fc: avg("forecast_vol_ann") };
  });
  const max = Math.max(...groups.flatMap((g) => g.members.map((s) => Math.max(s.forecast_vol_ann, s.current_vol_ann)))) * 1.08;
  const ordered = groups.every((g, i) => i === 0 || groups[i - 1].fc <= g.fc);

  return (
    <Card title="By super-sector"
          sub="Average forecast per group, with each member marked · lower is calmer">
      <div className="ss">
        {groups.map((g) => (
          <div key={g.key} className="ss-group">
            <div className="ss-head">
              <strong>{g.label}</strong>
              <span className="mono">{g.fc.toFixed(1)}%</span>
              <span className={`mono ${g.fc < g.now ? "pos" : "neg"}`}>{signed(((g.fc / g.now) - 1) * 100, 0)}%</span>
            </div>
            <div className="ss-track">
              <span className="ss-fill" style={{ width: `${(g.fc / max) * 100}%` }} />
              <span className="ss-now" style={{ left: `${(g.now / max) * 100}%` }} title={`now ${g.now.toFixed(1)}%`} />
              {g.members.map((s) => (
                <a key={s.ticker} href={`#sectors/${s.ticker}`} className="ss-pin"
                   style={{ left: `${(s.forecast_vol_ann / max) * 100}%`, background: regimeColor(s.regime) }}
                   title={`${s.sector} ${s.forecast_vol_ann.toFixed(1)}%`} aria-label={`${s.sector} ${s.forecast_vol_ann.toFixed(1)} percent`} />
              ))}
            </div>
            <div className="ss-members">
              {g.members.map((s) => (
                <a key={s.ticker} href={`#sectors/${s.ticker}`} className="mono">{s.ticker}</a>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className={`callout ${ordered ? "good" : ""}`} style={{ marginTop: 14 }}>
        {ordered
          ? "Usual ordering holds: defensives are expected to be calmest, sensitives the most volatile."
          : "The usual ordering is broken today — a group that normally sits higher is forecast calmer than one below it."}
      </p>
    </Card>
  );
}

/* ------------------------------------------------------ news vs volatility map */

/**
 * Each sector placed by how loud its news is (x) against how volatile the model expects
 * it to be (y). Descriptive: the news layer is not a model input.
 */
export function NewsVolMap({ sectors, news }) {
  const [hover, setHover] = useState(null);
  if (!news?.available) return null;
  const byTk = Object.fromEntries((news.sectors || []).map((n) => [n.ticker, n]));
  const pts = sectors.ranking
    .map((s) => ({ s, p: byTk[s.ticker]?.pressure }))
    .filter((d) => d.p != null);
  if (!pts.length) return null;

  const W = 560, H = 380, P = { l: 48, r: 18, t: 18, b: 40 };
  const xs = pts.map((d) => d.p), ys = pts.map((d) => d.s.forecast_vol_ann);
  const x0 = Math.max(0, Math.floor(Math.min(...xs) / 10) * 10 - 5), x1 = Math.min(100, Math.ceil(Math.max(...xs) / 10) * 10 + 5);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) / 5) * 5 - 2), y1 = Math.ceil(Math.max(...ys) / 5) * 5 + 2;
  const X = (v) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - ((v - y0) / (y1 - y0)) * (H - P.t - P.b);
  const med = (a) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
  const mx = med(xs), my = med(ys);
  const xt = []; for (let v = Math.ceil(x0 / 10) * 10; v <= x1; v += 10) xt.push(v);
  const yt = []; for (let v = Math.ceil(y0 / 5) * 5; v <= y1; v += 5) yt.push(v);
  const h = hover != null ? pts[hover] : null;

  // Greedy label placement: try right, left, above, below each dot and keep the first
  // spot that clears every dot and every label already placed.
  const LW = (t) => t.length * 7.6 + 2, LH = 13;
  const boxes = pts.map((d) => ({ x: X(d.p) - 9, y: Y(d.s.forecast_vol_ann) - 9, w: 18, h: 18 }));
  const hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  const labels = pts.map((d) => {
    const cx = X(d.p), cy = Y(d.s.forecast_vol_ann), w = LW(d.s.ticker);
    const tries = [
      { x: cx + 12, y: cy - LH / 2, anchor: "start", tx: 12, ty: 4 },
      { x: cx - 12 - w, y: cy - LH / 2, anchor: "end", tx: -12, ty: 4 },
      { x: cx - w / 2, y: cy - 12 - LH, anchor: "middle", tx: 0, ty: -13 },
      { x: cx - w / 2, y: cy + 12, anchor: "middle", tx: 0, ty: 22 },
    ];
    const pick = tries.find((t) => {
      const b = { x: t.x, y: t.y, w, h: LH };
      return !boxes.some((o) => hit(b, o)) && b.x > 0 && b.x + w < W;
    }) || tries[0];
    boxes.push({ x: pick.x, y: pick.y, w, h: LH });
    return pick;
  });

  return (
    <Card title="News pressure vs forecast volatility"
          sub="Each sector by how loud its headlines are and how volatile it is expected to be · lines at the medians">
      <div className="scatter-wrap">
        <svg className="scatter" viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Scatter of news pressure against forecast volatility per sector">
          {xt.map((v) => <line key={`x${v}`} x1={X(v)} x2={X(v)} y1={P.t} y2={H - P.b} stroke={C.line} strokeDasharray="3 4" />)}
          {yt.map((v) => (
            <g key={`y${v}`}>
              <line x1={P.l} x2={W - P.r} y1={Y(v)} y2={Y(v)} stroke={C.line} strokeDasharray="3 4" />
              <text x={P.l - 8} y={Y(v) + 4} textAnchor="end" className="axis-t">{v}%</text>
            </g>
          ))}
          {xt.map((v) => <text key={`xt${v}`} x={X(v)} y={H - P.b + 18} textAnchor="middle" className="axis-t">{v}</text>)}
          <text x={(P.l + W - P.r) / 2} y={H - 4} textAnchor="middle" className="axis-l">news pressure →</text>
          <line x1={X(mx)} x2={X(mx)} y1={P.t} y2={H - P.b} stroke={C.muted} strokeOpacity=".5" />
          <line x1={P.l} x2={W - P.r} y1={Y(my)} y2={Y(my)} stroke={C.muted} strokeOpacity=".5" />
          <text x={W - P.r - 6} y={P.t + 14} textAnchor="end" className="quad">loud &amp; volatile</text>
          <text x={P.l + 8} y={P.t + 14} className="quad">quiet but volatile</text>
          <text x={W - P.r - 6} y={H - P.b - 8} textAnchor="end" className="quad">loud but calm</text>
          <text x={P.l + 8} y={H - P.b - 8} className="quad">quiet &amp; calm</text>
          {pts.map((d, i) => (
            <g key={d.s.ticker} className="sc-pt" transform={`translate(${X(d.p)},${Y(d.s.forecast_vol_ann)})`}
               tabIndex={0} role="link"
               aria-label={`${d.s.sector}: news pressure ${d.p.toFixed(0)}, forecast ${d.s.forecast_vol_ann.toFixed(1)}%`}
               onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
               onFocus={() => setHover(i)} onBlur={() => setHover(null)}
               onClick={() => open(d.s.ticker)} onKeyDown={keyOpen(d.s.ticker)}>
              <circle r="16" fill="transparent" />
              <circle r={hover === i ? 9 : 7.5} fill={regimeColor(d.s.regime)} stroke="var(--bg-2)" strokeWidth="2" />
              <text x={labels[i].tx} y={labels[i].ty} textAnchor={labels[i].anchor}
                    className="sc-lbl">{d.s.ticker}</text>
            </g>
          ))}
        </svg>
        {h && (
          <div className="sc-tip" style={{ left: `${(X(h.p) / W) * 100}%`, top: `${(Y(h.s.forecast_vol_ann) / H) * 100}%` }}>
            <strong>{h.s.sector}</strong>
            <span>forecast {pct(h.s.forecast_vol_ann)} <Chip regime={h.s.regime} /></span>
            <span>news pressure {h.p.toFixed(0)}</span>
          </div>
        )}
      </div>
      <p className="sub" style={{ margin: "8px 0 0" }}>
        Top-right is where loud news and high expected volatility agree. Top-left and
        bottom-right are the disagreements worth reading about. Context only — the news
        layer is not an input to the forecast.
      </p>
    </Card>
  );
}
