import React, { useEffect, useMemo, useState } from "react";
import { getHistory } from "../api";
import { Card, Chip } from "./ui";
import { HistoryChart } from "./Charts";
import { Headline, PressureBar, ToneBar, themeLabel } from "./NewsPanel";
import { ArrowLeft, ArrowRight, Direction } from "./icons";
import { C, pressureColor, regimeColor } from "../theme";
import { pct, pct01, signed, SECTOR_BLURB, SUPER_SECTORS, superSectorOf } from "../format";

// Histories are fetched once per session and shared between the board and detail pages.
const historyCache = new Map();
function useHistory(ticker, limit) {
  const key = `${ticker}:${limit}`;
  const [h, setH] = useState(() => historyCache.get(key) || null);
  useEffect(() => {
    if (!ticker) return;
    if (historyCache.has(key)) return setH(historyCache.get(key));
    let live = true;
    getHistory("sector", ticker, limit)
      .then((r) => { historyCache.set(key, r); if (live) setH(r); })
      .catch(() => live && setH({ series: [] }));
    return () => { live = false; };
  }, [key, ticker, limit]);
  return h;
}

/** Recent out-of-sample record as a compact line pair: realized (grey) vs forecast. */
function Trend({ ticker, width = 168, height = 40 }) {
  const h = useHistory(ticker, 130);
  if (!h) return <div className="trend skeleton" style={{ width, height }} />;
  const s = h.series || [];
  if (s.length < 2) return <div className="trend" style={{ width, height }} />;
  const vals = s.flatMap((d) => [d.forecast, d.realized]).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i) => (i / (s.length - 1)) * (width - 2) + 1;
  const y = (v) => height - 3 - ((v - lo) / (hi - lo || 1)) * (height - 6);
  const line = (k) => s.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(d[k]).toFixed(1)}`).join("");
  return (
    <svg className="trend" width={width} height={height} viewBox={`0 0 ${width} ${height}`}
         role="img" aria-label={`${ticker}: forecast versus realized volatility, last ${s.length} sessions`}>
      <path d={line("realized")} stroke={C.muted} strokeOpacity=".7" strokeWidth="1.2" fill="none" />
      <path d={line("forecast")} stroke={C.accent} strokeWidth="1.6" fill="none" />
    </svg>
  );
}

function SectorRow({ s, newsBy, horizon }) {
  const nw = newsBy[s.ticker];
  const rising = s.direction === "rising";
  return (
    <a className="srow" href={`#sectors/${s.ticker}`}
       aria-label={`${s.sector}: forecast ${pct(s.forecast_vol_ann)}, ${s.regime}. Open details`}>
      <span className="srow-name">
        <span className="srow-dot" style={{ background: regimeColor(s.regime) }} />
        <span>
          <strong>{s.sector}</strong>
          <span className="mono dim srow-tk">{s.ticker}</span>
        </span>
      </span>
      <Trend ticker={s.ticker} />
      <span className="srow-fc">
        <span className="mono srow-big">{pct(s.forecast_vol_ann)}</span>
        <span className={`srow-chg ${rising ? "bad-txt" : "good-txt"}`}>
          <Direction rising={rising} size={12} /> {signed(s.change_pct)}%
        </span>
      </span>
      <span className="srow-reg"><Chip regime={s.regime} /></span>
      <span className="srow-news">{nw ? <PressureBar value={nw.pressure} /> : <span className="dim">—</span>}</span>
      <span className="srow-hit mono muted">
        {s.track_record ? pct01(s.track_record.dir_accuracy) : "—"}
      </span>
      <ArrowRight size={14} className="icon srow-go" />
    </a>
  );
}

export function SectorBoard({ sectors, news, horizon }) {
  const by = Object.fromEntries(sectors.ranking.map((s) => [s.ticker, s]));
  const newsBy = Object.fromEntries((news?.sectors || []).map((s) => [s.ticker, s]));
  return (
    <>
      <div className="page-head">
        <h2 className="page-title">Sectors</h2>
        <p className="section-note">
          Each of the eleven S&amp;P sectors, grouped by how tied its earnings are to the
          economic cycle. The line is the model's out-of-sample record over the last six
          months — accent is what it forecast, grey is what happened.
        </p>
      </div>
      <div className="board-head" aria-hidden="true">
        <span>Sector</span><span>Last 6 months</span><span>Forecast · {horizon}d</span>
        <span>Outlook</span><span>News</span><span>Hit rate</span><span />
      </div>
      {SUPER_SECTORS.map((g) => {
        const rows = g.tickers.map((t) => by[t]).filter(Boolean)
          .sort((a, b) => a.forecast_vol_ann - b.forecast_vol_ann);
        const avg = rows.reduce((a, r) => a + r.forecast_vol_ann, 0) / Math.max(rows.length, 1);
        return (
          <section key={g.id} className="group" aria-labelledby={`g-${g.id}`}>
            <div className="group-head">
              <h3 id={`g-${g.id}`}>{g.name}</h3>
              <p>{g.blurb}</p>
              <span className="mono muted group-avg">avg {pct(avg)}</span>
            </div>
            <div className="board">
              {rows.map((s) => <SectorRow key={s.ticker} s={s} newsBy={newsBy} horizon={horizon} />)}
            </div>
          </section>
        );
      })}
    </>
  );
}

function PeerBars({ sectors, ticker }) {
  const rows = [...sectors.ranking].sort((a, b) => b.forecast_vol_ann - a.forecast_vol_ann);
  const max = rows[0]?.forecast_vol_ann || 1;
  return (
    <div className="peers" role="list">
      {rows.map((s) => (
        <a key={s.ticker} role="listitem" href={`#sectors/${s.ticker}`}
           className={`peer ${s.ticker === ticker ? "on" : ""}`}
           aria-current={s.ticker === ticker ? "page" : undefined}>
          <span className="peer-tk mono">{s.ticker}</span>
          <span className="peer-track">
            <span className="peer-fill" style={{ width: `${(s.forecast_vol_ann / max) * 100}%`,
              background: regimeColor(s.regime) }} />
          </span>
          <span className="peer-v mono">{pct(s.forecast_vol_ann)}</span>
        </a>
      ))}
    </div>
  );
}

export function SectorDetail({ sectors, news, horizon, ticker }) {
  const s = sectors.ranking.find((r) => r.ticker === ticker);
  // An unknown ticker gets the recovery message below, not a request that can only 404.
  const history = useHistory(s ? ticker : null, 756);
  const order = useMemo(() => SUPER_SECTORS.flatMap((g) => g.tickers)
    .filter((t) => sectors.ranking.some((r) => r.ticker === t)), [sectors]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, textarea, select")) return;
      const i = order.indexOf(ticker);
      if (e.key === "ArrowRight") window.location.hash = `#sectors/${order[(i + 1) % order.length]}`;
      if (e.key === "ArrowLeft") window.location.hash = `#sectors/${order[(i - 1 + order.length) % order.length]}`;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [order, ticker]);

  if (!s) {
    return (
      <div className="callout">
        No sector called <strong>{ticker}</strong>. <a href="#sectors">Back to all sectors</a>.
      </div>
    );
  }

  const group = superSectorOf(ticker);
  const nw = news?.available ? news.sectors.find((x) => x.ticker === ticker) : null;
  const heads = news?.available ? news.headlines.filter((h) => h.sectors.includes(ticker)) : [];
  const themes = (() => {
    const acc = {};
    heads.forEach((h) => h.themes.forEach((t) => { acc[t] = (acc[t] || 0) + 1; }));
    return Object.entries(acc).sort((a, b) => b[1] - a[1]).slice(0, 6);
  })();
  const tone = { bearish: 0, neutral: 0, bullish: 0 };
  heads.forEach((h) => { tone[h.label.toLowerCase()] += 1; });
  const rising = s.direction === "rising";
  const i = order.indexOf(ticker);
  const prev = order[(i - 1 + order.length) % order.length];
  const next = order[(i + 1) % order.length];
  const name = (t) => sectors.ranking.find((r) => r.ticker === t)?.sector || t;
  const tr = s.track_record || {};

  return (
    <>
      <nav className="crumbs" aria-label="Breadcrumb">
        <a href="#sectors"><ArrowLeft size={13} /> All sectors</a>
      </nav>

      <header className="sector-head">
        <div>
          <h2 className="page-title">
            {s.sector} <span className="mono dim">{s.ticker}</span>
          </h2>
          <p className="muted">
            {SECTOR_BLURB[ticker]}{group && <> · <span className="chip ghost">{group.name}</span></>}
          </p>
        </div>
        <Chip regime={s.regime} />
      </header>

      <dl className="facts">
        <div>
          <dt>Forecast, next {horizon} days</dt>
          <dd><span className="mono facts-big">{pct(s.forecast_vol_ann)}</span>
            <span className={`facts-chg ${rising ? "bad-txt" : "good-txt"}`}>
              <Direction rising={rising} size={12} /> {signed(s.change_pct)}%
            </span></dd>
        </div>
        <div><dt>Realized now</dt><dd className="mono">{pct(s.current_vol_ann)}</dd></div>
        <div><dt>Stability rank</dt><dd className="mono">{s.stability_rank} <span className="dim">of {sectors.ranking.length}</span></dd></div>
        <div><dt>Direction hit rate</dt><dd className="mono">{pct01(tr.dir_accuracy)}</dd></div>
        <div><dt>Skill vs persistence</dt><dd className="mono">{tr.skill_vs_persistence != null ? signed(tr.skill_vs_persistence * 100, 0) + "%" : "—"}</dd></div>
        <div><dt>News pressure</dt>
          <dd className="mono" style={{ color: pressureColor(nw?.pressure) }}>{nw?.pressure != null ? nw.pressure.toFixed(0) : "—"}</dd></div>
      </dl>

      <div className="grid side">
        {history ? (
          history.series?.length ? (
            <HistoryChart history={history} title="Forecast against what happened"
                          sub={`${s.ticker} · out-of-sample, through ${history.series.at(-1).date}`} />
          ) : (
            <Card title="Forecast against what happened"><p className="muted">No out-of-sample history for this sector yet.</p></Card>
          )
        ) : (
          <Card title="Forecast against what happened"><div className="skeleton" style={{ height: 320 }} /></Card>
        )}
        <Card title="Where it sits" sub={`All sectors by forecast vol · ${s.sector} highlighted`}>
          <PeerBars sectors={sectors} ticker={ticker} />
        </Card>
      </div>

      <h3 className="section-title">What the news is saying about {s.sector}</h3>
      {!news?.available ? (
        <p className="muted">The news overlay has not been built.</p>
      ) : (
        <div className="grid side">
          <Card title="Headlines" sub={`${heads.length} stories in the last ${news.window_hours}h`}>
            {heads.length === 0 ? (
              <p className="muted">No stories tagged to this sector in the window. A quiet tape reads
                as "no data" here, never as calm.</p>
            ) : (
              <div className="feed" style={{ maxHeight: 560, overflowY: "auto", paddingRight: 6 }}>
                {heads.map((h) => <Headline key={h.title + h.published} h={h} />)}
              </div>
            )}
          </Card>
          <div className="stack">
            <Card title="Tone">
              {heads.length ? <ToneBar m={tone} /> : <p className="muted">—</p>}
              {nw?.divergence && (
                <p className="callout" style={{ marginTop: 14 }}>
                  {nw.divergence === "news_hotter"
                    ? "News is louder than the forecast: this sector ranks much higher on news pressure than on forecast volatility."
                    : "News is quieter than the forecast: the model expects more movement than the coverage suggests."}
                  {" "}A prompt to look, not a signal — the overlay is not validated.
                </p>
              )}
            </Card>
            <Card title="Themes">
              {themes.length ? (
                <ul className="theme-list">
                  {themes.map(([t, n]) => (
                    <li key={t}><span>{themeLabel(t)}</span><span className="mono muted">{n}</span></li>
                  ))}
                </ul>
              ) : <p className="muted">No risk themes detected.</p>}
            </Card>
          </div>
        </div>
      )}

      <nav className="pager" aria-label="Other sectors">
        <a href={`#sectors/${prev}`}><ArrowLeft size={13} /> {name(prev)}</a>
        <span className="dim"><kbd>←</kbd> <kbd>→</kbd> move between sectors</span>
        <a href={`#sectors/${next}`}>{name(next)} <ArrowRight size={13} /></a>
      </nav>
    </>
  );
}
