import React, { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card } from "./ui";
import { ArrowRight, Bolt, Close } from "./icons";
import { AXIS, C, pressureColor, tooltipStyle } from "../theme";
import { pct, signed } from "../format";

const THEME_LABEL = {
  monetary_policy: "Central banks & rates", inflation: "Inflation", labor: "Jobs & labour",
  earnings: "Earnings", geopolitics: "Geopolitics", trade: "Trade & tariffs",
  credit: "Credit & bonds", politics: "Politics", regulation: "Regulation & legal",
  ai: "AI", energy: "Energy & oil",
};
export const themeLabel = (t) => THEME_LABEL[t] || t;

const ago = (iso) => {
  const m = Math.max(0, (Date.now() - new Date(iso).getTime()) / 60000);
  if (m < 60) return `${Math.round(m)}m ago`;
  if (m < 48 * 60) return `${Math.round(m / 60)}h ago`;
  return `${Math.round(m / 1440)}d ago`;
};

export function PressureBar({ value }) {
  if (value == null) return <span className="dim">—</span>;
  return (
    <div className="pbar">
      <div className="track">
        <div className="fill" style={{ width: `${value}%`, background: pressureColor(value) }} />
      </div>
      <span className="v" style={{ color: pressureColor(value) }}>{value.toFixed(0)}</span>
    </div>
  );
}

export function ToneBar({ m }) {
  const tot = Math.max(1, m.bearish + m.neutral + m.bullish);
  return (
    <>
      <div className="tonebar">
        <div style={{ width: `${(m.bearish / tot) * 100}%`, background: C.stressed }} />
        <div style={{ width: `${(m.neutral / tot) * 100}%`, background: C.dim }} />
        <div style={{ width: `${(m.bullish / tot) * 100}%`, background: C.calm }} />
      </div>
      <div className="tonebar-legend">
        <span className="bad-txt">{m.bearish} bearish</span>
        <span>{m.neutral} neutral</span>
        <span className="good-txt">{m.bullish} bullish</span>
      </div>
    </>
  );
}

export function Headline({ h }) {
  return (
    <div className="feed-item">
      <span className={`tone ${h.label}`} title={`${h.label} · ${(h.confidence * 100).toFixed(0)}% confidence`} />
      <div>
        {h.link ? (
          <a className="t" href={h.link} target="_blank" rel="noreferrer noopener">{h.title}</a>
        ) : (
          <span className="t">{h.title}</span>
        )}
        <div className="meta">
          <span>{h.channels.join(", ")}</span>
          <span>· {ago(h.published)}</span>
          {h.sectors.map((s) => <a key={s} className="tag sector" href={`#sectors/${s}`}>{s}</a>)}
          {h.themes.slice(0, 3).map((t) => <span key={t} className="tag">{themeLabel(t)}</span>)}
          {h.shock > 0 && <span className="tag shock" title="shock-language score"><Bolt size={11} /> {h.shock}</span>}
        </div>
      </div>
    </div>
  );
}

/** Compact card for the overview tab. */
export function NewsPulse({ news, onOpen }) {
  if (!news || !news.available) {
    return (
      <Card title="News pulse" sub="What the news channels are saying">
        <p className="muted" style={{ fontSize: 13.5 }}>
          News overlay not built yet. Run <code className="mono">python -m volradar.cli news</code>.
        </p>
      </Card>
    );
  }
  const m = news.market;
  return (
    <Card
      title="News pulse"
      sub={`${news.n_headlines} stories · ${news.channels.length} channels · last ${news.window_hours}h`}
      action={<button className="link-btn" onClick={onOpen}>Open news radar <ArrowRight size={12} /></button>}
    >
      <div className="pressure-big">
        <span className="n" style={{ color: pressureColor(m.pressure) }}>{m.pressure.toFixed(0)}</span>
        <span className="of">/ 100 news pressure</span>
        {m.vs_typical != null && (
          <span className="chip ghost">{signed(m.vs_typical, 1)}σ vs typical</span>
        )}
      </div>
      <ToneBar m={m} />
      <div className="feed" style={{ marginTop: 10 }}>
        {m.top.slice(0, 3).map((h) => <Headline key={h.title} h={h} />)}
      </div>
      <p className="sub" style={{ margin: "10px 0 0" }}>
        Descriptive overlay — not an input to the forecast.
      </p>
    </Card>
  );
}

/** Full News radar tab. */
export function NewsRadar({ news }) {
  const [sector, setSector] = useState("all");
  const [tone, setTone] = useState("all");

  const feed = useMemo(() => {
    if (!news?.available) return [];
    return news.headlines.filter(
      (h) => (sector === "all" || h.sectors.includes(sector)) && (tone === "all" || h.label === tone)
    );
  }, [news, sector, tone]);

  if (!news || !news.available) {
    return (
      <div className="callout">
        The news overlay has not been built. Run <code className="mono">python -m volradar.cli news</code>
        {" "}(needs network). {news?.reason}
      </div>
    );
  }

  const m = news.market;
  const v = news.validation || {};
  const timeline = news.timeline.map((d) => ({
    t: new Date(d.t).toLocaleString(undefined, { weekday: "short", hour: "numeric" }),
    Bearish: d.bearish, Neutral: d.neutral, Bullish: d.bullish,
  }));
  const themes = m.themes.map((t) => ({ name: themeLabel(t.theme), weight: t.weight, n: t.n }));
  const sectors = [...news.sectors].sort((a, b) => (b.pressure ?? -1) - (a.pressure ?? -1));

  return (
    <>
      <div className="callout info" style={{ marginBottom: 18 }}>
        <strong>What this is.</strong> Every headline from {news.channels.length} news channels
        in the last {news.window_hours} hours, scored for tone by a fine-tuned DistilBERT model
        and tagged for sector, risk theme and shock language. <strong>It is not an input to the
        forecast</strong>: free feeds only carry days of history, so it cannot yet be
        walk-forward validated. Every headline is archived so that it can be.
      </div>

      <div className="grid hero">
        <Card className="feature" title="Market news pressure"
              sub={`${news.n_headlines} stories · scored by ${news.scorer}`}>
          <div className="pressure-big">
            <span className="n" style={{ color: pressureColor(m.pressure) }}>{m.pressure.toFixed(0)}</span>
            <span className="of">/ 100</span>
          </div>
          <ToneBar m={m} />
          <div className="stat-row">
            <div className="stat"><div className="k">Net tone</div>
              <div className={`v ${m.net_tone < 0 ? "bad-txt" : "good-txt"}`}>{signed(m.net_tone, 2)}</div></div>
            <div className="stat"><div className="k">Shock language</div>
              <div className="v">{m.intensity.toFixed(2)}<small>/story</small></div></div>
            <div className="stat"><div className="k">vs typical</div>
              <div className="v">{m.vs_typical != null ? `${signed(m.vs_typical, 1)}σ` : "—"}
                {m.vs_typical == null && <small>needs 10 days</small>}</div></div>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 14 }}>
            {news.channels.map((c) => (
              <span key={c.channel} className="chip ghost">{c.channel} <span className="mono dim">{c.n}</span></span>
            ))}
          </div>
        </Card>

        <Card title="Tone over time" sub="Stories per 6-hour window, by tone">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={timeline} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <CartesianGrid stroke={C.line} strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="t" tick={AXIS} tickLine={false} axisLine={{ stroke: C.line }} minTickGap={20} />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(94,177,255,.06)" }} />
              <Bar dataKey="Bearish" stackId="a" fill={C.stressed} fillOpacity={0.85} />
              <Bar dataKey="Neutral" stackId="a" fill={C.dim} fillOpacity={0.6} />
              <Bar dataKey="Bullish" stackId="a" fill={C.calm} fillOpacity={0.85} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <h3 className="section-title">News against the forecast, by sector</h3>
      <p className="section-note">
        Pressure combines how negative a sector's coverage is with how loud it is (0–100).
        A <span className="warn-txt">divergence</span> flags a sector whose news-pressure rank
        and forecast-volatility rank sit five or more places apart — a prompt to look, not a
        signal to act on.
      </p>
      <div className="grid side">
        <Card title="Sector news radar" sub="Sorted by news pressure · click a row to see its headlines">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Sector</th>
                  <th>News pressure</th>
                  <th className="num">Stories</th>
                  <th className="num">Share</th>
                  <th className="num">Tone</th>
                  <th className="num">Forecast vol</th>
                  <th>Flag</th>
                </tr>
              </thead>
              <tbody>
                {sectors.map((s) => (
                  <tr key={s.ticker} className={`clickable ${sector === s.ticker ? "on" : ""}`}
                      onClick={() => setSector(sector === s.ticker ? "all" : s.ticker)}>
                    <td>{s.sector} <span className="muted mono" style={{ fontSize: 11.5, marginLeft: 5 }}>{s.ticker}</span></td>
                    <td><PressureBar value={s.pressure} /></td>
                    <td className="num muted">{s.n}</td>
                    <td className="num muted">{(s.attention * 100).toFixed(0)}%</td>
                    <td className={`num ${s.net_tone == null ? "" : s.net_tone < 0 ? "bad-txt" : "good-txt"}`}>
                      {s.net_tone == null ? "—" : signed(s.net_tone, 2)}</td>
                    <td className="num">{pct(s.forecast_vol_ann)} {s.regime && <span className={`tone`} style={{ background: s.regime === "calm" ? C.calm : s.regime === "stressed" ? C.stressed : C.normal, marginLeft: 4 }} />}</td>
                    <td>
                      {s.divergence === "news_hotter" && <span className="badge-div news_hotter">news louder</span>}
                      {s.divergence === "news_calmer" && <span className="badge-div news_calmer">news quieter</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="What the coverage is about" sub="Risk themes, recency-weighted">
          <ResponsiveContainer width="100%" height={Math.max(180, themes.length * 34)}>
            <BarChart data={themes} layout="vertical" margin={{ top: 0, right: 12, left: 6, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="name" tick={{ ...AXIS, fill: C.muted }} width={130}
                     tickLine={false} axisLine={false} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(94,177,255,.06)" }}
                       formatter={(val, _n, p) => [`${p.payload.n} stories`, "coverage"]} />
              <Bar dataKey="weight" radius={[0, 4, 4, 0]} barSize={14}>
                {themes.map((t, i) => <Cell key={t.name} fill={i < 3 ? C.accent : C.accent2} fillOpacity={1 - i * 0.07} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid side" style={{ marginTop: 18 }}>
        <Card title="Headlines" sub={`${feed.length} shown · newest first`}>
          <div className="filters">
            {["all", "Bearish", "Neutral", "Bullish"].map((t) => (
              <button key={t} className={`fbtn ${tone === t ? "on" : ""}`} onClick={() => setTone(t)}>
                {t === "all" ? "All tones" : t}
              </button>
            ))}
            <span style={{ width: 10 }} />
            <button className={`fbtn ${sector === "all" ? "on" : ""}`} onClick={() => setSector("all")}>All sectors</button>
            {sector !== "all" && (
              <button className="fbtn on" onClick={() => setSector("all")} aria-label={`Clear ${sector} filter`}>
                {sector} <Close size={11} />
              </button>
            )}
          </div>
          <div className="feed" style={{ maxHeight: 620, overflowY: "auto", paddingRight: 6 }}>
            {feed.length === 0 && <p className="muted">No headlines match.</p>}
            {feed.map((h) => <Headline key={h.title + h.published} h={h} />)}
          </div>
        </Card>

        <div style={{ display: "grid", gap: 18, alignContent: "start" }}>
          <Card title="Validation status" sub="When can news become a forecast input?">
            <div className="kv"><span className="k">Archived days</span>
              <span className="v">{v.archived_days} / {v.days_needed}</span></div>
            <div className="progress"><div style={{ width: `${Math.min(100, (v.archived_days / v.days_needed) * 100)}%` }} /></div>
            <div className="kv"><span className="k">Stories archived</span><span className="v">{news.n_archived?.toLocaleString()}</span></div>
            <div className="kv"><span className="k">Status</span>
              <span className="v"><span className={`chip ${v.status === "testable" ? "calm" : "info"}`}>{v.status}</span></span></div>
            {v.spearman_pressure_vs_residual != null && (
              <div className="kv"><span className="k">ρ(pressure, forecast error)</span>
                <span className="v">{v.spearman_pressure_vs_residual} (t={v.t})</span></div>
            )}
            <p className="sub" style={{ margin: "10px 0 0" }}>{v.message}</p>
          </Card>

          <Card title="How the reading is built" sub="Every number traces to a rule">
            <div className="kv"><span className="k">Pressure</span><span className="v" style={{ fontSize: 11 }}>{news.method.pressure}</span></div>
            <div className="kv"><span className="k">Weighting</span><span className="v" style={{ fontSize: 11 }}>{news.method.weighting}</span></div>
            <div className="kv"><span className="k">Divergence</span><span className="v" style={{ fontSize: 11 }}>{news.method.divergence}</span></div>
            <div className="kv"><span className="k">Tone model</span><span className="v" style={{ fontSize: 11 }}>{news.scorer}</span></div>
            <div className="kv"><span className="k">Feeds reached</span>
              <span className="v">{news.coverage.filter((c) => !c.error).length} / {news.coverage.length}</span></div>
          </Card>

          {sector !== "all" && (
            <a className="fbtn on link-row" href={`#sectors/${sector}`}>
              Open the {sector} sector page <ArrowRight size={12} />
            </a>
          )}
        </div>
      </div>
    </>
  );
}

