import React, { useEffect, useState } from "react";
import { getEquity, getForecast, getHistory, getNews, getPerformance, getSectors } from "./api";
import { MarketCard, DriversCard } from "./components/MarketCard";
import { SectorTable } from "./components/SectorTable";
import { HistoryChart, EquityChart } from "./components/Charts";
import { TrackRecord } from "./components/TrackRecord";
import { RadarSweep, RadarLegend } from "./components/RadarSweep";
import { NewsPulse, NewsRadar } from "./components/NewsPanel";
import { Card, Chip, Logo, State } from "./components/ui";
import { folds } from "./format";
import { pressureColor } from "./theme";

const TABS = [
  ["dashboard", "Overview"],
  ["news", "News radar"],
  ["record", "Track record"],
  ["method", "Method"],
];

// The tab lives in the URL hash so a link can open straight onto it.
const tabFromHash = () => {
  const h = window.location.hash.replace("#", "");
  return TABS.some(([id]) => id === h) ? h : "dashboard";
};

export default function App() {
  const [tab, setTabState] = useState(tabFromHash);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [sectorTicker, setSectorTicker] = useState(null);
  const [sectorHistory, setSectorHistory] = useState(null);

  const setTab = (t) => {
    setTabState(t);
    window.history.replaceState(null, "", `#${t}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  useEffect(() => {
    // getNews resolves to null when the overlay is absent, so it can never block the page.
    Promise.all([getForecast(), getSectors(), getPerformance(), getHistory("market"), getEquity(), getNews()])
      .then(([forecast, sectors, perf, history, equity, news]) =>
        setData({ forecast, sectors, perf, history, equity, news })
      )
      .catch((e) => setErr(e.message));
  }, []);

  useEffect(() => {
    if (!sectorTicker) return setSectorHistory(null);
    getHistory("sector", sectorTicker).then(setSectorHistory).catch(() => setSectorHistory(null));
  }, [sectorTicker]);

  if (err)
    return (
      <State>
        <p>Could not load forecasts.</p>
        <p className="muted" style={{ fontSize: 13.5 }}>{err}</p>
        <p style={{ marginTop: 18 }}>
          Generate the artifacts first: <code>python -m volradar.cli all</code>
        </p>
      </State>
    );
  if (!data) return <State loading>Loading forecasts…</State>;

  const { forecast, sectors, perf, history, equity, news } = data;
  const horizon = forecast.meta?.horizon ?? 10;
  const selected = sectors.ranking.find((s) => s.ticker === sectorTicker);
  const foldCount = folds(perf, "sector", horizon)?.n;
  const hasNews = Boolean(news?.available);

  const pickSector = (t) => {
    setSectorTicker(t);
    if (t && tab !== "dashboard") setTab("dashboard");
  };

  return (
    <>
      <header className="top">
        <div className="top-inner">
          <div className="brand">
            <Logo />
            <div>
              <h1>Volatility <span>Radar</span></h1>
              <p>Market &amp; sector volatility, {horizon} trading days ahead — with its own track record.</p>
            </div>
          </div>
          <div className="status-strip">
            <span className="pill"><span className="live" />as of <strong>{forecast.as_of}</strong></span>
            <span className="pill">SPY <strong>{forecast.forecast_vol_ann.toFixed(1)}%</strong> <Chip regime={forecast.regime} /></span>
            {hasNews && (
              <span className="pill">news pressure{" "}
                <strong style={{ color: pressureColor(news.market.pressure) }}>
                  {news.market.pressure.toFixed(0)}
                </strong>
              </span>
            )}
          </div>
        </div>
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? "on" : ""}`} onClick={() => setTab(id)}>
              {label}
              {id === "news" && hasNews && <span className="count">{news.n_headlines}</span>}
            </button>
          ))}
        </nav>
      </header>

      <div className="wrap">
        {tab === "dashboard" && (
          <>
            <div className="grid hero">
              <MarketCard f={forecast} horizon={horizon} />
              <Card
                title="Sector volatility radar"
                sub={`Forecast vol per sector · next ${horizon} days · hover a blip, click to drill in`}
              >
                <RadarSweep sectors={sectors} news={news} selected={sectorTicker} onSelect={pickSector} />
                <RadarLegend hasNews={hasNews} />
              </Card>
            </div>

            <div className="grid two" style={{ marginTop: 18 }}>
              <DriversCard drivers={forecast.drivers} />
              <NewsPulse news={news} onOpen={() => setTab("news")} />
            </div>

            <h3 className="section-title">Which sectors are stable?</h3>
            <p className="section-note">
              Forecast volatility for each of the eleven S&amp;P sectors. Lower is more
              stable. Defensive sectors normally sit at the top and cyclicals at the bottom —
              when that ordering breaks, something has changed.
            </p>
            <SectorTable
              sectors={sectors}
              horizon={horizon}
              selected={sectorTicker}
              onSelect={setSectorTicker}
              news={news}
              foldCount={foldCount}
            />

            <h3 className="section-title">Forecast against what actually happened</h3>
            <p className="section-note">
              Out-of-sample only. Each point was produced by a model that had seen nothing
              after the previous year end, so this is the record a user would really have got.
            </p>
            <HistoryChart
              history={sectorHistory && selected ? sectorHistory : history}
              title={sectorHistory && selected ? `${selected.sector} — forecast vs realized` : "Market — forecast vs realized"}
              sub={
                sectorHistory && selected
                  ? `${selected.ticker} · click the row or blip again to return to the market view`
                  : "SPY · select a sector above to drill in"
              }
            />
          </>
        )}

        {tab === "news" && <NewsRadar news={news} onSelectSector={pickSector} />}

        {tab === "record" && <TrackRecord perf={perf} />}

        {tab === "record" && equity && (
          <div style={{ marginTop: 16 }}>
            <EquityChart equity={equity} />
          </div>
        )}

        {tab === "method" && <Method perf={perf} forecast={forecast} />}

        <footer>
          Educational research project — <strong>not investment advice</strong>. Forecasts are
          of volatility (how much prices move), never of direction (which way). Prices from
          Yahoo Finance and Cboe; headlines from public RSS feeds. Model last fitted{" "}
          {forecast.meta?.fitted_utc?.slice(0, 10)}
          {hasNews && <> · news read {news.generated_utc.slice(0, 16).replace("T", " ")} UTC</>}.
        </footer>
      </div>
    </>
  );
}

function Method({ perf, forecast }) {
  const meta = forecast.meta || {};
  // Read from the walk-forward folds rather than written into the copy: this count grew
  // from 13 to 15 once and the page did not notice.
  const foldCount = folds(perf, "market", meta.horizon ?? 10)?.n;
  return (
    <>
      <h3 className="section-title">How it works</h3>
      <div className="grid two">
        <div className="card">
          <h2>What is being predicted</h2>
          <p className="sub">And, just as importantly, what is not.</p>
          <p style={{ fontSize: 14 }}>
            The forecast is <strong>realized volatility</strong>: how much prices will move
            over the next {meta.horizon ?? 10} trading days, expressed as an annualized
            percentage. It says nothing about <em>direction</em>. That is deliberate —
            short-horizon direction is very close to unpredictable, while volatility is
            strongly persistent and genuinely forecastable. Volatility clusters: calm
            follows calm, turbulence follows turbulence.
          </p>
          <p style={{ fontSize: 14 }}>
            A rising forecast means bigger swings are expected — in either direction. It is a
            risk signal, not a market call.
          </p>
        </div>

        <div className="card">
          <h2>Keeping it honest</h2>
          <p className="sub">The methodology that makes the track record meaningful.</p>
          <ul style={{ fontSize: 14, paddingLeft: 18, margin: 0, lineHeight: 1.65 }}>
            <li><strong>Chronological only.</strong> Train on the past, test on the future. Days are never shuffled.</li>
            <li><strong>No lookahead.</strong> Every input uses only data up to its own day, enforced by a test that alters a future bar and asserts no past feature moves.</li>
            <li><strong>Boundary embargo.</strong> A 10-day outcome spans 10 days, so rows whose outcome window crosses a train/test boundary are purged rather than leaked.</li>
            <li><strong>Walk-forward.</strong> Retrained every year, scored on the following year — {foldCount ?? "multiple"} independent out-of-sample windows, reported with their spread.</li>
            <li><strong>Baselines everywhere.</strong> Persistence for the level, majority for the classes. A score without its baseline is not reported.</li>
            <li><strong>Frozen regime bands.</strong> Calm/normal/stressed thresholds come from training data only, so they cannot drift with the period being scored.</li>
            <li><strong>Negative results kept.</strong> Macro data hurt and was dropped; the 21-day horizon was unreliable and is not shown.</li>
          </ul>
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        <div className="card">
          <h2>Model</h2>
          <p className="sub">Chosen by validation, not preference.</p>
          <div className="kv"><span className="k">Estimator</span><span className="v">gradient-boosted trees</span></div>
          <div className="kv"><span className="k">Training rows</span><span className="v">{meta.n_train?.toLocaleString()}</span></div>
          <div className="kv"><span className="k">Features</span><span className="v">{meta.n_features}</span></div>
          <div className="kv"><span className="k">Training window</span><span className="v">{meta.train_start} → {meta.train_end}</span></div>
          <div className="kv"><span className="k">Horizons served</span><span className="v">{perf.horizons_served?.join(", ")} days</span></div>
          <div className="kv"><span className="k">Horizons withheld</span><span className="v">{perf.horizons_withheld?.join(", ") || "none"} days</span></div>
        </div>

        <div className="card">
          <h2>Inputs</h2>
          <p className="sub">Only families that measurably earned their place survived.</p>
          <p style={{ fontSize: 14 }}>
            <strong>Price history — and, it turns out, only price history.</strong> Realized
            volatility at four timescales, range-based estimators that recover intraday
            information from daily bars, downside volatility, momentum, drawdown and volume.
          </p>
          <p style={{ fontSize: 14 }} className="muted">
            <strong>Built, tested, and not shipped:</strong> the VIX complex (level, term
            structure, vol-of-vol, variance risk premium); cross-sectional dispersion and
            average correlation across ~80 large caps; rates, credit spreads and the dollar;
            sector dispersion. Each is a plausible volatility predictor. None of them
            reliably improved the forecast under the project's single adoption rule, so the
            simplest model was kept. They remain implemented and are re-tested on every run.
          </p>
          <p style={{ fontSize: 14 }} className="muted">
            That is a real result, not a shortcut. Realized volatility is a remarkably
            complete summary of a market's own risk state, and the things that look like
            independent signals mostly turn out to be reflections of it.
          </p>
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        <div className="card">
          <h2>News semantics overlay</h2>
          <p className="sub">What the news channels say — kept beside the forecast, not inside it.</p>
          <p style={{ fontSize: 14 }}>
            Headlines from CNBC, MarketWatch, the FT, BBC, Nasdaq, Seeking Alpha, Investing.com,
            the Federal Reserve, Google News and per-sector Yahoo Finance feeds are pulled on every
            refresh, de-duplicated across channels, scored Bearish / Bullish / Neutral by a
            DistilBERT model fine-tuned on financial news text (88% accuracy, 0.85 macro-F1 on its
            validation split), and tagged with the sectors, risk themes and shock language they
            contain.
          </p>
          <p style={{ fontSize: 14 }} className="muted">
            It is <strong>not</strong> a model input. Free feeds carry days of history, not years,
            so there is no honest way yet to show it improves a walk-forward forecast — and
            back-filling history from today's articles would leak the future. Every headline is
            archived with its first-seen timestamp instead, so the test can be run under the same
            paired rule as every other candidate feature once the archive is long enough.
          </p>
        </div>
        <div className="card">
          <h2>Refreshing</h2>
          <p className="sub">One command brings everything to the latest close.</p>
          <p style={{ fontSize: 14 }}>
            <code className="mono">python -m volradar.cli refresh</code> re-downloads prices,
            rebuilds the feature panels, refits the production models and rebuilds the news
            radar. The backtest is not re-run: split boundaries are absolute dates, so extending
            the sample cannot change any reported track-record number.
          </p>
        </div>
      </div>
    </>
  );
}
