// Presentation helpers, kept out of the components.
//
// The feature dictionary is the important bit: model column names like `vix_rv_gap` are
// meaningless to a reader, and a dashboard that shows raw column names is asking the user
// to trust a black box. Every driver the model can surface has a plain-language sentence
// here saying what it measures and why it relates to volatility.

export const pct = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(d)}%`;

export const signed = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

export const num = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d);

export const pct01 = (v, d = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;

// Walk-forward fold counts, read from the API rather than written into the copy.
//
// These were hardcoded, and the walk-forward grew from 13 years to 15 without them: the
// dashboard published "8 of 13 years" for the withheld horizon while the config file and
// the README said "12 of 15". Deriving them means the page cannot drift again — the number
// on screen is the number in `walkforward_folds.csv`.
export const folds = (perf, panel, horizon) => {
  const row = (perf?.stability || []).find(
    (r) => r.panel === panel && Number(r.horizon) === Number(horizon)
  );
  return row ? { n: row.n, positive: row.years_positive } : null;
};

// "12 of 15" — or nothing at all, rather than a stale guess.
export const foldsPhrase = (perf, panel, horizon) => {
  const f = folds(perf, panel, horizon);
  return f ? `${f.positive} of ${f.n}` : null;
};

export const FEATURES = {
  log_rv_d: ["Yesterday's move", "How far the market moved in the last session."],
  log_rv_w: ["1-week realized vol", "Actual volatility over the past five sessions."],
  log_rv_m: ["1-month realized vol", "Actual volatility over the past month — the anchor of the forecast."],
  log_rv_q: ["3-month realized vol", "The slow-moving baseline level of volatility."],
  log_pk_d: ["Daily high-low range", "Parkinson estimator: how wide the day's trading range was."],
  log_gk_d: ["Daily range + gap", "Garman-Klass: the day's range including its open-to-close move."],
  log_gk_w: ["1-week range vol", "Range-based volatility averaged over a week — less noisy than close-to-close."],
  log_gk_m: ["1-month range vol", "Range-based volatility averaged over a month."],
  rv_slope_wm: ["Short vs medium vol", "Is this week more volatile than this month? Rising short-term vol leads."],
  rv_slope_mq: ["Medium vs long vol", "Is this month more volatile than this quarter?"],
  log_semivol_m: ["Downside volatility", "Volatility of losing days only. Markets fall faster than they rise."],
  ret_5: ["1-week return", "Recent direction — selloffs and volatility travel together."],
  ret_21: ["1-month return", "Medium-term direction."],
  drawdown_63: ["Drawdown from 3-month high", "How far below its recent peak the market is trading."],
  turnover_z: ["Unusual volume", "Today's volume against its own recent normal."],

  vix_log: ["VIX level", "The market's own forward-looking volatility estimate."],
  vix_chg_5: ["VIX 1-week change", "Is priced-in fear rising or falling?"],
  vix_rel_21: ["VIX vs its month average", "Elevated relative to its own recent norm."],
  vix_term: ["VIX term structure", "3-month vs spot VIX. Negative means near-term stress is priced above longer-term."],
  vix_term_short: ["Near-term VIX slope", "30-day vs 9-day VIX. The first part of the curve to invert when a shock lands."],
  vix_term_long: ["Far-term VIX slope", "6-month vs 3-month VIX — how much stress is priced beyond the immediate."],
  vix_curve: ["VIX curve shape", "Whether the term structure is bowed, not just tilted."],
  vvix_log: ["Vol-of-vol (VVIX)", "How unstable the volatility estimate itself is."],
  vix_vol_pk: ["VIX intraday range", "How far the fear gauge itself swung within the day."],
  skew_log: ["Option skew (SKEW)", "What the option market pays for crash protection, beyond the VIX level."],
  vix_rv_gap: ["Variance risk premium", "Implied minus realized: trouble priced in that hasn't shown up in returns yet."],
  vix_rv_gap_short: ["Near-term variance premium", "The same gap at the 9-day horizon, matched to shorter forecasts."],
  rvx_vix_spread: ["Small-cap vol premium", "Russell 2000 vs S&P implied vol — is stress broad or concentrated?"],
  vxn_vix_spread: ["Nasdaq vol premium", "Nasdaq-100 vs S&P implied vol."],
  ovx_log: ["Oil volatility (OVX)", "Crude-oil implied vol — the cross-asset shock channel."],

  log_bpv_m: ["Smooth volatility", "Bipower variation: the part of volatility left after jumps are stripped out."],
  jump_share_m: ["Jump share", "How much of the past month's variance came from sudden gaps rather than drift."],
  log_jump_d: ["Today's jump", "The discontinuous part of today's move."],
  rq_ratio: ["Estimate reliability", "How heavy-tailed recent returns were — high means today's vol reading is noisy."],
  log_rs_down_m: ["Down-day variance", "Variance contributed by falling days only."],
  log_rs_up_m: ["Up-day variance", "Variance contributed by rising days only."],
  semivar_skew_m: ["Down/up asymmetry", "How much more of the variance came from falls than rises."],
  log_rs_d: ["Drift-robust range", "Rogers-Satchell: the day's range, corrected for a trending market."],
  log_rs_m: ["1-month drift-robust range", "Rogers-Satchell averaged over a month."],
  log_yz_w: ["1-week Yang-Zhang vol", "The most efficient daily estimator, including the overnight gap."],
  log_yz_m: ["1-month Yang-Zhang vol", "Same estimator over a month — the cleanest read on the current level."],
  yz_gap_share: ["Overnight share", "How much of the month's variance happened between the close and the next open."],

  disp_log: ["Stock dispersion", "How differently individual stocks moved today."],
  disp_chg_5: ["Dispersion change", "Is the cross-section spreading out or compressing?"],
  avg_corr: ["Average correlation", "How much stocks move together. Correlation spikes precede index vol spikes."],
  corr_chg_21: ["Correlation change", "Whether stocks have started moving as one over the past month."],
  breadth_down: ["Share of stocks falling", "Market breadth, smoothed over a week."],
  xs_vol_skew: ["Volatile-tail spread", "How far the most volatile names sit above the typical name."],

  yield_10y: ["10-year Treasury yield", "The long-term rate backdrop."],
  curve_slope: ["Yield curve slope", "10-year minus 3-month. Inversion has historically led stress."],
  credit_ratio_z: ["Credit stress", "High-yield vs investment-grade bonds — risk appetite in credit markets."],
  credit_chg_21: ["Credit 1-month change", "Whether credit has been deteriorating."],
  dollar_chg_21: ["Dollar 1-month change", "A rising dollar tightens global financial conditions."],

  sec_disp_log: ["Sector dispersion", "Are sectors rotating, or all moving together?"],
  sec_vol_spread: ["Calmest-to-wildest sector gap", "The spread between the most and least volatile sectors."],
  sec_avg_rv: ["Average sector volatility", "The typical sector's realized volatility."],
  sec_rv_rel: ["This sector vs peers", "How volatile this sector is relative to the average sector."],
  sec_rv_rank: ["Volatility rank", "Where this sector sits in the volatility ordering today."],
  sec_beta_63: ["Beta to the market", "How strongly this sector amplifies market moves."],
};

export const featureLabel = (k) => (FEATURES[k] ? FEATURES[k][0] : k);
export const featureBlurb = (k) => (FEATURES[k] ? FEATURES[k][1] : "");

export const REGIME_COPY = {
  calm: "Volatility is forecast to sit in the bottom third of its historical range.",
  normal: "Volatility is forecast in its typical range — neither unusually quiet nor stressed.",
  stressed: "Volatility is forecast in the top fifth of its historical range.",
};

// Morningstar's three super-sectors: how sensitive each sector's earnings are to the
// economic cycle. The grouping is the reading frame for the sector board — defensive
// sectors are normally the calm ones, so a defensive sector forecast as stressed is news.
export const SUPER_SECTORS = [
  {
    id: "defensive",
    name: "Defensive",
    blurb: "Demand holds up in any economy. Usually the calmest sectors.",
    tickers: ["XLP", "XLU", "XLV"],
  },
  {
    id: "cyclical",
    name: "Cyclical",
    blurb: "Rises and falls with the economy and with interest rates.",
    tickers: ["XLY", "XLF", "XLB", "XLRE"],
  },
  {
    id: "sensitive",
    name: "Sensitive",
    blurb: "Moderately tied to the cycle; often where volatility starts.",
    tickers: ["XLK", "XLC", "XLE", "XLI"],
  },
];

export const superSectorOf = (ticker) =>
  SUPER_SECTORS.find((g) => g.tickers.includes(ticker)) || null;

export const SECTOR_BLURB = {
  XLK: "Software, semiconductors and hardware. The largest weight in the S&P 500.",
  XLC: "Internet platforms, media and telecom carriers.",
  XLE: "Oil and gas producers, refiners and services — moves with crude.",
  XLI: "Aerospace, defence, machinery, airlines and freight.",
  XLY: "Retail, autos, travel and restaurants — discretionary spending.",
  XLF: "Banks, insurers, brokers and payment networks.",
  XLB: "Chemicals, metals, mining and packaging.",
  XLRE: "REITs: offices, data centres, towers, housing. Rate-sensitive.",
  XLP: "Food, beverages, household goods and grocers.",
  XLU: "Electric, gas and water utilities. Bond-like and rate-sensitive.",
  XLV: "Pharma, biotech, insurers and medical devices.",
};
