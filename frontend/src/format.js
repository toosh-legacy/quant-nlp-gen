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
  vvix_log: ["Vol-of-vol (VVIX)", "How unstable the volatility estimate itself is."],
  vix_rv_gap: ["Variance risk premium", "Implied minus realized: trouble priced in that hasn't shown up in returns yet."],

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
