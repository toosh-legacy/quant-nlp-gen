// Two backends, one contract. Served by FastAPI, the dashboard calls /api/* live. Built with
// VITE_STATIC=1 (GitHub Pages), it reads the same responses pre-exported as JSON files by
// scripts/export_static.py, so nothing below the fetch layer knows which mode it is in.
const STATIC = import.meta.env.VITE_STATIC === "1";
const BASE = import.meta.env.BASE_URL; // "/" live, "./" static

const j = async (url) => {
  const r = await fetch(url);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json();
};

const route = (name) => (STATIC ? `${BASE}api/${name}.json` : `/api/${name}`);

export const getForecast = () => j(route("forecast"));
export const getSectors = () => j(route("sectors"));
export const getPerformance = () => j(route("performance"));
export const getEquity = () => j(route("equity"));
export const getHistory = (panel = "market", ticker = null, limit = 1500) => {
  if (!STATIC) {
    return j(`/api/history?panel=${panel}&limit=${limit}${ticker ? `&ticker=${ticker}` : ""}`);
  }
  const file = panel === "market" ? "market" : `${panel}/${ticker}`;
  return j(`${BASE}api/history/${file}.json`).then((h) => ({ ...h, series: h.series.slice(-limit) }));
};
// Optional overlay: resolves to null rather than failing the page when it is absent.
export const getNews = () => j(route("news")).catch(() => null);
