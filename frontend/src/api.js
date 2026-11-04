const j = async (url) => {
  const r = await fetch(url);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return r.json();
};

export const getForecast = () => j("/api/forecast");
export const getSectors = () => j("/api/sectors");
export const getPerformance = () => j("/api/performance");
export const getEquity = () => j("/api/equity");
export const getHistory = (panel = "market", ticker = null, limit = 1500) =>
  j(`/api/history?panel=${panel}&limit=${limit}${ticker ? `&ticker=${ticker}` : ""}`);
// Optional overlay: resolves to null rather than failing the page when it is absent.
export const getNews = () => j("/api/news").catch(() => null);
