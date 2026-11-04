// Chart colours in one place. CSS custom properties cannot be read by Recharts props
// without a DOM lookup, so the palette is mirrored here from styles.css.
export const C = {
  text: "#e8edf6",
  muted: "#8e9bb2",
  dim: "#5f6c84",
  line: "#222c3e",
  panel2: "#1a2232",
  calm: "#34d399",
  normal: "#f2b84b",
  stressed: "#f6656b",
  accent: "#5eb1ff",
  accent2: "#8b7cff",
};

export const REGIME_COLOR = { calm: C.calm, normal: C.normal, stressed: C.stressed };
export const regimeColor = (r) => REGIME_COLOR[r] || C.muted;

// News pressure 0-100 -> colour on the same calm/normal/stressed scale as volatility.
export const pressureColor = (p) =>
  p == null ? C.dim : p >= 55 ? C.stressed : p >= 38 ? C.normal : C.calm;

export const AXIS = { stroke: C.dim, fontSize: 11 };

export const tooltipStyle = {
  contentStyle: {
    background: "rgba(14,19,29,.96)", border: `1px solid ${C.line}`,
    borderRadius: 10, fontSize: 12.5, boxShadow: "0 10px 30px rgba(0,0,0,.5)",
  },
  labelStyle: { color: C.muted, marginBottom: 4 },
  cursor: { stroke: C.line },
};
