import React, { useMemo, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card } from "./ui";
import { AXIS, C, tooltipStyle } from "../theme";

const GRID = C.line;

/**
 * Forecast against what actually happened, on out-of-sample data only.
 *
 * This is the chart that earns or loses the user's trust, so it shows the misses as
 * plainly as the hits. Every point was produced by a model trained only on data before
 * that year.
 */
// Trading sessions per range. "All" keeps 2020 in view, which flattens everything after it,
// so the default is the last three years and the y-axis rescales to what is visible.
const RANGES = [["6M", 126], ["1Y", 252], ["3Y", 756], ["All", Infinity]];

export function RangeTabs({ value, onChange, options = RANGES, label = "Time range" }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([label]) => (
        <button key={label} type="button" className={value === label ? "on" : ""}
                aria-pressed={value === label} onClick={() => onChange(label)}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function HistoryChart({ history, title, sub, height = 420, defaultRange = "3Y" }) {
  const [range, setRange] = useState(defaultRange);
  const n = RANGES.find(([l]) => l === range)[1];
  const series = useMemo(() => history.series.slice(-Math.min(n, history.series.length)),
                         [history, n]);
  // Log by default: volatility spikes are multiplicative (and the model forecasts log vol),
  // so on a linear axis one crash flattens every other year into the floor.
  const [scale, setScale] = useState("Log");
  const hasBand = series.some((d) => d.lo != null);
  const first = series[0]?.date, last = series.at(-1)?.date;
  const vals = series.flatMap((d) => [d.forecast, d.realized, d.lo, d.hi]).filter((v) => v > 0);
  const yLo = Math.max(1, Math.min(...vals) * 0.9), yHi = Math.max(...vals) * 1.1;
  const logTicks = [2.5, 5, 10, 20, 40, 80, 160, 320].filter((t) => t >= yLo && t <= yHi);
  const yAxis = scale === "Log"
    ? { scale: "log", domain: [yLo, yHi], ticks: logTicks, allowDataOverflow: true }
    : { domain: [0, "auto"] };
  const data = series.map((d) => ({
    date: d.date,
    Forecast: d.forecast,
    Realized: d.realized,
    // Recharts stacks an area from a [lo, hi] pair, which draws the band directly.
    Range: d.lo != null && d.hi != null ? [d.lo, d.hi] : null,
  }));

  return (
    <Card title={title} sub={sub} action={
      <div className="seg-row">
        <RangeTabs value={scale} onChange={setScale} options={[["Log"], ["Linear"]]} label="Axis scale" />
        <RangeTabs value={range} onChange={setRange} />
      </div>
    }>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 4, right: 22, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }}
                 minTickGap={80} tickFormatter={shortDate} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} {...yAxis}
                 tickFormatter={(v) => `${+Number(v).toFixed(1)}%`} width={52} />
          <Tooltip {...tooltipStyle} labelFormatter={longDate}
                   formatter={(v) => Array.isArray(v)
                     ? `${Number(v[0]).toFixed(1)}–${Number(v[1]).toFixed(1)}%`
                     : `${Number(v).toFixed(1)}%`} />
          <Legend wrapperStyle={{ fontSize: 12.5, paddingTop: 8 }} iconType="plainline" />
          {hasBand && (
            <Area type="monotone" dataKey="Range" stroke="none" fill={C.accent}
                  fillOpacity={0.15} name="80% range" />
          )}
          <Line type="monotone" dataKey="Realized" stroke={C.muted} dot={false}
                strokeWidth={1.4} />
          <Line type="monotone" dataKey="Forecast" stroke={C.accent} dot={false}
                strokeWidth={1.8} />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="sub" style={{ margin: "10px 0 0" }}>
        {first && <>{longDate(first)} to {longDate(last)} · {series.length} sessions. </>}
        Annualized volatility, percent. Grey is what actually happened; blue is what the
        model said in advance. The shaded band is the calibrated 80% range — it is allowed
        to be wrong one day in five, and the Track record tab reports how often it actually
        was.
      </p>
    </Card>
  );
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const shortDate = (d) => `${MONTHS[+d.slice(5, 7) - 1]} ’${d.slice(2, 4)}`;
export const longDate = (d) => `${+d.slice(8, 10)} ${MONTHS[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;

/** Growth of $1 under each sizing rule in the volatility-targeting simulation. */
export function EquityChart({ equity }) {
  const data = equity.series.map((d) => ({
    date: d.date,
    "Buy & hold": d.buy_and_hold,
    "Sized on current vol": d.persistence_sized,
    "Sized on forecast": d.model_sized,
  }));

  return (
    <Card
      title="Would acting on it have helped?"
      sub="Growth of $1 · volatility targeting at 15% annualized · 1bp cost per trade · out-of-sample"
    >
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={data} margin={{ top: 4, right: 22, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="gModel" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={C.accent} stopOpacity={0.32} />
              <stop offset="100%" stopColor={C.accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }}
                 minTickGap={80} tickFormatter={shortDate} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52}
                 tickFormatter={(v) => `$${Number(v).toFixed(2)}`} domain={["auto", "auto"]} />
          <Tooltip {...tooltipStyle} labelFormatter={longDate} formatter={(v) => `$${Number(v).toFixed(2)}`} />
          <Legend wrapperStyle={{ fontSize: 12.5, paddingTop: 8 }} iconType="plainline" />
          <Area type="monotone" dataKey="Buy & hold" stroke={C.dim} fill="none"
                strokeWidth={1.3} />
          <Area type="monotone" dataKey="Sized on current vol" stroke={C.normal} fill="none"
                strokeWidth={1.3} strokeDasharray="4 3" />
          <Area type="monotone" dataKey="Sized on forecast" stroke={C.accent}
                fill="url(#gModel)" strokeWidth={1.9} />
        </AreaChart>
      </ResponsiveContainer>
      <p className="sub" style={{ margin: "10px 0 0" }}>
        The honest comparison is blue against <em>amber</em>, not against grey. Amber already
        uses a volatility estimate — it just measures current volatility instead of
        forecasting it. Beating it is what shows the forecast added something.
      </p>
    </Card>
  );
}
