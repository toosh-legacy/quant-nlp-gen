import React from "react";
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card } from "./ui";

const AXIS = { stroke: "#5d6b83", fontSize: 11 };
const GRID = "#263042";

const tooltipStyle = {
  contentStyle: {
    background: "#1b2130", border: "1px solid #263042",
    borderRadius: 8, fontSize: 12.5,
  },
  labelStyle: { color: "#8b98ae", marginBottom: 4 },
};

/**
 * Forecast against what actually happened, on out-of-sample data only.
 *
 * This is the chart that earns or loses the user's trust, so it shows the misses as
 * plainly as the hits. Every point was produced by a model trained only on data before
 * that year.
 */
export function HistoryChart({ history, title, sub }) {
  const hasBand = history.series.some((d) => d.lo != null);
  const data = history.series.map((d) => ({
    date: d.date,
    Forecast: d.forecast,
    Realized: d.realized,
    // Recharts stacks an area from a [lo, hi] pair, which draws the band directly.
    Range: d.lo != null && d.hi != null ? [d.lo, d.hi] : null,
  }));

  return (
    <Card title={title} sub={sub}>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }}
                 minTickGap={70} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false}
                 tickFormatter={(v) => `${v}%`} width={52} />
          <Tooltip {...tooltipStyle}
                   formatter={(v) => Array.isArray(v)
                     ? `${Number(v[0]).toFixed(1)}–${Number(v[1]).toFixed(1)}%`
                     : `${Number(v).toFixed(1)}%`} />
          <Legend wrapperStyle={{ fontSize: 12.5, paddingTop: 8 }} iconType="plainline" />
          {hasBand && (
            <Area type="monotone" dataKey="Range" stroke="none" fill="#5aa9ff"
                  fillOpacity={0.15} name="80% range" />
          )}
          <Line type="monotone" dataKey="Realized" stroke="#8b98ae" dot={false}
                strokeWidth={1.4} />
          <Line type="monotone" dataKey="Forecast" stroke="#5aa9ff" dot={false}
                strokeWidth={1.8} />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="sub" style={{ margin: "10px 0 0" }}>
        Annualized volatility, percent. Grey is what actually happened; blue is what the
        model said in advance. The shaded band is the calibrated 80% range — it is allowed
        to be wrong one day in five, and the Track record tab reports how often it actually
        was.
      </p>
    </Card>
  );
}

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
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="gModel" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#5aa9ff" stopOpacity={0.32} />
              <stop offset="100%" stopColor="#5aa9ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }}
                 minTickGap={70} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52}
                 tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
          <Tooltip {...tooltipStyle} formatter={(v) => `$${Number(v).toFixed(2)}`} />
          <Legend wrapperStyle={{ fontSize: 12.5, paddingTop: 8 }} iconType="plainline" />
          <Area type="monotone" dataKey="Buy & hold" stroke="#5d6b83" fill="none"
                strokeWidth={1.3} />
          <Area type="monotone" dataKey="Sized on current vol" stroke="#d9a441" fill="none"
                strokeWidth={1.3} strokeDasharray="4 3" />
          <Area type="monotone" dataKey="Sized on forecast" stroke="#5aa9ff"
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
