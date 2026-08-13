import React from "react";
import { Card, Chip } from "./ui";
import { pct, pct01, signed } from "../format";

const barColor = (regime) =>
  regime === "calm" ? "var(--calm)" : regime === "stressed" ? "var(--stressed)" : "var(--normal)";

/**
 * The sector stability ranking — the product's second headline.
 *
 * Two decisions worth noting. Rows are sorted by forecast volatility ascending, so the
 * most stable sector is at the top and the ordering IS the answer. And every row carries
 * its own measured accuracy: the model does not forecast all sectors equally well, and a
 * ranking presented without that invites the reader to trust each row equally.
 */
export function SectorTable({ sectors, horizon, selected, onSelect }) {
  const rows = sectors.ranking;
  const max = Math.max(...rows.map((r) => r.forecast_vol_ann));

  return (
    <Card
      title="Sector stability ranking"
      sub={`Forecast volatility over the next ${horizon} trading days · most stable first · as of ${sectors.as_of}`}
    >
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th className="rank">#</th>
              <th>Sector</th>
              <th>Forecast volatility</th>
              <th className="num">Now</th>
              <th className="num">Change</th>
              <th>Outlook</th>
              <th className="num" title="Out-of-sample rise/fall accuracy for this sector, across every walk-forward year">
                Hit rate
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr
                key={s.ticker}
                className={`clickable ${selected === s.ticker ? "on" : ""}`}
                onClick={() => onSelect(selected === s.ticker ? null : s.ticker)}
              >
                <td className="rank">{s.stability_rank}</td>
                <td>
                  {s.sector}
                  <span className="muted mono" style={{ fontSize: 11.5, marginLeft: 7 }}>
                    {s.ticker}
                  </span>
                </td>
                <td className="barcell">
                  <span
                    className="bar"
                    style={{
                      width: `${(s.forecast_vol_ann / max) * 100}%`,
                      background: barColor(s.regime),
                    }}
                  />
                  <span className="val">{pct(s.forecast_vol_ann)}</span>
                </td>
                <td className="num muted">{pct(s.current_vol_ann)}</td>
                <td className={`num ${s.change_pct >= 0 ? "bad-txt" : "good-txt"}`}>
                  {signed(s.change_pct)}%
                </td>
                <td>
                  <Chip regime={s.regime} />
                </td>
                <td className="num muted">
                  {s.track_record ? pct01(s.track_record.dir_accuracy) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="sub" style={{ margin: "14px 0 0" }}>
        Click a sector to load its out-of-sample forecast history below. "Hit rate" is how
        often the model called that sector's volatility direction correctly across 13
        walk-forward years — not a claim about today.
      </p>
    </Card>
  );
}
