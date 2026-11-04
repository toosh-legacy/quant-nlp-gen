import React from "react";

// One small authored icon set: 16px grid, 1.6 stroke, round caps. Glyphs (▲ ⚡ ✕ →) render
// differently on every platform and font; these render the same everywhere.
const Svg = ({ size = 14, children, title, ...rest }) => (
  <svg viewBox="0 0 16 16" width={size} height={size} fill="none" stroke="currentColor"
       strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
       aria-hidden={title ? undefined : true} role={title ? "img" : undefined}
       className="icon" {...rest}>
    {title && <title>{title}</title>}
    {children}
  </svg>
);

export const ArrowUp = (p) => <Svg {...p}><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" /></Svg>;
export const ArrowDown = (p) => <Svg {...p}><path d="M8 3v10M3.5 8.5 8 13l4.5-4.5" /></Svg>;
export const ArrowRight = (p) => <Svg {...p}><path d="M3 8h10M8.5 3.5 13 8l-4.5 4.5" /></Svg>;
export const ArrowLeft = (p) => <Svg {...p}><path d="M13 8H3M7.5 3.5 3 8l4.5 4.5" /></Svg>;
export const Close = (p) => <Svg {...p}><path d="M4 4l8 8M12 4l-8 8" /></Svg>;
export const Bolt = (p) => <Svg {...p}><path d="M9 1.8 3.5 9h4L7 14.2 12.5 7h-4L9 1.8Z" /></Svg>;
export const External = (p) => <Svg {...p}><path d="M9.5 2.5h4v4M13.5 2.5 7.5 8.5M12 9.5V13a.5.5 0 0 1-.5.5h-8A.5.5 0 0 1 3 13V5a.5.5 0 0 1 .5-.5H7" /></Svg>;
export const Grid = (p) => <Svg {...p}><path d="M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z" /></Svg>;

export const Direction = ({ rising, size }) =>
  rising ? <ArrowUp size={size} title="rising" /> : <ArrowDown size={size} title="falling" />;
