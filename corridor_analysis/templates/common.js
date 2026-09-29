/* Shared helpers for the corridor pages: formatting, tooltip, mark paths, resizing. */
const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString("en-US");
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const css = (v) => `var(--${v})`;
const hm = (m) => `${Math.floor(m / 60)} h ${String(Math.round(m % 60)).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const CLASSES = [
  { key: "normal_minor", label: "Normal or minor (<20 min)", color: "sev-normal" },
  { key: "moderate", label: "Moderate (20–59 min)", color: "sev-moderate" },
  { key: "severe", label: "Severe (60+ min)", color: "sev-severe" },
  { key: "canceled", label: "Canceled", color: "sev-canceled" },
];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const mlabel = (s) => `${MONTHS[+s.slice(5) - 1]} ${s.slice(0, 4)}`;
const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/* Tooltip: any element with data-tip (HTML, built from our own data) shows it. */
const tip = $("#tip");
function showTip(html, x, y) {
  tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
}
document.addEventListener("pointermove", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) showTip(t.getAttribute("data-tip"), e.clientX, e.clientY);
  else if (!e.target.closest || !e.target.closest("[data-cross]")) tip.hidden = true;
});
document.addEventListener("focusin", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) { const r = t.getBoundingClientRect(); showTip(t.getAttribute("data-tip"), r.right, r.top); }
});
document.addEventListener("focusout", () => { tip.hidden = true; });

function niceMax(v, steps = 4) {
  const raw = v / steps, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = [1, 2, 2.5, 5, 10].find((k) => k * mag >= raw) * mag;
  return { max: n * steps, step: n };
}
// Column with 4px rounded data end, square at the baseline.
function colPath(x, y, w, h, r = 4) {
  if (h <= 0) return "";
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
// Horizontal bar with 4px rounded data end.
function barPath(x, y, w, h, r = 4) {
  if (w <= 0) return "";
  r = Math.min(r, h / 2, w);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}
function legend(el, items) {
  el.innerHTML = items.map((i) => `<span><i style="background:${css(i.color)}"></i>${i.label}</span>`).join("");
}
const charts = [];
function register(fn) { charts.push(fn); fn(); }
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => charts.forEach((f) => f()), 120); });

/* 100% bar of severity classes. */
function classBar(el, counts, total, H = 54, labels = true) {
  const W = el.clientWidth;
  let x = 0, out = "";
  CLASSES.forEach((c, i) => {
    const n = counts[c.key] || 0, w = (n / total) * W, gap = i < CLASSES.length - 1 ? 2 : 0;
    const d = i === 0 ? `M${x + 4},0H${x + w - gap}V28H${x + 4}Q${x},28 ${x},24V4Q${x},0 ${x + 4},0Z`
      : i === CLASSES.length - 1 ? barPath(x, 0, w, 28) : `M${x},0H${x + w - gap}V28H${x}Z`;
    out += `<g class="hov" data-tip="<b>${c.label}</b><br>${fmt(n)} journeys · ${pct(n / total)}"><path class="mark" d="${d}" fill="${css(c.color)}"/></g>`;
    if (labels && w > 54) out += `<text class="val" x="${x + 2}" y="46">${pct(n / total)}</text>`;
    x += w;
  });
  el.innerHTML = `<svg width="${W}" height="${H}">${out}</svg>`;
}

/* Line chart with a shared crosshair: series = [{label, color, points:[{x,y}]}]. */
function lineChart(el, series, o) {
  const W = el.clientWidth, H = o.height || 260, m = { l: 48, r: 12, t: 12, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = series[0].points.map((p) => p.x);
  const x0 = o.xMin ?? Math.min(...xs), x1 = o.xMax ?? Math.max(...xs);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * iw, Y = (v) => m.t + ih - ((v - o.yMin) / (o.yMax - o.yMin)) * ih;
  let g = "";
  for (let v = o.yMin; v <= o.yMax + 1e-9; v += o.yStep) g += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${o.yFmt(v)}</text>`;
  // Ticks at the plot edges are anchored inward so they stay clear of the y labels and the frame.
  (o.xTicks(W) || []).forEach((t) => { const x = X(t.v), a = x < m.l + 24 ? "start" : x > W - m.r - 24 ? "end" : "middle"; g += `<text x="${x}" y="${H - m.b + 16}" text-anchor="${a}">${t.label}</text>`; });
  (o.refs || []).forEach((r) => { g += `<line x1="${X(r.x)}" x2="${X(r.x)}" y1="${m.t}" y2="${m.t + ih}" stroke="var(--axis)" stroke-dasharray="3 3"/><text x="${X(r.x) + 4}" y="${m.t + 10}">${r.label}</text>`; });
  const lines = series.map((s) => {
    // A missing value breaks the line rather than bridging the gap.
    const d = s.points.map((p, i) => p.y == null ? "" : `${i && s.points[i - 1].y != null ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="${css(s.color)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join("");
  const id = el.id;
  el.innerHTML = `<svg width="${W}" height="${H}">${g}<line class="baseline" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>${lines}<g id="${id}-cross" pointer-events="none"></g><rect data-cross="1" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
  const rect = el.querySelector("rect[data-cross]"), cross = el.querySelector(`#${id}-cross`);
  const move = (clientX, clientY) => {
    const b = rect.getBoundingClientRect();
    const v = x0 + ((clientX - b.left) / b.width) * (x1 - x0);
    let best = xs[0]; xs.forEach((x) => { if (Math.abs(x - v) < Math.abs(best - v)) best = x; });
    const cx = X(best);
    cross.innerHTML = `<line x1="${cx}" x2="${cx}" y1="${m.t}" y2="${m.t + ih}" stroke="var(--ink-2)" stroke-width="1"/>` +
      series.map((s) => { const p = s.points.find((q) => q.x === best); return p && p.y != null ? `<circle cx="${cx}" cy="${Y(p.y)}" r="4.5" fill="${css(s.color)}" stroke="var(--surface)" stroke-width="2"/>` : ""; }).join("");
    showTip(`<b>${o.xFmt(best)}</b><br>` + series.map((s) => { const p = s.points.find((q) => q.x === best); return `${s.label}: ${p && p.y != null ? o.tipFmt(p) : "–"}`; }).join("<br>"), clientX, clientY);
  };
  rect.addEventListener("pointermove", (e) => move(e.clientX, e.clientY));
  rect.addEventListener("pointerleave", () => { cross.innerHTML = ""; tip.hidden = true; });
}
