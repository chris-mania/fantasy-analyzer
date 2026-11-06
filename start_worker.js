// Fantasy Mania start/sit pages: /start/ and /start/<player>-vs-<player>/
// Runs on Cloudflare Pages only for /start addresses (see _routes.json). Everything else on the
// site is served exactly as before. Numbers come from /start-data.json, rebuilt about once an hour.
// Source lives in the repo as start_worker.js; seo_pages.py copies it to public/_worker.js.

const SITE = "https://fantasymania.org";
let CACHE = { at: 0, data: null, by: null };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname === "fantasymania.pages.dev") return Response.redirect(SITE + url.pathname + url.search, 301);
    if (!url.pathname.startsWith("/start")) return env.ASSETS.fetch(request);
    if (url.pathname === "/start") return Response.redirect(url.origin + "/start/", 301);
    try {
      const data = await load(env, url);
      const rest = url.pathname.slice("/start/".length).replace(/\/+$/, "");
      if (!rest) return html(hub(data), 200);
      if (!url.pathname.endsWith("/")) return Response.redirect(url.origin + url.pathname + "/" + url.search, 301);
      const i = rest.indexOf("-vs-");
      const a = i > 0 ? data.by[rest.slice(0, i)] : null, b = i > 0 ? data.by[rest.slice(i + 4)] : null;
      if (!a || !b || a === b) return html(notFound(data), 404);
      const canon = [a.s, b.s].sort().join("-vs-");
      if (canon !== rest) return Response.redirect(url.origin + "/start/" + canon + "/", 301);
      return html(matchup(data, a, b), 200);
    } catch (e) {
      return new Response("Start/sit pages are updating. Try again in a minute.", { status: 503, headers: { "content-type": "text/plain; charset=utf-8", "retry-after": "60" } });
    }
  },
};

async function load(env, url) {
  if (CACHE.data && Date.now() - CACHE.at < 120000) return CACHE.data;
  const r = await env.ASSETS.fetch(new Request(url.origin + "/start-data.json"));
  if (!r.ok) throw new Error("no data");
  const d = await r.json();
  d.by = {};
  d.players.forEach((p) => { d.by[p.s] = p; });
  CACHE = { at: Date.now(), data: d };
  return d;
}

function html(body, status) {
  return new Response(body, { status, headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=300" } });
}

// ---------- helpers ----------
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f1 = (v) => (v == null || isNaN(v) ? "—" : (+v).toFixed(1));
const posName = (p) => (p === "DST" ? "D/ST" : p);
const ring = (v) => (v == null ? "#9aa3ad" : v >= 80 ? "#22a45d" : v >= 60 ? "#e3a21a" : "#e2505b");
const ink = (v) => (v == null ? "#40474f" : v >= 80 ? "#16803f" : v >= 60 ? "#9a6700" : "#c0303c");
const tier = (v) => (v >= 90 ? "Elite" : v >= 82 ? "Excellent" : v >= 74 ? "Strong" : v >= 65 ? "Starter" : "Depth");
const initials = (p) => (p.pos === "DST" ? p.tm : p.n.replace(/\b(Jr\.|Sr\.|II|III|IV)\b/g, "").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase());
function short(p) {
  if (p.pos === "DST") return p.n.replace(/\s*D\/ST$/, "");
  const w = p.n.replace(/,?\s+(Jr\.|Sr\.|II|III|IV)$/, "").split(/\s+/);
  return w.length > 1 ? w.slice(1).join(" ") : p.n;
}
const withNick = (p) => (p.nk && p.nk.length ? `${p.n} (${p.nk[0]})` : p.n);
const pts = (p) => (p.pj ? p.pj.p : null);
const playing = (p) => !p.bye && !p.out;
const START_AT = { QB: 12, RB: 24, WR: 30, TE: 12, K: 12, DST: 12 };

function listed(d, a, b) {
  if (!(a.in && b.in)) return false;
  const same = a.pos === b.pos, flex = (a.pos === "RB" && b.pos === "WR") || (a.pos === "WR" && b.pos === "RB");
  if (!same && !flex) return false;
  if (same && (a.pos === "K" || a.pos === "DST")) return true;
  return pts(a) != null && pts(b) != null && Math.abs(pts(a) - pts(b)) < d.gap;
}

function verdict(d, a, b) {
  const wk = d.next;
  if (!playing(a) && !playing(b)) return { w: null, l: null, tag: "Neither", line: `Neither ${a.n} nor ${b.n} is expected to play in Week ${wk}. Look for another option.` };
  for (const [x, y] of [[a, b], [b, a]]) {
    if (!playing(x) && playing(y)) {
      const why = x.bye ? "is on bye" : "is ruled out";
      return { w: y, l: x, tag: "Easy call", line: `Start ${y.n}. ${x.n} ${why} in Week ${wk}.` };
    }
  }
  let w = a, l = b;
  const sa = a.wr == null ? -1 : a.wr, sb = b.wr == null ? -1 : b.wr;
  if (sb > sa || (sb === sa && (pts(b) || 0) > (pts(a) || 0))) { w = b; l = a; }
  const gap = Math.abs(sa - sb), pgap = Math.abs((pts(a) || 0) - (pts(b) || 0));
  const tag = gap < 1.5 && pgap < 1 ? "Coin flip" : gap < 4 && pgap < 2.5 ? "Slight edge" : gap < 8 && pgap < 5 ? "Lean" : "Clear start";
  const vsTxt = (p) => (p.opp ? ` against ${p.opp}` : "");
  let line = `Start ${w.n} over ${l.n} in Week ${wk}. Fantasy Mania projects ${short(w)} for ${f1(pts(w))} points${vsTxt(w)}, versus ${f1(pts(l))} for ${short(l)}${vsTxt(l)}.`;
  line += ` Week ${wk} Rating: ${f1(w.wr)} to ${f1(l.wr)}.`;
  if (pts(l) != null && pts(w) != null && pts(l) > pts(w)) line += ` ${short(l)} projects slightly higher, but ${short(w)} rates better once matchup and role are counted.`;
  return { w, l, tag, line };
}

function startTake(d, p) {
  const wk = d.next, pn = posName(p.pos);
  if (p.bye) return `${p.n} is on bye in Week ${wk}.`;
  if (p.out) return `${p.n} is ruled out for Week ${wk}.`;
  const at = START_AT[p.pos] || 12;
  const call = p.wk == null ? "a deep-league option" : p.wk <= at ? "a solid start" : p.wk <= at * 2 ? (p.pos === "RB" || p.pos === "WR" ? "a flex option" : "a streaming option") : "a bench or deep-league play";
  let s = `${p.n} is ${call} in Week ${wk}: ${pn}${p.wk || "—"} by Fantasy Mania's Week ${wk} Rating (${f1(p.wr)}), projected for ${f1(pts(p))} points${p.opp ? " against " + p.opp : ""}`;
  if (p.pj && p.pj.lo != null && p.pj.hi != null) s += `, with a range of ${f1(p.pj.lo)} to ${f1(p.pj.hi)}`;
  s += ".";
  if (p.inj && p.inj.l) s += ` Injury report: ${p.inj.l}${p.inj.t ? " (" + p.inj.t + ")" : ""}.`;
  return s;
}

function reasons(d, w, l) {
  if (!w || !l) return [];
  const out = [], a = w, b = l;
  if (pts(a) != null && pts(b) != null) out.push(`<b>Projection:</b> ${esc(short(a))} ${f1(pts(a))} vs ${esc(short(b))} ${f1(pts(b))} points.`);
  if (a.ma != null && b.ma != null && Math.abs(a.ma - b.ma) >= 0.5) {
    const m = a.ma > b.ma ? a : b, o = m === a ? b : a;
    out.push(`<b>Better matchup:</b> ${esc(short(m))}${m.opp ? " vs " + esc(m.opp) : ""} (${m.ma > 0 ? "+" : ""}${f1(m.ma)} to his Week ${d.next} Rating, vs ${o.ma > 0 ? "+" : ""}${f1(o.ma)} for ${esc(short(o))}).`.replace("his Week", a.pos === "DST" ? "its Week" : "his Week"));
  }
  if (a.pj && b.pj && a.pj.hi != null && b.pj.hi != null) {
    const c = a.pj.hi >= b.pj.hi ? a : b, f = (a.pj.lo || 0) >= (b.pj.lo || 0) ? a : b;
    out.push(c === f
      ? `<b>Ceiling and floor:</b> ${esc(short(c))} has both the higher ceiling (up to ${f1(c.pj.hi)}) and the safer floor (${f1(c.pj.lo)}).`
      : `<b>Ceiling and floor:</b> ${esc(short(c))} has the higher ceiling (up to ${f1(c.pj.hi)}); ${esc(short(f))} has the safer floor (${f1(f.pj.lo)}). Need a big week? Lean ${esc(short(c))}.`);
  }
  if (a.sh != null && b.sh != null && (a.pos === b.pos) && (a.pos === "RB" || a.pos === "WR" || a.pos === "TE")) {
    const what = a.pos === "RB" ? "of team touches" : "of team targets";
    out.push(`<b>Role:</b> ${esc(short(a))} gets ${f1(a.sh)}% ${what}, ${esc(short(b))} ${f1(b.sh)}%.`);
  }
  const avg = (x) => { const v = (x.l3 || []).filter((n) => n != null); return v.length ? v.reduce((s, n) => s + n, 0) / v.length : null; };
  if (avg(a) != null && avg(b) != null) out.push(`<b>Last 3 games:</b> ${esc(short(a))} ${f1(avg(a))} a game, ${esc(short(b))} ${f1(avg(b))}.`);
  [a, b].forEach((p) => { if (p.inj && p.inj.l) out.push(`<b>Injury report:</b> ${esc(p.n)}, ${esc(p.inj.l)}${p.inj.t ? " (" + esc(p.inj.t) + ")" : ""}.`); });
  [a, b].forEach((p) => { if (p.pl && p.pl.f != null) out.push(`<b>Already played:</b> ${esc(p.n)} scored ${f1(p.pl.f)} (${esc(p.pl.s)}).`); });
  return out;
}

function partners(d, p, skip) {
  const at = pts(p);
  return d.players
    .filter((q) => q !== p && q !== skip && q.pos === p.pos && listed(d, p, q))
    .sort((x, y) => (Math.abs((x.wk || 999) - (p.wk || 999)) + Math.abs((pts(x) || 0) - at) / 10) - (Math.abs((y.wk || 999) - (p.wk || 999)) + Math.abs((pts(y) || 0) - at) / 10))
    .slice(0, 6);
}

// ---------- page pieces ----------
const CA = "#2563eb", CB = "#d9731f"; // line colors for player A and B
const BOOM = { QB: [25, 12], RB: [20, 8], WR: [20, 8], TE: [15, 5], K: [12, 5], DST: [12, 3] };
const POSW = { QB: "quarterbacks", RB: "running backs", WR: "wide receivers", TE: "tight ends", K: "kickers", DST: "defenses" };

const CSS = `
:root{--bg:#e9edf2;--card:#fff;--ink:#0b1830;--muted:#5a616a;--line:#dde3ea;--blue:#2563eb;--blue2:#1d4ed8;--green:#16803f;--greenbg:#e7f6ed}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
a{color:var(--blue2)}
.top{background:#e7edf3;border-bottom:1px solid #c4ccd6;position:sticky;top:0;z-index:5}
.nav{max-width:1100px;margin:auto;height:58px;padding:0 16px;display:flex;align-items:center;gap:22px}
.brand{font-weight:900;font-style:italic;font-size:22px;letter-spacing:-.6px;text-decoration:none;color:#060709;white-space:nowrap}.brand b{color:#1d4ed8;font-weight:900}
.links{display:flex;gap:18px;overflow-x:auto;scrollbar-width:none}.links a{color:#40474f;text-decoration:none;font-weight:600;font-size:15px;white-space:nowrap;padding:6px 0}.links a.on{color:#060709;box-shadow:inset 0 -2px #060709}
.wrap{max-width:1100px;margin:auto;padding:20px 16px 48px}
.crumb{font-size:13px;color:var(--muted);margin-bottom:10px}.crumb a{color:var(--muted)}
.kicker{font-size:11px;color:var(--blue2);font-weight:800;text-transform:uppercase;letter-spacing:1.2px;display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.ppr{display:inline-block;font-size:10.5px;letter-spacing:.4px;color:#40474f;background:#fff;border:1px solid #d9dfe7;border-radius:99px;padding:2px 8px;text-transform:none;font-weight:700}
h1{font-size:clamp(28px,6vw,44px);line-height:1.08;letter-spacing:-1px;margin:6px 0 6px;font-weight:800}
.sub{color:var(--muted);margin:0 0 16px;font-size:15px}
.card{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 26px -20px rgba(20,40,80,.55);border:1px solid #e4eaf3}
.vsWrap{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:6px;margin:6px 0 14px}
.vsMid{font-weight:900;font-style:italic;font-size:34px;color:#c0303c;letter-spacing:-2px;text-shadow:0 2px 0 rgba(0,0,0,.08)}
.pl{text-align:center;padding:18px 12px 16px;position:relative}
.av{width:84px;height:84px;border-radius:50%;margin:0 auto 10px;display:grid;place-items:center;font-weight:800;font-size:26px;color:#5a616a;background:radial-gradient(circle at 50% 35%,#f4f6f9,#d8dee5);box-shadow:0 0 0 5px var(--r) inset,0 0 0 9px #fff inset}
.pl .nm{font-weight:800;font-size:clamp(17px,3.4vw,24px);text-decoration:none;color:var(--ink);letter-spacing:-.3px}.pl .nm:hover{text-decoration:underline}
.pl .meta{color:var(--muted);font-size:13px;font-weight:700;margin-top:2px}
.pl .big{font-size:clamp(34px,7vw,52px);font-weight:900;letter-spacing:-1.5px;margin-top:6px;line-height:1}
.pl .lbl{font-size:10.5px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-weight:800;margin-top:4px}
.mr{margin-top:10px;font-size:14px;font-weight:700;color:#272c33}.mr b{font-size:17px}
.pl.win{outline:3px solid #22a45d;outline-offset:-3px}.winB{position:absolute;top:10px;right:10px;background:#22a45d;color:#fff;font-size:11px;font-weight:800;border-radius:99px;padding:3px 10px;letter-spacing:.5px}
.iBtn{display:inline-flex;width:28px;height:28px;align-items:center;justify-content:center;border-radius:50%;color:var(--blue2);font-weight:800;text-decoration:none;margin-left:2px;vertical-align:middle}
.mHelp{background:#f3f6fb;border:1px solid #d9e2f1;border-radius:14px;padding:4px 16px;margin:0 0 6px}.mHelp summary{cursor:pointer;font-weight:800;color:var(--blue2);padding:10px 0;min-height:44px;display:flex;align-items:center}.mHelp p{margin:0 0 10px;font-size:14.5px;color:#272c33}
details.info{display:inline}details.info summary{display:inline-flex;list-style:none;cursor:pointer;color:var(--blue2);font-weight:800;margin-left:4px;width:22px;height:22px;align-items:center;justify-content:center;border-radius:50%}details.info summary::-webkit-details-marker{display:none}
details.info[open] .pop{display:block}.pop{display:none;text-align:left;font-size:13.5px;font-weight:500;color:#272c33;background:#f3f6fb;border:1px solid #d9e2f1;border-radius:12px;padding:10px 12px;margin-top:8px;line-height:1.5}
.tabs{display:flex;gap:4px;background:#e1e7ef;border-radius:14px;padding:4px;margin:14px 0 12px;overflow-x:auto;scrollbar-width:none}
.tabs button{flex:1 0 auto;border:0;background:none;font:inherit;font-weight:700;font-size:14px;color:#40474f;padding:10px 14px;border-radius:10px;cursor:pointer;min-height:44px;white-space:nowrap}
.tabs button[aria-selected="true"]{background:#0b1830;color:#fff}.tabs .ts{display:none}
.js .tp{display:none}.js .tp.on{display:block}.tp>h2.tpH{margin-top:0}.js .tp>h2.tpH{position:absolute;left:-9999px}
.verdict{border-left:6px solid #22a45d}.vTag{display:inline-block;background:var(--greenbg);color:var(--green);font-weight:800;font-size:12px;border-radius:99px;padding:4px 10px;margin-bottom:8px}
.vHead{font-size:clamp(22px,5vw,32px);font-weight:800;letter-spacing:-.6px;margin:0 0 8px}.vHead span{color:var(--green)}.vLine{margin:0;font-size:16px}
ul.why{margin:14px 0 0;padding-left:20px}ul.why li{margin:7px 0}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:10px 8px;border-top:1px solid var(--line);text-align:center}th{font-weight:600;color:var(--muted);font-size:13px}
thead th,thead td{border-top:0;font-weight:800;font-size:13px;color:var(--ink)}td.w{color:var(--green);font-weight:800}tbody th{text-align:center}
.chart{width:100%;max-width:760px;height:auto;display:block;margin:0 auto}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;font-weight:700;margin:6px 0 4px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mini h3{margin:0 0 8px;font-size:16px}.stat{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-top:1px solid var(--line);font-size:14.5px}.stat:first-of-type{border-top:0}.stat b{font-weight:800}
.rng{position:relative;height:8px;border-radius:99px;background:#eef2f6;margin:8px 0 4px}.rng i{position:absolute;top:0;bottom:0;border-radius:99px}.rng b{position:absolute;top:-4px;width:4px;height:16px;border-radius:2px;background:#0b1830}
.small{font-size:12.5px;color:var(--muted)}
h2{font-size:20px;margin:26px 0 10px;letter-spacing:-.3px}
.bkRow{display:grid;grid-template-columns:44px 1fr 44px;align-items:center;gap:10px;padding:9px 0;border-top:1px solid var(--line)}.bkRow:first-child{border-top:0}
.bkL{font-weight:700;font-size:14px;text-align:center;margin-bottom:4px}.bars{display:grid;grid-template-columns:1fr 1fr;gap:4px}.bar{height:10px;border-radius:99px;background:#eef2f6;position:relative;overflow:hidden}.bar i{position:absolute;top:0;bottom:0;border-radius:99px}.bar.l i{right:0}.bar.r i{left:0}
.bkN{font-weight:800;font-size:15px;text-align:center}.bkN.w{color:var(--green)}
.faq h3{font-size:16px;margin:16px 0 4px}.faq h3:first-child{margin-top:0}.faq p{margin:0;color:#272c33}
.chips{display:flex;flex-wrap:wrap;gap:8px;list-style:none;padding:0;margin:0}.chips a{display:inline-flex;align-items:center;min-height:40px;padding:8px 14px;border-radius:99px;background:#fff;border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700;font-size:14px}.chips a:hover{border-color:#2563eb;color:#1d4ed8}
.btns{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.btn{display:inline-flex;align-items:center;min-height:44px;padding:10px 18px;border-radius:12px;background:#0b1830;color:#fff;text-decoration:none;font-weight:700}.btn.alt{background:#fff;color:#0b1830;border:1px solid #d9dfe7}
.pick{display:grid;grid-template-columns:1fr auto 1fr auto;gap:10px;align-items:center}.pick input{width:100%;min-height:48px;border-radius:12px;border:1px solid #c4ccd6;padding:10px 14px;font:inherit;font-size:16px;background:#fff}.pick .vs{font-weight:800;color:var(--muted)}.pick button{min-height:48px;border:0;border-radius:12px;background:#0b1830;color:#fff;font:inherit;font-weight:800;padding:0 20px;cursor:pointer}
.err{color:#c0303c;font-size:14px;margin-top:8px;min-height:20px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.grid2 h3{margin:0 0 10px;font-size:16px}
.disc{font-size:12.5px;color:var(--muted);margin-top:26px;border-left:3px solid #c7d7f7;padding:10px 12px;background:#eef3fb;border-radius:0 10px 10px 0}
@media(max-width:700px){.links{display:none}.wrap{padding-top:14px}.card{padding:14px;border-radius:16px}.av{width:60px;height:60px;font-size:19px}.vsMid{font-size:22px}.vsWrap{gap:2px}.pl{padding:14px 6px 12px}.winB{position:static;display:inline-block;margin-top:8px}
 table{font-size:13.5px}th,td{padding:9px 4px}.pick{grid-template-columns:1fr}.pick .vs{text-align:center}.two{grid-template-columns:1fr}.tabs{gap:2px;padding:3px}.tabs button{padding:10px 4px;font-size:13px;flex:1 1 auto;min-width:0}.tabs .tl{display:none}.tabs .ts{display:inline}.mr{font-size:12.5px}.mr b{font-size:15px}}
`;

function shell({ title, desc, path, crumbs, body, robots, nav }) {
  const ld = { "@context": "https://schema.org", "@type": "BreadcrumbList", itemListElement: crumbs.map((c, i) => ({ "@type": "ListItem", position: i + 1, name: c[0], item: SITE + c[1] })) };
  const links = [["Rankings", "/rankings/"], ["Start/Sit", "/start/"], ["Compare", "/compare/"], ["Trade", "/trade/"], ["Team", "/team/"], ["Insights", "/insights/"]];
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(title)}</title><meta name="description" content="${esc(desc)}"><link rel="canonical" href="${SITE}${path}">${robots ? `<meta name="robots" content="${robots}">` : ""}
<meta property="og:type" content="website"><meta property="og:site_name" content="Fantasy Mania"><meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(desc)}"><meta property="og:url" content="${SITE}${path}">
<meta name="theme-color" content="#e7edf3"><link rel="icon" href="/apple-touch-icon.png"><link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>${CSS}</style><script type="application/ld+json">${JSON.stringify(ld).replace(/</g, "\\u003c")}</script></head><body>
<header class="top"><div class="nav"><a class="brand" href="/">FANTASY <b>MANIA</b></a><nav class="links">${links.map(([n, u]) => `<a href="${u}"${nav === u ? ' class="on"' : ""}>${n}</a>`).join("")}</nav></div></header>
<main class="wrap"><div class="crumb">${crumbs.map((c, i) => (i < crumbs.length - 1 ? `<a href="${c[1]}">${esc(c[0])}</a>` : esc(c[0]))).join(" › ")}</div>${body}
<p class="disc">Fantasy Mania ratings and projections are estimates, not guarantees. Injuries and game flow can change anything, so check final inactives before you set your lineup.</p></main></body></html>`;
}

function picker(d, a, b) {
  const opts = d.players.map((p) => `<option value="${esc(p.n)}">`).join("") + d.players.filter((p) => p.nk && p.nk.length).map((p) => p.nk.map((k) => `<option value="${esc(k)}">`).join("")).join("");
  const map = {}; d.players.forEach((p) => { map[p.n.toLowerCase()] = p.s; (p.nk || []).forEach((k) => { map[k.toLowerCase()] = p.s; }); });
  return `<form class="pick" id="pick" autocomplete="off"><input list="fmNames" id="pa" placeholder="First player" aria-label="First player" value="${a ? esc(a.n) : ""}"><span class="vs">or</span><input list="fmNames" id="pb" placeholder="Second player" aria-label="Second player" value="${b ? esc(b.n) : ""}"><button>Who do I start?</button></form><div class="err" id="perr" role="status"></div><datalist id="fmNames">${opts}</datalist>
<script>(function(){var M=${JSON.stringify(map).replace(/</g, "\\u003c")};document.getElementById('pick').addEventListener('submit',function(e){e.preventDefault();var a=M[document.getElementById('pa').value.trim().toLowerCase()],b=M[document.getElementById('pb').value.trim().toLowerCase()];var er=document.getElementById('perr');if(!a||!b){er.textContent='Pick two players from the list.';return}if(a===b){er.textContent='Pick two different players.';return}location.href='/start/'+[a,b].sort().join('-vs-')+'/'})})();</script>`;
}

function maniaInfo(d, p) {
  const cats = (p.bk || []).map((b) => b[0]);
  const list = cats.length ? cats.slice(0, -1).join(", ") + (cats.length > 1 ? " and " : "") + cats[cats.length - 1] : "production, usage, role and efficiency";
  return `<details class="mHelp" id="maniaHelp"><summary>What's the Mania Rating?</summary><p>It's a player's overall grade from 0 to 100, kind of like a Madden overall. Fantasy Mania combines ${esc(list.toLowerCase())}, each compared with every other ${esc(POSW[p.pos] ? POSW[p.pos].replace(/s$/, "") : "player")}, into one number. 90+ is Elite, 82+ Excellent, 74+ Strong, 65+ Starter.</p><p>The <b>Week ${d.next} projection</b> is different: it's how many points we expect this week, after the matchup and the injury report. That's the number to use for start/sit.</p></details>`;
}

function card(d, p, isWin) {
  const proj = playing(p) ? f1(pts(p)) : p.bye ? "BYE" : "OUT";
  return `<div class="card pl${isWin ? " win" : ""}">${isWin ? `<span class="winB">START</span>` : ""}<div class="av" style="--r:${ring(p.mr)}">${esc(initials(p))}</div>
<a class="nm" href="/players/${p.s}/">${esc(p.n)}</a><div class="meta">${esc(p.tm)} · ${posName(p.pos)}${p.bye ? " · Bye" : p.opp ? " · vs " + esc(p.opp) : ""}</div>
<div class="big">${proj}</div><div class="lbl">Week ${d.next} projection</div>
<div class="mr">Mania Rating <b style="color:${ink(p.mr)}">${f1(p.mr)}</b><a class="iBtn" href="#maniaHelp" aria-controls="maniaHelp" aria-label="What is the Mania Rating?">ⓘ</a></div></div>`;
}

function rng(p, lo, hi, color) {
  if (!p.pj || p.pj.lo == null || p.pj.hi == null || hi <= lo) return "";
  const L = ((p.pj.lo - lo) / (hi - lo)) * 100, W = ((p.pj.hi - p.pj.lo) / (hi - lo)) * 100, P = ((p.pj.p - lo) / (hi - lo)) * 100;
  return `<div class="rng" aria-hidden="true"><i style="left:${L.toFixed(1)}%;width:${W.toFixed(1)}%;background:${color}55"></i><b style="left:calc(${P.toFixed(1)}% - 2px)"></b></div>`;
}

function lineChart(d, a, b) {
  const ga = (a.gl || []).filter((x) => x[2] != null), gb = (b.gl || []).filter((x) => x[2] != null);
  if (!ga.length && !gb.length) return "";
  const weeks = [...new Set(ga.concat(gb).map((x) => x[0]))].sort((x, y) => x - y);
  const maxW = Math.max(...weeks), minW = Math.min(...weeks);
  const vals = ga.concat(gb).map((x) => x[2]);
  const top = Math.max(10, Math.ceil(Math.max(...vals) / 5) * 5), bot = Math.min(0, Math.floor(Math.min(...vals) / 5) * 5);
  const W = 520, H = 250, L = 34, R = 12, T = 14, B = 32;
  const x = (w) => L + (maxW === minW ? (W - L - R) / 2 : ((w - minW) / (maxW - minW)) * (W - L - R));
  const y = (v) => T + (1 - (v - bot) / (top - bot)) * (H - T - B);
  let g = "";
  for (let v = bot; v <= top; v += top - bot > 30 ? 10 : 5) g += `<line x1="${L}" x2="${W - R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="#e4e9f0"/><text x="${L - 6}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end" font-size="13" fill="#5a616a">${v}</text>`;
  for (let w = minW; w <= maxW; w++) g += `<text x="${x(w).toFixed(1)}" y="${H - 10}" text-anchor="middle" font-size="13" fill="#5a616a">W${w}</text>`;
  const line = (arr, c) => {
    const s = arr.slice().sort((p, q) => p[0] - q[0]);
    let path = "", prev = null;
    s.forEach(([w, , v]) => { path += (prev != null && w === prev + 1 ? "L" : "M") + x(w).toFixed(1) + "," + y(v).toFixed(1); prev = w; });
    return `<path d="${path}" fill="none" stroke="${c}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>` + s.map(([w, , v]) => `<circle cx="${x(w).toFixed(1)}" cy="${y(v).toFixed(1)}" r="4.5" fill="#fff" stroke="${c}" stroke-width="2.5"><title>Week ${w}: ${f1(v)}</title></circle>`).join("");
  };
  return `<div class="legend"><span><i style="background:${CA}"></i>${esc(a.n)}</span><span><i style="background:${CB}"></i>${esc(b.n)}</span></div>
<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Fantasy points by week for ${esc(a.n)} and ${esc(b.n)}">${g}${line(ga, CA)}${line(gb, CB)}</svg>`;
}

function gameLog(d, a, b) {
  const map = (p) => { const m = {}; (p.gl || []).forEach(([w, o, v]) => { m[w] = [o, v]; }); return m; };
  const ma = map(a), mb = map(b);
  const weeks = [...new Set(Object.keys(ma).concat(Object.keys(mb)).map(Number))].sort((x, y) => y - x);
  const cell = (e) => (e ? `${f1(e[1])}${e[0] ? ` <span class="small">${esc(e[0])}</span>` : ""}` : `<span class="small">—</span>`);
  const rows = weeks.map((w) => { const A = ma[w], B = mb[w], wa = A && B && A[1] > B[1], wb = A && B && B[1] > A[1];
    return `<tr><th>Week ${w}</th><td class="${wa ? "w" : ""}">${cell(A)}</td><td class="${wb ? "w" : ""}">${cell(B)}</td></tr>`; }).join("");
  return `${lineChart(d, a, b)}<div style="overflow-x:auto;margin-top:8px"><table><thead><tr><th></th><td>${esc(short(a))}</td><td>${esc(short(b))}</td></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function matchupTab(d, a, b) {
  const one = (p, c) => {
    const dv = d.dvp && p.opp && d.dvp[p.opp] ? d.dvp[p.opp][p.pos] : null;
    let s = `<div class="card mini"><h3 style="color:${c}">${esc(p.n)}</h3>`;
    if (p.bye) return s + `<p>On bye in Week ${d.next}.</p></div>`;
    s += `<div class="stat"><span>Opponent</span><b>${esc(p.opp || "—")}</b></div>`;
    if (p.ma != null) s += `<div class="stat"><span>Matchup boost to his Week ${d.next} Rating</span><b style="color:${p.ma > 0 ? "#16803f" : p.ma < 0 ? "#c0303c" : "inherit"}">${p.ma > 0 ? "+" : ""}${f1(p.ma)}</b></div>`;
    if (dv) s += `<div class="stat"><span>${esc(p.opp)} allows to ${POSW[p.pos]}</span><b>${f1(dv[0])} a game</b></div><div class="stat"><span>Rank (1 = gives up the most)</span><b>#${dv[1]} of ${dv[2]}</b></div>`;
    if (p.fx && p.fx.s != null) s += `<div class="stat"><span>Game script effect on projection</span><b>${p.fx.s > 0 ? "+" : ""}${f1(p.fx.s)}</b></div>`;
    const easy = dv ? (dv[1] <= 10 ? "a good matchup" : dv[1] >= 23 ? "a tough matchup" : "an average matchup") : null;
    if (easy) s += `<p class="small" style="margin:8px 0 0">${esc(p.opp)} gives up the ${ord(dv[1])} most points to ${POSW[p.pos]}: ${easy}.</p>`;
    return s + "</div>";
  };
  return `<div class="two">${one(a, CA)}${one(b, CB)}</div>`;
}
const ord = (n) => n + (n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th");

function boomTab(d, a, b) {
  const lo = Math.min(0, (a.pj && a.pj.lo) || 0, (b.pj && b.pj.lo) || 0), hi = Math.max((a.pj && a.pj.hi) || 0, (b.pj && b.pj.hi) || 0);
  const one = (p, c) => {
    const [bm, bs] = BOOM[p.pos] || [20, 8];
    const v = (p.gl || []).map((x) => x[2]).filter((x) => x != null);
    const nb = v.filter((x) => x >= bm).length, nd = v.filter((x) => x < bs).length;
    const best = v.length ? Math.max(...v) : null, worst = v.length ? Math.min(...v) : null;
    return `<div class="card mini"><h3 style="color:${c}">${esc(p.n)}</h3>
<div class="stat"><span>Boom games (${bm}+ points)</span><b>${nb} of ${v.length}</b></div>
<div class="stat"><span>Bust games (under ${bs})</span><b>${nd} of ${v.length}</b></div>
<div class="stat"><span>Best game</span><b>${f1(best)}</b></div><div class="stat"><span>Worst game</span><b>${f1(worst)}</b></div>
${p.pj && p.pj.lo != null ? `<div class="stat"><span>Week ${d.next} floor to ceiling</span><b>${f1(p.pj.lo)} to ${f1(p.pj.hi)}</b></div>${rng(p, lo, hi, c)}` : ""}</div>`;
  };
  return `<div class="two">${one(a, CA)}${one(b, CB)}</div><p class="small" style="margin-top:10px">Floor and ceiling are the low and high ends of what Fantasy Mania expects this week. The dark mark is the projection.</p>`;
}

function projTab(d, a, b) {
  const one = (p, c) => {
    if (!playing(p)) return `<div class="card mini"><h3 style="color:${c}">${esc(p.n)}</h3><p>${p.bye ? "On bye" : "Ruled out"} in Week ${d.next}.</p></div>`;
    const j = p.pj || {}, rows = [];
    const add = (l, v, fmt = f1) => { if (v != null) rows.push(`<div class="stat"><span>${l}</span><b>${fmt(v)}</b></div>`); };
    add(`Projected points`, j.p);
    if (p.pos === "QB") { add("Pass yards", j.pyd, Math.round); add("Pass TDs", j.ptd); add("Interceptions", j.int); add("Rush yards", j.ruy, Math.round); }
    else if (p.pos === "RB") { add("Carries", j.car); add("Rush yards", j.ruy, Math.round); add("Targets", j.tgt); add("Receptions", j.rec); add("Receiving yards", j.ryd, Math.round); add("Touchdowns", j.td); }
    else if (p.pos === "WR" || p.pos === "TE") { add("Targets", j.tgt); add("Receptions", j.rec); add("Receiving yards", j.ryd, Math.round); add("Touchdowns", j.td); }
    if (p.fx) { const sg = (v) => (v > 0 ? "+" : "") + f1(v); if (p.fx.m) add("Matchup adjustment", p.fx.m, sg); if (p.fx.s) add("Game script adjustment", p.fx.s, sg); if (p.fx.i) add("Injury adjustment", p.fx.i, sg); }
    if (p.inj && p.inj.l) rows.push(`<div class="stat"><span>Injury report</span><b>${esc(p.inj.l)}${p.inj.t ? " (" + esc(p.inj.t) + ")" : ""}</b></div>`);
    return `<div class="card mini"><h3 style="color:${c}">${esc(p.n)}</h3>${rows.join("")}</div>`;
  };
  return `<div class="two">${one(a, CA)}${one(b, CB)}</div>`;
}

function breakdown(d, a, b) {
  if (!(a.bk && a.bk.length) || !(b.bk && b.bk.length)) return "";
  const help = (h) => (h ? `<details class="info"><summary aria-label="What does this mean?">ⓘ</summary><div class="pop">${esc(h)}</div></details>` : "");
  const bar = (v, c, side) => `<div class="bar ${side}"><i style="width:${Math.max(2, Math.min(100, v || 0))}%;background:${c}"></i></div>`;
  if (a.pos === b.pos) {
    const rows = a.bk.map(([l, va, h], i) => { const m = b.bk.find((x) => x[0] === l) || b.bk[i] || []; const vb = m[1];
      return `<div><div class="bkL">${esc(l)}${help(h)}</div><div class="bkRow"><span class="bkN${va > vb ? " w" : ""}">${Math.round(va)}</span><div class="bars">${bar(va, CA, "l")}${bar(vb, CB, "r")}</div><span class="bkN${vb > va ? " w" : ""}">${vb == null ? "—" : Math.round(vb)}</span></div></div>`; }).join("");
    return `<div class="card"><div class="legend" style="justify-content:space-between"><span><i style="background:${CA}"></i>${esc(short(a))}</span><span><i style="background:${CB}"></i>${esc(short(b))}</span></div><p class="small" style="margin:0 0 6px">Percentile among ${POSW[a.pos]}. 100 is the best, 50 is average.</p>${rows}</div>`;
  }
  const one = (p, c) => `<div class="card mini"><h3 style="color:${c}">${esc(p.n)}</h3>${p.bk.map(([l, v, h]) => `<div class="stat"><span>${esc(l)}${help(h)}</span><b>${Math.round(v)}</b></div>`).join("")}</div>`;
  return `<div class="two">${one(a, CA)}${one(b, CB)}</div>`;
}

const STATS = {
  QB: [["ppr", "Points per game"], ["pyd", "Pass yards"], ["ptd", "Pass TDs"], ["int", "Interceptions", 1], ["att", "Attempts"], ["cmp", "Completion %"], ["ypa", "Yards per attempt"], ["ruy", "Rush yards"], ["rtd", "Rush TDs"]],
  RB: [["ppr", "Points per game"], ["car", "Carries"], ["rushy", "Rush yards"], ["tgt", "Targets"], ["rec", "Receptions"], ["recy", "Receiving yards"], ["td", "Touchdowns"], ["touch", "Touch share %"], ["snap", "Snap %"]],
  WR: [["ppr", "Points per game"], ["tgt", "Targets"], ["rec", "Receptions"], ["recy", "Receiving yards"], ["td", "Touchdowns"], ["tshare", "Target share %"], ["ashare", "Air yards share %"], ["snap", "Snap %"]],
  K: [["fgp", "FG %"], ["fgm", "FG made per game"], ["fga", "FG tries per game"], ["xpg", "Extra points per game"], ["lg", "Longest FG"]],
};
STATS.TE = STATS.WR;
function statVals(p) {
  if (p.pos === "DST") {
    const k = p.kl || [], n = k.length || 1, av = (i) => (k.length ? k.reduce((s, r) => s + (r[i] || 0), 0) / n : null);
    return { ppg: p.ppg, pa: av(0), ya: av(1), sk: av(2), to: av(3), dtd: av(4) };
  }
  const m = Object.assign({}, p.ms || {}); if (p.pos === "K") m.ppg = p.ppg; return m;
}
function statTable(d, a, b) {
  const defs = a.pos === "DST" || b.pos === "DST" ? [["ppg", "Points per game"], ["pa", "Points allowed", 1], ["ya", "Yards allowed", 1], ["sk", "Sacks"], ["to", "Takeaways"], ["dtd", "Defensive TDs"]]
    : a.pos === "K" ? [["ppg", "Points per game"]].concat(STATS.K)
    : a.pos === b.pos ? STATS[a.pos] : [["ppr", "Points per game"], ["tgt", "Targets"], ["rec", "Receptions"], ["recy", "Receiving yards"], ["car", "Carries"], ["rushy", "Rush yards"], ["td", "Touchdowns"], ["snap", "Snap %"]];
  const va = statVals(a), vb = statVals(b);
  const rows = (defs || []).map(([k, l, low]) => { const x = va[k], y = vb[k]; if (x == null && y == null) return "";
    const wa = x != null && y != null && (low ? x < y : x > y), wb = x != null && y != null && (low ? y < x : y > x);
    return `<tr><td class="${wa ? "w" : ""}">${x == null ? "—" : f1(x)}</td><th>${l}</th><td class="${wb ? "w" : ""}">${y == null ? "—" : f1(y)}</td></tr>`; }).join("");
  return `<div class="card" style="padding:8px 12px;overflow-x:auto"><table><thead><tr><td>${esc(a.n)}</td><th>Per game, this season</th><td>${esc(b.n)}</td></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function matchup(d, a, b) {
  const wk = d.next, v = verdict(d, a, b);
  const both = `${a.n} or ${b.n}`;
  const title = `${both}? Who to Start in Week ${wk} (PPR) | Fantasy Mania`;
  const desc = v.w ? `${v.line.split(". Week")[0].replace(` in Week ${wk}.`, ` in Week ${wk} (PPR).`)}. Week ${wk} projections, ratings, matchups, game logs, floor and ceiling. Free.` : v.line;
  const path = `/start/${[a.s, b.s].sort().join("-vs-")}/`;
  const idx = listed(d, a, b);
  const why = reasons(d, v.w, v.l);
  const more = (p, o) => { const ps = partners(d, p, o); return ps.length ? `<h3>More ${esc(short(p))} start/sit</h3><ul class="chips">${ps.map((q) => `<li><a href="/start/${[p.s, q.s].sort().join("-vs-")}/">${esc(short(p))} or ${esc(q.n)}</a></li>`).join("")}</ul>` : ""; };
  const tabs = [["ss", "Start/Sit", "Start/Sit"], ["gl", "Game log", "Games"], ["mu", "Matchup", "Matchup"], ["bb", "Boom/Bust", "Boom/Bust"], ["pj", "Projection", "Proj."]];
  const panel = (id, name, inner) => `<section class="tp${id === "ss" ? " on" : ""}" id="tp-${id}" role="tabpanel" aria-labelledby="tb-${id}"><h2 class="tpH">${name}</h2>${inner}</section>`;
  const body = `<div class="kicker"><span>Week ${wk} start/sit · Updated ${esc(d.built)}</span><span class="ppr">PPR scoring</span></div>
<h1>${esc(withNick(a))} or ${esc(withNick(b))}?</h1>
<p class="sub">${esc(short(a))} or ${esc(short(b))}, ${esc(short(b))} or ${esc(short(a))}: here's who Fantasy Mania would start this week.</p>
<div class="vsWrap">${card(d, a, v.w === a)}<div class="vsMid" aria-hidden="true">VS</div>${card(d, b, v.w === b)}</div>${maniaInfo(d, a)}
<div class="tabs" role="tablist">${tabs.map(([id, n, sn], i) => `<button type="button" role="tab" id="tb-${id}" aria-controls="tp-${id}" aria-selected="${i === 0}"><span class="tl">${n}</span><span class="ts" aria-hidden="true">${sn}</span></button>`).join("")}</div>
${panel("ss", "Who should you start?", `<div class="card verdict"><span class="vTag">${esc(v.tag)}</span><p class="vHead">${v.w ? `Start <span>${esc(v.w.n)}</span>` : "Neither is playing"}</p><p class="vLine">${esc(v.line)}</p>${why.length ? `<ul class="why">${why.map((x) => `<li>${x}</li>`).join("")}</ul>` : ""}</div>`)}
${panel("gl", "Game log", `<div class="card">${gameLog(d, a, b)}</div>`)}
${panel("mu", "Matchup", matchupTab(d, a, b))}
${panel("bb", "Boom/Bust", boomTab(d, a, b))}
${panel("pj", `Week ${wk} projection`, projTab(d, a, b))}
<script>document.documentElement.classList.add('js');(function(){document.querySelectorAll('.iBtn').forEach(function(b){b.addEventListener('click',function(e){e.preventDefault();var h=document.getElementById('maniaHelp');h.open=!h.open;if(h.open)h.scrollIntoView({block:'nearest',behavior:'smooth'})})});var bs=document.querySelectorAll('.tabs button');bs.forEach(function(b){b.addEventListener('click',function(){bs.forEach(function(x){var on=x===b;x.setAttribute('aria-selected',on);document.getElementById(x.getAttribute('aria-controls')).classList.toggle('on',on)})})})})();</script>
${a.bk && b.bk ? `<h2>Compare the Mania breakdown</h2>${breakdown(d, a, b)}` : ""}
<h2>Stat by stat</h2>${statTable(d, a, b)}
<h2>Common questions</h2><div class="card faq">
<h3>Who should I start in Week ${wk}, ${esc(a.n)} or ${esc(b.n)}?</h3><p>${esc(v.line)}</p>
<h3>${esc(b.n)} or ${esc(a.n)}: who has the better matchup?</h3><p>${a.ma != null && b.ma != null ? esc(`${(a.ma >= b.ma ? a : b).n} has the better Week ${wk} matchup by Fantasy Mania's numbers (${f1(Math.max(a.ma, b.ma))} vs ${f1(Math.min(a.ma, b.ma))} points of rating boost)${(a.ma >= b.ma ? a : b).opp ? ", facing " + (a.ma >= b.ma ? a : b).opp : ""}.`) : esc(`${a.n} faces ${a.opp || "no one (bye)"} and ${b.n} faces ${b.opp || "no one (bye)"} in Week ${wk}.`)}</p>
<h3>Is ${esc(a.n)} a good start in Week ${wk}?</h3><p>${esc(startTake(d, a))}</p>
<h3>Is ${esc(b.n)} a good start in Week ${wk}?</h3><p>${esc(startTake(d, b))}</p></div>
<div class="btns"><a class="btn" href="/players/${a.s}/">${esc(short(a))}'s full profile</a><a class="btn" href="/players/${b.s}/">${esc(short(b))}'s full profile</a><a class="btn alt" href="/compare/">Compare in the app</a></div>
<h2>More start/sit questions</h2><div class="grid2"><div class="card">${more(a, b) || "<p class='small'>No other close calls this week.</p>"}</div><div class="card">${more(b, a) || "<p class='small'>No other close calls this week.</p>"}</div></div>
<h2>Check another matchup</h2><div class="card">${picker(d, null, null)}</div>`;
  return shell({ title, desc, path, crumbs: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"], [both, path]], body, robots: idx ? "" : "noindex,follow", nav: "/start/" });
}

function hub(d) {
  const wk = d.next;
  const pool = d.players.filter((p) => p.in);
  const sec = [["QB", "Quarterbacks"], ["RB", "Running backs"], ["WR", "Wide receivers"], ["TE", "Tight ends"], ["K", "Kickers"], ["DST", "Defenses"]].map(([pos, label]) => {
    const ps = pool.filter((p) => p.pos === pos).sort((x, y) => (x.wk || 999) - (y.wk || 999)).slice(0, 24);
    const pairs = [];
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) if (listed(d, ps[i], ps[j])) pairs.push([Math.abs((pts(ps[i]) || 0) - (pts(ps[j]) || 0)) + (i + j) * 0.15, ps[i], ps[j]]);
    pairs.sort((x, y) => x[0] - y[0]);
    const top = pairs.slice(0, 12);
    return top.length ? `<div class="card"><h3>${label}</h3><ul class="chips">${top.map(([, x, y]) => `<li><a href="/start/${[x.s, y.s].sort().join("-vs-")}/">${esc(short(x))} or ${esc(short(y))}</a></li>`).join("")}</ul></div>` : "";
  }).join("");
  const body = `<div class="kicker"><span>Week ${wk} · Updated ${esc(d.built)}</span><span class="ppr">PPR scoring</span></div><h1>Who Should I Start? Week ${wk} Start/Sit</h1>
<p class="sub">Pick any two players and Fantasy Mania tells you who to start, with Week ${wk} ratings, projections, matchups, floor and ceiling. Free, PPR, every position including kickers and defenses.</p>
<div class="card">${picker(d)}</div><h2>This week's closest calls</h2><div class="grid2">${sec}</div>`;
  return shell({ title: `Who Should I Start? Week ${wk} Fantasy Football Start/Sit Tool (PPR) | Fantasy Mania`, desc: `Free Week ${wk} fantasy football start/sit tool. Compare any two players and get a clear pick with projections, matchups and Week ${wk} ratings. QB, RB, WR, TE, K and D/ST.`, path: "/start/", crumbs: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"]], body, robots: "", nav: "/start/" });
}

function notFound(d) {
  const body = `<h1>We couldn't find that matchup</h1><p class="sub">One of those players isn't in this week's Fantasy Mania numbers. Pick two players below.</p><div class="card">${picker(d)}</div>`;
  return shell({ title: "Matchup not found | Fantasy Mania", desc: "Pick two players to see who to start.", path: "/start/", crumbs: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"]], body, robots: "noindex,follow", nav: "/start/" });
}
