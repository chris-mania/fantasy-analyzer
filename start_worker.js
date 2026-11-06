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
  let line = `Start ${w.n} over ${l.n} in Week ${wk} (PPR). Fantasy Mania projects ${short(w)} for ${f1(pts(w))} points${vsTxt(w)}, versus ${f1(pts(l))} for ${short(l)}${vsTxt(l)}.`;
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
  let s = `${p.n} is ${call} in Week ${wk}: ${pn}${p.wk || "—"} by Fantasy Mania's Week ${wk} Rating (${f1(p.wr)}), projected for ${f1(pts(p))} PPR points${p.opp ? " against " + p.opp : ""}`;
  if (p.pj && p.pj.lo != null && p.pj.hi != null) s += `, with a range of ${f1(p.pj.lo)} to ${f1(p.pj.hi)}`;
  s += ".";
  if (p.inj && p.inj.l) s += ` Injury report: ${p.inj.l}${p.inj.t ? " (" + p.inj.t + ")" : ""}.`;
  return s;
}

function reasons(d, w, l) {
  if (!w || !l) return [];
  const out = [], a = w, b = l;
  if (pts(a) != null && pts(b) != null) out.push(`<b>Projection:</b> ${esc(short(a))} ${f1(pts(a))} vs ${esc(short(b))} ${f1(pts(b))} PPR points.`);
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
const CSS = `
:root{--bg:#e9edf2;--card:#fff;--ink:#0b1830;--muted:#5a616a;--line:#dde3ea;--blue:#2563eb;--blue2:#1d4ed8;--green:#16803f;--greenbg:#e7f6ed}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
a{color:var(--blue2)}
.top{background:#e7edf3;border-bottom:1px solid #c4ccd6;position:sticky;top:0;z-index:5}
.nav{max-width:1100px;margin:auto;height:58px;padding:0 16px;display:flex;align-items:center;gap:22px}
.brand{font-weight:900;font-style:italic;font-size:22px;letter-spacing:-.6px;text-decoration:none;color:#060709;white-space:nowrap}.brand b{color:#1d4ed8;font-weight:900}
.links{display:flex;gap:18px;overflow-x:auto;scrollbar-width:none}.links a{color:#40474f;text-decoration:none;font-weight:600;font-size:15px;white-space:nowrap;padding:6px 0}.links a.on{color:#060709;box-shadow:inset 0 -2px #060709}
.wrap{max-width:1100px;margin:auto;padding:22px 16px 48px}
.crumb{font-size:13px;color:var(--muted);margin-bottom:10px}.crumb a{color:var(--muted)}
.kicker{font-size:11px;color:var(--blue2);font-weight:800;text-transform:uppercase;letter-spacing:1.2px}
h1{font-size:clamp(28px,6vw,44px);line-height:1.08;letter-spacing:-1px;margin:6px 0 6px;font-weight:800}
.sub{color:var(--muted);margin:0 0 18px;font-size:15px}
.card{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 26px -20px rgba(20,40,80,.55);border:1px solid #e4eaf3}
.verdict{border-left:6px solid #22a45d;margin-bottom:16px}
.vTag{display:inline-block;background:var(--greenbg);color:var(--green);font-weight:800;font-size:12px;border-radius:99px;padding:4px 10px;margin-bottom:8px}
.vHead{font-size:clamp(24px,5vw,34px);font-weight:800;letter-spacing:-.6px;margin:0 0 8px}.vHead span{color:var(--green)}
.vLine{margin:0;font-size:16px}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}
.pl{text-align:center;padding:16px 10px}
.av{width:68px;height:68px;border-radius:50%;margin:0 auto 8px;display:grid;place-items:center;font-weight:800;font-size:22px;color:#5a616a;background:radial-gradient(circle at 50% 35%,#f4f6f9,#d8dee5);box-shadow:0 0 0 4px var(--r) inset,0 0 0 7px #fff inset}
.pl .nm{font-weight:800;font-size:18px;text-decoration:none;color:var(--ink)}.pl .nm:hover{text-decoration:underline}
.pl .meta{color:var(--muted);font-size:13px;font-weight:600;margin-top:2px}
.pl .big{font-size:38px;font-weight:800;letter-spacing:-1px;margin-top:8px;line-height:1}.pl .lbl{font-size:10.5px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-weight:800;margin-top:4px}
.pl.win{outline:3px solid #22a45d;outline-offset:-3px}.winB{display:inline-block;margin-top:8px;background:#22a45d;color:#fff;font-size:11px;font-weight:800;border-radius:99px;padding:3px 10px;letter-spacing:.5px}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:11px 8px;border-top:1px solid var(--line);text-align:center}th{font-weight:600;color:var(--muted);text-align:left;font-size:13px}
thead td{border-top:0;font-weight:800;font-size:13px}td.w{color:var(--green);font-weight:800}
.rng{position:relative;height:8px;border-radius:99px;background:#eef2f6;margin:6px 0 2px}.rng i{position:absolute;top:0;bottom:0;border-radius:99px;background:linear-gradient(90deg,#c7d7f7,#2563eb)}.rng b{position:absolute;top:-4px;width:4px;height:16px;border-radius:2px;background:#0b1830}
.small{font-size:12px;color:var(--muted)}
h2{font-size:20px;margin:28px 0 10px;letter-spacing:-.3px}
ul.why{margin:0;padding-left:20px}ul.why li{margin:7px 0}
.faq h3{font-size:16px;margin:16px 0 4px}.faq p{margin:0;color:#272c33}
.chips{display:flex;flex-wrap:wrap;gap:8px;list-style:none;padding:0;margin:0}.chips a{display:inline-flex;align-items:center;min-height:40px;padding:8px 14px;border-radius:99px;background:#fff;border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700;font-size:14px}.chips a:hover{border-color:#2563eb;color:#1d4ed8}
.btns{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.btn{display:inline-flex;align-items:center;min-height:44px;padding:10px 18px;border-radius:12px;background:#0b1830;color:#fff;text-decoration:none;font-weight:700}.btn.alt{background:#fff;color:#0b1830;border:1px solid #d9dfe7}
.pick{display:grid;grid-template-columns:1fr auto 1fr auto;gap:10px;align-items:center}.pick input{width:100%;min-height:48px;border-radius:12px;border:1px solid #c4ccd6;padding:10px 14px;font:inherit;font-size:16px;background:#fff}.pick .vs{font-weight:800;color:var(--muted)}.pick button{min-height:48px;border:0;border-radius:12px;background:#0b1830;color:#fff;font:inherit;font-weight:800;padding:0 20px;cursor:pointer}
.err{color:#c0303c;font-size:14px;margin-top:8px;min-height:20px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.grid2 h3{margin:0 0 10px;font-size:16px}
.disc{font-size:12.5px;color:var(--muted);margin-top:26px;border-left:3px solid #c7d7f7;padding:10px 12px;background:#eef3fb;border-radius:0 10px 10px 0}
@media(max-width:700px){.links{display:none}.wrap{padding-top:16px}.card{padding:16px;border-radius:16px}.pl .big{font-size:30px}.av{width:56px;height:56px;font-size:18px}table{font-size:14px}th,td{padding:10px 4px}.pick{grid-template-columns:1fr}.pick .vs{text-align:center}}
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
<p class="disc">Fantasy Mania ratings and projections are estimates, not guarantees. Injuries and game flow can change anything, so check final inactives before you set your lineup. PPR scoring.</p></main></body></html>`;
}

function picker(d, a, b) {
  const opts = d.players.map((p) => `<option value="${esc(p.n)}">`).join("") + d.players.filter((p) => p.nk && p.nk.length).map((p) => p.nk.map((k) => `<option value="${esc(k)}">`).join("")).join("");
  const map = {}; d.players.forEach((p) => { map[p.n.toLowerCase()] = p.s; (p.nk || []).forEach((k) => { map[k.toLowerCase()] = p.s; }); });
  return `<form class="pick" id="pick" autocomplete="off"><input list="fmNames" id="pa" placeholder="First player" aria-label="First player" value="${a ? esc(a.n) : ""}"><span class="vs">or</span><input list="fmNames" id="pb" placeholder="Second player" aria-label="Second player" value="${b ? esc(b.n) : ""}"><button>Who do I start?</button></form><div class="err" id="perr" role="status"></div><datalist id="fmNames">${opts}</datalist>
<script>(function(){var M=${JSON.stringify(map).replace(/</g, "\\u003c")};document.getElementById('pick').addEventListener('submit',function(e){e.preventDefault();var a=M[document.getElementById('pa').value.trim().toLowerCase()],b=M[document.getElementById('pb').value.trim().toLowerCase()];var er=document.getElementById('perr');if(!a||!b){er.textContent='Pick two players from the list.';return}if(a===b){er.textContent='Pick two different players.';return}location.href='/start/'+[a,b].sort().join('-vs-')+'/'})})();</script>`;
}

function playerCard(d, p, isWin) {
  const r = playing(p) ? p.wr : null;
  return `<div class="card pl${isWin ? " win" : ""}"><div class="av" style="--r:${ring(p.mr)}">${esc(initials(p))}</div>
<a class="nm" href="/players/${p.s}/">${esc(p.n)}</a><div class="meta">${esc(p.tm)} · ${posName(p.pos)}${p.bye ? " · Bye" : p.opp ? " · vs " + esc(p.opp) : ""}</div>
<div class="big" style="color:${ink(r)}">${playing(p) ? f1(p.wr) : p.bye ? "BYE" : "OUT"}</div><div class="lbl">Week ${d.next} Rating</div>
${isWin ? `<span class="winB">START</span>` : ""}</div>`;
}

function rangeCell(p, lo, hi) {
  if (!p.pj || p.pj.lo == null || p.pj.hi == null || hi <= lo) return "—";
  const L = ((p.pj.lo - lo) / (hi - lo)) * 100, W = ((p.pj.hi - p.pj.lo) / (hi - lo)) * 100, P = ((p.pj.p - lo) / (hi - lo)) * 100;
  return `<div class="rng" aria-hidden="true"><i style="left:${L.toFixed(1)}%;width:${W.toFixed(1)}%"></i><b style="left:calc(${P.toFixed(1)}% - 2px)"></b></div><span class="small">${f1(p.pj.lo)} to ${f1(p.pj.hi)}</span>`;
}

function table(d, a, b) {
  const rows = [];
  const num = (label, va, vb, fmt, higher = true) => {
    const wa = va != null && vb != null && (higher ? va > vb : va < vb), wb = va != null && vb != null && (higher ? vb > va : vb < va);
    rows.push(`<tr><th>${label}</th><td class="${wa ? "w" : ""}">${va == null ? "—" : fmt(va)}</td><td class="${wb ? "w" : ""}">${vb == null ? "—" : fmt(vb)}</td></tr>`);
  };
  const txt = (label, ta, tb) => rows.push(`<tr><th>${label}</th><td>${ta}</td><td>${tb}</td></tr>`);
  num(`Week ${d.next} Rating`, playing(a) ? a.wr : null, playing(b) ? b.wr : null, f1);
  num("Projected points", playing(a) ? pts(a) : null, playing(b) ? pts(b) : null, f1);
  const lo = Math.min(a.pj && a.pj.lo != null ? a.pj.lo : 0, b.pj && b.pj.lo != null ? b.pj.lo : 0, 0), hi = Math.max(a.pj && a.pj.hi || 0, b.pj && b.pj.hi || 0);
  txt("Range (floor to ceiling)", rangeCell(a, lo, hi), rangeCell(b, lo, hi));
  txt("Opponent", a.bye ? "Bye" : esc(a.opp || "—"), b.bye ? "Bye" : esc(b.opp || "—"));
  num("Matchup boost", a.ma, b.ma, (v) => (v > 0 ? "+" : "") + f1(v));
  num(`${posName(a.pos) === posName(b.pos) ? posName(a.pos) : "Position"} rank this week`, a.wk, b.wk, (v) => "#" + v, false);
  num("Mania Rating (season)", a.mr, b.mr, f1);
  num("Points per game", a.ppg, b.ppg, f1);
  const l3 = (p) => (p.l3 && p.l3.length ? p.l3.map(f1).join(", ") : "—");
  txt("Last 3 games", l3(a), l3(b));
  const st = (p) => {
    const j = p.pj || {}; const bits = [];
    if (p.pos === "QB") { if (j.pyd != null) bits.push(Math.round(j.pyd) + " pass yds"); if (j.ptd != null) bits.push(f1(j.ptd) + " TD"); if (j.ruy) bits.push(Math.round(j.ruy) + " rush yds"); }
    else if (p.pos === "RB") { if (j.car != null) bits.push(f1(j.car) + " car"); if (j.ruy != null) bits.push(Math.round(j.ruy) + " rush yds"); if (j.rec != null) bits.push(f1(j.rec) + " rec"); }
    else if (p.pos === "WR" || p.pos === "TE") { if (j.tgt != null) bits.push(f1(j.tgt) + " tgt"); if (j.rec != null) bits.push(f1(j.rec) + " rec"); if (j.ryd != null) bits.push(Math.round(j.ryd) + " yds"); }
    return bits.length ? bits.join(" · ") : "—";
  };
  if (a.pos !== "K" && a.pos !== "DST") txt("Projected stats", `<span class="small">${st(a)}</span>`, `<span class="small">${st(b)}</span>`);
  const inj = (p) => (p.inj && p.inj.l ? esc(p.inj.l) + (p.inj.t ? `<br><span class="small">${esc(p.inj.t)}</span>` : "") : "No report");
  txt("Injury report", inj(a), inj(b));
  return `<div class="card" style="padding:8px 12px;overflow-x:auto"><table><thead><tr><td></td><td>${esc(short(a))}</td><td>${esc(short(b))}</td></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function matchup(d, a, b) {
  const wk = d.next, v = verdict(d, a, b);
  const both = `${a.n} or ${b.n}`;
  const title = `${both}? Who to Start in Week ${wk} (PPR) | Fantasy Mania`;
  const desc = v.w ? `${v.line.split(". Week")[0]}. Week ${wk} ratings, projections, matchups, floor and ceiling. Free.` : v.line;
  const path = `/start/${[a.s, b.s].sort().join("-vs-")}/`;
  const idx = listed(d, a, b);
  const names = `${withNick(a)} or ${withNick(b)}`;
  const why = reasons(d, v.w, v.l);
  const more = (p, o) => { const ps = partners(d, p, o); return ps.length ? `<h3>More ${esc(short(p))} start/sit</h3><ul class="chips">${ps.map((q) => `<li><a href="/start/${[p.s, q.s].sort().join("-vs-")}/">${esc(short(p))} or ${esc(q.n)}</a></li>`).join("")}</ul>` : ""; };
  const body = `<div class="kicker">Week ${wk} start/sit · PPR · Updated ${esc(d.built)}</div>
<h1>${esc(names)}?</h1>
<p class="sub">${esc(short(a))} or ${esc(short(b))}, ${esc(short(b))} or ${esc(short(a))}: here's who Fantasy Mania would start this week.</p>
<div class="card verdict"><span class="vTag">${esc(v.tag)}</span><p class="vHead">${v.w ? `Start <span>${esc(v.w.n)}</span>` : "Neither is playing"}</p><p class="vLine">${esc(v.line)}</p></div>
<div class="duo">${playerCard(d, a, v.w === a)}${playerCard(d, b, v.w === b)}</div>
${table(d, a, b)}
${why.length ? `<h2>The breakdown</h2><div class="card"><ul class="why">${why.map((x) => `<li>${x}</li>`).join("")}</ul></div>` : ""}
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
  const body = `<div class="kicker">Week ${wk} · PPR · Updated ${esc(d.built)}</div><h1>Who Should I Start? Week ${wk} Start/Sit</h1>
<p class="sub">Pick any two players and Fantasy Mania tells you who to start, with Week ${wk} ratings, projections, matchups, floor and ceiling. Free, PPR, every position including kickers and defenses.</p>
<div class="card">${picker(d)}</div><h2>This week's closest calls</h2><div class="grid2">${sec}</div>`;
  return shell({ title: `Who Should I Start? Week ${wk} Fantasy Football Start/Sit Tool (PPR) | Fantasy Mania`, desc: `Free Week ${wk} fantasy football start/sit tool. Compare any two players and get a clear pick with projections, matchups and Week ${wk} ratings. QB, RB, WR, TE, K and D/ST.`, path: "/start/", crumbs: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"]], body, robots: "", nav: "/start/" });
}

function notFound(d) {
  const body = `<h1>We couldn't find that matchup</h1><p class="sub">One of those players isn't in this week's Fantasy Mania numbers. Pick two players below.</p><div class="card">${picker(d)}</div>`;
  return shell({ title: "Matchup not found | Fantasy Mania", desc: "Pick two players to see who to start.", path: "/start/", crumbs: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"]], body, robots: "noindex,follow", nav: "/start/" });
}
