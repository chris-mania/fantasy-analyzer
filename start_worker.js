// Fantasy Mania start/sit pages: /start/ and /start/<player>-vs-<player>/
// Runs on Cloudflare Pages only for /start addresses (see _routes.json). Everything else on the
// site is served exactly as before.
//
// A matchup page is the real Fantasy Mania app (index.html), opened straight to Compare with the
// two players picked, so photos, rings, badges (and their pop-ups), tabs, menus and profiles all work
// as normal. On top of that: its own title/description/canonical for Google, the winner's card
// outlined with "Start <name>", the Week projection as the big number with the Mania Rating as an
// "overall" badge on the ring, and a written answer + common questions below for search engines.
// Numbers for the written part come from /start-data.json (rebuilt about once an hour). The pick
// uses the app's own rule: higher projected points, unless one player is on bye, out or already played.
// Source lives in the repo as start_worker.js; seo_pages.py copies it to public/_worker.js.

const SITE = "https://fantasymania.org";
let CACHE = { at: 0, data: null };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname === "fantasymania.pages.dev") return Response.redirect(SITE + url.pathname + url.search, 301);
    if (!url.pathname.startsWith("/start")) return env.ASSETS.fetch(request);
    if (url.pathname === "/start") return Response.redirect(url.origin + "/start/", 301);
    try {
      const data = await load(env, url);
      const rest = url.pathname.slice("/start/".length).replace(/\/+$/, "");
      if (!rest) {
        // /start/?a=<id>&b=<id> (homepage box, copied links): go to that pair's page, or the app's Compare if unknown
        const ia = url.searchParams.get("a"), ib = url.searchParams.get("b");
        if (ia && ib) {
          const pa = data.byId[ia], pb = data.byId[ib];
          if (pa && pb && pa !== pb) return Response.redirect(url.origin + "/start/" + [pa.s, pb.s].sort().join("-vs-") + "/", 302);
          return Response.redirect(url.origin + "/#c=" + encodeURIComponent(ia) + "," + encodeURIComponent(ib), 302);
        }
        return html(hub(data), 200);
      }
      if (!url.pathname.endsWith("/")) return Response.redirect(url.origin + url.pathname + "/" + url.search, 301);
      const i = rest.indexOf("-vs-");
      const a = i > 0 ? data.by[rest.slice(0, i)] : null, b = i > 0 ? data.by[rest.slice(i + 4)] : null;
      if (!a || !b || a === b) return html(notFound(data), 404);
      const canon = [a.s, b.s].sort().join("-vs-");
      if (canon !== rest) return Response.redirect(url.origin + "/start/" + canon + "/", 301);
      return await matchupApp(env, url, data, a, b);
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
  d.by = {}; d.byId = {};
  d.players.forEach((p) => {
    // People search "Bears defense", not "Bears D/ST".
    if (p.pos === "DST" && / ?D\/ST$/.test(p.n)) p.n = p.n.replace(/\s*D\/ST$/, "") + " Defense";
    d.by[p.s] = p; d.byId[p.id] = p;
  });
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
function short(p) {
  if (p.pos === "DST") return p.n.replace(/\s*(D\/ST|Defense)$/, "");
  const w = p.n.replace(/,?\s+(Jr\.|Sr\.|II|III|IV)$/, "").split(/\s+/);
  return w.length > 1 ? w.slice(1).join(" ") : p.n;
}
const withNick = (p) => (p.nk && p.nk.length ? `${p.n} (${p.nk[0]})` : p.n);
const pts = (p) => (p.pj ? p.pj.p : null);
const unavailable = (p) => p.bye || p.out || !!p.pl;
const START_AT = { QB: 12, RB: 24, WR: 30, TE: 12, K: 12, DST: 12 };

function listed(d, a, b) {
  if (!(a.in && b.in)) return false;
  const same = a.pos === b.pos, flex = (a.pos === "RB" && b.pos === "WR") || (a.pos === "WR" && b.pos === "RB");
  if (!same && !flex) return false;
  if (same && (a.pos === "K" || a.pos === "DST")) return true;
  return pts(a) != null && pts(b) != null && Math.abs(pts(a) - pts(b)) < d.gap;
}

// Same rule as the app's Compare "Our pick".
function verdict(d, a, b) {
  const wk = d.next, why = (p) => (p.bye ? "is on bye" : p.out ? "is ruled out" : "has already played");
  if (unavailable(a) && unavailable(b)) return { w: null, tag: "Neither", line: `Neither ${a.n} nor ${b.n} is available to start in Week ${wk}. ${a.n} ${why(a)} and ${b.n} ${why(b)}.` };
  if (unavailable(a) !== unavailable(b)) {
    const w = unavailable(a) ? b : a, l = w === a ? b : a;
    return { w, tag: "Easy call", line: `Start ${w.n}. ${l.n} ${why(l)} in Week ${wk}.` };
  }
  const pa = pts(a), pb = pts(b);
  const w = pa != null && pb != null ? (pa >= pb ? a : b) : ((a.wr || 0) >= (b.wr || 0) ? a : b), l = w === a ? b : a;
  const vs = (p) => (p.opp ? ` against ${p.opp}` : "");
  return { w, line: `Start ${w.n} over ${l.n} in Week ${wk}. Fantasy Mania projects ${short(w)} for ${f1(pts(w))} points${vs(w)}, versus ${f1(pts(l))} for ${short(l)}${vs(l)}.` };
}

function startTake(d, p) {
  const wk = d.next, pn = posName(p.pos);
  if (p.bye) return `${p.n} is on bye in Week ${wk}.`;
  if (p.out) return `${p.n} is ruled out for Week ${wk}.`;
  if (p.pl) return `${p.n} already played in Week ${wk} and scored ${f1(p.pl.f)} points (${p.pl.s}).`;
  const at = START_AT[p.pos] || 12;
  const call = p.wk == null ? "a deep-league option" : p.wk <= at ? "a solid start" : p.wk <= at * 2 ? (p.pos === "RB" || p.pos === "WR" ? "a flex option" : "a streaming option") : "a bench or deep-league play";
  let s = `${p.n} is ${call} in Week ${wk}, projected for ${f1(pts(p))} points${p.opp ? " against " + p.opp : ""} (${pn}${p.wk || "—"} this week)`;
  if (p.pj && p.pj.lo != null && p.pj.hi != null) s += `, with a range of ${f1(p.pj.lo)} to ${f1(p.pj.hi)}`;
  s += ".";
  if (p.inj && p.inj.l) s += ` Injury report: ${p.inj.l}${p.inj.t ? " (" + p.inj.t + ")" : ""}.`;
  return s;
}

function partners(d, p, skip) {
  const at = pts(p);
  return d.players
    .filter((q) => q !== p && q !== skip && q.pos === p.pos && listed(d, p, q))
    .sort((x, y) => (Math.abs((x.wk || 999) - (p.wk || 999)) + Math.abs((pts(x) || 0) - at) / 10) - (Math.abs((y.wk || 999) - (p.wk || 999)) + Math.abs((pts(y) || 0) - at) / 10))
    .slice(0, 6);
}

// ---------- matchup page: the real app, opened to Compare ----------
const APP_CSS = `
#fmSeo{max-width:1240px;margin:8px auto 48px;padding:0 20px}#fmSeo[hidden]{display:none}
#fmSeo h2{font-size:22px;font-weight:800;letter-spacing:-.3px;margin:30px 0 12px}
#fmSeo .fmCard{background:#fff;border-radius:20px;padding:20px 22px;border:1px solid #e4eaf3;box-shadow:0 10px 26px -20px rgba(20,40,80,.5)}
#fmSeo h3{font-size:16px;margin:16px 0 4px}#fmSeo h3:first-child{margin-top:0}#fmSeo p{margin:0;color:#272c33;line-height:1.55}
#fmSeo .fmChips{display:flex;flex-wrap:wrap;gap:8px;list-style:none;padding:0;margin:0}#fmSeo .fmChips a{display:inline-flex;align-items:center;min-height:40px;padding:8px 14px;border-radius:99px;background:#fff;border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700;font-size:14px}#fmSeo .fmChips a:hover{border-color:#2563eb;color:#1d4ed8}
#fmSeo .fmGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}#fmSeo .fmGrid h3{margin-top:0}
.fmPpr{display:inline-block;margin:2px 0 6px;font-size:12px;font-weight:700;color:#40474f;background:#fff;border:1px solid #d9dfe7;border-radius:99px;padding:3px 10px}
.comparePlayer.fmWin{outline:3px solid #22a45d;outline-offset:-3px;border-radius:24px;position:relative}
.fmStart{display:inline-block;background:#22a45d;color:#fff;font-weight:800;font-size:13px;letter-spacing:.3px;border-radius:99px;padding:5px 14px;margin:0 auto 10px}
.cmpPortrait{position:relative}
.fmOvr{position:absolute;right:-6px;bottom:-4px;width:66px;height:66px;border-radius:50%;background:#fff;border:0;box-shadow:0 6px 18px -6px rgba(10,30,60,.45);display:grid;place-content:center;cursor:pointer;padding:0;z-index:2;font:inherit}
.fmOvr b{font-size:19px;font-weight:900;line-height:1;letter-spacing:-.5px}.fmOvr small{font-size:7.5px;font-weight:800;letter-spacing:.6px;color:#40474f;margin-top:2px}
.fmProjLab{font-weight:700}
#fmMHelp{max-width:760px;margin:12px auto 0;text-align:left;background:#f3f6fb;border:1px solid #d9e2f1;border-radius:14px;padding:14px 16px;font-size:14.5px;line-height:1.55;color:#272c33}#fmMHelp[hidden]{display:none}#fmMHelp b{color:#0b1830}
@media(max-width:800px){.fmOvr{width:52px;height:52px;right:-8px;bottom:-6px}.fmOvr b{font-size:15px}.fmOvr small{font-size:6px}#fmSeo{padding:0 14px}.fmStart{font-size:12px;padding:4px 12px}}
`;

function seoSection(d, a, b, v) {
  const wk = d.next;
  const more = (p, o) => { const ps = partners(d, p, o); return ps.length ? `<div class="fmCard"><h3>More ${esc(short(p))} start/sit</h3><ul class="fmChips">${ps.map((q) => `<li><a href="/start/${[p.s, q.s].sort().join("-vs-")}/">${esc(short(p))} or ${esc(q.n)}</a></li>`).join("")}</ul></div>` : ""; };
  const better = a.ma != null && b.ma != null ? (a.ma >= b.ma ? a : b) : null;
  return `<section id="fmSeo" aria-labelledby="fmSeoH">
<h2 id="fmSeoH">Who should I start: ${esc(withNick(a))} or ${esc(withNick(b))}?</h2>
<div class="fmCard"><p><b data-fmv="head">${v.w ? "Start " + esc(v.w.n) + "." : "Neither is available."}</b> <span data-fmv="line">${esc(v.line)}</span> ${esc(short(a))} or ${esc(short(b))}, ${esc(short(b))} or ${esc(short(a))}: this is Fantasy Mania's pick for Week ${wk}.</p></div>
<h2>Common questions</h2><div class="fmCard">
<h3>Who should I start in Week ${wk}, ${esc(a.n)} or ${esc(b.n)}?</h3><p data-fmv="line">${esc(v.line)}</p>
<h3>${esc(b.n)} or ${esc(a.n)}: who has the better matchup?</h3><p>${better ? esc(`${better.n} has the better Week ${wk} matchup by Fantasy Mania's numbers (${f1(Math.max(a.ma, b.ma))} vs ${f1(Math.min(a.ma, b.ma))} points of rating boost)${better.opp ? ", facing " + better.opp : ""}.`) : esc(`${a.n} faces ${a.opp || "no one (bye)"} and ${b.n} faces ${b.opp || "no one (bye)"} in Week ${wk}.`)}</p>
<h3>Is ${esc(a.n)} a good start in Week ${wk}?</h3><p>${esc(startTake(d, a))}</p>
<h3>Is ${esc(b.n)} a good start in Week ${wk}?</h3><p>${esc(startTake(d, b))}</p></div>
<h2>More start/sit questions</h2><div class="fmGrid">${more(a, b)}${more(b, a)}</div>
<p style="margin-top:14px"><a href="/start/">Compare any two players →</a></p>
</section>`;
}

function opener(d, a, b, v) {
  const cfg = { a: a.id, b: b.id, wk: d.next, win: v.w ? v.w.id : null };
  return `<script>(function(){var C=${JSON.stringify(cfg).replace(/</g, "\\u003c")},touched=false,tries=0;
var POSW={QB:'quarterback',RB:'running back',WR:'wide receiver',TE:'tight end',K:'kicker',DST:'defense'};
['pointerdown','keydown'].forEach(function(t){addEventListener(t,function(){touched=true},{capture:true,once:true})});
function ready(){return typeof renderCompare==='function'&&typeof byId==='function'&&typeof showView==='function'&&typeof DB!=='undefined'}
function seoShow(){var s=document.getElementById('fmSeo'),v=document.getElementById('compare');if(s)s.hidden=!(window.__fmStart&&v&&v.classList.contains('on'))}
function tweak(){if(!window.__fmStart)return;var hero=document.querySelector('#compare .compareHero');if(!hero||!A||!B)return;
 var cards=hero.querySelectorAll('.comparePlayer');if(cards.length<2)return;
 var pick=document.querySelector('#compare .cwin strong, #compare .kdVerdict strong'),pn=pick?dn(pick.textContent.trim()):null;
 [[cards[0],A],[cards[1],B]].forEach(function(x){var el=x[0],p=x[1];if(el.dataset.fm==='1')return;el.dataset.fm='1';
  var big=el.querySelector('.scorebig'),lab=el.querySelector('.scorelab'),mr=big?big.textContent.trim():'';
  var pj=p.proj&&p.proj.pts!=null?p.proj.pts:null,bye=p.opp==='BYE'||(p.r&&p.r.bye);
  if(big){big.className='scorebig';big.style.color='#0b1830';big.textContent=bye?'BYE':p.out?'OUT':pj!=null?(+pj).toFixed(1):'—'}
  if(lab){lab.textContent='Week '+C.wk+' projection';lab.classList.add('fmProjLab')}
  var por=el.querySelector('.cmpPortrait');
  if(mr&&mr!=='—'){var bt=document.createElement('button');bt.type='button';bt.className='fmOvr';bt.setAttribute('aria-label','Mania Rating '+mr+' overall. Tap for what it means.');
   var col=+mr>=80?'#16803f':+mr>=60?'#9a6700':'#c0303c';bt.innerHTML='<b style="color:'+col+'">'+mr+'</b><small>MANIA OVERALL</small>';
   bt.addEventListener('click',function(e){e.stopPropagation();var h=document.getElementById('fmMHelp');if(h){h.hidden=!h.hidden;if(!h.hidden)h.scrollIntoView({block:'nearest',behavior:'smooth'})}});
   if(por)por.appendChild(bt);else el.insertBefore(bt,el.firstChild)}
  var name=dn(p.name||(p.r&&p.r.name));el.dataset.fmName=name;el.dataset.fmPj=pj==null?'':pj;el.dataset.fmOpp=p.opp||'';
  if((pn&&name===pn)||(!pn&&p.id===C.win)){el.classList.add('fmWin');var s=document.createElement('div');s.className='fmStart';s.textContent='Start '+name;el.insertBefore(s,el.firstChild)}
 });
 fixAnswer(cards,pn);
 if(!document.getElementById('fmMHelp')){var h=document.createElement('div');h.id='fmMHelp';h.hidden=true;
  h.innerHTML='<b>What is the Mania Rating?</b> It is a player\\'s overall grade from 0 to 100, kind of like a Madden overall. Fantasy Mania combines how much he produces, how much he is used, his role, his high-value chances (like red-zone looks) and his efficiency, each compared with every other '+(POSW[A.pos]||'player')+', into one number. 90+ is Elite, 82+ Excellent, 74+ Strong, 65+ Starter.<br><br>The big number is the <b>Week '+C.wk+' projection</b>: the points we expect this week after the matchup and the injury report. That is the number to use for start/sit.';
  hero.parentNode.insertBefore(h,hero.nextSibling)}
}
function last(n){if(n.slice(-4)==='D/ST')return n.slice(0,-4).trim();if(n.slice(-8)===' Defense')return n.slice(0,-8);var w=n.split(' '),suf=['Jr.','Sr.','II','III','IV'];if(w.length>2&&suf.indexOf(w[w.length-1])>=0)w.pop();return w.length>1?w.slice(1).join(' '):n}
function dn(n){n=n||'';return n.slice(-4)==='D/ST'?n.slice(0,-4).trim()+' Defense':n}
function fixAnswer(cards,pn){pn=dn(pn);if(!pn)return;var h=document.querySelector('#fmSeo [data-fmv="head"]');if(!h||h.textContent==='Start '+pn+'.')return;
 var w=[].find.call(cards,function(c){return c.dataset.fmName===pn}),l=[].find.call(cards,function(c){return c!==w});if(!w||!l)return;
 var vs=function(c){return c.dataset.fmOpp&&c.dataset.fmOpp!=='BYE'?' against '+c.dataset.fmOpp:''},f=function(c){return c.dataset.fmPj===''?'—':(+c.dataset.fmPj).toFixed(1)};
 var line='Start '+pn+' over '+l.dataset.fmName+' in Week '+C.wk+'. Fantasy Mania projects '+last(pn)+' for '+f(w)+' points'+vs(w)+', versus '+f(l)+' for '+last(l.dataset.fmName)+vs(l)+'.';
 h.textContent='Start '+pn+'.';document.querySelectorAll('#fmSeo [data-fmv="line"]').forEach(function(e){e.textContent=line})}
function hook(){if(window.__fmHooked)return;window.__fmHooked=1;var r=renderCompare;renderCompare=function(){var o=r.apply(this,arguments);try{tweak()}catch(e){}return o};
 var sv=showView;showView=function(){var o=sv.apply(this,arguments);try{seoShow()}catch(e){}return o}}
function open(){if(!ready()){if(++tries<200)setTimeout(open,40);return}hook();
 var a=byId(C.a),b=byId(C.b);if(!a||!b){seoShow();return}A=a;B=b;selA=null;selB=null;try{cmpOpen.start=true}catch(e){}
 window.__fmOpening=1;window.__fmStart=1;try{showView('compare');renderCompare();seoShow()}finally{window.__fmOpening=0}try{scrollTo(0,0)}catch(e){}}
open();
addEventListener('load',function(){[300,900].forEach(function(t){setTimeout(function(){var v=document.getElementById('compare');if(!touched&&v&&!v.classList.contains('on'))open()},t)})});
})();</script>`;
}

// What Google shows under the title: the pick, both projections, both position ranks and how big the edge is.
function snippet(d, v, a, b) {
  const wk = d.next, w = v.w, l = w === a ? b : a;
  if (v.tag === "Easy call") return `${v.line} Week ${wk} PPR start/sit from Fantasy Mania.`;
  const gap = Math.abs((pts(w) || 0) - (pts(l) || 0));
  const call = gap < 1 ? "It's a close call" : gap < 3 ? "A solid edge" : "An easy call";
  const rk = (p) => (p.wk ? `${posName(p.pos)}${p.wk}` : null);
  const ranks = rk(w) && rk(l) ? ` (${rk(w)} vs ${rk(l)} this week)` : "";
  return `Start ${w.n} over ${l.n} in Week ${wk} (PPR). ${call}: ${short(w)} projects ${f1(pts(w))} points vs ${f1(pts(l))} for ${short(l)}${ranks}. Matchups, game logs and boom/bust inside.`;
}

async function matchupApp(env, url, d, a, b) {
  const wk = d.next, v = verdict(d, a, b);
  const both = `${a.n} or ${b.n}`;
  const title = `${both} | Who Should I Start? Week ${wk} | Fantasy Mania`;
  const desc = v.w ? snippet(d, v, a, b) : v.line;
  const path = `/start/${[a.s, b.s].sort().join("-vs-")}/`;
  const ld = { "@context": "https://schema.org", "@type": "BreadcrumbList", itemListElement: [["Fantasy Mania", "/"], ["Start/Sit", "/start/"], [both, path]].map((c, i) => ({ "@type": "ListItem", position: i + 1, name: c[0], item: SITE + c[1] })) };
  const headExtra = `${listed(d, a, b) ? "" : '<meta name="robots" content="noindex,follow">'}<meta property="og:type" content="website"><meta property="og:site_name" content="Fantasy Mania"><meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(desc)}"><meta property="og:url" content="${SITE}${path}"><meta property="og:image" content="https://fantasymania.org/og.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://fantasymania.org/og.png"><script type="application/ld+json">${JSON.stringify(ld).replace(/</g, "\\u003c")}</script><style>${APP_CSS}</style>`;
  const app = await env.ASSETS.fetch(new Request(url.origin + "/index.html"));
  if (!app.ok) throw new Error("no app");
  const res = new HTMLRewriter()
    .on("head", { element(e) { e.prepend('<base href="/">', { html: true }); e.append(headExtra, { html: true }); } })
    .on("head > title", { element(e) { e.setInnerContent(title); } })
    .on('head > meta[property^="og:"]', { element(e) { e.remove(); } })
    .on('head > meta[name^="twitter:"]', { element(e) { e.remove(); } })
    .on('meta[name="description"]', { element(e) { e.setAttribute("content", desc); } })
    .on('link[rel="canonical"]', { element(e) { e.setAttribute("href", SITE + path); } })
    // Send the page with Compare already showing (not the homepage), so search engines see the matchup first.
    .on("section#home", { element(e) { e.setAttribute("class", (e.getAttribute("class") || "").replace(/\bon\b/, "").trim()); } })
    .on("section#compare", { element(e) { const c = e.getAttribute("class") || "view"; if (!/\bon\b/.test(c)) e.setAttribute("class", c + " on"); } })
    .on("#compare h1.title", { element(e) { e.setInnerContent(`${both}: Who Should I Start?`); e.after(`<p class="fmPpr">Week ${wk} start/sit · PPR scoring</p>`, { html: true }); } })
    .on("body", { element(e) { e.append(seoSection(d, a, b, v) + opener(d, a, b, v), { html: true }); } })
    .transform(app);
  const h = new Headers(res.headers);
  h.set("content-type", "text/html; charset=utf-8");
  h.set("cache-control", "public, max-age=300");
  return new Response(res.body, { status: 200, headers: h });
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
<meta property="og:type" content="website"><meta property="og:site_name" content="Fantasy Mania"><meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(desc)}"><meta property="og:url" content="${SITE}${path}"><meta property="og:image" content="https://fantasymania.org/og.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://fantasymania.org/og.png">
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
