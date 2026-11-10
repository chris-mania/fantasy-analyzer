"""Player and rankings pages that Google can read.

Runs after build.py and site_extras.py. Opens the finished app (public/index.html) in a
hidden browser, opens every player's real profile and the rankings, and saves what is on
screen as plain pages:

    /players/<name>/     the player's profile, exactly as the app draws it
    /rankings/           overall, plus /rankings/qb/ /rb/ /wr/ /te/

The saved page looks like the app. The first tap on a tab, arrow or button opens the
live profile in the app (/#p=<id>), so everything interactive still works. Player photos
are left out (the app's initials show instead); the app itself keeps the photos.

If anything goes wrong this script changes nothing and the normal site still publishes.
"""
import hashlib, html, json, os, re, shutil, sys, time, unicodedata

SITE = "https://fantasymania.org"
PUB = "public"
TMP = os.path.join(PUB, "_seo_tmp")
OUT_DIRS = ("players", "rankings", "a", "trade", "compare", "team", "insights", "teams")
ONLY = [x for x in os.environ.get("SEO_ONLY", "").split(",") if x]  # testing: limit players

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000105d3a1d00000000049454e44ae426082")


POSN = {"DST": "D/ST"}
PLURAL = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs", "K": "kickers", "DST": "defenses"}
# Weekly titles match how people search ("week 10 kicker rankings", "best defenses this week").
RANKS = [("ALL", "", "Week %s Fantasy Football Rankings (PPR): Top Players This Week"),
         ("QB", "qb/", "Week %s QB Rankings: Best Fantasy Quarterbacks This Week"),
         ("RB", "rb/", "Week %s RB Rankings: Best Fantasy Running Backs This Week (PPR)"),
         ("WR", "wr/", "Week %s WR Rankings: Top Fantasy Receivers This Week (PPR)"),
         ("TE", "te/", "Week %s TE Rankings: Best Fantasy Tight Ends This Week (PPR)"),
         ("K", "k/", "Week %s Kicker Rankings: Best Kickers to Start This Week"),
         ("DST", "dst/", "Week %s Defense Rankings: Best Fantasy Defenses to Start & Stream")]
CITY = {"ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo", "CAR": "Carolina", "CHI": "Chicago",
        "CIN": "Cincinnati", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
        "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville", "KC": "Kansas City", "LV": "Las Vegas",
        "LAC": "Los Angeles", "LA": "Los Angeles", "LAR": "Los Angeles", "MIA": "Miami", "MIN": "Minnesota",
        "NE": "New England", "NO": "New Orleans", "NYG": "New York", "NYJ": "New York", "PHI": "Philadelphia",
        "PIT": "Pittsburgh", "SF": "San Francisco", "SEA": "Seattle", "TB": "Tampa Bay", "TEN": "Tennessee",
        "WAS": "Washington", "WSH": "Washington"}
WORD = {"ALL": "players", "QB": "quarterbacks", "RB": "running backs", "WR": "wide receivers", "TE": "tight ends",
        "K": "kickers", "DST": "defenses"}


def dname(n):
    """'Bills D/ST' -> 'Bills Defense' (people search 'defense', not 'D/ST')."""
    return n[:-4].strip() + " Defense" if n.endswith("D/ST") else n


TOOLS = [
    ("trade", "Fantasy Football Trade Analyzer: Free Trade Calculator (%s PPR) | Fantasy Mania",
     "Free fantasy football trade analyzer. See who wins a trade by rest-of-season value, schedule strength and Mania Ratings. Works with ESPN and Sleeper leagues."),
    ("compare", "Fantasy Football Player Compare: Any Two Players (%s) | Fantasy Mania",
     "Free fantasy football player compare. Put any two players side by side: Mania Ratings, Week projections, usage, game logs and matchups. Compare whole fantasy teams too."),
    ("team", "Fantasy Football Team Analyzer: Start/Sit Help (%s) | Fantasy Mania",
     "Enter your fantasy football roster and get your best lineup, weekly projections and where your team is strong or weak. Free, PPR."),
    ("insights", "Fantasy Football Injuries, Depth Charts & Matchups (%s) | Fantasy Mania",
     "Weekly fantasy football reports: injuries, depth charts, team share, defense vs. position, schedule strength and play style. Free."),
]
TOOL_NAMES = {"trade": "Trade analyzer", "compare": "Player compare", "team": "Team analyzer", "insights": "Insights"}


def slugify(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "player"


def esc(s):
    return html.escape(str(s), quote=True)


class Assets:
    """data: URIs pulled out into shared files so each page stays small."""
    def __init__(self, root):
        self.root = os.path.join(root, "a")
        os.makedirs(self.root, exist_ok=True)

    def save(self, data, ext):
        name = hashlib.sha1(data).hexdigest()[:14] + "." + ext
        path = os.path.join(self.root, name)
        if not os.path.exists(path):
            open(path, "wb").write(data)
        return "/a/" + name

    def from_data_uri(self, m):
        import base64
        mime, b64, body = m.group(1), m.group(2), m.group(3)
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/svg+xml": "svg", "image/webp": "webp",
               "image/gif": "gif"}.get(mime)
        if not ext:
            return m.group(0)
        try:
            from urllib.parse import unquote
            data = base64.b64decode(body) if b64 else unquote(body).encode()
        except Exception:
            return m.group(0)
        return 'src="' + self.save(data, ext) + '"'


DATA_SRC = re.compile(r'src="data:([a-z/+.-]+)(;base64)?,([^"]*)"')
NFL_IMG = re.compile(r'<img\b[^>]*\bsrc="https://static\.www\.nfl\.com[^"]*"[^>]*>')
ON_ATTR = re.compile(r'\s(on[a-z]+)="[^"]*"')
SCRIPT = re.compile(r"<script\b[\s\S]*?</script>", re.I)
NAV_BTN = re.compile(r'<button\b([^>]*)\bdata-v="([a-z]+)"([^>]*)>([\s\S]*?)</button>')


def clean(fragment, assets):
    fragment = SCRIPT.sub("", fragment)
    fragment = NFL_IMG.sub("", fragment)
    fragment = ON_ATTR.sub("", fragment)
    fragment = DATA_SRC.sub(assets.from_data_uri, fragment)
    return fragment


def nav_links(topbar):
    def rep(m):
        attrs = (m.group(1) + m.group(3)).replace('type="button"', "")
        return '<a href="/#v=%s"%s data-v="%s">%s</a>' % (m.group(2), attrs.rstrip(), m.group(2), m.group(4))
    return NAV_BTN.sub(rep, topbar)


HANDOFF = """<script>
(function(){if(location.hostname==='fantasymania.pages.dev'){location.replace('%(site)s'+location.pathname);return}
var APP=%(app)s;
try{var nv0=performance.getEntriesByType('navigation')[0];if(nv0&&nv0.type==='reload'){location.replace(APP);return}}catch(e){}
document.addEventListener('click',function(e){var t=e.target;if(!t.closest)return;
 if(t.closest('a[href]'))return;
 if(t.closest('#profCopy')){e.preventDefault();e.stopPropagation();var cu=location.origin+location.pathname,cb=t.closest('#profCopy');
  if(navigator.share&&matchMedia('(pointer:coarse)').matches){navigator.share({title:document.title,url:cu}).catch(function(){});return}
  if(navigator.clipboard)navigator.clipboard.writeText(cu).then(function(){cb.textContent='Copied!'},function(){prompt('Copy this link',cu)});else prompt('Copy this link',cu);return}
 var nv=t.closest('.topbar [data-v]');if(nv){e.preventDefault();e.stopPropagation();location.href='/#v='+nv.dataset.v;return}
 if(t.closest('.topbar .logo,.topbar .brand')){location.href='/';return}
 var hit=t.closest('#seoMain button,#seoMain [data-k],#seoMain [data-board],#seoMain [role=button],#seoMain summary,#seoMain select,#seoMain input,#seoMain .backBtn');
 if(!hit)return;e.preventDefault();e.stopPropagation();
 if(hit.classList.contains('backBtn')&&document.body.dataset.back){location.href=document.body.dataset.back;return}
 location.href=APP},true);
})();
</script>"""


def set_nav(topbar, view):
    topbar = topbar.replace('class="navb on"', 'class="navb"')
    return topbar.replace('class="navb" data-v="%s"' % view, 'class="navb on" data-v="%s"' % view)


def page(*, title, desc, path, crumbs, body_class, topbar, sprite, main, css_href, fonts, app_url, back, extra=""):
    view = ("players" if path.startswith("/players/") else "rankings" if path.startswith("/rankings/")
            else "insights" if path.startswith(("/teams/", "/insights/")) else path.strip("/"))
    topbar = set_nav(topbar, view)
    ld = {
        "@context": "https://schema.org", "@type": "BreadcrumbList",
        "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, "item": SITE + u}
                            for i, (n, u) in enumerate(crumbs)]}
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>%s</title><meta name=\"description\" content=\"%s\">"
        '<link rel="canonical" href="%s%s">'
        '<meta property="og:type" content="website"><meta property="og:site_name" content="Fantasy Mania">'
        '<meta property="og:title" content="%s"><meta property="og:description" content="%s">'
        '<meta property="og:url" content="%s%s">'
        '<meta property="og:image" content="https://fantasymania.org/og.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://fantasymania.org/og.png"><meta name="theme-color" content="#ffffff">'
        '<link rel="icon" href="/apple-touch-icon.png"><link rel="apple-touch-icon" href="/apple-touch-icon.png">'
        "%s"
        '<link rel="stylesheet" href="%s">'
        '<style>#seoMore{max-width:1240px;margin:8px auto 40px;padding:0 20px}#seoMore h2{font-size:13px;letter-spacing:1.2px;text-transform:uppercase;color:#40474f;font-weight:800;margin:28px 0 12px}'
        '#seoMore ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:8px}'
        '#seoMore a{display:inline-flex;align-items:center;min-height:40px;padding:8px 14px;border-radius:99px;background:#fff;border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700;font-size:14px}'
        '#seoMore a:hover{border-color:#2563eb;color:#1d4ed8}#seoMore p a{display:inline;min-height:0;padding:0;border:0;background:none;border-radius:0;color:#1d4ed8;font-size:inherit}#seoMore h3{color:#0b1830}'
        '.seoSecH{font-size:26px!important;margin:40px 0 6px!important}#seoMore p{color:#5a616a;font-size:14px;line-height:1.55;max-width:760px}'
        '@media(max-width:800px){.seoD{display:none}}@media(min-width:801px){.seoM{display:none}}#seoMain .seoName{color:inherit;text-decoration:none}#seoMain .seoName:hover{text-decoration:underline}</style>'
        '<script type="application/ld+json">%s</script>'
        "%s"
        '</head><body class="%s" data-back="%s">%s%s<main class="wrap" id="seoMain">%s</main>%s</body></html>\n'
    ) % (esc(title), esc(desc), SITE, path, esc(title), esc(desc), SITE, path, fonts, css_href,
         json.dumps(ld, ensure_ascii=False).replace("</", "<\\/"),
         HANDOFF % {"site": SITE, "app": json.dumps(app_url)},
         esc(body_class), esc(back), sprite, topbar, main, extra)


LINKER = """() => { window.__seoLink = (root, names, own) => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); const hits = [];
    let n; while ((n = w.nextNode())) { const t = n.textContent.trim(); const u = names[t];
        if (u && u !== own && !n.parentElement.closest('a,h1,.hero2,.dkTop,option,title')) hits.push([n, u]); }
    hits.forEach(([n, u]) => { const a = document.createElement('a'); a.href = u; a.className = 'seoName';
        a.textContent = n.textContent; n.parentNode.replaceChild(a, n); });
} }"""


def both(desk, phone, assets):
    return ('<div class="seoD">%s</div><div class="seoM">%s</div>'
            % (clean(desk["html"], assets), clean(phone["html"], assets)))


def tool_chips(skip=None):
    return chips([("Who should I start?", "/start/")] + [(TOOL_NAMES[t], "/%s/" % t) for t, _, _ in TOOLS if t != skip])


def chips(links):
    return "<ul>" + "".join('<li><a href="%s">%s</a></li>' % (esc(u), esc(n)) for n, u in links) + "</ul>"


def run():
    if not os.path.exists(os.path.join(PUB, "index.html")):
        print("seo_pages: no public/index.html, skipping"); return
    from playwright.sync_api import sync_playwright

    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    assets = Assets(TMP)
    pub_abs = os.path.abspath(PUB)
    t0 = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, service_workers="block",
                                  reduced_motion="reduce")
        ctx_m = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
                                    device_scale_factor=2, service_workers="block", reduced_motion="reduce")

        def route(r):
            from urllib.parse import urlparse
            u = urlparse(r.request.url)
            if u.hostname == "fantasymania.org":
                p = u.path.lstrip("/") or "index.html"
                f = os.path.normpath(os.path.join(pub_abs, p))
                if f.startswith(pub_abs) and os.path.isfile(f):
                    ct = "text/html" if f.endswith(".html") else ("application/json" if f.endswith(".json") else None)
                    return r.fulfill(path=f, content_type=ct) if ct else r.fulfill(path=f)
                return r.fulfill(status=404, body="")
            if r.request.resource_type == "image" and "nfl.com" not in (u.hostname or ""):
                return r.fulfill(status=200, content_type="image/png", body=TINY_PNG)  # team logos keep their place
            return r.abort()

        ctx.route("**/*", route)
        ctx_m.route("**/*", route)
        errors = []

        def open_app(c):
            g = c.new_page()
            g.on("pageerror", lambda e: errors.append(str(e)))
            g.goto(SITE + "/", wait_until="load")
            g.wait_for_function("typeof DB!=='undefined' && typeof openPlayer==='function'", timeout=60000)
            g.wait_for_timeout(1500)
            return g

        pg, pg_m = open_app(ctx), open_app(ctx_m)

        shell = pg.evaluate("""() => { const lg = (p) => ((p && p.logs) || []).map(l => [l.w, l.ppr != null ? l.ppr : (l.pts != null ? l.pts : l.fp)]).filter(x => x[1] != null);
            return {
            css: [...document.querySelectorAll('style')].map(s => s.textContent).join('\\n'),
            fonts: [...document.head.querySelectorAll('link[rel=stylesheet],link[rel=preconnect]')].map(l => l.outerHTML).join(''),
            sprite: document.body.firstElementChild.tagName.toLowerCase() === 'svg' ? document.body.firstElementChild.outerHTML : '',
            topbar: (document.querySelector('.topbar') || {outerHTML: ''}).outerHTML,
            week: META.week, next: META.next_week, season: META.season, built: META.built || '',
            teams: [...new Set(DB.map(p => p.team))].filter(Boolean).sort().map(t => [t, ((XTRA.logos || {})[t] || {}).n || t]),
            players: DB.map(p => [p.id, p.name, p.pos, p.team, p.mania, p.pos_rank, p.proj ? p.proj.pts : null, p.opp || null, p.games, lg(p)])
              .concat((META.qbs || []).map(p => [p.id, p.name, 'QB', p.team, p.mania, p.pos_rank, p.proj ? p.proj.pts : null, p.opp || null, p.games, lg(p)]))
              .concat(((META.kd && META.kd.k) || []).concat((META.kd && META.kd.d) || []).map(x => {
                  const q = byId(x.id), r = (q && q.r) || {};
                  return [x.id, x.name, x.pos, x.team, r.rating == null ? null : r.rating, r.rank == null ? null : r.rank, null, null, x.games, lg(q || x)]; }))
        } }""")
        css_href = assets.save(shell["css"].encode(), "css").replace("/a/", "/a/app-", 1)
        os.rename(os.path.join(TMP, "a", os.path.basename(css_href).replace("app-", "", 1)),
                  os.path.join(TMP, "a", os.path.basename(css_href)))
        topbar = clean(shell["topbar"], assets)
        sprite = shell["sprite"]
        fonts = shell["fonts"]
        week, nxt, season = shell["week"], shell["next"], shell["season"]

        players, seen = [], {}
        shell["teams"] = [(t, ("%s %s" % (CITY[t], n)) if t in CITY and not n.startswith(CITY[t]) else n) for t, n in shell["teams"]]
        team_name = dict(shell["teams"])
        team_slug = {t: slugify(n) for t, n in shell["teams"]}
        for pid, name, pos, team, mania, prank, pts, opp, games, logs in shell["players"]:
            if pid in seen:
                continue
            s = slugify(name)
            if s in {p["slug"] for p in players}:
                s = s + "-" + slugify(team or pos)
            seen[pid] = s
            players.append(dict(id=pid, name=name, pos=pos, team=team, mania=mania, prank=prank,
                                pts=pts, opp=opp, games=games, slug=s, logs=logs))
        by_pos = {}
        for p in players:
            by_pos.setdefault(p["pos"], []).append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: (p["prank"] is None, p["prank"] or 999, -(p["mania"] or 0)))

        names = {p["name"]: "/players/%s/" % p["slug"] for p in players}
        pg.evaluate(LINKER); pg_m.evaluate(LINKER)

        # Start/sit data for the /start/ pages. If this part fails, everything else still gets made.
        start_links, start_ok, rows = {}, False, []
        try:
            import start_sit
            rows = pg.evaluate(start_sit.PICK_JS)
            start_links = start_sit.build(rows, {p["id"]: p["slug"] for p in players},
                                          {"season": season, "week": week, "next": nxt, "built": shell.get("built")},
                                          TMP, os.path.dirname(os.path.abspath(__file__)))
            start_ok = True
        except Exception as e:
            print("seo_pages: start/sit data skipped:", repr(e)[:200])
        wkinfo = {r["id"]: r for r in rows}
        slug_of = {p["id"]: p["slug"] for p in players}

        def week_top(pos, n):
            """This week's best plays at a position by projected points (healthy, not on bye)."""
            lst = [r for r in rows if (pos == "ALL" and r["pos"] not in ("K", "DST") or r["pos"] == pos)
                   and not r.get("out") and not r.get("bye") and (r.get("pj") or {}).get("p") is not None
                   and r["id"] in slug_of]
            lst.sort(key=lambda r: -r["pj"]["p"])
            return lst[:n]

        def plays(lst, start=1):
            return "; ".join('%d. <a class="seoName" href="/players/%s/">%s</a> (%s, %.1f projected)' % (
                i + start, slug_of[r["id"]], esc(dname(r["n"])), ("vs " + r["opp"]) if r.get("opp") else r["pos"], r["pj"]["p"])
                for i, r in enumerate(lst))

        def grab_player(g, pid, own):
            return g.evaluate("""([id, names, own]) => new Promise(res => {
                const p = byId(id); if (!p) return res(null);
                try { skipPush = true } catch (e) {}
                openPlayer(p);
                const v = document.getElementById('profile');
                let last = '', same = 0, tries = 0;
                const tick = () => {
                    const t = v.innerText;
                    same = (t === last) ? same + 1 : 0; last = t;
                    if ((same >= 2 && tries >= 2) || tries > 60) {
                        if (!v.classList.contains('on')) return res(null);
                        const bk = [...v.querySelectorAll('.maniaPanel .barrow')].map(r => {
                            const l = r.querySelector('.lblBtn') || r.querySelector('button') || r.querySelector('span'), num = r.querySelector('.pctNum') || r.querySelector('b');
                            const hid = l && l.dataset ? l.dataset.help : null, h = hid ? v.querySelector('#' + CSS.escape(hid)) : null;
                            return [(l ? l.textContent : '').replace('ⓘ', '').replace(/\\s*\\d+%\\s*$/, '').trim(), num ? +num.textContent : null,
                                    h ? h.textContent.trim().slice(0, 260) : ''];
                        }).filter(x => x[0] && x[1] != null);
                        const c = v.cloneNode(true); c.querySelectorAll('.glX').forEach(e => e.classList.add('go')); window.__seoLink(c, names, own);
                        return res({html: c.outerHTML, body: document.body.className, bk: bk});
                    }
                    tries++; setTimeout(tick, 60);
                };
                requestAnimationFrame(() => setTimeout(tick, 60));
            })""", [pid, names, own])

        def grab_log(g, own):
            """The profile that is open now, cut down to its header and the game log tile, opened."""
            return g.evaluate("""([names, own]) => {
                const v = document.getElementById('profile'); if (!v || !v.classList.contains('on')) return null;
                const c = v.cloneNode(true), body = c.querySelector('#profileBody'); if (!body) return null;
                const t = body.querySelector('.ptile[data-k="log"]'); if (!t) return null;
                [...body.children].forEach(ch => { if (!ch.matches('.topRow,.hero2,.dkTop,.ptiles')) ch.remove(); });
                c.querySelectorAll('.glX').forEach(e => e.classList.add('go'));
                body.querySelectorAll('.ptile').forEach(e => { if (e !== t) e.remove(); });
                t.classList.add('open'); t.querySelectorAll('.pprev,.ptx').forEach(e => e.remove());
                body.removeAttribute('data-pf'); body.querySelectorAll('[data-pfg]').forEach(e => e.removeAttribute('data-pfg'));
                c.querySelectorAll('.dkTop button').forEach(x => x.classList.toggle('on', /game log/i.test(x.textContent)));
                c.querySelectorAll('h1').forEach(h => { h.textContent = h.textContent.trim() + ' game log'; });
                window.__seoLink(c, names, own);
                return {html: c.outerHTML, body: document.body.className};
            }""", [names, own])

        todo = [p for p in players if not ONLY or p["slug"] in ONLY or p["id"] in ONLY]
        made, breakdowns = [], {}
        for i, p in enumerate(todo):
            try:
                own = "/players/%s/" % p["slug"]
                snap = grab_player(pg, p["id"], own)
                snap_m = grab_player(pg_m, p["id"], own)
            except Exception as e:
                print("seo_pages: skipped", p["name"], str(e)[:120]); continue
            if not snap or not snap_m:
                print("seo_pages: no profile for", p["name"]); continue
            if snap.get("bk"):
                breakdowns[p["id"]] = snap["bk"]
            peers = by_pos[p["pos"]]
            k = peers.index(p)
            near = [q for q in peers[max(0, k - 3):k + 4] if q is not p][:6]
            posl = p["pos"].lower()
            pn = POSN.get(p["pos"], p["pos"])
            rank_txt = ("%s%s" % (pn, p["prank"])) if p["prank"] else pn
            proj_txt = ("Week %s vs %s: %.1f PPR projected." % (nxt, p["opp"], p["pts"])) if p["pts"] is not None and p["opp"] else ""
            nm = dname(p["name"])
            wi = wkinfo.get(p["id"]) or {}
            wk_txt = (" %s%s this week." % (pn, wi["wk"])) if wi.get("wk") else ""
            title = "%s Week %s: Projection, Matchup & Fantasy Stats | Fantasy Mania" % (nm, nxt)
            desc = ("Should you start %s in Week %s? %s%s %s Mania Rating (%s). Free %s PPR stats, game log, matchup and trade value."
                    % (nm, nxt, proj_txt or "%s, %s." % (pn, p["team"]), wk_txt,
                       "%.1f" % p["mania"] if p["mania"] is not None else "—", rank_txt, season)).replace("  ", " ")
            sl = start_links.get(p["id"]) or []
            start_html = ("<h2>Start/sit this week</h2>" + chips([("%s or %s" % (p["name"], n), u) for n, u in sl])) if sl else ""
            own_links = [("%s game log & fantasy points" % nm, "/players/%s/game-log/" % p["slug"])]
            if p["team"] in team_slug:
                own_links.append(("%s depth chart & snap counts" % team_name[p["team"]], "/teams/%s/" % team_slug[p["team"]]))
            own_links.append(("Week %s %s rankings" % (nxt, pn), "/rankings/%s/" % posl))
            more = ('<section id="seoMore"><h2>More on %s</h2>%s' + start_html + '<h2>More %s</h2>%s<h2>Rankings</h2>%s'
                    '<p>Mania Ratings and projections through Week %s of the %s season, updated during the week. '
                    'Tap anything above to open the full interactive profile in Fantasy Mania.</p></section>') % (
                esc(nm), chips(own_links), PLURAL.get(p["pos"], p["pos"]), chips([(dname(q["name"]), "/players/%s/" % q["slug"]) for q in near]),
                chips([("All rankings", "/rankings/"), ("Open Fantasy Mania", "/")]) + "<h2>Tools</h2>" + tool_chips(), week, season)
            html_out = page(
                title=title, desc=desc, path="/players/%s/" % p["slug"],
                crumbs=[("Fantasy Mania", "/"), ("%s Rankings" % pn, "/rankings/%s/" % posl),
                        (p["name"], "/players/%s/" % p["slug"])],
                body_class=snap["body"], topbar=topbar, sprite=sprite, main=both(snap, snap_m, assets),
                css_href=css_href, fonts=fonts, app_url="/#p=" + p["id"], back="/rankings/%s/" % posl, extra=more)
            d = os.path.join(TMP, "players", p["slug"])
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(html_out)
            made.append("/players/%s/" % p["slug"])

            # Game log page: the profile's own game log, for "<player> fantasy points" / "<player> game log".
            try:
                gl, gl_m = grab_log(pg, own), grab_log(pg_m, own)
            except Exception as e:
                gl = gl_m = None
            if gl and gl_m:
                lg = sorted(p["logs"] or [], key=lambda x: -x[0])
                recent = ", ".join("W%s %.1f" % (w, v) for w, v in lg[:4])
                avg = (sum(v for _, v in lg) / len(lg)) if lg else None
                gpath = "/players/%s/game-log/" % p["slug"]
                gtitle = "%s Fantasy Points & Game Log %s (PPR) | Fantasy Mania" % (nm, season)
                gdesc = ("%s fantasy points every game in %s (PPR)%s.%s Full stats each week, plus the Week %s projection."
                         % (nm, season, (": " + recent) if recent else "",
                            (" Averaging %.1f points a game." % avg) if avg is not None else "", nxt))
                gmore = ('<section id="seoMore"><h2>More on %s</h2>%s<h2>Rankings</h2>%s<h2>Tools</h2>%s</section>') % (
                    esc(nm), chips([("Full %s profile" % nm, own)] + own_links[1:] + [(q, u) for q, u in
                                   [("%s or %s" % (nm, n2), u2) for n2, u2 in sl[:4]]]),
                    chips([("All rankings", "/rankings/")]), tool_chips())
                gh = page(title=gtitle, desc=gdesc, path=gpath,
                          crumbs=[("Fantasy Mania", "/"), (nm, own), ("Game log", gpath)],
                          body_class=gl["body"], topbar=topbar, sprite=sprite, main=both(gl, gl_m, assets),
                          css_href=css_href, fonts=fonts, app_url="/#p=" + p["id"], back=own, extra=gmore)
                os.makedirs(os.path.join(d, "game-log"), exist_ok=True)
                open(os.path.join(d, "game-log", "index.html"), "w", encoding="utf-8").write(gh)
                made.append(gpath)

        # Rankings: overall plus each position, drawn by the app's own rankings view.
        rank_pages = RANKS
        def grab_rank(g, key, names):
            g.goto(SITE + "/", wait_until="load")
            g.wait_for_function("typeof DB!=='undefined' && typeof showView==='function'", timeout=60000)
            g.wait_for_timeout(800)
            g.evaluate(LINKER)
            return g.evaluate("""([key, names]) => new Promise(res => {
                showView('rankings');
                const v = document.getElementById('rankings');
                const pm = v.querySelector('button.rankmode[data-mode="proj"]'); if (pm) pm.click();
                if (key !== 'ALL') { const b = v.querySelector('button[data-pos="' + key + '"],button[data-k="' + key + '"]'); if (!b) return res(null); b.click(); }
                setTimeout(() => {
                    const on = [...v.querySelectorAll('button[data-pos].on')].map(b => b.dataset.pos);
                    const c = v.cloneNode(true); window.__seoLink(c, names, '');
                    res({html: c.outerHTML, body: document.body.className, on: on});
                }, 600);
            })""", [key, names])

        for key, sub, label in rank_pages:
            try:
                snap = grab_rank(pg, key, names)
                snap_m = grab_rank(pg_m, key, names)
            except Exception as e:
                print("seo_pages: rankings", key, "skipped", str(e)[:120]); continue
            if not snap or not snap_m:
                continue
            path = "/rankings/" + sub
            title = label % nxt
            pos_word = {"ALL": "", "K": "kicker ", "DST": "defense "}.get(key, key + " ")
            top = week_top(key, 10)
            lead = ", ".join(dname(r["n"]) for r in top[:3])
            desc = (("Week %s fantasy football %srankings" + ("" if key in ("K", "DST") else " (PPR)") + "%s. Every %s ranked by this week's projected points, "
                    "with matchups and Mania Ratings. Updated through Week %s.") % (
                        nxt, pos_word, (": " + lead + " lead the way") if lead else "", WORD[key][:-1] if key != "ALL" else "player", week))
            w = WORD[key]
            faq = ""
            if top:
                qs = [("Who are the best %s to start in Week %s?" % (w, nxt),
                       "Fantasy Mania's top Week %s %s by projected%s points: %s." % (nxt, w, "" if key in ("K", "DST") else " PPR", plays(top[:5])))]
                if key in ("K", "DST") and len(top) > 5:
                    qs.append(("Which %s should I stream in Week %s?" % ("defense" if key == "DST" else "kicker", nxt),
                               "If your top options are gone, the next best Week %s plays are %s." % (nxt, plays(top[5:10], 6))))
                elif key == "ALL":
                    qs.append(("Who are the top fantasy players this week?", "After the top five, the next best Week %s plays are %s." % (nxt, plays(top[5:10], 6))))
                else:
                    qs.append(("Who are the top %s %s this week?" % (len(top), w), "Picks 6 to %d for Week %s: %s." % (len(top), nxt, plays(top[5:10], 6))))
                qs.append(("How are these Week %s rankings made?" % nxt,
                           ("Each %s is ranked by Fantasy Mania's Week %s projection, built from how it has played this season and how many points "
                            "the opponent gives up. Rankings update during the week as injuries and news come in." % ("defense" if key == "DST" else "kicker", nxt))
                           if key in ("K", "DST") else
                           "Players are ranked by Fantasy Mania's Week %s projection in PPR scoring, which blends each player's recent role, "
                           "his season-long Mania Rating (a 0-100 overall grade, like a Madden overall) and how many points his opponent allows. "
                           "Rankings update during the week as injuries and news come in." % nxt))
                faq = "<h2>Common questions</h2>" + "".join("<h3 style=\"font-size:17px;margin:18px 0 6px\">%s</h3><p>%s</p>" % (esc(q), a) for q, a in qs)
            others = [("All rankings", "/rankings/")] + [("%s rankings" % ("Defense" if k == "DST" else "Kicker" if k == "K" else k), "/rankings/" + sb)
                                                        for k, sb, _ in RANKS if sb]
            more = '<section id="seoMore">%s<h2>More Week %s rankings</h2>%s<h2>Tools</h2>%s</section>' % (
                faq, nxt, chips([o for o in others if o[1] != path] + [("NFL depth charts", "/teams/"), ("Open Fantasy Mania", "/")]), tool_chips())
            crumbs = [("Fantasy Mania", "/"), ("Rankings", "/rankings/")] + ([] if key == "ALL" else [("%s Rankings" % POSN.get(key, key), path)])
            html_out = page(title=title, desc=desc, path=path, crumbs=crumbs, body_class=snap["body"], topbar=topbar,
                            sprite=sprite, main=both(snap, snap_m, assets), css_href=css_href, fonts=fonts,
                            app_url="/#v=rankings", back="/", extra=more)
            d = os.path.join(TMP, "rankings", sub.strip("/")) if sub else os.path.join(TMP, "rankings")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(html_out)
            made.append(path)

        def grab_view(g, view):
            g.goto(SITE + "/", wait_until="load")
            g.wait_for_function("typeof DB!=='undefined' && typeof showView==='function'", timeout=60000)
            g.wait_for_timeout(800)
            g.evaluate(LINKER)
            return g.evaluate("""([view, names]) => new Promise(res => {
                showView(view);
                setTimeout(() => {
                    const v = document.getElementById(view); if (!v || !v.classList.contains('on')) return res(null);
                    const c = v.cloneNode(true); window.__seoLink(c, names, '');
                    res({html: c.outerHTML, body: document.body.className});
                }, 700);
            })""", [view, names])

        for view, title_t, desc in TOOLS:
            try:
                snap, snap_m = grab_view(pg, view), grab_view(pg_m, view)
            except Exception as e:
                print("seo_pages: tool", view, "skipped", str(e)[:120]); continue
            if not snap or not snap_m:
                continue
            path = "/%s/" % view
            more = ('<section id="seoMore"><h2>More tools</h2>%s<h2>Rankings</h2>%s<p>%s Tap anything above to use it in Fantasy Mania.</p></section>'
                    % (tool_chips(view), chips([("All rankings", "/rankings/")] + [("%s rankings" % POSN.get(k, k), "/rankings/" + sb) for k, sb, _ in RANKS if sb]), esc(desc)))
            html_out = page(title=title_t % season, desc=desc, path=path,
                            crumbs=[("Fantasy Mania", "/"), (TOOL_NAMES[view], path)], body_class=snap["body"],
                            topbar=topbar, sprite=sprite, main=both(snap, snap_m, assets), css_href=css_href,
                            fonts=fonts, app_url="/#v=" + view, back="/", extra=more)
            d = os.path.join(TMP, view)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(html_out)
            made.append(path)

        # Team pages: depth chart (incl. kicker), snap share, target share, game logs against. One per team.
        TEAM_JS = """([t, names, L]) => new Promise(async res => {
            const wait = (ms) => new Promise(r => setTimeout(r, ms)), q = (s) => document.querySelector(s);
            const tidy = (s) => s;
            showView('insights'); showInsight('share');
            const pick = document.getElementById('teamPick'); if (!pick || ![...pick.options].some(o => o.value === t)) return res(null);
            pick.value = t; q('.shareMode[data-share="snap"]').click(); await wait(150);
            const snap = q('#teamShareBody').innerHTML;
            q('.shareMode[data-share="target"]').click(); await wait(150);
            const target = q('#teamShareBody').innerHTML;
            q('.shareMode[data-share="snap"]').click();
            showInsight('depth'); let depth = '', st = '';
            if (window.dcOpenTeam) { q('#dcSeg button[data-s="off"]').click(); dcOpenTeam(t); await wait(120); depth = q('#dcBody').innerHTML;
                q('#dcSeg button[data-s="st"]').click(); await wait(120); st = q('#dcBody').innerHTML; q('#dcSeg button[data-s="off"]').click(); }
            showInsight('vs'); let vs = '';
            try { VST.team = t; VST.pos = 'ALL'; VST.role = 'ALL'; vsRender(); await wait(150); vs = q('#vsBody').innerHTML; } catch (e) {}
            const nick = L.split(' ').slice(-1)[0];
            const sec = (id, h, inner) => inner ? '<div class="insPane on" id="' + id + '"><h2 class="title seoSecH">' + h + '</h2>' + inner + '</div>' : '';
            const box = document.createElement('section'); box.id = 'insights'; box.className = 'view on';
            box.innerHTML = '<div class="kicker">Through Week ' + META.week + '</div><h1 class="title">' + L + ' depth chart, snap counts &amp; target share</h1>'
              + '<p class="sub">Who starts for the ' + L + ' at every fantasy position, who is on the field each week and who gets the ball. Tap any player for his full Fantasy Mania profile.</p>'
              + sec('ins-depth', L + ' depth chart', depth ? '<div id="dcBody">' + depth + '</div>' : '')
              + sec('ins-depth', nick + ' kicker &amp; special teams', st ? '<div id="dcBody">' + st + '</div>' : '')
              + sec('ins-share', nick + ' snap share by week', '<div id="teamShareBody">' + snap + '</div>')
              + sec('ins-share', nick + ' target &amp; rush share by week', '<div id="teamShareBody">' + target + '</div>')
              + sec('ins-vs', 'Game logs against the ' + nick + ' defense', vs ? '<div id="vsBody">' + vs + '</div>' : '');
            box.querySelectorAll('select,.vsCtl').forEach(e => e.remove());
            window.__seoLink(box, names, '');
            res({html: box.outerHTML, body: document.body.className});
        })"""
        team_made = []
        for g in (pg, pg_m):
            g.goto(SITE + "/", wait_until="load")
            g.wait_for_function("typeof DB!=='undefined' && typeof showInsight==='function'", timeout=60000)
            g.wait_for_timeout(800)
            g.evaluate(LINKER)
        team_players = {}
        for p in players:
            team_players.setdefault(p["team"], []).append(p)
        for t, full in shell["teams"]:
            try:
                snap, snap_m = pg.evaluate(TEAM_JS, [t, names, full]), pg_m.evaluate(TEAM_JS, [t, names, full])
            except Exception as e:
                print("seo_pages: team", t, "skipped", str(e)[:120]); continue
            if not snap or not snap_m:
                continue
            nick = full.split()[-1]
            path = "/teams/%s/" % team_slug[t]
            title = "%s Depth Chart, Snap Counts & Target Share (Week %s) | Fantasy Mania" % (full, nxt)
            if len(title) > 75:
                title = "%s Depth Chart, Snap Counts & Target Share (Week %s)" % (full, nxt)
            roster = sorted(team_players.get(t, []), key=lambda p: (p["pos"] != "QB", -(p["mania"] or 0)))
            stars = [dname(p["name"]) for p in roster if p["pos"] not in ("K", "DST")][:3]
            kick = [p["name"] for p in roster if p["pos"] == "K"]
            desc = ("%s fantasy football: who starts at QB, RB, WR, TE%s, snap share and target share every week, "
                    "and every player's game log against the %s defense. %sThrough Week %s." % (
                        full, " and kicker (%s)" % kick[0] if kick else " and kicker", nick,
                        ("Featuring " + ", ".join(stars) + ". ") if stars else "", week))
            others = [(n, "/teams/%s/" % team_slug[x]) for x, n in shell["teams"] if x != t]
            more = ('<section id="seoMore"><h2>%s players</h2>%s<h2>Rankings</h2>%s<h2>Other teams</h2>%s<h2>Tools</h2>%s</section>') % (
                esc(nick), chips([(dname(p["name"]), "/players/%s/" % p["slug"]) for p in roster[:24]]),
                chips([("Week %s %s rankings" % (nxt, "Defense" if k == "DST" else "Kicker" if k == "K" else k), "/rankings/" + sb) for k, sb, _ in RANKS if sb]),
                chips(others), tool_chips())
            html_out = page(title=title, desc=desc, path=path,
                            crumbs=[("Fantasy Mania", "/"), ("Teams", "/teams/"), (full, path)], body_class=snap["body"],
                            topbar=topbar, sprite=sprite, main=both(snap, snap_m, assets), css_href=css_href, fonts=fonts,
                            app_url="/#v=insights", back="/teams/", extra=more)
            d = os.path.join(TMP, "teams", team_slug[t])
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(html_out)
            made.append(path); team_made.append((full, path))
        if team_made:
            main = ('<section class="view on" id="teamsIdx"><div class="kicker">Week %s</div><h1 class="title">NFL depth charts, snap counts &amp; target share</h1>'
                    '<p class="sub">Pick a team to see its fantasy depth chart (including the kicker), who is on the field each week, who gets the targets and carries, '
                    'and every player\'s game log against that defense.</p><div id="seoMore" style="padding:0;margin:20px 0">%s</div></section>') % (week, chips(team_made))
            path = "/teams/"
            html_out = page(title="NFL Depth Charts for Fantasy Football: Snap Counts & Target Share (Week %s)" % nxt,
                            desc="Fantasy football depth charts for all 32 NFL teams, with weekly snap share, target share, kickers and game logs against every defense. Free, updated through Week %s." % week,
                            path=path, crumbs=[("Fantasy Mania", "/"), ("Teams", path)], body_class="", topbar=topbar, sprite=sprite,
                            main=main, css_href=css_href, fonts=fonts, app_url="/#v=insights", back="/",
                            extra='<section id="seoMore"><h2>Rankings</h2>%s<h2>Tools</h2>%s</section>' % (
                                chips([("All rankings", "/rankings/")] + [("%s rankings" % ("Defense" if k == "DST" else "Kicker" if k == "K" else k), "/rankings/" + sb) for k, sb, _ in RANKS if sb]), tool_chips()))
            os.makedirs(os.path.join(TMP, "teams"), exist_ok=True)
            open(os.path.join(TMP, "teams", "index.html"), "w", encoding="utf-8").write(html_out)
            made.append(path)

        # League-wide insight pages: schedule strength and defense vs. position, every position on one page.
        LEAGUE_JS = """([pane, btn, body, poss, names, h1, sub]) => new Promise(async res => {
            const wait = (ms) => new Promise(r => setTimeout(r, ms));
            showView('insights'); showInsight(pane); let html = '';
            for (const ps of poss) { const b = document.querySelector('.' + btn + '[data-v="' + ps + '"]'); if (!b) continue; b.click(); await wait(200);
                const el = document.getElementById(body); if (!el) continue;
                html += '<h2 class="title seoSecH">' + ps + '</h2><div class="panel" id="' + body + '">' + el.innerHTML + '</div>'; }
            if (!html) return res(null);
            const box = document.createElement('section'); box.id = 'insights'; box.className = 'view on';
            box.innerHTML = '<div class="kicker">Week ' + META.next_week + '</div><h1 class="title">' + h1 + '</h1><p class="sub">' + sub + '</p><div class="insPane on" id="ins-' + pane + '">' + html + '</div>';
            window.__seoLink(box, names, ''); res({html: box.outerHTML, body: document.body.className});
        })"""
        LEAGUE = [
            ("schedule-strength", "schedule", "scPos", "scheduleBody",
             "Fantasy football strength of schedule by position",
             "For every team, how easy the next four games look for its QBs, RBs, WRs and TEs, based on the points each upcoming defense gives up to that position. Easiest first.",
             "Fantasy Football Strength of Schedule by Position (Week %s)",
             "Fantasy football strength of schedule for QB, RB, WR and TE: which teams have the easiest and toughest matchups over the next four weeks, from points each defense allows. Free, Week %s."),
            ("defense-vs-position", "defense", "dvPos", "defenseBody",
             "Fantasy points allowed: defense vs. position",
             "How many fantasy points every defense gives up to quarterbacks, running backs, receivers and tight ends, and whether that makes the matchup easy or tough.",
             "Fantasy Points Allowed by Defense vs. Position: QB, RB, WR, TE (Week %s)",
             "Defense vs. position for fantasy football: fantasy points, yards and touchdowns every NFL defense allows to QBs, RBs, WRs and TEs, easiest matchups first. Free, Week %s."),
        ]
        for slug, pane, btn, body, h1, sub, ttl, dsc in LEAGUE:
            try:
                snap = pg.evaluate(LEAGUE_JS, [pane, btn, body, ["QB", "RB", "WR", "TE"], names, h1, sub])
                snap_m = pg_m.evaluate(LEAGUE_JS, [pane, btn, body, ["QB", "RB", "WR", "TE"], names, h1, sub])
            except Exception as e:
                print("seo_pages: insights", slug, "skipped", str(e)[:120]); continue
            if not snap or not snap_m:
                continue
            path = "/insights/%s/" % slug
            html_out = page(title=ttl % nxt, desc=dsc % nxt, path=path,
                            crumbs=[("Fantasy Mania", "/"), ("Insights", "/insights/"), (h1, path)], body_class=snap["body"],
                            topbar=topbar, sprite=sprite, main=both(snap, snap_m, assets), css_href=css_href, fonts=fonts,
                            app_url="/#v=insights", back="/insights/",
                            extra='<section id="seoMore"><h2>Teams</h2>%s<h2>Rankings</h2>%s<h2>Tools</h2>%s</section>' % (
                                chips(team_made), chips([("Week %s rankings" % nxt, "/rankings/")]), tool_chips()))
            d = os.path.join(TMP, "insights_x", slug)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(html_out)
            made.append(path)

        browser.close()

    if errors:
        print("seo_pages: page errors while drawing:", errors[:3])
    if len(made) < 20 and not ONLY:
        raise RuntimeError("only %d pages made, keeping the site as it was" % len(made))

    # Swap the new pages in, then list everything in the sitemap.
    xd = os.path.join(TMP, "insights_x")
    if os.path.isdir(xd) and os.path.isdir(os.path.join(TMP, "insights")):
        for n in os.listdir(xd):
            shutil.move(os.path.join(xd, n), os.path.join(TMP, "insights", n))
    shutil.rmtree(xd, ignore_errors=True)
    for d in OUT_DIRS:
        shutil.rmtree(os.path.join(PUB, d), ignore_errors=True)
        if os.path.isdir(os.path.join(TMP, d)):
            shutil.move(os.path.join(TMP, d), os.path.join(PUB, d))
    if start_ok:
        for f in ("start-data.json", "sitemap-start.xml", "_worker.js", "_routes.json"):
            shutil.move(os.path.join(TMP, f), os.path.join(PUB, f))
    shutil.rmtree(TMP, ignore_errors=True)
    urls = ["/"] + sorted(u for u in made if not u.startswith("/players")) + sorted(u for u in made if u.startswith("/players"))
    with open(os.path.join(PUB, "sitemap.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for u in urls:
            f.write("  <url><loc>%s%s</loc><lastmod>%s</lastmod></url>\n" % (SITE, u, time.strftime("%Y-%m-%d", time.gmtime())))
        f.write("</urlset>\n")
    print("seo_pages: %d pages in %.0fs" % (len(made), time.time() - t0))


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        shutil.rmtree(TMP, ignore_errors=True)
        print("seo_pages: FAILED, site publishes without new pages:", repr(e)[:300])
        sys.exit(0)
