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
OUT_DIRS = ("players", "rankings", "a", "trade", "compare", "team", "insights")
ONLY = [x for x in os.environ.get("SEO_ONLY", "").split(",") if x]  # testing: limit players

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000105d3a1d00000000049454e44ae426082")


POSN = {"DST": "D/ST"}
PLURAL = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs", "K": "kickers", "DST": "defenses"}
RANKS = [("ALL", "", "Fantasy Football Rankings"), ("QB", "qb/", "Fantasy Football QB Rankings"),
         ("RB", "rb/", "Fantasy Football RB Rankings"), ("WR", "wr/", "Fantasy Football WR Rankings"),
         ("TE", "te/", "Fantasy Football TE Rankings"), ("K", "k/", "Fantasy Football Kicker Rankings"),
         ("DST", "dst/", "Fantasy Football Defense (D/ST) Rankings")]


TOOLS = [
    ("trade", "Free Fantasy Football Trade Analyzer (%s PPR) | Fantasy Mania",
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
document.addEventListener('click',function(e){var t=e.target;if(!t.closest)return;
 if(t.closest('a[href]'))return;
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
    view = "players" if path.startswith("/players/") else "rankings" if path.startswith("/rankings/") else path.strip("/")
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
        '#seoMore a:hover{border-color:#2563eb;color:#1d4ed8}#seoMore p{color:#5a616a;font-size:14px;line-height:1.55;max-width:760px}'
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

        shell = pg.evaluate("""() => ({
            css: [...document.querySelectorAll('style')].map(s => s.textContent).join('\\n'),
            fonts: [...document.head.querySelectorAll('link[rel=stylesheet],link[rel=preconnect]')].map(l => l.outerHTML).join(''),
            sprite: document.body.firstElementChild.tagName.toLowerCase() === 'svg' ? document.body.firstElementChild.outerHTML : '',
            topbar: (document.querySelector('.topbar') || {outerHTML: ''}).outerHTML,
            week: META.week, next: META.next_week, season: META.season, built: META.built || '',
            players: DB.map(p => [p.id, p.name, p.pos, p.team, p.mania, p.pos_rank, p.proj ? p.proj.pts : null, p.opp || null, p.games])
              .concat((META.qbs || []).map(p => [p.id, p.name, 'QB', p.team, p.mania, p.pos_rank, p.proj ? p.proj.pts : null, p.opp || null, p.games]))
              .concat(((META.kd && META.kd.k) || []).concat((META.kd && META.kd.d) || []).map(x => {
                  const q = byId(x.id), r = (q && q.r) || {};
                  return [x.id, x.name, x.pos, x.team, r.rating == null ? null : r.rating, r.rank == null ? null : r.rank, null, null, x.games]; }))
        })""")
        css_href = assets.save(shell["css"].encode(), "css").replace("/a/", "/a/app-", 1)
        os.rename(os.path.join(TMP, "a", os.path.basename(css_href).replace("app-", "", 1)),
                  os.path.join(TMP, "a", os.path.basename(css_href)))
        topbar = clean(shell["topbar"], assets)
        sprite = shell["sprite"]
        fonts = shell["fonts"]
        week, nxt, season = shell["week"], shell["next"], shell["season"]

        players, seen = [], {}
        for pid, name, pos, team, mania, prank, pts, opp, games in shell["players"]:
            if pid in seen:
                continue
            s = slugify(name)
            if s in {p["slug"] for p in players}:
                s = s + "-" + slugify(team or pos)
            seen[pid] = s
            players.append(dict(id=pid, name=name, pos=pos, team=team, mania=mania, prank=prank,
                                pts=pts, opp=opp, games=games, slug=s))
        by_pos = {}
        for p in players:
            by_pos.setdefault(p["pos"], []).append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: (p["prank"] is None, p["prank"] or 999, -(p["mania"] or 0)))

        names = {p["name"]: "/players/%s/" % p["slug"] for p in players}
        pg.evaluate(LINKER); pg_m.evaluate(LINKER)

        # Start/sit data for the /start/ pages. If this part fails, everything else still gets made.
        start_links, start_ok = {}, False
        try:
            import start_sit
            rows = pg.evaluate(start_sit.PICK_JS)
            start_links = start_sit.build(rows, {p["id"]: p["slug"] for p in players},
                                          {"season": season, "week": week, "next": nxt, "built": shell.get("built")},
                                          TMP, os.path.dirname(os.path.abspath(__file__)))
            start_ok = True
        except Exception as e:
            print("seo_pages: start/sit data skipped:", repr(e)[:200])

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
                        const c = v.cloneNode(true); window.__seoLink(c, names, own);
                        return res({html: c.outerHTML, body: document.body.className, bk: bk});
                    }
                    tries++; setTimeout(tick, 60);
                };
                requestAnimationFrame(() => setTimeout(tick, 60));
            })""", [pid, names, own])

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
            title = "%s Fantasy Rating & Week %s Projection | Fantasy Mania" % (p["name"], nxt)
            desc = ("%s (%s, %s): %s Mania Rating, %s in Fantasy Mania. %s Free %s PPR stats, game log, matchup and trade value."
                    % (p["name"], pn, p["team"], "%.1f" % p["mania"] if p["mania"] is not None else "—",
                       rank_txt, proj_txt, season)).replace("  ", " ")
            sl = start_links.get(p["id"]) or []
            start_html = ("<h2>Start/sit this week</h2>" + chips([("%s or %s" % (p["name"], n), u) for n, u in sl])) if sl else ""
            more = ('<section id="seoMore">' + start_html + '<h2>More %s</h2>%s<h2>Rankings</h2>%s'
                    '<p>Mania Ratings and projections through Week %s of the %s season, updated during the week. '
                    'Tap anything above to open the full interactive profile in Fantasy Mania.</p></section>') % (
                PLURAL.get(p["pos"], p["pos"]), chips([(q["name"], "/players/%s/" % q["slug"]) for q in near]),
                chips([("%s rankings" % pn, "/rankings/%s/" % posl), ("All rankings", "/rankings/"),
                       ("Open Fantasy Mania", "/")]) + "<h2>Tools</h2>" + tool_chips(), week, season)
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
            title = "%s %s, Week %s (PPR) | Fantasy Mania" % (season, label, nxt)
            pos_word = "" if key == "ALL" else POSN.get(key, key) + " "
            desc = ("Free %s fantasy football %srankings by Mania Rating, with Week %s projections. "
                    "Updated through Week %s. Tap any player for the full profile." % (season, pos_word, nxt, week))
            others = [("All rankings", "/rankings/")] + [("%s rankings" % POSN.get(k, k), "/rankings/" + sb)
                                                        for k, sb, _ in RANKS if sb]
            more = '<section id="seoMore"><h2>More rankings</h2>%s<h2>Tools</h2>%s<p>%s</p></section>' % (
                chips([o for o in others if o[1] != path] + [("Open Fantasy Mania", "/")]), tool_chips(), esc(desc))
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

        browser.close()

    if errors:
        print("seo_pages: page errors while drawing:", errors[:3])
    if len(made) < 20 and not ONLY:
        raise RuntimeError("only %d pages made, keeping the site as it was" % len(made))

    # Swap the new pages in, then list everything in the sitemap.
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
