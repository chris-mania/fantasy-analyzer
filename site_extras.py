"""Runs after build.py, every build. Adds what the site needs no matter what
build.py looks like: the pages.dev -> fantasymania.org move (carries saved data,
skips home-screen apps), Google tags, robots.txt and sitemap.xml.
Kept separate from build.py so feature edits there can't remove it."""
import os

MOVE = r'''<script id="fmMove">(function(){var H=location.hostname;
if(H==='fantasymania.pages.dev'){
 try{if((window.matchMedia&&matchMedia('(display-mode: standalone)').matches)||navigator.standalone===true)return}catch(e){return}
 var d={},n=0,skip={faAlive:1,faRs:1,fmRs:1,faLast:1,fmNav2:1,fmMoved:1};
 try{for(var i=0;i<localStorage.length;i++){var k=localStorage.key(i);if(!skip[k]){d[k]=localStorage.getItem(k);n++}}}catch(e){return}
 var tail=location.hash;
 if(n){tail='#fmmove='+encodeURIComponent(JSON.stringify({d:d,h:location.hash}));if(tail.length>500000)return}
 location.replace('https://fantasymania.org'+location.pathname+location.search+tail);return}
if(H==='fantasymania.org'&&location.hash.indexOf('#fmmove=')===0){
 var raw=location.hash.slice(8),p=null;
 try{p=JSON.parse(decodeURIComponent(raw))}catch(e){}
 try{history.replaceState(null,'',location.pathname+location.search+(p&&p.h||''))}catch(e){}
 if(!p||!p.d)return;
 try{if(localStorage.getItem('fmMoved'))return;
  for(var k in p.d){if(localStorage.getItem(k)===null)localStorage.setItem(k,p.d[k])}
  localStorage.setItem('fmMoved','1')}catch(e){}}
})();</script>'''

GOOGLE = ('<link rel="canonical" href="https://fantasymania.org/">'
          '<script type="application/ld+json">{"@context":"https://schema.org","@type":"WebSite",'
          '"name":"Fantasy Mania","alternateName":["FantasyMania","Fantasy Mania Fantasy Football"],'
          '"url":"https://fantasymania.org/"}</script>')

def patch(path, extra_head=""):
    if not os.path.exists(path):
        print("site_extras: missing", path); return
    s = open(path, encoding="utf-8").read()
    if 'id="fmMove"' not in s:
        i = s.find("<head>")
        assert i >= 0, path + " has no <head>"
        s = s[:i+6] + MOVE + s[i+6:]
    if extra_head and 'rel="canonical"' not in s:
        i = s.find("</head>")
        assert i >= 0, path + " has no </head>"
        s = s[:i] + extra_head + s[i:]
    open(path, "w", encoding="utf-8").write(s)
    print("site_extras: patched", path)

patch("public/index.html", GOOGLE)
patch("public/fantasy.html")
open("public/robots.txt", "w").write("User-agent: *\nAllow: /\n\nSitemap: https://fantasymania.org/sitemap.xml\nSitemap: https://fantasymania.org/sitemap-start.xml\n")
open("public/sitemap.xml", "w").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <url><loc>https://fantasymania.org/</loc></url>\n</urlset>\n')
print("site_extras: robots.txt and sitemap.xml written")

# ---- Discovery helpers (each one is optional: if it fails, the site still publishes) ----
import html as _html, re as _re, shutil as _shutil
SITE = "https://fantasymania.org"
from site_extras_key import KEY as INDEXNOW_KEY  # public by design: Bing checks it at /<key>.txt

HOME_CSS = ('<style id="fmHomeLinks">.homeActions a.cta{text-decoration:none;display:inline-flex;align-items:center;justify-content:center}'
            '.fmExplore{margin:28px 0 8px}.fmExplore .maH{margin-bottom:10px}'
            '.fmExplore ul{list-style:none;margin:0 0 12px;padding:0;display:flex;flex-wrap:wrap;gap:8px}'
            '.fmExplore a{display:inline-flex;align-items:center;min-height:40px;padding:8px 14px;border-radius:99px;background:#fff;'
            'border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700;font-size:14px}'
            '.fmExplore a:hover{border-color:#2563eb;color:#1d4ed8}.fmExplore h3{font-size:12px;letter-spacing:1px;text-transform:uppercase;'
            'color:#5a6882;margin:14px 0 8px;font-weight:800}</style>')
EXPLORE = ('<nav class="fmExplore" aria-label="Explore Fantasy Mania"><div class="maH">Explore Fantasy Mania</div>'
           '<h3>Start/sit</h3><ul><li><a href="/start/">Who should I start this week?</a></li></ul>'
           '<h3>Rankings</h3><ul>' + "".join('<li><a href="/rankings/%s">%s</a></li>' % (u, n) for u, n in (
               ("", "All rankings"), ("qb/", "QB rankings"), ("rb/", "RB rankings"), ("wr/", "WR rankings"),
               ("te/", "TE rankings"), ("k/", "Kicker rankings"), ("dst/", "Defense rankings"))) + '</ul>'
           '<h3>Tools</h3><ul>' + "".join('<li><a href="/%s/">%s</a></li>' % (u, n) for u, n in (
               ("trade", "Trade analyzer"), ("compare", "Player compare"), ("team", "Team analyzer"), ("insights", "Insights"))) +
           '</ul></nav>')


def share_tags(title, desc, url):
    e = lambda s: _html.escape(s, quote=True)
    return ('<meta property="og:type" content="website"><meta property="og:site_name" content="Fantasy Mania">'
            '<meta property="og:title" content="%s"><meta property="og:description" content="%s">'
            '<meta property="og:url" content="%s"><meta property="og:image" content="%s/og.png">'
            '<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">'
            '<meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="%s/og.png">'
            % (e(title), e(desc), url, SITE, SITE))


def home_extras(path="public/index.html"):
    s = open(path, encoding="utf-8").read()
    if 'id="fmHomeLinks"' in s:
        return
    t = _re.search(r"<title>([^<]*)</title>", s)
    dsc = _re.search(r'<meta name="description" content="([^"]*)"', s)
    title = _html.unescape(t.group(1)) if t else "Fantasy Mania"
    desc = _html.unescape(dsc.group(1)) if dsc else "Free fantasy football ratings, projections and start/sit help."
    head = HOME_CSS + ("" if 'property="og:image"' in s else share_tags(title, desc, SITE + "/"))
    i = s.find("</head>")
    s = s[:i] + head + s[i:]
    btn = '<button class="cta" id="homeInsights">Insights</button>'
    if s.count(btn) == 1:
        s = s.replace(btn, btn + '<a class="cta" href="/start/">Who should I start?</a>')
    disc = '<p class="tinyDisc">'
    hs = s.find('<section id="home"')
    j = s.find(disc, hs)
    if hs >= 0 and j > hs:
        s = s[:j] + EXPLORE + s[j:]
    open(path, "w", encoding="utf-8").write(s)
    print("site_extras: homepage links and share preview added")


LLMS = """# Fantasy Mania

> Free fantasy football ratings, weekly projections, start/sit answers and trade help (PPR). Every player gets a 0-100 Mania Rating, an overall grade like a Madden overall, built from production, usage, role, high-value chances and efficiency, plus a projection for this week.

## Main pages
- [Home](https://fantasymania.org/): ratings, projections and tools
- [Who should I start?](https://fantasymania.org/start/): pick any two players for a start/sit answer; matchup pages live at /start/<player>-vs-<player>/
- [Rankings](https://fantasymania.org/rankings/): overall plus [QB](https://fantasymania.org/rankings/qb/), [RB](https://fantasymania.org/rankings/rb/), [WR](https://fantasymania.org/rankings/wr/), [TE](https://fantasymania.org/rankings/te/), [K](https://fantasymania.org/rankings/k/), [D/ST](https://fantasymania.org/rankings/dst/)
- Player pages: /players/<player-name>/ (for example https://fantasymania.org/players/jahmyr-gibbs/)
- [Trade analyzer](https://fantasymania.org/trade/), [Player compare](https://fantasymania.org/compare/), [Team analyzer](https://fantasymania.org/team/), [Insights](https://fantasymania.org/insights/)

## Notes
- Scoring is PPR. Numbers update during the week; projections are estimates, not guarantees.
- Full page lists: https://fantasymania.org/sitemap.xml and https://fantasymania.org/sitemap-start.xml
"""

def app_view_extras(path="public/fantasy.html"):
    """fantasy.html is the same app in a special view: point Google at the homepage and add the share preview."""
    s = open(path, encoding="utf-8").read()
    if 'rel="canonical"' in s:
        return
    i = s.find("</head>")
    t = _re.search(r"<title>([^<]*)</title>", s)
    s = s[:i] + '<link rel="canonical" href="%s/">' % SITE + share_tags(_html.unescape(t.group(1)) if t else "Fantasy Mania",
        "Free fantasy football ratings, projections and start/sit help.", SITE + "/") + s[i:]
    open(path, "w", encoding="utf-8").write(s)
    print("site_extras: fantasy.html points to the homepage")


NOT_FOUND = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Page not found | Fantasy Mania</title><meta name="robots" content="noindex"><link rel="icon" href="/apple-touch-icon.png">
<style>body{margin:0;background:#e9edf2;color:#0b1830;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:640px;margin:12vh auto;padding:0 20px}.card{background:#fff;border-radius:20px;padding:28px;border:1px solid #e4eaf3}
.brand{font-weight:900;font-style:italic;font-size:22px;letter-spacing:-.6px;text-decoration:none;color:#060709}.brand b{color:#1d4ed8}
h1{font-size:30px;letter-spacing:-.6px;margin:18px 0 8px}p{color:#40474f;line-height:1.55;margin:0 0 18px}
ul{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px}a.c{display:inline-flex;align-items:center;min-height:44px;padding:8px 16px;border-radius:99px;background:#fff;border:1px solid #d9dfe7;color:#0b1830;text-decoration:none;font-weight:700}
a.c.p{background:#0b1830;color:#fff;border-color:#0b1830}</style></head><body><main><div class="card">
<a class="brand" href="/">FANTASY <b>MANIA</b></a><h1>We couldn't find that page</h1>
<p>The link may be old or mistyped. Here's where to go instead.</p>
<ul><li><a class="c p" href="/">Open Fantasy Mania</a></li><li><a class="c" href="/start/">Who should I start?</a></li>
<li><a class="c" href="/rankings/">Rankings</a></li><li><a class="c" href="/compare/">Player compare</a></li><li><a class="c" href="/trade/">Trade analyzer</a></li></ul>
</div></main></body></html>
"""

for step in (home_extras, app_view_extras):
    try:
        step()
    except Exception as e:
        print("site_extras: skipped", step.__name__, repr(e)[:200])
try:
    if os.path.exists("og.png"):
        _shutil.copyfile("og.png", "public/og.png")
    open("public/llms.txt", "w").write(LLMS)
    open("public/404.html", "w").write(NOT_FOUND)
    open("public/%s.txt" % INDEXNOW_KEY, "w").write(INDEXNOW_KEY)
    print("site_extras: og.png, llms.txt and Bing key written")
except Exception as e:
    print("site_extras: extras skipped", repr(e)[:200])
