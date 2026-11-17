"""Builds the side page https://fantasymania.org/preview/ : the finished app page with the "Scoreboard" look laid on top.

Runs in the workflow after the normal build (and after site_extras/seo_pages), reads public/index.html and writes
public/preview/. It never edits the real pages. It makes no API calls. If anything fails the workflow keeps going
(the step is continue-on-error) and the live site publishes as usual.

The look itself is two plain files in look/ (look.css, look.js). Styling and small layout moves only; no content is removed.
"""
import os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(ROOT, "public")
SRC = os.path.join(ROOT, "look")
OUT = os.path.join(PUB, "preview")


def boost(css):
    """The app's own rules are scoped by page id (#team ...). Prefix every rule with :is(#team,body) so these win."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def rules(txt):
        res, j = [], 0
        while j < len(txt):
            k = txt.find("{", j)
            if k < 0:
                break
            sel, d, m = txt[j:k].strip(), 1, k + 1
            while d and m < len(txt):
                d += (txt[m] == "{") - (txt[m] == "}")
                m += 1
            body = txt[k + 1:m - 1]
            if sel.startswith(("@media", "@supports")):
                res.append(sel + "{" + rules(body) + "}")
            elif sel.startswith("@"):
                res.append(sel + "{" + body + "}")
            else:
                ss = []
                for x in sel.split(","):
                    x = x.strip()
                    if x:
                        ss.append(x if re.match(r"^(html|body)\b", x) else ":is(#team,body) " + x)
                res.append(",".join(ss) + "{" + body + "}")
            j = m
        return "\n".join(res)

    return rules(css)


def main():
    src = os.path.join(PUB, "index.html")
    if not os.path.exists(src):
        print("look_preview: public/index.html not found; nothing to do")
        return 0
    h = open(src, encoding="utf-8").read()
    if len(h) < 300000 or 'id="brandHome"' not in h and "class=\"topbar\"" not in h:
        print("look_preview: page does not look like the app; skipping")
        return 0
    os.makedirs(os.path.join(OUT, "look"), exist_ok=True)
    css = boost(open(os.path.join(SRC, "look.css"), encoding="utf-8").read())
    open(os.path.join(OUT, "look", "look.css"), "w", encoding="utf-8").write(css)
    js = open(os.path.join(SRC, "look.js"), encoding="utf-8").read()
    open(os.path.join(OUT, "look", "look.js"), "w", encoding="utf-8").write(js)
    v = str(abs(hash(css + js)) % 10**8)
    # keep Google and the "install" flow away from this side page
    h = re.sub(r'<link[^>]*rel="canonical"[^>]*>', "", h)
    h = re.sub(r'<link[^>]*rel="manifest"[^>]*>', "", h)
    h = re.sub(r'<meta[^>]*property="og:url"[^>]*>', "", h)
    h = h.replace('href="apple-touch-icon.png"', 'href="/apple-touch-icon.png"')
    h = h.replace("</head>", '<meta name="robots" content="noindex,nofollow"></head>', 1)
    i = h.rfind("</body>")
    h = h[:i] + '<link rel="stylesheet" href="/preview/look/look.css?v=' + v + '"><script src="/preview/look/look.js?v=' + v + '"></script>' + h[i:]
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(h)
    rb = os.path.join(PUB, "robots.txt")
    if os.path.exists(rb):
        t = open(rb, encoding="utf-8").read()
        if "/preview/" not in t:
            open(rb, "w", encoding="utf-8").write(t.rstrip("\n") + "\n\nUser-agent: *\nDisallow: /preview/\n")
    print("look_preview: wrote public/preview/ (%d bytes page, %d bytes css)" % (len(h), len(css)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never block the real site
        print("look_preview: skipped (%s)" % e)
        sys.exit(0)
