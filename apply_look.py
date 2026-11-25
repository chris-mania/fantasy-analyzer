"""Applies the "Scoreboard" look (look/look.css + look/look.js) straight to the real site.

Runs in the workflow right after site_extras.py and before seo_pages.py, so every page Google
sees (player profiles, rankings, teams, insights -- seo_pages.py just opens whatever
public/index.html already renders and saves the result) picks up the new look automatically.
Writes the finished CSS/JS to public/look/ and links them from public/index.html and
public/fantasy.html. Styling and small layout moves only; no content is removed. Runs
continue-on-error in the workflow so a problem here never blocks the real site publishing.
"""
import os, re, sys, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(ROOT, "public")
SRC = os.path.join(ROOT, "look")


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


def apply_to(path, head_tag, body_tag):
    if not os.path.exists(path):
        print("apply_look: missing", path)
        return
    h = open(path, encoding="utf-8").read()
    if "look/look.css" in h:
        print("apply_look: already applied to", path)
        return
    j = h.rfind("</head>")
    i = h.rfind("</body>")
    if j < 0 or i < 0:
        print("apply_look:", path, "has no </head> or </body>; skipping")
        return
    # look.css goes in <head> (render-blocking, like any normal stylesheet) so the browser applies
    # the "Scoreboard" look before first paint instead of flashing the plain base look and then
    # visibly swapping to it a moment later. look.js stays at the end of <body>: it patches DOM the
    # main app script just built, so it still needs to run after that script.
    h = h[:j] + head_tag + h[j:]
    i = h.rfind("</body>")
    h = h[:i] + body_tag + h[i:]
    open(path, "w", encoding="utf-8").write(h)
    print("apply_look: patched", path)


def main():
    css = boost(open(os.path.join(SRC, "look.css"), encoding="utf-8").read())
    js = open(os.path.join(SRC, "look.js"), encoding="utf-8").read()
    v = str(abs(hash(css + js)) % 10**8)
    os.makedirs(os.path.join(PUB, "look"), exist_ok=True)
    open(os.path.join(PUB, "look", "look.css"), "w", encoding="utf-8").write(css)
    open(os.path.join(PUB, "look", "look.js"), "w", encoding="utf-8").write(js)
    img_src = os.path.join(SRC, "img")
    if os.path.isdir(img_src):
        shutil.copytree(img_src, os.path.join(PUB, "look", "img"), dirs_exist_ok=True)
    # The Barlow font used to be added by a <link> that look.js injected at runtime -- by the time
    # that script tag was even reached (end of body) and then fetched the font, the page had already
    # painted with a fallback font and look.css's colors, so the body text visibly swapped fonts a
    # moment later. Loading it as a normal <link> in <head> lets the browser fetch it in parallel
    # with everything else, right alongside the stylesheet that names it.
    # Only the plain "Barlow" family goes here -- "Barlow Condensed" is already requested (at a
    # wider weight range) by build.py's own <link> near the top of the page, and asking Google
    # Fonts for it a second time here just adds a redundant request competing for the same
    # bandwidth while every headline is waiting on it to avoid the fallback-font flash.
    font_tag = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=optional">'
    head_tag = font_tag + '<link rel="stylesheet" href="/look/look.css?v=' + v + '">'
    body_tag = '<script src="/look/look.js?v=' + v + '"></script>'
    apply_to(os.path.join(PUB, "index.html"), head_tag, body_tag)
    apply_to(os.path.join(PUB, "fantasy.html"), head_tag, body_tag)
    print("apply_look: wrote public/look/ (%d bytes css, %d bytes js)" % (len(css), len(js)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never block the real site
        print("apply_look: skipped (%s)" % e)
        sys.exit(0)
