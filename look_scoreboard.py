"""Scoreboard look: turns the app's stylesheet into a black / brand-blue night-game version.

Only colours, corner shapes and heading fonts change. No content or layout is touched.
build.py calls make_scoreboard_css(css) and writes the result to public/css/fm-sb-<hash>.css.
The page loads that file only when the Scoreboard look is switched on (?look=sb), so the normal site is unchanged.
"""
import colorsys
import re

BLUE = (31, 107, 255)       # brand blue (buttons, selected)
BLUE_TXT = (88, 156, 255)   # brand blue readable on black
INK = (234, 240, 248)
NAMED = {"white": (255, 255, 255), "black": (0, 0, 0)}


def _hex(h):
    h = h.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
    return (r, g, b), a


def _parse(tok):
    t = tok.strip().lower()
    if t in NAMED:
        return NAMED[t], 1.0
    if t.startswith("#"):
        try:
            return _hex(t)
        except Exception:
            return None
    m = re.match(r"rgba?\(([^)]*)\)", t)
    if m:
        p = [x.strip() for x in re.split(r"[,\s/]+", m.group(1).strip()) if x.strip()]
        try:
            vals = []
            for x in p[:3]:
                vals.append(float(x[:-1]) * 2.55 if x.endswith("%") else float(x))
            a = float(p[3][:-1]) / 100 if len(p) > 3 and p[3].endswith("%") else float(p[3]) if len(p) > 3 else 1.0
            return tuple(int(max(0, min(255, v))) for v in vals), a
        except Exception:
            return None
    return None


def _fmt(rgb, a):
    r, g, b = [int(round(max(0, min(255, v)))) for v in rgb]
    return f"rgba({r},{g},{b},{round(a, 3)})" if a < 0.999 else "#%02x%02x%02x" % (r, g, b)


def _hsl(rgb):
    h, l, s = colorsys.rgb_to_hls(*[v / 255 for v in rgb])
    return h * 360, s, l


def _rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, max(0, min(1, l)), max(0, min(1, s)))
    return (r * 255, g * 255, b * 255)


def _is_blue(h, s):
    return 195 <= h <= 250 and s > 0.35


def map_bg(rgb, light_text=False):
    h, s, l = _hsl(rgb)
    if l >= 0.80:
        if s >= 0.40 and l < 0.97:                       # soft tint chips (green / yellow / red / blue tints)
            if _is_blue(h, s):
                return _rgb(216, 0.55, 0.17)
            return _rgb(h, min(s, 0.45), 0.17)
        nl = 0.03 + (l - 0.80) / 0.20 * 0.11             # page / card / inset levels keep their order
        return _rgb(214, 0.34, nl)
    if l >= 0.55:
        if s >= 0.40:
            return _rgb(h, min(s, 0.5), 0.22)
        return _rgb(214, 0.28, 0.30 - (l - 0.55) / 0.25 * 0.12)
    if s >= 0.35:                                         # saturated accents stay accents
        if _is_blue(h, s):
            return BLUE
        return _rgb(h, s, min(max(l, 0.34), 0.5))
    return BLUE if light_text else _rgb(214, 0.30, 0.19)  # filled dark neutral = selected / primary


def map_text(rgb):
    h, s, l = _hsl(rgb)
    if l >= 0.86:
        return rgb
    if s >= 0.35:
        if _is_blue(h, s):
            return BLUE_TXT
        return _rgb(h, min(1, s), max(l, 0.64))
    if l < 0.45:
        nl = 0.95 - (max(l, 0.12) - 0.12) / 0.33 * 0.27
        return _rgb(h or 214, min(s, 0.2), nl)
    return _rgb(h or 214, min(s, 0.22), max(l, 0.62))


def map_border(rgb):
    h, s, l = _hsl(rgb)
    if l >= 0.55:
        if s >= 0.40:
            return _rgb(h, min(s, 0.5), 0.30)
        return _rgb(214, 0.28, 0.13 + (l - 0.55) / 0.45 * 0.12)
    if s >= 0.35:
        if _is_blue(h, s):
            return BLUE_TXT
        return _rgb(h, s, max(l, 0.5))
    return _rgb(214, 0.2, 0.42)                           # dark line on light = strong line


COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|\bwhite\b|\bblack\b", re.I)


def _map_value(val, role, light_text=False):
    def rep(m):
        p = _parse(m.group(0))
        if not p:
            return m.group(0)
        rgb, a = p
        if a == 0:
            return m.group(0)
        if role == "shadow":
            return "rgba(0,0,0,%s)" % round(min(0.6, max(a, 0.25)), 2)
        f = {"bg": lambda c: map_bg(c, light_text), "text": map_text, "border": map_border}[role]
        return _fmt(f(rgb), a)
    return COLOR_RE.sub(rep, val)


def _split_decls(body):
    out, cur, depth, q = [], "", 0, ""
    for ch in body:
        if q:
            cur += ch
            if ch == q:
                q = ""
            continue
        if ch in "\"'":
            q = ch
            cur += ch
        elif ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == ";" and depth == 0:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _role(prop):
    p = prop.lower()
    if p in ("background", "background-color", "background-image", "stop-color", "flood-color"):
        return "bg"
    if p in ("color", "fill", "stroke", "caret-color", "text-decoration-color", "-webkit-text-fill-color", "column-rule-color"):
        return "text"
    if p.startswith("border") and not p.endswith(("radius", "width", "style", "collapse", "spacing", "image")):
        return "border"
    if p in ("outline", "outline-color"):
        return "border"
    if p in ("box-shadow", "text-shadow", "filter"):
        return "shadow"
    if p.startswith("--"):
        n = p
        if any(k in n for k in ("line", "border", "stroke")):
            return "border"
        if any(k in n for k in ("ink", "muted", "text", "fg", "txt")):
            return "text"
        if any(k in n for k in ("blue", "green", "yellow", "orange", "red", "accent", "good", "bad", "warn")):
            return "text"
        return "auto"
    return None


def _radius(val):
    def rep(m):
        n = float(m.group(1))
        return ("%gpx" % (4 if 5 <= n else n))
    if "%" in val:
        return val
    return re.sub(r"(\d+(?:\.\d+)?)px", rep, val)


def _transform_decls(body):
    decls = _split_decls(body)
    parsed = []
    props = {}
    for d in decls:
        if ":" not in d:
            parsed.append((None, d))
            continue
        k, v = d.split(":", 1)
        parsed.append((k.strip(), v.strip()))
        props[k.strip().lower()] = v.strip()
    light_text = False
    tv = props.get("color")
    if tv:
        pc = _parse(tv.split("!")[0])
        light_text = bool(pc and _hsl(pc[0])[2] >= 0.86)
    out = []
    changed = False
    fs = props.get("font-size", "")
    fw = props.get("font-weight", "")
    mfs = re.match(r"(\d+(?:\.\d+)?)px", fs)
    big = bool(mfs and float(mfs.group(1)) >= 22 and (fw.startswith(("700", "800", "900")) or fw in ("bold", "bolder")))
    if big and "font-family" not in props:
        out.append("font-family:'Barlow Condensed','Arial Narrow',Arial,sans-serif")
        changed = True
    for k, v in parsed:
        if k is None:
            continue
        lk = k.lower()
        nv = v
        role = _role(lk)
        if role == "auto" and COLOR_RE.search(v):
            pc = _parse(COLOR_RE.search(v).group(0))
            role = "text" if (pc and _hsl(pc[0])[1] >= 0.35) else "bg"
        if role and COLOR_RE.search(v):
            nv = _map_value(v, role, light_text)
        if lk == "font-family" and "Inter" in v:
            nv = "'Barlow Condensed','Arial Narrow',Arial,sans-serif" if big else "'Barlow',FM,'Helvetica Neue',Arial,sans-serif"
        if lk in ("border-radius", "border-top-left-radius", "border-top-right-radius", "border-bottom-left-radius", "border-bottom-right-radius"):
            nv = _radius(v)
        if nv != v:
            changed = True
            out.append(f"{k}:{nv}")
    return out if changed else []


def _split_top(css):
    """Yield (prelude, body) pairs at the top level of css."""
    i, n = 0, len(css)
    while i < n:
        j = css.find("{", i)
        if j < 0:
            break
        prelude = css[i:j].strip()
        depth, k, q = 1, j + 1, ""
        while k < n and depth:
            ch = css[k]
            if q:
                if ch == q:
                    q = ""
            elif ch in "\"'":
                q = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        yield prelude, css[j + 1:k - 1]
        i = k


def _prefix(sel):
    parts, cur, depth = [], "", 0
    for ch in sel:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    res = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith(":root"):
            res.append("html.sb" + p[5:])
        elif p.startswith("html"):
            res.append("html.sb" + p[4:])
        elif p == "*" or p.startswith("*,") or p.startswith("*::"):
            res.append("html.sb " + p)
        else:
            res.append("html.sb " + p)
    return ",".join(res)


def _walk(css, out):
    for prelude, body in _split_top(css):
        if prelude.startswith("@font-face") or prelude.startswith("@import") or prelude.startswith("@charset"):
            continue
        if prelude.startswith("@keyframes") or prelude.startswith("@-webkit-keyframes"):
            inner = []
            for fp, fb in _split_top(body):
                nd = _transform_decls(fb)
                if nd:
                    inner.append(f"{fp}{{{';'.join(nd)}}}")
            if inner:
                out.append(f"{prelude}{{{''.join(inner)}}}")
            continue
        if prelude.startswith("@media") or prelude.startswith("@supports"):
            sub = []
            _walk(body, sub)
            if sub:
                out.append(f"{prelude}{{{''.join(sub)}}}")
            continue
        if prelude.startswith("@"):
            continue
        nd = _transform_decls(body)
        if nd:
            out.append(f"{_prefix(prelude)}{{{';'.join(nd)}}}")


EXTRA = """
html.sb{color-scheme:dark;--bg:#070b12;--surface:#0e1520;--surface2:#141d2a;--line:#243247;--ink:#eaf0f8;--muted:#8fa0b6;--blue:#589cff;--blue2:#7db2ff;--green:#3ddc84;--yellow:#ffc24b;--orange:#ff9a4d;--red:#ff6b6b;--ff:'Barlow',FM,'Helvetica Neue',Arial,sans-serif}
html.sb,html.sb body{background:#070b12!important}
html.sb body{color:#eaf0f8;font-family:'Barlow',FM,'Helvetica Neue',Arial,sans-serif}
html.sb .sbHead{font-family:'Barlow Condensed','Arial Narrow',sans-serif}
html.sb h1,html.sb h2,html.sb .title,html.sb .sectionTitle,html.sb .kicker{font-family:'Barlow Condensed','Arial Narrow',Arial,sans-serif;text-transform:uppercase;letter-spacing:.03em}
html.sb h1,html.sb .title{font-weight:800;letter-spacing:.01em;color:#eaf0f8}
html.sb img.logo{filter:invert(1) hue-rotate(180deg)}
html.sb input,html.sb select,html.sb textarea{color:#eaf0f8;background-color:#0e1520;border-color:#2f3e52}
html.sb ::placeholder{color:#7d8da3}
html.sb ::-webkit-scrollbar{background:#0b1016}
html.sb ::-webkit-scrollbar-thumb{background:#2f3e52}
html.sb img{color-scheme:light}
"""


INLINE_RE = re.compile(r"style=\\?[\"']([^\"'<>]{3,400})")
DECL_RE = re.compile(r"((?:background(?:-color)?|color|border(?:-[a-z]+)*|fill|stroke)\s*:\s*(?:#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|white|black))", re.I)


def inline_rules(html):
    """Colours written straight into the page's tags (style="background:#e5f0ff") need their own rules."""
    seen, out = set(), []
    for m in INLINE_RE.finditer(html):
        for d in DECL_RE.findall(m.group(1)):
            d = d.strip()
            if d in seen:
                continue
            seen.add(d)
            prop = d.split(":", 1)[0].strip().lower()
            role = _role(prop)
            nd = _map_value(d.split(":", 1)[1], role) if role in ("bg", "text", "border") else None
            if nd and nd != d.split(":", 1)[1]:
                out.append('html.sb [style*="%s"]{%s:%s!important}' % (d.replace('"', ""), prop, nd.strip()))
    return "".join(out)


def make_scoreboard_css(css, html=""):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    _walk(css, out)
    return "".join(out) + (inline_rules(html) if html else "") + EXTRA
