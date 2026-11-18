"""Start/sit data for the /start/ matchup pages (used by seo_pages.py).

seo_pages.py pulls every player's Week numbers out of the built app and calls build() here.
This writes:
    public/start-data.json   compact numbers the /start/ pages are drawn from
    public/sitemap-start.xml the pairs we ask Google to look at
    public/_worker.js        the page maker (copied from start_worker.js), runs only for /start/
    public/_routes.json      keeps the page maker off every other address

Which pairs get listed for Google (Chris's rules):
    QB, K, D/ST: every pair of players with a game this week
    RB top 80, WR top 100, TE top 35 by Week rating
    within 8 projected points of each other (any K or D/ST pair is fine)
    same position only for now; RB vs WR pages work and can be found, but are not in the sitemap yet
Any pair still works if someone types it in; unlisted pairs just tell Google not to index them.
"""
import json, os, shutil
from datetime import datetime, timezone

SITE = "https://fantasymania.org"
TOP = {"RB": 80, "WR": 100, "TE": 35}
GAP = 8.0
FLEX_IN_SITEMAP = False  # turn on once Google is crawling the same-position pages

NICK = {
    "Christian McCaffrey": ["CMC"],
    "Jaxon Smith-Njigba": ["JSN"],
    "Amon-Ra St. Brown": ["ARSB"],
    "Brian Thomas Jr.": ["BTJ"],
    "Marvin Harrison Jr.": ["MHJ"],
    "A.J. Brown": ["AJB"],
    "Justin Jefferson": ["JJettas", "Jettas"],
    "DK Metcalf": ["DK"],
    "Terry McLaurin": ["T-Mac"],
    "T.J. Hockenson": ["TJ Hock"],
    "Marquise Brown": ["Hollywood Brown"],
}

PICK_JS = """() => {
  const ids = DB.map(p => p.id).concat((META.qbs || []).map(p => p.id))
     .concat(((META.kd && META.kd.k) || []).concat((META.kd && META.kd.d) || []).map(x => x.id));
  const raw = {}; ((META.kd && META.kd.k) || []).concat((META.kd && META.kd.d) || []).forEach(x => raw[x.id] = x);
  const n = v => (v == null || isNaN(v)) ? null : Math.round(+v * 10) / 10;
  const out = [];
  ids.forEach(id => {
    const p = byId(id); if (!p) return;
    const kd = raw[id], pr = p.proj || null, m = p.m || {};
    let last = (p.logs || []).slice(-3).map(l => n(l.ppr != null ? l.ppr : l.pts));
    if (kd && kd.log) last = kd.log.slice(-3).map(l => n(l.pts));
    const pl = p.played || null;
    out.push({
      id: p.id, n: p.name, pos: p.pos, tm: p.team, opp: p.opp || null,
      wr: n(p.start), mr: n(p.mania), pr: p.pos_rank || null, g: p.games || 0, out: !!p.out,
      bye: kd ? !!kd.bye : !p.opp,
      inj: p.inj ? {l: p.inj.label || '', t: p.inj.note || '', a: p.inj.avail} : null,
      pl: pl ? {s: pl.score || '', r: pl.res || '', f: n(pl.ppr != null ? pl.ppr : pl.pts)} : null,
      pj: pr ? {p: pr.pts == null ? null : Math.round(+pr.pts * 1000) / 1000, lo: n(pr.lo), hi: n(pr.hi), rec: n(pr.rec), ryd: n(pr.ryd), ruy: n(pr.ruy),
                td: n(pr.td), tgt: n(pr.tgt), car: n(pr.car), pyd: n(pr.pyd), ptd: n(pr.ptd), int: n(pr.int)} : null,
      ma: n(p.matchup_adj), tr: n(p.trend_pct), ppg: n(m.ppr != null ? m.ppr : p.ppr_pg),
      sh: n(p.pos === 'RB' ? m.touch : m.tshare), sn: n(m.snap), l3: last,
      dp: p.depth && p.depth.rank || null,
      fx: pr && pr.fx ? {s: n(pr.fx.script), m: n(pr.fx.matchup), i: n(pr.fx.inj)} : null,
      gl: (kd ? (kd.log || []).map(l => [l.w, l.opp || null, n(l.pts)])
              : (p.logs || []).map(l => [l.w, l.opp || ((META.opps || {})[p.team] || {})[l.w] || null, n(l.ppr)])),
      ms: kd ? (p.pos === 'K' ? kd.st || null : null) : m,
      kl: (kd && p.pos === 'DST') ? (kd.log || []).map(l => [l.pa, l.ya, l.sk, (l.int || 0) + (l.fr || 0), l.td || 0]) : null
    });
  });
  return out;
}"""


def pair_slug(a, b):
    x, y = sorted([a, b])
    return "%s-vs-%s" % (x, y)


def listed(a, b):
    """Would Chris want this pair as a page for Google?"""
    if not (a["in"] and b["in"]):
        return False
    same = a["pos"] == b["pos"]
    flex = {a["pos"], b["pos"]} == {"RB", "WR"}
    if not (same or flex):
        return False
    if a["pos"] in ("K", "DST") and same:
        return True
    pa, pb = (a["pj"] or {}).get("p"), (b["pj"] or {}).get("p")
    return pa is not None and pb is not None and abs(pa - pb) < GAP


def add_breakdowns(out_dir, bk):
    """Put each player's Mania breakdown (read from his profile in the app) into start-data.json."""
    path = os.path.join(out_dir, "start-data.json")
    data = json.load(open(path, encoding="utf-8"))
    n = 0
    for p in data["players"]:
        if p["id"] in bk:
            p["bk"] = bk[p["id"]]
            n += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print("start_sit: Mania breakdown added for %d players" % n)


def build(rows, slugs, meta, out_dir, here):
    """rows: from PICK_JS; slugs: id -> page slug. Returns {id: [(partner_name, url), ...]} for player pages."""
    for r in rows:
        if isinstance(r.get("ms"), dict):
            r["ms"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r["ms"].items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # How many PPR points each defense allows to each position (for the Matchup tab).
    tot = {}
    for r in rows:
        if r["pos"] in ("K", "DST"):
            continue
        for w, opp, f in r.get("gl") or []:
            if opp and f is not None:
                t = tot.setdefault(opp, {}).setdefault(r["pos"], {})
                t[w] = t.get(w, 0) + f
    dvp = {}
    for pos in ("QB", "RB", "WR", "TE"):
        avg = {tm: sum(v[pos].values()) / len(v[pos]) for tm, v in tot.items() if v.get(pos)}
        order = sorted(avg, key=lambda tm: -avg[tm])
        for i, tm in enumerate(order):
            dvp.setdefault(tm, {})[pos] = [round(avg[tm], 1), i + 1, len(order)]
    players = []
    for r in rows:
        if r["id"] not in slugs:
            continue
        r["s"] = slugs[r["id"]]
        r["nk"] = NICK.get(r["n"], [])
        players.append(r)
    by_pos = {}
    for r in players:
        by_pos.setdefault(r["pos"], []).append(r)
    for pos, lst in by_pos.items():
        playing = [r for r in lst if not r["bye"] and r["wr"] is not None]
        playing.sort(key=lambda r: -r["wr"])
        for i, r in enumerate(playing):
            r["wk"] = i + 1
        for r in lst:
            r.setdefault("wk", None)
            ok = r["wk"] is not None and not r["out"] and (r["pj"] or {}).get("p") is not None
            r["in"] = bool(ok and (r["wk"] <= TOP[pos] if pos in TOP else True))

    pool = [r for r in players if r["in"]]
    pairs = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a["pos"] != b["pos"] and not FLEX_IN_SITEMAP:
                continue
            if listed(a, b):
                pa, pb = a["pj"]["p"], b["pj"]["p"]
                pairs.append((-min(a["wr"], b["wr"]), abs(pa - pb), pair_slug(a["s"], b["s"]), a, b))
    pairs.sort(key=lambda x: (x[0], x[1]))

    partners = {}
    for _, gap, slug, a, b in pairs:
        near = abs((a["wk"] or 999) - (b["wk"] or 999)) + gap / 10.0  # similar weekly rank first
        partners.setdefault(a["id"], []).append((near, b["n"], slug))
        partners.setdefault(b["id"], []).append((near, a["n"], slug))
    links = {pid: [(n, "/start/%s/" % s) for _, n, s in sorted(v)[:6]] for pid, v in partners.items()}

    for r in players:  # the app draws charts, stats and breakdowns itself; keep the file small
        for k in ("gl", "ms", "kl", "bk", "fx", "sh", "sn", "dp"):
            r.pop(k, None)
    now = datetime.now(timezone.utc)
    data = {"v": 1, "season": meta["season"], "week": meta["week"], "next": meta["next"],
            "built": meta.get("built") or now.strftime("%a %b %-d"), "date": now.strftime("%Y-%m-%d"),
            "gap": GAP, "players": players}
    with open(os.path.join(out_dir, "start-data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(out_dir, "sitemap-start.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        f.write("  <url><loc>%s/start/</loc><lastmod>%s</lastmod></url>\n" % (SITE, data["date"]))
        for _, _, slug, _, _ in pairs[:49000]:
            f.write("  <url><loc>%s/start/%s/</loc><lastmod>%s</lastmod></url>\n" % (SITE, slug, data["date"]))
        f.write("</urlset>\n")
    shutil.copyfile(os.path.join(here, "start_worker.js"), os.path.join(out_dir, "_worker.js"))
    with open(os.path.join(out_dir, "_routes.json"), "w") as f:
        json.dump({"version": 1, "include": ["/start", "/start/*", "/trade/*"], "exclude": []}, f)
    print("start_sit: %d players, %d pairs listed for Google" % (len(players), len(pairs)))
    return links
