"""Integration reproduction for the "win-win trades are missing from suggestions" bug.

The trade suggester used to accept candidates on starting-lineup points only, so a deal that improves
both rosters' rest-of-season VALUE (the yardstick the manual trade checker uses) without changing this
week's starters never appeared in the suggested trades. This drives the built app in a headless browser,
imports a real public ESPN league, runs the suggester, and asserts that it now surfaces trades that are
win-win by that value yardstick (both sides' blended value up by more than 0.25).

Needs network + a public ESPN league. Run after a build:
    python tests/trade_winwin_repro.py [public_dir] [leagueId] [year]
Defaults: public_dir=public, a known public league, current season. Requires: pip install playwright && python -m playwright install chromium
"""
import sys, json, time, threading, functools, http.server, urllib.request
from playwright.sync_api import sync_playwright

PUBLIC = sys.argv[1] if len(sys.argv) > 1 else "public"
LEAGUE = sys.argv[2] if len(sys.argv) > 2 else "2073417915"
YEAR = sys.argv[3] if len(sys.argv) > 3 else "2026"
BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/%s/segments/0/leagues/%s" % (YEAR, LEAGUE)

def get(q):
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE + q, headers={"User-Agent": "Mozilla/5.0"})).read())

feed = {"base": get("?view=mSettings&view=mTeam&view=mRoster&view=mStandings"), "proj": {}}
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=PUBLIC))
srv.RequestHandlerClass.log_message = lambda *a: None
threading.Thread(target=srv.serve_forever, daemon=True).start()

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_context(**p.devices["iPhone 13"]).new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:%d/index.html" % srv.server_address[1])
    pg.wait_for_function("typeof trEspnFromApi==='function'&&typeof lgIdeasAll==='function'&&typeof lgApply==='function'&&typeof DB!=='undefined'")
    # Drive the suggester for EVERY team, not one arbitrary pick: whether a given roster has a win-win
    # available swings week to week, but across the league there is always at least one, and old main
    # (no tier-8 pass) surfaces zero no matter which team you look at. trEspnFromApi + lgApply is exactly
    # the path the team picker uses, so this exercises the real code without the one-shot modal.
    t0 = time.perf_counter()
    out = pg.evaluate("""s=>{
      let r=trEspnFromApi(JSON.parse(s),Object.values(trCtx().V).map(v=>v.p),'');TR.espnPj=r.pj||{};
      let vs=(a,af)=>{let wk=trLineup(af,'now').tot-trLineup(a,'now').tot;return .3*(.6*wk+.4*trTVd(a,af,'now'))+.7*trTVd(a,af,'ros')};
      let teams=0,total=0,tier8=0,winWin=0;
      r.teams.forEach((t,i)=>{lgApply(r.teams,i,false);teams++;
        let sug=lgIdeasAll(TR.my);total+=sug.length;
        sug.forEach(d=>{if(d.tier===8)tier8++;
          let sa=d.A.map(x=>x.p.id),sb=d.B.map(x=>x.p.id),their=TR.lg[d.ti].ids,
              myA=TR.my.filter(i=>sa.indexOf(i)<0).concat(sb),thA=their.filter(i=>sb.indexOf(i)<0).concat(sa);
          if(vs(TR.my,myA)>0.25&&vs(their,thA)>0.25)winWin++;});});
      return {teams, total, winWin, tier8};
    }""", json.dumps(feed))
    ms_per_team = round((time.perf_counter() - t0) * 1000 / max(1, out["teams"]))
    b.close()

assert not errs, "page errors: %r" % errs
assert out["total"] > 0, "suggester returned nothing"
assert out["tier8"] > 0, "the win-win pass produced no trades league-wide: %r" % out
assert out["winWin"] > 0, "no win-win trades surfaced league-wide: %r" % out
assert ms_per_team < 6000, "suggester too slow: %dms/team" % ms_per_team
print("trade win-win repro: ok (%d teams, %d suggestions, %d win-win, %d from the win-win pass, %dms/team)"
      % (out["teams"], out["total"], out["winWin"], out["tier8"], ms_per_team))
