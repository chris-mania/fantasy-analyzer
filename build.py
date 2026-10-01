import json
import math
import os
import numpy as np
import pandas as pd

SEASON = 2026
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# ===== TUNING KNOBS: change these to change the rankings =====
OPP = {"target": 1.6, "carry": 0.7, "rz_target": 1.5, "rz_carry": 0.5, "gl_carry": 2.0}
SCORE_W = {"opp": 0.35, "snap": 0.20, "share": 0.20, "rz": 0.15, "ppr": 0.10}
QUALIFY_SNAP_PCT = 30   # avg snap % needed to count in position percentiles
LUCK_GAP = 3.0          # PPR/G above or below usage-expected before flagging

print("1. Fetching data...")
stats = pd.read_csv(f"{BASE}/stats_player/stats_player_week_{SEASON}.csv", low_memory=False)
snaps = pd.read_csv(f"{BASE}/snap_counts/snap_counts_{SEASON}.csv", low_memory=False)
if "season_type" in stats.columns:
    stats = stats[stats["season_type"] == "REG"]
if "game_type" in snaps.columns:
    snaps = snaps[snaps["game_type"] == "REG"]

name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
team_col = "team" if "team" in stats.columns else "recent_team"
for c in ["targets", "carries", "receiving_air_yards", "fantasy_points_ppr"]:
    if c not in stats.columns:
        stats[c] = 0
    stats[c] = stats[c].fillna(0)

# Team totals (all positions) for share stats
team_tot = (
    stats.groupby(team_col)[["targets", "carries", "receiving_air_yards"]].sum()
    .rename(columns={"targets": "t_tgt", "carries": "t_car", "receiving_air_yards": "t_air"})
    .reset_index().rename(columns={team_col: "team"})
)

keep = ["player_id", name_col, team_col, "position", "week",
        "targets", "carries", "receiving_air_yards", "fantasy_points_ppr"]
s = stats[stats["position"].isin(["WR", "RB", "TE"])][keep].rename(
    columns={name_col: "player_name", team_col: "team"})

sn = snaps[["player", "team", "week", "offense_snaps", "offense_pct"]].drop_duplicates(
    ["player", "team", "week"])
merged = s.merge(sn, left_on=["player_name", "team", "week"],
                 right_on=["player", "team", "week"], how="left")
merged["offense_snaps"] = merged["offense_snaps"].fillna(0)
merged["offense_pct"] = merged["offense_pct"].fillna(0.0)

print("2. Fetching red zone data (play-by-play)...")
rz_ok = True
try:
    want = ["season_type", "week", "yardline_100", "pass_attempt", "rush_attempt",
            "receiver_player_id", "rusher_player_id"]
    try:
        pbp = pd.read_csv(f"{BASE}/pbp/play_by_play_{SEASON}.csv.gz",
                          usecols=lambda c: c in want, low_memory=False)
    except Exception:
        pbp = pd.read_csv(f"{BASE}/pbp/play_by_play_{SEASON}.csv",
                          usecols=lambda c: c in want, low_memory=False)
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["yardline_100"] <= 20)]
    rec = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec["player_id"] = rec["receiver_player_id"]
    rec["rz_targets"] = 1
    rus = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    rus["player_id"] = rus["rusher_player_id"]
    rus["rz_carries"] = 1
    rus["gl_carries"] = (rus["yardline_100"] <= 5).astype(int)
    rz = (pd.concat([rec[["player_id", "week", "rz_targets"]],
                     rus[["player_id", "week", "rz_carries", "gl_carries"]]])
          .fillna(0).groupby(["player_id", "week"]).sum().reset_index())
except Exception as e:
    print("WARNING: red zone data unavailable:", e)
    rz_ok = False

if rz_ok:
    merged = merged.merge(rz, on=["player_id", "week"], how="left")
for c in ["rz_targets", "rz_carries", "gl_carries"]:
    if c not in merged.columns:
        merged[c] = 0
    merged[c] = merged[c].fillna(0)

merged["opp_pts"] = (
    merged["targets"] * OPP["target"] + merged["carries"] * OPP["carry"]
    + merged["rz_targets"] * OPP["rz_target"] + merged["rz_carries"] * OPP["rz_carry"]
    + merged["gl_carries"] * OPP["gl_carry"]
)

print("3. Calculating metrics, ranks and Usage Score...")
g = merged.groupby(["player_id", "player_name", "team", "position"]).agg(
    games=("week", "count"), targets=("targets", "sum"), carries=("carries", "sum"),
    air=("receiving_air_yards", "sum"), rz_t=("rz_targets", "sum"),
    rz_c=("rz_carries", "sum"), gl_c=("gl_carries", "sum"),
    snap_pct=("offense_pct", "mean"), ppr_ppg=("fantasy_points_ppr", "mean"),
    opp_pg=("opp_pts", "mean")).reset_index()
g = g.merge(team_tot, on="team", how="left")
for c in ["t_tgt", "t_car", "t_air"]:
    g[c] = g[c].fillna(0)

def div(a, b):
    return np.where(b > 0, a / b.where(b > 0, 1) * 100, 0)

g["snap_pct"] = g["snap_pct"] * 100
g["targets_pg"] = g["targets"] / g["games"]
g["carries_pg"] = g["carries"] / g["games"]
g["tgt_share"] = div(g["targets"], g["t_tgt"])
g["air_share"] = div(g["air"], g["t_air"])
g["touch_share"] = div(g["targets"] + g["carries"], g["t_tgt"] + g["t_car"])
g["share"] = np.where(g["position"] == "RB", g["touch_share"], g["tgt_share"])
g["rz_pg"] = (g["rz_t"] + g["rz_c"]) / g["games"]
g["gl_pg"] = g["gl_c"] / g["games"]
g["rz_val"] = (g["rz_t"] + 0.5 * g["rz_c"] + 2 * g["gl_c"]) / g["games"]
g["qualified"] = g["snap_pct"] >= QUALIFY_SNAP_PCT

def pctile(df, col):
    out = pd.Series(0.0, index=df.index)
    for pos, grp in df.groupby("position"):
        ref = np.sort(grp.loc[grp["qualified"], col].values)
        if len(ref) == 0:
            continue
        vals = grp[col].values
        p = np.minimum(np.searchsorted(ref, vals, side="right") / len(ref) * 100, 100)
        p[vals <= 0] = 0
        out.loc[grp.index] = p
    return out

METRICS = ["targets_pg", "carries_pg", "opp_pg", "snap_pct", "share",
           "air_share", "rz_pg", "gl_pg", "ppr_ppg"]
for col in METRICS + ["rz_val"]:
    g["p_" + col] = pctile(g, col)
    g["t_" + col] = g.groupby(["team", "position"])[col].rank(
        ascending=False, method="min").astype(int)

w = dict(SCORE_W)
if not rz_ok:
    w.pop("rz")
tot = sum(w.values())
pmap = {"opp": "p_opp_pg", "snap": "p_snap_pct", "share": "p_share",
        "rz": "p_rz_val", "ppr": "p_ppr_ppg"}
g["score"] = sum(w[k] / tot * g[pmap[k]] for k in w)

g["pos_rank"] = g.groupby("position")["score"].transform(
    lambda x: x.where(g.loc[x.index, "qualified"]).rank(ascending=False, method="min")
).fillna(0).astype(int)
pos_n = g[g["qualified"]].groupby("position").size().to_dict()
g["team_rank"] = g.groupby(["team", "position"])["score"].rank(
    ascending=False, method="min").astype(int)
g["role"] = g["position"] + g["team_rank"].astype(str)

# Usage-expected PPR (for buy low / sell high)
q = g[g["qualified"]].groupby("position")[["ppr_ppg", "opp_pg"]].sum()
kmap = (q["ppr_ppg"] / q["opp_pg"].clip(lower=0.1)).to_dict()
g["exp_ppr"] = g["opp_pg"] * g["position"].map(kmap).fillna(1)
g["luck"] = g["ppr_ppg"] - g["exp_ppr"]

max_week = int(merged["week"].max())
small_cut = max(2, math.ceil(0.5 * max_week))

players = []
for _, r in g.sort_values("score", ascending=False).iterrows():
    d = merged[(merged["player_id"] == r["player_id"]) & (merged["team"] == r["team"])].sort_values("week")
    logs = [{"w": int(x.week), "t": int(x.targets), "c": int(x.carries),
             "s": int(x.offense_snaps), "sp": round(float(x.offense_pct) * 100, 1),
             "rt": int(x.rz_targets), "rc": int(x.rz_carries), "gc": int(x.gl_carries),
             "o": round(float(x.opp_pts), 1), "p": round(float(x.fantasy_points_ppr), 1)}
            for x in d.itertuples()]
    small = bool(r["games"] < small_cut)
    flag = ""
    if not small and rz_ok:
        flag = "SELL HIGH" if r["luck"] >= LUCK_GAP else ("BUY LOW" if r["luck"] <= -LUCK_GAP else "")
    players.append({
        "id": r["player_id"], "name": r["player_name"], "team": r["team"],
        "pos": r["position"], "role": r["role"], "team_rank": int(r["team_rank"]),
        "games": int(r["games"]), "score": round(float(r["score"]), 1),
        "pos_rank": int(r["pos_rank"]), "pos_n": int(pos_n.get(r["position"], 0)),
        "small": small, "flag": flag, "luck": round(float(r["luck"]), 1),
        "exp": round(float(r["exp_ppr"]), 1),
        "l3_opp": round(float(d["opp_pts"].tail(3).mean()), 1),
        "m": {k: [round(float(r[k]), 1), int(round(r["p_" + k])), int(r["t_" + k])] for k in METRICS},
        "logs": logs,
    })

print(f"4. Building app with {len(players)} players through Week {max_week}...")
payload = json.dumps(players)
meta = json.dumps({"week": max_week, "rz": rz_ok})

html_code = (
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Fantasy Usage & Role Analyzer (2026)</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#c9d1d9;--hi:#f0f6fc;--ac:#58a6ff;--gr:#3fb950;--or:#f0883e;--mu:#8b949e}
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
body{background:var(--bg);color:var(--tx);padding:24px}
.wrap{max-width:1000px;margin:0 auto}
header{text-align:center;margin-bottom:22px}h1{color:var(--hi);font-size:2rem}.sub{color:var(--mu);margin-top:4px}
.tabs{display:flex;justify-content:center;gap:12px;margin-bottom:22px}
.tab{background:var(--card);border:1px solid var(--bd);color:var(--tx);padding:9px 20px;border-radius:6px;cursor:pointer;font-weight:600}
.tab.on{background:var(--ac);color:#0d1117;border-color:var(--ac)}
.sb{position:relative;margin-bottom:20px}
input{width:100%;background:var(--card);border:1px solid var(--bd);color:var(--hi);padding:14px 18px;border-radius:8px;font-size:1.05rem;outline:none}
input:focus{border-color:var(--ac)}
.dd{position:absolute;top:calc(100% + 4px);left:0;right:0;background:var(--card);border:1px solid var(--bd);border-radius:8px;max-height:250px;overflow-y:auto;z-index:100;display:none}
.dd div{padding:12px 18px;cursor:pointer;border-bottom:1px solid #21262d}.dd div:hover{background:#21262d;color:var(--ac)}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:24px;margin-bottom:22px}
.top{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--bd);padding-bottom:16px;margin-bottom:18px}
h2{color:var(--hi);font-size:1.7rem}.mu{color:var(--mu)}
.badge{display:inline-block;padding:6px 12px;border-radius:20px;font-size:.85rem;font-weight:700;margin-left:6px}
.t1{background:rgba(63,185,80,.15);color:var(--gr);border:1px solid var(--gr)}
.t2{background:rgba(88,166,255,.15);color:var(--ac);border:1px solid var(--ac)}
.t3{background:rgba(240,136,62,.15);color:var(--or);border:1px solid var(--or)}
.buy{background:rgba(63,185,80,.2);color:var(--gr);border:1px solid var(--gr)}
.sell{background:rgba(248,81,73,.2);color:#f85149;border:1px solid #f85149}
.score{display:flex;align-items:center;gap:20px;margin-bottom:18px}
.big{font-size:3rem;font-weight:800;color:var(--hi);line-height:1}
.warn{background:rgba(240,136,62,.12);border:1px solid var(--or);color:var(--or);padding:10px 14px;border-radius:8px;margin-bottom:14px;font-size:.9rem}
table{width:100%;border-collapse:collapse;margin-top:10px}
th,td{padding:10px 8px;text-align:center;border-bottom:1px solid var(--bd);font-size:.9rem}
th{color:var(--mu);text-transform:uppercase;font-size:.72rem;background:#0d1117}td.l{text-align:left}
.bar{display:inline-block;width:110px;height:8px;background:#0d1117;border-radius:4px;vertical-align:middle;overflow:hidden}
.fill{height:100%}.g{background:var(--gr)}.b{background:var(--ac)}.o{background:var(--or)}
.pc{margin-left:8px;font-size:.8rem;color:var(--mu)}
.win{color:var(--gr);font-weight:700}.lose{color:var(--mu)}
.note{color:var(--mu);font-size:.8rem;margin-top:14px}
.scroll{overflow-x:auto}
</style></head><body><div class="wrap">
<header><h1>Fantasy Usage & Role Analyzer</h1>
<p class="sub" id="sub"></p></header>
<div class="tabs"><button class="tab on" id="t1" onclick="mode('s')">Player Profile</button>
<button class="tab" id="t2" onclick="mode('c')">Head-to-Head</button></div>
<div id="S"><div class="sb"><input id="q" placeholder="Search a player..." oninput="find(this.value,'s')"/><div class="dd" id="dds"></div></div><div id="P"></div></div>
<div id="C" style="display:none"><div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div class="sb"><input placeholder="Player A..." oninput="find(this.value,'a')"/><div class="dd" id="dda"></div></div>
<div class="sb"><input placeholder="Player B..." oninput="find(this.value,'b')"/><div class="dd" id="ddb"></div></div></div><div id="K"></div></div>
</div><script>
const DB="""
    + payload
    + """;
const META="""
    + meta
    + """;
const KEYS=["targets_pg","carries_pg","opp_pg","snap_pct","share","air_share","rz_pg","gl_pg","ppr_ppg"];
const LAB={targets_pg:"Targets / G",carries_pg:"Carries / G",opp_pg:"Opportunity Pts / G",snap_pct:"Snap Share %",share:"Team Share %",air_share:"Air Yard Share %",rz_pg:"Red Zone Touches / G",gl_pg:"Goal-Line Carries / G",ppr_ppg:"PPR Pts / G"};
const lab=(k,p)=>k==="share"?(p.pos==="RB"?"Touch Share % (of team)":"Target Share % (of team)"):LAB[k];
const keysFor=p=>KEYS.filter(k=>!((k==="carries_pg"||k==="gl_pg")&&p.pos!=="RB")&&!(k==="air_share"&&p.pos==="RB")&&!((k==="rz_pg"||k==="gl_pg")&&!META.rz));
const tier=s=>s>=85?"Elite Usage":s>=70?"Strong Starter":s>=50?"Startable":s>=30?"Flex / Depth":"Fringe";
const bc=v=>v>=75?"g":v>=40?"b":"o";
let A=null,B=null;
document.getElementById("sub").innerHTML="2026 NFL Season &bull; Through Week "+META.week+" &bull; Auto-Updated Weekly";
window.onload=()=>{if(DB.length)profile(DB[0])};
function mode(m){S.style.display=m==="s"?"block":"none";C.style.display=m==="c"?"block":"none";t1.classList.toggle("on",m==="s");t2.classList.toggle("on",m==="c")}
function find(v,t){
  const d=document.getElementById("dd"+t),q=v.toLowerCase().trim();
  if(!q){d.style.display="none";return}
  const m=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,8);
  d.innerHTML="";d.style.display=m.length?"block":"none";
  m.forEach(p=>{const e=document.createElement("div");e.textContent=p.name+" ("+p.team+" - "+p.role+")";
    e.onclick=()=>{d.style.display="none";if(t==="s")profile(p);else{if(t==="a")A=p;else B=p;compare()}};d.appendChild(e)});
}
function profile(p){
  const rows=keysFor(p).map(k=>{const[v,pc,tr]=p.m[k];
    return `<tr><td class="l">${lab(k,p)}</td><td><b>${v}</b></td><td>#${tr} ${p.pos} on ${p.team}</td><td><div class="bar"><div class="fill ${bc(pc)}" style="width:${pc}%"></div></div><span class="pc">${pc}th</span></td></tr>`}).join("");
  const logs=p.logs.map(l=>`<tr><td>Wk ${l.w}</td><td>${l.t}</td><td>${l.c}</td><td>${l.sp}%</td>${META.rz?`<td>${l.rt}</td><td>${l.rc}</td><td>${l.gc}</td>`:""}<td>${l.o}</td><td class="win">${l.p}</td></tr>`).join("");
  const tc=p.team_rank===1?"t1":p.team_rank===2?"t2":"t3";
  const flag=p.flag?`<span class="badge ${p.flag==="BUY LOW"?"buy":"sell"}">${p.flag}</span>`:"";
  const small=p.small?`<div class="warn">Small sample: only ${p.games} game(s) played, so treat these numbers carefully.</div>`:"";
  const luck=META.rz?`<p class="mu" style="margin-bottom:14px">Scoring <b>${p.m.ppr_ppg[0]}</b> PPR/G vs <b>${p.exp}</b> expected from usage (${p.luck>0?"+":""}${p.luck}).</p>`:"";
  const trend=p.games>=4?`<p class="mu" style="margin-bottom:14px">Last 3 weeks: ${p.l3_opp} opp pts/G ${p.l3_opp>=p.m.opp_pg[0]?"&#9650;":"&#9660;"} (season ${p.m.opp_pg[0]})</p>`:"";
  P.innerHTML=`<div class="card"><div class="top"><div><h2>${p.name}</h2><div class="mu">${p.team} &bull; ${p.pos} &bull; ${p.games} games</div></div>
  <div><span class="badge ${tc}">${p.role} ON TEAM</span>${flag}</div></div>
  <div class="score"><div class="big">${p.score}</div><div><b style="color:var(--hi)">Usage Score</b><div class="mu">${tier(p.score)} &bull; ${p.pos_rank?p.pos+p.pos_rank+" of "+p.pos_n:"Unranked (low snaps)"}</div></div></div>
  ${small}${luck}${trend}
  <table><thead><tr><th style="text-align:left">Metric</th><th>Value</th><th>Team Rank</th><th>Position Percentile</th></tr></thead><tbody>${rows}</tbody></table>
  <h3 style="color:var(--hi);margin-top:24px">Game Logs</h3><div class="scroll"><table><thead><tr><th>Game</th><th>Tgt</th><th>Car</th><th>Snap%</th>${META.rz?"<th>RZ Tgt</th><th>RZ Car</th><th>GL Car</th>":""}<th>Opp Pts</th><th>PPR</th></tr></thead><tbody>${logs}</tbody></table></div>
  <p class="note">Usage Score = percentile blend vs same-position players: 35% opportunity points, 20% snap share, 20% team share, 15% red zone usage, 10% PPR points. Percentiles only compare against players averaging 30%+ of snaps.</p></div>`;
}
function compare(){
  if(!A||!B)return;
  const ks=KEYS.filter(k=>keysFor(A).includes(k)&&keysFor(B).includes(k));
  const rows=[["Usage Score",A.score,B.score]].concat(ks.map(k=>[lab(k,A),A.m[k][0],B.m[k][0]]));
  let wa=0,wb=0;
  const html=rows.map(([l,a,b])=>{if(a>b)wa++;else if(b>a)wb++;
    return `<tr><td class="${a>b?"win":b>a?"lose":""}">${a}</td><td class="l" style="text-align:center">${l}</td><td class="${b>a?"win":a>b?"lose":""}">${b}</td></tr>`}).join("");
  const w=wa>wb?A.name:wb>wa?B.name:"Tie";
  const fl=p=>p.flag?" ("+p.flag+")":"";
  K.innerHTML=`<div class="card"><h2 style="text-align:center;margin-bottom:6px">Start / Sit Matrix</h2>
  <p style="text-align:center;margin-bottom:16px" class="mu">${w==="Tie"?"Dead even":"<b class='win'>"+w+"</b> wins "+Math.max(wa,wb)+" of "+rows.length+" categories"}</p>
  <table><thead><tr><th>${A.name}<br><span class="mu">${A.team} ${A.role}${fl(A)}</span></th><th></th><th>${B.name}<br><span class="mu">${B.team} ${B.role}${fl(B)}</span></th></tr></thead><tbody>${html}</tbody></table></div>`;
}
</script></body></html>"""
)

os.makedirs("public", exist_ok=True)
with open("public/index.html", "w", encoding="utf-8") as f:
    f.write(html_code)
print("5. Done: public/index.html written.")
