import json
import os
import pandas as pd

SEASON = 2026
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

print(f"1. Fetching {SEASON} NFL data from nflverse...")

stats_url = f"{BASE}/stats_player/stats_player_week_{SEASON}.csv"
snaps_url = f"{BASE}/snap_counts/snap_counts_{SEASON}.csv"

# No silent fallbacks: if this fails, the build fails loudly
stats_df = pd.read_csv(stats_url, low_memory=False)
snaps_df = pd.read_csv(snaps_url, low_memory=False)

# Regular season only
if "season_type" in stats_df.columns:
    stats_df = stats_df[stats_df["season_type"] == "REG"]
if "game_type" in snaps_df.columns:
    snaps_df = snaps_df[snaps_df["game_type"] == "REG"]

# Clean column headers
name_col = (
    "player_display_name"
    if "player_display_name" in stats_df.columns
    else "player_name"
)
team_col = "team" if "team" in stats_df.columns else "recent_team"

# Filter to skill positions
stats_df = stats_df[stats_df["position"].isin(["WR", "RB", "TE"])].copy()

cols = [
    "player_id",
    name_col,
    team_col,
    "position",
    "week",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "fantasy_points_ppr",
]
stats_clean = stats_df[[c for c in cols if c in stats_df.columns]].copy()
stats_clean.rename(
    columns={name_col: "player_name", team_col: "team"}, inplace=True
)

# Clean snap stats
snaps_clean = snaps_df[
    ["player", "team", "week", "offense_snaps", "offense_pct"]
].copy()
snaps_clean = snaps_clean.drop_duplicates(subset=["player", "team", "week"])

merged = pd.merge(
    stats_clean,
    snaps_clean,
    left_on=["player_name", "team", "week"],
    right_on=["player", "team", "week"],
    how="left",
)

# Fill empty values
merged["offense_snaps"] = merged["offense_snaps"].fillna(0)
merged["offense_pct"] = merged["offense_pct"].fillna(0.0)
merged["targets"] = merged["targets"].fillna(0)
merged["carries"] = merged["carries"].fillna(0)
merged["fantasy_points_ppr"] = merged["fantasy_points_ppr"].fillna(0.0)

print("2. Calculating season metrics & depth chart roles...")

season_summary = (
    merged.groupby(["player_id", "player_name", "team", "position"])
    .agg(
        games=("week", "count"),
        targets_pg=("targets", "mean"),
        carries_pg=("carries", "mean"),
        snap_pct=("offense_pct", "mean"),
        ppr_ppg=("fantasy_points_ppr", "mean"),
    )
    .reset_index()
)

# Assign depth chart role
season_summary["rank"] = (
    season_summary.groupby(["team", "position"])["targets_pg"]
    .rank(ascending=False, method="min")
    .astype(int)
)
rb_mask = season_summary["position"] == "RB"
season_summary.loc[rb_mask, "rank"] = (
    season_summary[rb_mask]
    .groupby("team")["carries_pg"]
    .rank(ascending=False, method="min")
    .astype(int)
)

season_summary["role"] = (
    season_summary["position"] + season_summary["rank"].astype(str)
)

# Show top scorers first
season_summary = season_summary.sort_values("ppr_ppg", ascending=False)

players_list = []
for _, row in season_summary.iterrows():
    p_id = row["player_id"]
    p_logs = merged[merged["player_id"] == p_id].sort_values("week")

    history = []
    for _, g in p_logs.iterrows():
        history.append({
            "week": int(g["week"]),
            "targets": int(g["targets"]),
            "carries": int(g["carries"]),
            "snaps": int(g["offense_snaps"]),
            "snap_pct": round(float(g["offense_pct"]) * 100, 1),
            "ppr": round(float(g["fantasy_points_ppr"]), 1),
        })

    players_list.append({
        "id": row["player_id"],
        "name": row["player_name"],
        "team": row["team"],
        "pos": row["position"],
        "role": row["role"],
        "games": int(row["games"]),
        "targets_pg": round(float(row["targets_pg"]), 1),
        "carries_pg": round(float(row["carries_pg"]), 1),
        "snap_pct": round(float(row["snap_pct"]) * 100, 1),
        "ppr_ppg": round(float(row["ppr_ppg"]), 1),
        "game_logs": history,
    })

max_week = int(merged["week"].max())
print(f"3. Building app with {len(players_list)} players through Week {max_week}...")

json_payload = json.dumps(players_list)

html_code = (
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fantasy Volume & Role Analyzer (2026)</title>
  <style>
    :root {
      --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
      --text: #c9d1d9; --text-bright: #f0f6fc; --accent: #58a6ff;
      --green: #3fb950; --orange: #f0883e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .container { max-width: 1050px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 24px; }
    h1 { color: var(--text-bright); font-size: 2.1rem; margin-bottom: 6px; }
    p.sub { color: #8b949e; font-size: 0.95rem; }
    .tabs { display: flex; justify-content: center; gap: 12px; margin-bottom: 24px; }
    .tab-btn {
      background: var(--card-bg); border: 1px solid var(--border); color: var(--text);
      padding: 9px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.95rem;
    }
    .tab-btn.active { background: var(--accent); color: #0d1117; border-color: var(--accent); }
    .search-box { position: relative; margin-bottom: 24px; }
    input[type="text"] {
      width: 100%; background: var(--card-bg); border: 1px solid var(--border);
      color: var(--text-bright); padding: 14px 18px; border-radius: 8px; font-size: 1.05rem; outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    .autocomplete-list {
      position: absolute; top: calc(100% + 4px); left: 0; right: 0; background: var(--card-bg);
      border: 1px solid var(--border); border-radius: 8px; max-height: 250px; overflow-y: auto;
      z-index: 100; display: none; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .autocomplete-item { padding: 12px 18px; cursor: pointer; border-bottom: 1px solid #21262d; }
    .autocomplete-item:hover { background: #21262d; color: var(--accent); }
    .profile-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .profile-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
    .player-title h2 { color: var(--text-bright); font-size: 1.8rem; }
    .player-title .subhead { color: #8b949e; margin-top: 4px; }
    .badge { display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; }
    .badge-tier1 { background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid var(--green); }
    .badge-tier2 { background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid var(--accent); }
    .badge-depth { background: rgba(240, 136, 62, 0.15); color: var(--orange); border: 1px solid var(--orange); }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .stat-box { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 18px; text-align: center; }
    .stat-box .label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; margin-bottom: 6px; }
    .stat-box .val { font-size: 1.8rem; font-weight: 700; color: var(--text-bright); }
    .stat-box .sub-val { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { padding: 12px 10px; text-align: center; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
    th { color: #8b949e; text-transform: uppercase; font-size: 0.75rem; background: #0d1117; }
    tr:hover { background: #1f242c; }
    .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .winner { color: var(--green); font-weight: bold; }
    .loser { color: #8b949e; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Fantasy Usage & Role Analyzer</h1>
    <p class="sub">2026 NFL Season &bull; Through Week """
    + str(max_week)
    + """ &bull; Auto-Updated Weekly</p>
  </header>
  <div class="tabs">
    <button class="tab-btn active" id="singleTabBtn" onclick="switchMode('single')">Single Player Profile</button>
    <button class="tab-btn" id="compareTabBtn" onclick="switchMode('compare')">Head-to-Head Start/Sit</button>
  </div>
  <div id="singleSection">
    <div class="search-box">
      <input type="text" id="playerInput" placeholder="Search 2026 player..." oninput="handleSearch(this.value, 'single')" />
      <div class="autocomplete-list" id="singleDropdown"></div>
    </div>
    <div id="profileContainer"></div>
  </div>
  <div id="compareSection" style="display: none;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
      <div class="search-box">
        <input type="text" placeholder="Select Player A..." oninput="handleSearch(this.value, 'compA')" />
        <div class="autocomplete-list" id="compADropdown"></div>
      </div>
      <div class="search-box">
        <input type="text" placeholder="Select Player B..." oninput="handleSearch(this.value, 'compB')" />
        <div class="autocomplete-list" id="compBDropdown"></div>
      </div>
    </div>
    <div id="compareContainer"></div>
  </div>
</div>
<script>
const database = """
    + json_payload
    + """;
let compareA = null, compareB = null;
window.onload = () => {
  if (database.length > 0) renderProfile(database[0]);
};
function switchMode(m) {
  document.getElementById('singleSection').style.display = m === 'single' ? 'block' : 'none';
  document.getElementById('compareSection').style.display = m === 'compare' ? 'block' : 'none';
  document.getElementById('singleTabBtn').classList.toggle('active', m === 'single');
  document.getElementById('compareTabBtn').classList.toggle('active', m === 'compare');
}
function handleSearch(val, target) {
  const query = val.toLowerCase().trim();
  const dropdown = document.getElementById(target === 'single' ? 'singleDropdown' : (target === 'compA' ? 'compADropdown' : 'compBDropdown'));
  if (!query) { dropdown.style.display = 'none'; return; }
  const matches = database.filter(p => p.name.toLowerCase().includes(query)).slice(0, 8);
  dropdown.innerHTML = '';
  if (matches.length > 0) {
    dropdown.style.display = 'block';
    matches.forEach(p => {
      const item = document.createElement('div');
      item.className = 'autocomplete-item';
      item.textContent = `${p.name} (${p.team} - ${p.role})`;
      item.onclick = () => {
        dropdown.style.display = 'none';
        if (target === 'single') renderProfile(p);
        else if (target === 'compA') { compareA = p; renderComparison(); }
        else if (target === 'compB') { compareB = p; renderComparison(); }
      };
      dropdown.appendChild(item);
    });
  } else { dropdown.style.display = 'none'; }
}
function getBadgeClass(role) {
  if (role.endsWith('1')) return 'badge-tier1';
  if (role.endsWith('2')) return 'badge-tier2';
  return 'badge-depth';
}
function renderProfile(p) {
  const logsHtml = p.game_logs.map(l => `
    <tr><td>Week ${l.week}</td><td><strong>${l.targets}</strong></td><td>${l.carries}</td><td>${l.snaps}</td><td>${l.snap_pct}%</td><td style="color:var(--green); font-weight:bold;">${l.ppr}</td></tr>
  `).join('');
  document.getElementById('profileContainer').innerHTML = `
    <div class="profile-card">
      <div class="profile-header">
        <div class="player-title"><h2>${p.name}</h2><div class="subhead">${p.team} &bull; ${p.pos} &bull; ${p.games} Games Analyzed</div></div>
        <span class="badge ${getBadgeClass(p.role)}">${p.role} ON DEPTH CHART</span>
      </div>
      <div class="stat-grid">
        <div class="stat-box"><div class="label">Targets / Game</div><div class="val">${p.targets_pg}</div><div class="sub-val">${p.pos === 'RB' ? p.carries_pg + ' Carries/G' : 'Primary Target Floor'}</div></div>
        <div class="stat-box"><div class="label">Average Snap Share</div><div class="val">${p.snap_pct}%</div><div class="sub-val">Field Involvement Rate</div></div>
        <div class="stat-box"><div class="label">PPR Points / Game</div><div class="val" style="color:var(--green);">${p.ppr_ppg}</div><div class="sub-val">Per Game Fantasy Average</div></div>
      </div>
      <h3 style="color:var(--text-bright); margin-bottom: 12px;">2026 Game Logs</h3>
      <table><thead><tr><th>Game</th><th>Targets</th><th>Carries</th><th>Snaps</th><th>Snap %</th><th>PPR Pts</th></tr></thead><tbody>${logsHtml}</tbody></table>
    </div>`;
}
function renderComparison() {
  if (!compareA || !compareB) return;
  const cmp = (a, b, f='') => a > b ? [`<span class="winner">${a}${f}</span>`, `<span class="loser">${b}${f}</span>`] : (b > a ? [`<span class="loser">${a}${f}</span>`, `<span class="winner">${b}${f}</span>`] : [`${a}${f}`, `${b}${f}`]);
  const [tA, tB] = cmp(compareA.targets_pg, compareB.targets_pg);
  const [cA, cB] = cmp(compareA.carries_pg, compareB.carries_pg);
  const [sA, sB] = cmp(compareA.snap_pct, compareB.snap_pct, '%');
  const [pA, pB] = cmp(compareA.ppr_ppg, compareB.ppr_ppg);
  document.getElementById('compareContainer').innerHTML = `
    <div class="profile-card">
      <h2 style="text-align:center; color:var(--text-bright); margin-bottom:20px;">Start / Sit Decision Matrix</h2>
      <div class="compare-grid">
        <div style="text-align:center; border-right:1px solid var(--border); padding-right:16px;">
          <h3>${compareA.name}</h3><p style="color:#8b949e; margin-bottom:14px;">${compareA.team} - ${compareA.role}</p>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Targets / Game</div><div class="val">${tA}</div></div>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Carries / Game</div><div class="val">${cA}</div></div>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Snap Share</div><div class="val">${sA}</div></div>
          <div class="stat-box"><div class="label">PPR PPG</div><div class="val">${pA}</div></div>
        </div>
        <div style="text-align:center; padding-left:16px;">
          <h3>${compareB.name}</h3><p style="color:#8b949e; margin-bottom:14px;">${compareB.team} - ${compareB.role}</p>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Targets / Game</div><div class="val">${tB}</div></div>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Carries / Game</div><div class="val">${cB}</div></div>
          <div class="stat-box" style="margin-bottom:10px;"><div class="label">Snap Share</div><div class="val">${sB}</div></div>
          <div class="stat-box"><div class="label">PPR PPG</div><div class="val">${pB}</div></div>
        </div>
      </div>
    </div>`;
}
</script>
</body>
</html>"""
)

os.makedirs("public", exist_ok=True)
with open("public/index.html", "w") as f:
    f.write(html_code)

print("4. Successfully generated public/index.html!")
