import json
import math
import os
import warnings
from urllib.error import HTTPError

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# FANTASY MANIA — 2026
# Autonomous build pipeline for GitHub Pages
# ============================================================

SEASON = 2026
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Core tuning
QUALIFY_SNAP = 0.30
PARTIAL_RATIO = 0.55       # <55% of normal snap share = likely shortened game
PARTIAL_MIN_NORMAL = 0.45  # only detect shortened games for players with regular roles
MIN_PARTIAL_WEIGHT = 0.20
MATCHUP_CAP = 6.0          # Max Start Rating swing from matchup
RECENT_CAP = 2.5           # Max swing from recent volume trend
CONFIDENCE_SHRINK = 0.12   # Sample size shrink factor

# Position-specific Mania bucket weights
BUCKET_W = {
    "RB": {"production": .30, "opportunity": .30, "role": .20, "high_value": .125, "efficiency": .075},
    "WR": {"production": .30, "opportunity": .30, "role": .225, "high_value": .10, "efficiency": .075},
    "TE": {"production": .30, "opportunity": .30, "role": .225, "high_value": .10, "efficiency": .075},
}

def safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else 0.0

def weighted_sum(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok] * w[ok])) if ok.any() else 0.0

def percentile_series(df, col, qualify_col="qualified"):
    out = pd.Series(0.0, index=df.index)
    for pos, grp in df.groupby("position"):
        ref = np.sort(grp.loc[grp[qualify_col], col].replace([np.inf, -np.inf], np.nan).dropna().values)
        if len(ref) == 0:
            continue
        vals = grp[col].fillna(0).values
        pct = np.searchsorted(ref, vals, side="right") / len(ref) * 100
        out.loc[grp.index] = np.clip(pct, 0, 100)
    return out

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

print("1. Loading current 2026 player stats + snaps...")
stats_url = f"{BASE}/stats_player/stats_player_week_{SEASON}.csv"
snaps_url = f"{BASE}/snap_counts/snap_counts_{SEASON}.csv"

stats = pd.read_csv(stats_url, low_memory=False)
snaps = pd.read_csv(snaps_url, low_memory=False)

if "season_type" in stats.columns:
    stats = stats[stats["season_type"].astype(str).eq("REG")].copy()
if "game_type" in snaps.columns:
    snaps = snaps[snaps["game_type"].astype(str).eq("REG")].copy()

name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
team_col = "team" if "team" in stats.columns else "recent_team"

needed = [
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "carries", "rushing_yards", "rushing_tds",
    "fantasy_points_ppr"
]
for c in needed:
    if c not in stats.columns:
        stats[c] = 0
    stats[c] = safe_num(stats[c])

keep = ["player_id", name_col, team_col, "position", "week"] + needed
s = stats[stats["position"].isin(["RB", "WR", "TE"])][keep].copy()
s = s.rename(columns={name_col: "player_name", team_col: "team"})
s["week"] = safe_num(s["week"]).astype(int)

# Offense snap percentages
for c in ["offense_snaps", "offense_pct"]:
    if c not in snaps.columns:
        snaps[c] = 0
    snaps[c] = safe_num(snaps[c])

snap_name = "player" if "player" in snaps.columns else ("player_name" if "player_name" in snaps.columns else None)
if snap_name:
    sn = snaps[[snap_name, "team", "week", "offense_snaps", "offense_pct"]].copy()
    sn = sn.rename(columns={snap_name: "snap_name"})
    sn["week"] = safe_num(sn["week"]).astype(int)
    sn = sn.drop_duplicates(["snap_name", "team", "week"])
    merged = s.merge(
        sn, left_on=["player_name", "team", "week"],
        right_on=["snap_name", "team", "week"], how="left"
    )
else:
    merged = s.copy()
    merged["offense_snaps"] = 0
    merged["offense_pct"] = 0

merged["offense_snaps"] = safe_num(merged["offense_snaps"])
merged["offense_pct"] = safe_num(merged["offense_pct"])

max_week = int(merged["week"].max())
print(f"   Stats loaded through Week {max_week}.")

print("   Loading player headshots...")
headshot_map = {}
try:
    player_info = pd.read_csv(f"{BASE}/players/players.csv", low_memory=False)
    id_col = "gsis_id" if "gsis_id" in player_info.columns else ("player_id" if "player_id" in player_info.columns else None)
    hs_col = "headshot" if "headshot" in player_info.columns else ("headshot_url" if "headshot_url" in player_info.columns else None)
    if id_col and hs_col:
        headshot_map = (
            player_info[[id_col, hs_col]].dropna()
            .drop_duplicates(id_col).set_index(id_col)[hs_col].astype(str).to_dict()
        )
except Exception as e:
    print("   Headshots fallback will be initials:", e)

# ============================================================
# RED ZONE / GOAL LINE DATA
# ============================================================
print("2. Loading red-zone usage from PBP...")
rz_ok = True
try:
    want = [
        "season_type", "week", "yardline_100", "pass_attempt", "rush_attempt",
        "receiver_player_id", "rusher_player_id", "air_yards"
    ]
    pbp = None
    pbp_urls = [
        f"{BASE}/pbp/play_by_play_{SEASON}.csv.gz",
        f"{BASE}/pbp/play_by_play_{SEASON}.csv",
    ]
    last_err = None
    for u in pbp_urls:
        try:
            pbp = pd.read_csv(u, usecols=lambda c: c in want, low_memory=False)
            break
        except Exception as e:
            last_err = e
    if pbp is None:
        raise last_err or RuntimeError("PBP unavailable")

    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"].astype(str).eq("REG")]
    pbp["yardline_100"] = safe_num(pbp["yardline_100"])
    pbp = pbp[(pbp["week"] <= max_week) & (pbp["yardline_100"] <= 20) & (pbp["yardline_100"] > 0)].copy()

    rec = pbp[(safe_num(pbp["pass_attempt"]) == 1) & pbp["receiver_player_id"].notna()].copy()
    rec["player_id"] = rec["receiver_player_id"]
    rec["rz_targets"] = 1
    rec["endzone_targets"] = ((safe_num(rec["air_yards"]) if "air_yards" in rec.columns else pd.Series(0, index=rec.index)) >= rec["yardline_100"]).astype(int)

    rus = pbp[(safe_num(pbp["rush_attempt"]) == 1) & pbp["rusher_player_id"].notna()].copy()
    rus["player_id"] = rus["rusher_player_id"]
    rus["rz_carries"] = 1
    rus["gl_carries"] = (rus["yardline_100"] <= 5).astype(int)

    rz = pd.concat([
        rec[["player_id", "week", "rz_targets", "endzone_targets"]],
        rus[["player_id", "week", "rz_carries", "gl_carries"]],
    ]).fillna(0)
    rz = rz.groupby(["player_id", "week"], as_index=False).sum()
    merged = merged.merge(rz, on=["player_id", "week"], how="left")
except Exception as e:
    print("   WARNING: PBP red-zone data unavailable; fallback enabled:", e)
    rz_ok = False

for c in ["rz_targets", "endzone_targets", "rz_carries", "gl_carries"]:
    if c not in merged.columns:
        merged[c] = 0
    merged[c] = safe_num(merged[c])

# ============================================================
# TEAM-WEEK TOTALS & ACTIVE SHARES
# ============================================================
print("3. Building active-game team totals & participation weights...")

team_week = (
    stats.groupby([team_col, "week"], as_index=False)[
        ["targets", "carries", "receiving_air_yards"]
    ].sum()
    .rename(columns={
        team_col: "team",
        "targets": "team_targets",
        "carries": "team_carries",
        "receiving_air_yards": "team_air",
    })
)
team_week["week"] = safe_num(team_week["week"]).astype(int)
merged = merged.merge(team_week, on=["team", "week"], how="left")

for c in ["team_targets", "team_carries", "team_air"]:
    merged[c] = safe_num(merged[c])

merged["target_share_game"] = np.where(
    merged["team_targets"] > 0, merged["targets"] / merged["team_targets"] * 100, 0
)
merged["rush_share_game"] = np.where(
    merged["team_carries"] > 0, merged["carries"] / merged["team_carries"] * 100, 0
)
merged["air_share_game"] = np.where(
    merged["team_air"] > 0, merged["receiving_air_yards"] / merged["team_air"] * 100, 0
)
merged["touch_share_game"] = np.where(
    (merged["team_targets"] + merged["team_carries"]) > 0,
    (merged["targets"] + merged["carries"]) /
    (merged["team_targets"] + merged["team_carries"]) * 100, 0
)

# Automated partial-game detection without manual injury lists
merged["normal_snap"] = 0.0
merged["participation_ratio"] = 1.0
merged["game_weight"] = 1.0
merged["partial"] = False

for pid, idx in merged.groupby("player_id").groups.items():
    ix = list(idx)
    snaps_pct = merged.loc[ix, "offense_pct"].values.astype(float)
    positive = snaps_pct[snaps_pct > 0]
    if len(positive):
        cutoff = np.median(positive)
        upper = positive[positive >= cutoff]
        normal = float(np.median(upper)) if len(upper) else float(np.max(positive))
    else:
        normal = 0.0

    merged.loc[ix, "normal_snap"] = normal
    if normal > 0:
        ratio = np.clip(snaps_pct / normal, 0, 1.25)
        merged.loc[ix, "participation_ratio"] = ratio
        is_partial = (normal >= PARTIAL_MIN_NORMAL) & (ratio < PARTIAL_RATIO) & (snaps_pct > 0)
        merged.loc[ix, "partial"] = is_partial
        weights = np.where(is_partial, np.clip(ratio, MIN_PARTIAL_WEIGHT, 0.60), 1.0)
        merged.loc[ix, "game_weight"] = weights

# ============================================================
# PLAYER PROFILE SUMMARY
# ============================================================
def summarize_player(d):
    d = d.sort_values("week").copy()
    w = d["game_weight"].values
    games = len(d)
    effective_games = float(np.sum(w))

    def wm(col):
        return weighted_mean(d[col].values, w)

    targets_pg = wm("targets")
    rec_pg = wm("receptions")
    rec_yards_pg = wm("receiving_yards")
    rec_td_pg = wm("receiving_tds")
    carries_pg = wm("carries")
    rush_yards_pg = wm("rushing_yards")
    rush_td_pg = wm("rushing_tds")
    ppr_pg = wm("fantasy_points_ppr")
    snap_pct = wm("offense_pct") * 100
    tgt_share = wm("target_share_game")
    rush_share = wm("rush_share_game")
    air_share = wm("air_share_game")
    touch_share = wm("touch_share_game")
    rz_t_pg = wm("rz_targets")
    ez_t_pg = wm("endzone_targets")
    rz_c_pg = wm("rz_carries")
    gl_c_pg = wm("gl_carries")

    scrim_pg = rec_yards_pg + rush_yards_pg
    total_td_pg = rec_td_pg + rush_td_pg

    ypt = weighted_sum(d["receiving_yards"], w) / max(weighted_sum(d["targets"], w), 1)
    catch_rate = weighted_sum(d["receptions"], w) / max(weighted_sum(d["targets"], w), 1) * 100
    ypc = weighted_sum(d["rushing_yards"], w) / max(weighted_sum(d["carries"], w), 1)
    opp = weighted_sum(d["carries"], w) + 2.0 * weighted_sum(d["targets"], w)
    fp_per_opp = weighted_sum(d["fantasy_points_ppr"], w) / max(opp, 1)

    recent = d.tail(min(2, games))
    recent_opp = float((recent["targets"] * 2.0 + recent["carries"]).mean()) if games else 0
    season_opp = targets_pg * 2.0 + carries_pg
    trend_pct = ((recent_opp / season_opp) - 1) * 100 if season_opp > 0 else 0

    return {
        "games": games,
        "effective_games": effective_games,
        "partial_games": int(d["partial"].sum()),
        "targets_pg": targets_pg,
        "rec_pg": rec_pg,
        "rec_yards_pg": rec_yards_pg,
        "rec_td_pg": rec_td_pg,
        "carries_pg": carries_pg,
        "rush_yards_pg": rush_yards_pg,
        "rush_td_pg": rush_td_pg,
        "scrim_pg": scrim_pg,
        "td_pg": total_td_pg,
        "ppr_pg": ppr_pg,
        "snap_pct": snap_pct,
        "target_share": tgt_share,
        "rush_share": rush_share,
        "air_share": air_share,
        "touch_share": touch_share,
        "rz_targets_pg": rz_t_pg,
        "endzone_targets_pg": ez_t_pg,
        "rz_carries_pg": rz_c_pg,
        "gl_carries_pg": gl_c_pg,
        "yards_per_target": ypt,
        "catch_rate": catch_rate,
        "yards_per_carry": ypc,
        "fp_per_weighted_opp": fp_per_opp,
        "trend_pct": trend_pct,
    }

rows = []
for keys, d in merged.groupby(["player_id", "player_name", "team", "position"]):
    pid, name, team, pos = keys
    x = summarize_player(d)
    x.update({"player_id": pid, "player_name": name, "team": team, "position": pos})
    rows.append(x)

g = pd.DataFrame(rows)
g["qualified"] = (g["snap_pct"] >= QUALIFY_SNAP * 100) | (g["ppr_pg"] >= 6)

# ============================================================
# MANIA RATING CALCULATION
# ============================================================
print("4. Calculating Mania Rating...")

PCT_COLS = [
    "ppr_pg", "scrim_pg", "rec_yards_pg", "td_pg",
    "targets_pg", "rec_pg", "carries_pg",
    "target_share", "air_share", "touch_share", "snap_pct",
    "rz_targets_pg", "endzone_targets_pg", "rz_carries_pg", "gl_carries_pg",
    "yards_per_target", "catch_rate", "yards_per_carry", "fp_per_weighted_opp"
]
for c in PCT_COLS:
    g["pct_" + c] = percentile_series(g, c)

bucket_values = []
for _, r in g.iterrows():
    pos = r["position"]

    if pos == "RB":
        production = (
            .55 * r["pct_ppr_pg"] +
            .30 * r["pct_scrim_pg"] +
            .15 * r["pct_td_pg"]
        )
        opportunity = (
            .45 * r["pct_carries_pg"] +
            .35 * r["pct_targets_pg"] +
            .20 * r["pct_rec_pg"]
        )
        role = (
            .60 * r["pct_touch_share"] +
            .40 * r["pct_snap_pct"]
        )
        high_value = (
            .40 * r["pct_rz_carries_pg"] +
            .35 * r["pct_gl_carries_pg"] +
            .25 * r["pct_rz_targets_pg"]
        )
        efficiency = (
            .55 * r["pct_fp_per_weighted_opp"] +
            .25 * r["pct_yards_per_carry"] +
            .20 * r["pct_catch_rate"]
        )
    else:
        production = (
            .55 * r["pct_ppr_pg"] +
            .30 * r["pct_rec_yards_pg"] +
            .15 * r["pct_td_pg"]
        )
        opportunity = (
            .50 * r["pct_targets_pg"] +
            .30 * r["pct_rec_pg"] +
            .20 * r["pct_target_share"]
        )
        role = (
            .45 * r["pct_target_share"] +
            .30 * r["pct_air_share"] +
            .25 * r["pct_snap_pct"]
        )
        high_value = (
            .60 * r["pct_rz_targets_pg"] +
            .40 * r["pct_endzone_targets_pg"]
        )
        efficiency = (
            .45 * r["pct_yards_per_target"] +
            .30 * r["pct_catch_rate"] +
            .25 * r["pct_fp_per_weighted_opp"]
        )

    W = BUCKET_W[pos]
    raw = (
        W["production"] * production +
        W["opportunity"] * opportunity +
        W["role"] * role +
        W["high_value"] * high_value +
        W["efficiency"] * efficiency
    )

    bucket_values.append((production, opportunity, role, high_value, efficiency, raw))

g[["production", "opportunity", "role_score", "high_value", "efficiency", "raw_profile"]] = pd.DataFrame(
    bucket_values, index=g.index
)

# Standardized Madden-like 0-100 fantasy scale
g["mania_base"] = 45 + 0.52 * g["raw_profile"]

season_games = max_week
g["confidence"] = np.clip(g["effective_games"] / max(season_games, 1), 0, 1)
shrink = (1 - g["confidence"]) * CONFIDENCE_SHRINK
g["mania"] = g["mania_base"] * (1 - shrink) + 72 * shrink
g["mania"] = g["mania"].clip(0, 99.5)

# Overall & team ranks
g["pos_rank"] = g.groupby("position")["mania"].rank(ascending=False, method="min").astype(int)
g["overall_rank"] = g["mania"].rank(ascending=False, method="min").astype(int)
g["team_pos_rank"] = g.groupby(["team", "position"])["mania"].rank(ascending=False, method="min").astype(int)

def confidence_label(r):
    if r["games"] >= max(3, math.ceil(max_week * .75)) and r["partial_games"] == 0:
        return "High"
    if r["effective_games"] >= 2:
        return "Medium"
    return "Low"

g["confidence_label"] = g.apply(confidence_label, axis=1)

# ============================================================
# SCHEDULE & SIMILARITY MATCHUP ENGINE
# ============================================================
print("5. Loading schedule & calculating matchup projections...")
schedule_ok = True
games = None
schedule_urls = [
    f"{BASE}/schedules/games.csv",
    f"{BASE}/schedules/games.csv.gz",
]
for u in schedule_urls:
    try:
        games = pd.read_csv(u, low_memory=False)
        break
    except Exception:
        pass

if games is None:
    schedule_ok = False
else:
    if "season" in games.columns:
        games = games[safe_num(games["season"]).astype(int) == SEASON].copy()
    if "game_type" in games.columns:
        games = games[games["game_type"].astype(str).eq("REG")].copy()

next_week = max_week + 1

opp_map = {}
if schedule_ok and {"week", "home_team", "away_team"}.issubset(games.columns):
    nw = games[safe_num(games["week"]).astype(int) == next_week]
    for _, game in nw.iterrows():
        opp_map[str(game["home_team"])] = str(game["away_team"])
        opp_map[str(game["away_team"])] = str(game["home_team"])

profile_cols = {
    "RB": ["targets_pg", "carries_pg", "touch_share", "snap_pct", "ppr_pg", "rz_carries_pg", "gl_carries_pg"],
    "WR": ["targets_pg", "rec_pg", "target_share", "air_share", "snap_pct", "ppr_pg", "rz_targets_pg"],
    "TE": ["targets_pg", "rec_pg", "target_share", "air_share", "snap_pct", "ppr_pg", "rz_targets_pg"],
}

completed_opp = {}
if schedule_ok and {"week", "home_team", "away_team"}.issubset(games.columns):
    for _, game in games[safe_num(games["week"]).astype(int) <= max_week].iterrows():
        wk = int(game["week"])
        h, a = str(game["home_team"]), str(game["away_team"])
        completed_opp[(h, wk)] = a
        completed_opp[(a, wk)] = h

merged["opponent"] = [completed_opp.get((str(t), int(w)), "") for t, w in zip(merged["team"], merged["week"])]

zprofiles = {}
for pos, grp in g.groupby("position"):
    cols = profile_cols[pos]
    X = grp[cols].astype(float)
    mu = X.mean()
    sd = X.std(ddof=0).replace(0, 1)
    Z = (X - mu) / sd
    for idx in grp.index:
        zprofiles[g.loc[idx, "player_id"]] = Z.loc[idx].values.astype(float)

def similarity(a, b):
    if a not in zprofiles or b not in zprofiles:
        return 0.0
    dist = float(np.sqrt(np.mean((zprofiles[a] - zprofiles[b]) ** 2)))
    return math.exp(-0.75 * dist)

summary_by_id = g.set_index("player_id", drop=False)

def matchup_adjustment(player_row, opponent):
    if not opponent:
        return 0.0, []

    pos = player_row["position"]
    pid = player_row["player_id"]
    candidates = merged[(merged["position"] == pos) & (merged["opponent"] == opponent)].copy()
    if candidates.empty:
        return 0.0, []

    examples = []
    weighted_deltas = []
    weights = []

    for _, game in candidates.iterrows():
        other_id = game["player_id"]
        if other_id == pid or other_id not in summary_by_id.index:
            continue
        other = summary_by_id.loc[other_id]
        sim = similarity(pid, other_id)
        if sim < 0.35:
            continue
        normal = float(other["ppr_pg"])
        if normal < 3:
            continue
        actual = float(game["fantasy_points_ppr"])
        delta_pct = (actual / normal - 1) * 100
        wt = sim ** 2 * float(game.get("game_weight", 1))
        weighted_deltas.append(delta_pct)
        weights.append(wt)
        examples.append({
            "id": str(other_id),
            "name": str(other["player_name"]),
            "sim": round(sim * 100),
            "normal": round(normal, 1),
            "actual": round(actual, 1),
            "tgt": round(float(other["targets_pg"]), 2),
            "rec": round(float(other["rec_pg"]), 2),
            "snap": round(float(other["snap_pct"]), 2),
            "share": round(float(other["touch_share"] if pos == "RB" else other["target_share"]), 2),
            "rz": round(float(other["rz_carries_pg"] if pos == "RB" else other["rz_targets_pg"]), 2),
        })

    if not weights or sum(weights) <= 0:
        return 0.0, []

    effect_pct = float(np.average(weighted_deltas, weights=weights))
    evidence_games = len(weights)
    maturity = min(1.0, max_week / 8.0)
    evidence = min(1.0, evidence_games / 6.0)
    adj = np.clip(effect_pct / 25.0 * MATCHUP_CAP, -MATCHUP_CAP, MATCHUP_CAP)
    adj *= maturity * (0.45 + 0.55 * evidence)

    examples = sorted(examples, key=lambda x: x["sim"], reverse=True)[:12]
    return float(adj), examples

start_vals = []
match_examples = {}
for _, r in g.iterrows():
    opp = opp_map.get(str(r["team"]), "")
    madj, ex = matchup_adjustment(r, opp)

    trend_adj = np.clip(r["trend_pct"] / 20.0, -1, 1) * RECENT_CAP
    if r["games"] < 2:
        trend_adj *= 0.25

    confidence_adj = {"High": 0.0, "Medium": -0.4, "Low": -0.8}[r["confidence_label"]]
    context_adj = madj + trend_adj
    if r["position"] == "TE":
        context_adj *= 0.35
    start = clamp(r["mania"] + context_adj + confidence_adj)
    start_vals.append((start, madj, trend_adj, opp))
    match_examples[r["player_id"]] = ex

g[["start_rating", "matchup_adj", "trend_adj", "opponent"]] = pd.DataFrame(start_vals, index=g.index)
g["start_pos_rank"] = g.groupby("position")["start_rating"].rank(ascending=False, method="min").astype(int)
g["start_overall_rank"] = g["start_rating"].rank(ascending=False, method="min").astype(int)

# ============================================================
# PACKAGE DATA PAYLOADS
# ============================================================
print("6. Formatting JSON datasets & payload structures...")

players = []
for _, r in g.sort_values("mania", ascending=False).iterrows():
    d = merged[(merged["player_id"] == r["player_id"]) & (merged["team"] == r["team"])].sort_values("week")

    logs = []
    for x in d.itertuples():
        logs.append({
            "w": int(x.week),
            "tgt": int(x.targets),
            "rec": int(x.receptions),
            "ry": round(float(x.receiving_yards), 1),
            "rtd": round(float(x.receiving_tds), 2),
            "air": round(float(x.receiving_air_yards), 1),
            "car": int(x.carries),
            "ruy": round(float(x.rushing_yards), 1),
            "rutd": round(float(x.rushing_tds), 2),
            "ppr": round(float(x.fantasy_points_ppr), 1),
            "snap": round(float(x.offense_pct) * 100, 1),
            "tshare": round(float(x.target_share_game), 1),
            "rshare": round(float(x.rush_share_game), 1),
            "ashare": round(float(x.air_share_game), 1),
            "touch": round(float(x.touch_share_game), 1),
            "rzt": int(x.rz_targets),
            "ez": int(x.endzone_targets),
            "rzc": int(x.rz_carries),
            "gl": int(x.gl_carries),
            "partial": bool(x.partial),
            "weight": round(float(x.game_weight), 2),
        })

    pos = r["position"]
    room = g[(g["team"] == r["team"]) & (g["position"] == pos)].sort_values("mania", ascending=False)
    room_data = []
    for _, rr in room.iterrows():
        rd = merged[(merged["player_id"] == rr["player_id"]) & (merged["team"] == rr["team"])].sort_values("week")
        weekly = [{
            "w": int(xx.week),
            "tgt": int(xx.targets),
            "car": int(xx.carries),
            "snap": round(float(xx.offense_pct) * 100, 1),
            "tshare": round(float(xx.target_share_game), 1),
            "rshare": round(float(xx.rush_share_game), 1),
            "share": round(float(xx.touch_share_game if pos == "RB" else xx.target_share_game), 1),
            "ppr": round(float(xx.fantasy_points_ppr), 1),
        } for xx in rd.itertuples()]
        room_data.append({
            "id": rr["player_id"], "name": rr["player_name"],
            "mania": round(float(rr["mania"]), 1),
            "ppr": round(float(rr["ppr_pg"]), 1),
            "car": round(float(rr["carries_pg"]), 1),
            "rshare": round(float(rr["rush_share"]), 1),
            "tgt": round(float(rr["targets_pg"]), 1),
            "tshare": round(float(rr["target_share"]), 1),
            "snap": round(float(rr["snap_pct"]), 1),
            "weekly": weekly,
        })

    metrics = {
        "ppr": round(float(r["ppr_pg"]), 1),
        "tgt": round(float(r["targets_pg"]), 1),
        "rec": round(float(r["rec_pg"]), 1),
        "recy": round(float(r["rec_yards_pg"]), 1),
        "car": round(float(r["carries_pg"]), 1),
        "rushy": round(float(r["rush_yards_pg"]), 1),
        "scrim": round(float(r["scrim_pg"]), 1),
        "td": round(float(r["td_pg"]), 2),
        "snap": round(float(r["snap_pct"]), 1),
        "tshare": round(float(r["target_share"]), 1),
        "rshare": round(float(r["rush_share"]), 1),
        "ashare": round(float(r["air_share"]), 1),
        "touch": round(float(r["touch_share"]), 1),
        "rzt": round(float(r["rz_targets_pg"]), 2),
        "ez": round(float(r["endzone_targets_pg"]), 2),
        "rzc": round(float(r["rz_carries_pg"]), 2),
        "gl": round(float(r["gl_carries_pg"]), 2),
        "ypt": round(float(r["yards_per_target"]), 1),
        "ypc": round(float(r["yards_per_carry"]), 1),
    }

    players.append({
        "id": r["player_id"],
        "name": r["player_name"],
        "team": r["team"],
        "pos": pos,
        "games": int(r["games"]),
        "partial_games": int(r["partial_games"]),
        "confidence": r["confidence_label"],
        "mania": round(float(r["mania"]), 1),
        "raw": round(float(r["mania_base"]), 1),
        "rank": int(r["overall_rank"]),
        "pos_rank": int(r["pos_rank"]),
        "team_rank": int(r["team_pos_rank"]),
        "start": round(float(r["start_rating"]), 1),
        "start_rank": int(r["start_overall_rank"]),
        "start_pos_rank": int(r["start_pos_rank"]),
        "opp": str(r["opponent"]) if r["opponent"] else "TBD",
        "headshot": headshot_map.get(r["player_id"], ""),
        "matchup_adj": round(float(r["matchup_adj"]), 1),
        "trend_adj": round(float(r["trend_adj"]), 1),
        "trend_pct": round(float(r["trend_pct"]), 1),
        "buckets": {
            "Fantasy Production": round(float(r["production"]), 1),
            "Target Volume" if pos != "RB" else "Touch Volume": round(float(r["opportunity"]), 1),
            "Team Target Role" if pos != "RB" else "Backfield Control": round(float(r["role_score"]), 1),
            "Red-Zone Threat" if pos != "RB" else "Goal-Line / Red-Zone": round(float(r["high_value"]), 1),
            "Per-Target Efficiency" if pos != "RB" else "Per-Touch Efficiency": round(float(r["efficiency"]), 1),
        },
        "m": metrics,
        "logs": logs,
        "room": room_data,
        "similar": match_examples.get(r["player_id"], []),
    })

pct_refs = {}
for pos in ["RB", "WR", "TE"]:
    pct_refs[pos] = {}
    gp = g[(g["position"] == pos) & g["qualified"]]
    for c in PCT_COLS:
        pct_refs[pos][c] = [round(float(v), 5) for v in np.sort(gp[c].replace([np.inf, -np.inf], np.nan).dropna().values)]
refs_json = json.dumps(pct_refs, separators=(",", ":"))

payload = json.dumps(players, separators=(",", ":"))
meta = json.dumps({
    "season": SEASON,
    "week": max_week,
    "next_week": next_week,
    "rz": rz_ok,
    "schedule": schedule_ok,
    "players": len(players),
})

# ============================================================
# HTML TEMPLATE
# ============================================================
html = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Mania</title>
<style>
:root{--bg:#070d14;--surface:#0e1722;--surface2:#132030;--line:#1d2c3e;--ink:#f4f7fb;--muted:#8e9cb0;--blue:#3b92ff;--blue2:#72b7ff;--green:#37c979;--yellow:#e2b43b;--red:#ef5a5a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.5}button,input,select{font:inherit}
.topbar{background:#060b12;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50}.nav{max-width:1300px;margin:auto;height:66px;padding:0 24px;display:flex;align-items:center;justify-content:space-between}.brand{font-size:24px;font-weight:900;letter-spacing:-.4px;color:#fff}.brand span{color:var(--blue)}.links{display:flex;height:100%}.navb{border:0;background:transparent;color:#94a2b3;font-size:14px;font-weight:800;padding:0 16px;cursor:pointer;border-bottom:2px solid transparent}.navb.on,.navb:hover{color:#fff;border-bottom-color:var(--blue)}
.wrap{max-width:1300px;margin:auto;padding:34px 24px 80px}.view{display:none}.view.on{display:block}.kicker{font-size:12px;color:var(--muted);font-weight:750;letter-spacing:.3px;margin-bottom:6px}.title{font-size:42px;font-weight:800;letter-spacing:-1px;margin:4px 0 6px}.sub{color:var(--muted);font-size:16px;margin:0 0 20px;line-height:1.5}
.search{position:relative}.search input{width:100%;height:52px;padding:0 18px;border:1px solid var(--line);border-radius:6px;background:#0b131c;color:#fff;outline:none;font-size:16px}.search input:focus{border-color:#486383}.dd{display:none;position:absolute;top:56px;left:0;right:0;background:#0d1824;border:1px solid var(--line);box-shadow:0 18px 45px #000a;z-index:60;border-radius:6px;overflow:hidden}.ddi{padding:14px 16px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;align-items:center;justify-content:space-between}.ddi:hover{background:#132233}.ddiName{font-weight:750}.ddiMeta{font-size:13px;color:var(--muted);margin-left:8px}.ddiRate{font-size:18px;font-weight:800;color:var(--green)}
.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin:18px 0}.pill{background:transparent;border:1px solid #28394e;color:#aab7c7;border-radius:4px;padding:8px 14px;font-size:13px;font-weight:800;cursor:pointer}.pill.on,.pill:hover{background:#f4f7fb;border-color:#f4f7fb;color:#07111b}.panel{border-top:1px solid var(--line);margin-top:24px;padding-top:22px}.ph{font-size:12px;letter-spacing:1px;color:#a0afc0;font-weight:800;text-transform:uppercase;margin-bottom:14px}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}table{width:100%;border-collapse:collapse;white-space:nowrap}th{color:#79889a;font-size:11px;text-transform:uppercase;letter-spacing:.7px;cursor:pointer}th,td{padding:13px 10px;border-bottom:1px solid var(--line);text-align:right;font-size:14px}th:nth-child(2),td:nth-child(2){text-align:left}.playerlink{font-weight:750;color:#eef4fb;cursor:pointer}.playerlink:hover{color:var(--blue2)}.tag{font-size:10px;background:#152231;color:#8ba0b7;padding:3px 6px;border-radius:3px;margin-left:6px;font-weight:700}.grade{font-weight:800}.g90,.g80{color:var(--green)}.g70,.g60{color:var(--yellow)}.g0{color:var(--red)}

/* Player Page Redesign */
.playerHeader{display:flex;gap:36px;align-items:center;padding:12px 0 28px}
.pPortraitBox{width:180px;height:180px;border-radius:50%;position:relative;display:grid;place-items:center;background:#111c28;flex-shrink:0}.pPortraitBox:before{content:"";position:absolute;inset:-6px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#1c2a38 0);z-index:-2}.pPortraitBox:after{content:"";position:absolute;inset:-1px;border-radius:50%;background:var(--bg);z-index:-1}.pPortraitBox img{width:176px;height:176px;object-fit:cover;object-position:center top;border-radius:50%}.initials{font-size:52px;font-weight:800;color:#5a6b82}
.pHeaderInfo{flex:1;min-width:0}
.pName{font-size:46px;font-weight:800;letter-spacing:-1.2px;line-height:1;margin:0 0 10px}
.pMeta{font-size:15px;color:var(--muted);font-weight:700;margin-bottom:6px}.pRankLine{font-size:14px;color:#c6d2df;font-weight:700;margin-bottom:20px}
.pScoresRow{display:flex;gap:48px;align-items:flex-end}
.pScoreBox{display:flex;flex-direction:column}
.pBigNum{font-size:58px;font-weight:800;line-height:.88;letter-spacing:-1px}
.pScoreTitle{font-size:13px;font-weight:800;color:#fff;margin-top:8px;text-transform:uppercase;letter-spacing:.5px}
.pScoreSub{font-size:13px;color:var(--muted);margin-top:2px}
.statStrip{display:grid;grid-template-columns:repeat(6,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:10px 0 28px}.sCell{padding:18px 10px;text-align:center}.sVal{font-size:30px;font-weight:800;line-height:1;letter-spacing:-.5px}.sKey{font-size:11px;color:var(--muted);text-transform:uppercase;font-weight:800;letter-spacing:.5px;margin-top:5px}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:36px}
.barrow{display:grid;grid-template-columns:190px 1fr 54px;align-items:center;gap:12px;margin:14px 0;font-size:14px}
.track{height:9px;background:#142130;border-radius:5px;overflow:hidden}.fill{height:100%;border-radius:5px}
.barValue{text-align:right;font-weight:800;font-size:14px}
.lblBtn{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-weight:750;color:#e4ecf5}.lblBtn:hover{color:#fff;text-decoration:underline}.lblBtn .chev{color:#77889d;font-size:13px}
.bucketHelp{display:none;margin:-4px 0 14px;padding:12px 14px;background:#0c1520;border-left:3px solid #37485c;border-radius:0 4px 4px 0;font-size:13px;line-height:1.55;color:#b4c1cf}
.bucketHelp.open{display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:-1px}

.roomtable th,.roomtable td{text-align:center;padding:11px 8px;font-size:13px}.roomtable th:first-child,.roomtable td:first-child{text-align:left}.roomtable td:first-child{font-weight:750}.roomtable .you{color:var(--blue2);font-weight:800}
.matchScore{font-size:36px;font-weight:800;letter-spacing:-.5px}.matchHead{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:12px}
.matchTable th:nth-child(2),.matchTable td:nth-child(2){text-align:right}
.good{color:var(--green);font-weight:800}.bad{color:var(--red);font-weight:800}

.usageGrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:14px}
.usageItem{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px}
.usageItem .uv{font-size:28px;font-weight:800;line-height:1}
.usageItem .ul{font-size:13px;color:#d5e1ee;font-weight:750;margin-top:6px}
.usageItem .uavg{font-size:12px;color:#8e9cb0;margin-top:6px;font-weight:700}
.usageItem .ud{font-size:12px;color:#718296;margin-top:6px;line-height:1.4}
.ubar{height:5px;background:#142130;border-radius:3px;margin-top:10px;overflow:hidden}.ubar i{display:block;height:100%;border-radius:3px}

.gameControls{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}
.gamechip{border:1px solid #28394e;background:#0b131c;color:#a8b6c6;padding:7px 11px;border-radius:4px;font-size:12px;font-weight:700;cursor:pointer}
.gamechip:has(input:checked){border-color:var(--blue);color:#fff;background:#102235}
.gamechip input{display:none}.reset{border:1px solid #33475d;background:transparent;color:#a9b8c8;border-radius:4px;padding:6px 10px;cursor:pointer;font-size:11px;font-weight:800}

/* Team Share View */
.teamControls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0 24px}
.teamSelect{min-width:200px;height:44px;background:#0c1520;color:#fff;border:1px solid #2e3f53;border-radius:5px;padding:0 12px;font-size:15px;font-weight:700}
.shareSection{margin:28px 0 40px}.shareSection h2{font-size:24px;margin:0 0 10px;letter-spacing:-.4px}
.shareTable th,.shareTable td{font-size:14px;padding:12px 10px;text-align:center}
.shareTable th:first-child,.shareTable td:first-child{text-align:left;position:sticky;left:0;background:var(--bg);z-index:2}
.shareTable td.season{background:#0e1824;font-weight:800;color:#fff}
.shareTable td.up{color:var(--green);font-weight:750}.shareTable td.down{color:var(--red);font-weight:750}
.shareSubHeader{font-size:11px;color:#708092;font-weight:700}

/* Home & Compare */
.homeHero{padding:32px 0}.homeHero h1{font-size:58px;line-height:1.02;letter-spacing:-2px;margin:8px 0 12px}.homeHero .sub{max-width:760px;font-size:18px}
.homeGrid{display:grid;grid-template-columns:1fr 1fr;gap:48px;padding-top:28px}
.miniRow{display:grid;grid-template-columns:30px 1fr 80px 80px;gap:8px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line);font-size:15px}
.miniRow .n{color:#6c7d91}.miniRow .pn{font-weight:750;cursor:pointer}.miniRow .score{text-align:right;font-weight:800}
.howBox{font-size:14px;line-height:1.6;color:#a9b7c7;margin-bottom:16px}.howBox b{display:block;font-size:16px;color:#fff;margin-bottom:3px}
.compareHero{display:grid;grid-template-columns:1fr 60px 1fr;align-items:center;margin-top:24px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:26px 0}
.comparePlayer{text-align:center}.cmpPortrait{width:160px;height:160px;margin:0 auto 12px;border-radius:50%;position:relative;display:grid;place-items:center;background:#111c28}
.cmpPortrait:before{content:"";position:absolute;inset:-5px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#1c2a38 0);z-index:-2}
.cmpPortrait:after{content:"";position:absolute;inset:-1px;border-radius:50%;background:var(--bg);z-index:-1}.cmpPortrait img{width:154px;height:154px;object-fit:cover;object-position:center top;border-radius:50%}
.vs{font-size:18px;font-weight:800;color:#5a6c82;text-align:center}.edge{text-align:center;padding:18px 0;border-bottom:1px solid var(--line)}.edge strong{font-size:30px}

@media(max-width:850px){
 .wrap{padding:20px 14px 60px}
 .playerHeader{flex-direction:column;text-align:center;gap:18px}
 .pPortraitBox{margin:0 auto}
 .pScoresRow{justify-content:center;gap:32px}
 .statStrip{grid-template-columns:repeat(3,1fr)}
 .grid2,.homeGrid,.compareHero{grid-template-columns:1fr}
 .barrow{grid-template-columns:1fr 50px}.barrow .track{grid-column:1/3}
 .usageGrid{grid-template-columns:1fr}
}
</style></head><body>
<div class="topbar"><div class="nav"><div class="brand">FANTASY <span>MANIA</span></div><div class="links"><button class="navb" data-v="home">HOME</button><button class="navb" data-v="players">PLAYERS</button><button class="navb on" data-v="rankings">RANKINGS</button><button class="navb" data-v="teamshare">TEAM SHARE</button><button class="navb" data-v="compare">COMPARE</button></div></div></div>
<main class="wrap">

<section id="home" class="view">
 <div class="homeHero"><div class="kicker">2026 Fantasy Football • Through Week <span id="homeWeek"></span></div><h1>Know the role. <span style="color:var(--blue)">Make the call.</span></h1><p class="sub">Autonomous NFL opportunity indexing. Analyze usage depth, active shares, red-zone equity and opponent comps.</p><div class="search" style="max-width:740px"><input id="homeQ" placeholder="Search any RB, WR or TE..."><div class="dd" id="homeDD"></div></div></div>
 <div class="homeGrid">
  <div><div class="ph">Top Performers Right Now</div><div id="homeLeaders"></div></div>
  <div><div class="ph">Core Framework</div>
   <div class="howBox"><b>Mania Score</b>Overall position-relative score evaluating baseline output, touches, backfield/target command, scoring-zone work, and efficiency.</div>
   <div class="howBox"><b>Week Score</b>Matchup Start Rating combining core profile with defensive opponent performance against similar archetypes.</div>
   <div class="howBox"><b>Green / Yellow / Red</b>All grades are position-relative percentiles. Green represents elite/high-end production, yellow is baseline average, red is below typical starter thresholds.</div>
  </div>
 </div>
</section>

<section id="players" class="view">
 <h1 class="title">Players</h1><p class="sub">Search any player to pull their full usage diagnostics and matchup model.</p>
 <div class="search"><input id="playerQ" placeholder="Search player..."><div class="dd" id="playerDD"></div></div>
</section>

<section id="rankings" class="view on">
 <div class="kicker">Through Week <span id="wk"></span></div><h1 class="title" id="rankTitle">Overall Rankings</h1><p class="sub" id="rankSub"></p>
 <div class="toolbar"><button class="pill rankmode on" data-mode="mania">MANIA SCORE</button><button class="pill rankmode" data-mode="start">WEEK START</button><span style="width:10px"></span><button class="pill posf on" data-pos="ALL">ALL</button><button class="pill posf" data-pos="RB">RB</button><button class="pill posf" data-pos="WR">WR</button><button class="pill posf" data-pos="TE">TE</button></div>
 <div class="tablewrap"><table><thead id="rankHead"></thead><tbody id="rankBody"></tbody></table></div>
</section>

<section id="teamshare" class="view">
 <div class="kicker">Through Week <span id="shareWeek"></span></div><h1 class="title">Team Share</h1><p class="sub">Full offense command by week. Running backs track carries, rush share, targets, target share, and snaps. WR/TEs focus on targets, target share, and snaps.</p>
 <div class="teamControls"><select id="teamPick" class="teamSelect"></select></div>
 <div id="teamShareBody"></div>
</section>

<section id="profile" class="view"><div id="profileBody"></div></section>

<section id="compare" class="view">
 <h1 class="title">Compare Players</h1><p class="sub">Head-to-head usage profile comparison and Week start recommendation.</p>
 <div class="grid2"><div class="search"><input id="aQ" placeholder="Player A..."><div class="dd" id="aDD"></div></div><div class="search"><input id="bQ" placeholder="Player B..."><div class="dd" id="bDD"></div></div></div>
 <div id="compareBody"></div>
</section>

</main>
<script>
const DB=__PAYLOAD__, META=__META__, REFS=__REFS__;
const $=id=>document.getElementById(id), byId=id=>DB.find(p=>p.id===id), fmt=n=>Math.round(Number(n)*10)/10;
$('wk').textContent=META.week; $('shareWeek').textContent=META.week;
let posFilter='ALL', rankMode='mania', sortKey='mania', sortDir=-1, A=null, B=null, selA=null, selB=null;

function grade(v){return v>=80?'g90':v>=60?'g70':'g0'}
function ring(v){return v>=80?'#37c979':v>=60?'#e2b43b':'#ef5a5a'}
function tier(v){return v>=90?'Elite':v>=82?'Excellent':v>=74?'Strong':v>=65?'Starter':'Depth'}

function showView(v){
 document.querySelectorAll('.view').forEach(x=>x.classList.toggle('on',x.id===v));
 document.querySelectorAll('.navb').forEach(x=>x.classList.toggle('on',x.dataset.v===v));
 window.scrollTo(0,0);
}
document.querySelectorAll('.navb').forEach(b=>b.onclick=()=>showView(b.dataset.v));
document.addEventListener('click',e=>{let t=e.target.closest('[data-open]');if(t&&t.dataset.open){let q=byId(t.dataset.open);if(q)openPlayer(q)}});

function initials(p){return p.name.split(' ').slice(0,2).map(x=>x[0]).join('')}
function pic(p){return p.headshot?`<img src="${p.headshot}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><span class="initials" style="display:none">${initials(p)}</span>`:`<span class="initials">${initials(p)}</span>`}

function searchBox(inp,dd,cb){
 $(inp).oninput=e=>{
  let q=e.target.value.toLowerCase().trim(),d=$(dd);
  if(!q){d.style.display='none';return}
  let m=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);
  d.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><div><span class="ddiName">${p.name}</span><span class="ddiMeta">${p.team} • ${p.pos}</span></div><span class="ddiRate">${p.mania}</span></div>`).join('');
  d.style.display=m.length?'block':'none';
  d.querySelectorAll('.ddi').forEach(x=>x.onclick=()=>{let p=byId(x.dataset.id);d.style.display='none';$(inp).value=p.name;cb(p)});
 };
}
searchBox('homeQ','homeDD',openPlayer);
searchBox('playerQ','playerDD',openPlayer);
searchBox('aQ','aDD',p=>{A=p;selA=p.logs.map((_,i)=>i);renderCompare()});
searchBox('bQ','bDD',p=>{B=p;selB=p.logs.map((_,i)=>i);renderCompare()});

// ==========================================
// TEAM SHARE RENDERER (RB: Carries/RushShare/Tgts/TgtShare/Snaps; WR/TE: Tgts/TgtShare/Snaps)
// ==========================================
function renderTeamShare(){
 let teams=[...new Set(DB.map(p=>p.team))].sort(), pick=$('teamPick');
 if(!pick.options.length) pick.innerHTML=teams.map(t=>`<option>${t}</option>`).join('');
 let team=pick.value||teams[0]; if(!pick.value) pick.value=team;
 let ps=DB.filter(p=>p.team===team), weeks=[...new Set(ps.flatMap(p=>p.logs.map(l=>l.w)))].sort((a,b)=>a-b);

 let html='';
 ['RB','WR','TE'].forEach(pos=>{
  let q=ps.filter(p=>p.pos===pos&&(p.m.snap>=8||p.m.tshare>=3||(pos==='RB'&&p.m.car>=2)));
  if(!q.length) return;
  q.sort((a,b)=>b.m.snap-a.m.snap);
  let isRB=pos==='RB';

  html+=`<div class="shareSection"><h2>${pos==='RB'?'Running Backs':pos==='WR'?'Wide Receivers':'Tight Ends'}</h2><div class="tablewrap"><table class="shareTable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>Wk ${w}<div class="shareSubHeader">${isRB?'Car(%) • Tgt(%) • Snap':'Tgt(%) • Snap'}</div></th>`).join('')}<th>Season Avg<div class="shareSubHeader">${isRB?'Car(%) • Tgt(%) • Snap':'Tgt(%) • Snap'}</div></th></tr></thead><tbody>`;

  q.forEach(p=>{
   let cells=weeks.map(w=>{
    let l=p.logs.find(x=>x.w===w);
    if(!l) return '<td>—</td>';
    if(isRB){
     return `<td><b>${l.car}</b> (${fmt(l.rshare)}%) • <b>${l.tgt}</b> (${fmt(l.tshare)}%) • <b>${fmt(l.snap)}%</b></td>`;
    }else{
     return `<td><b>${l.tgt}</b> (${fmt(l.tshare)}%) • <b>${fmt(l.snap)}%</b></td>`;
    }
   }).join('');

   let seasonCol = isRB ?
    `<td><b>${fmt(p.m.car)}</b> (${fmt(p.m.rshare)}%) • <b>${fmt(p.m.tgt)}</b> (${fmt(p.m.tshare)}%) • <b>${fmt(p.m.snap)}%</b></td>`:
    `<td><b>${fmt(p.m.tgt)}</b> (${fmt(p.m.tshare)}%) • <b>${fmt(p.m.snap)}%</b></td>`;

   html+=`<tr><td><span class="playerlink" data-open="${p.id}">${p.name}</span></td>${cells}<td class="season">${seasonCol}</td></tr>`;
  });
  html+=`</tbody></table></div></div>`;
 });
 $('teamShareBody').innerHTML=html;
}
$('teamPick').onchange=renderTeamShare;

// ==========================================
// RANKINGS RENDERER
// ==========================================
const COLS=[['rank','#'],['name','Player'],['mania','Mania'],['start','Start'],['m.ppr','PPR/G'],['m.tgt','TGT/G'],['m.rec','REC/G'],['m.recy','REC YD/G'],['m.car','CAR/G'],['m.rushy','RUSH YD/G'],['m.snap','SNAP%'],['share','SHARE'],['rz','RZ']];
function val(p,k){
 if(k==='rank') return rankMode==='mania'?p.rank:p.start_rank;
 if(k==='name') return p.name;
 if(k==='share') return p.pos==='RB'?p.m.touch:p.m.tshare;
 if(k==='rz') return p.pos==='RB'?p.m.rzc:p.m.rzt;
 if(k.startsWith('m.')) return p.m[k.slice(2)];
 return p[k];
}

function renderRanks(){
 let arr=DB.filter(p=>posFilter==='ALL'||p.pos===posFilter);
 arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);return typeof x==='string'?sortDir*x.localeCompare(y):sortDir*(x-y)});
 $('rankTitle').textContent=rankMode==='mania'?'Overall Rankings — Mania Score':`Week ${META.next_week} Score Rankings`;
 $('rankSub').textContent=rankMode==='mania'?'Core usage profile grade. Click any stat header to sort.':'Matchup-adjusted start expectation for this upcoming week.';
 $('rankHead').innerHTML='<tr>'+COLS.map(([k,l])=>`<th data-k="${k}">${l}${sortKey===k?(sortDir<0?' ↓':' ↑'):''}</th>`).join('')+'</tr>';
 $('rankBody').innerHTML=arr.map((p,i)=>`<tr><td>${i+1}</td><td><span class="playerlink" data-open="${p.id}">${p.name}</span><span class="tag">${p.team} ${p.pos}</span></td><td class="grade ${grade(p.mania)}">${p.mania}</td><td class="grade ${grade(p.start)}">${p.start}</td><td>${p.m.ppr}</td><td>${p.m.tgt}</td><td>${p.m.rec}</td><td>${p.m.recy}</td><td>${p.m.car}</td><td>${p.m.rushy}</td><td>${p.m.snap}%</td><td>${val(p,'share')}%</td><td>${val(p,'rz')}</td></tr>`).join('');
 $('rankHead').querySelectorAll('th').forEach(th=>th.onclick=()=>{let k=th.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==='name'?1:-1}renderRanks()});
}
document.querySelectorAll('.rankmode').forEach(b=>b.onclick=()=>{rankMode=b.dataset.mode;sortKey=rankMode;sortDir=-1;document.querySelectorAll('.rankmode').forEach(x=>x.classList.toggle('on',x===b));renderRanks()});
document.querySelectorAll('.posf').forEach(b=>b.onclick=()=>{posFilter=b.dataset.pos;document.querySelectorAll('.posf').forEach(x=>x.classList.toggle('on',x===b));renderRanks()});

// ==========================================
// MATH CALCULATION ENGINE (SANDBOX / CUSTOM GAME)
// ==========================================
function percentile(pos,key,v){let a=(REFS[pos]||{})[key]||[];if(!a.length)return 0;let lo=0,hi=a.length;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<=v)lo=m+1;else hi=m}return Math.min(100,lo/a.length*100)}
function avg(a,k,w=true){let den=0,num=0;a.forEach(x=>{let z=w?(x.weight||1):1;num+=Number(x[k]||0)*z;den+=z});return den?num/den:0}
function sum(a,k){return a.reduce((s,x)=>s+Number(x[k]||0)*(x.weight||1),0)}

function customMatch(p,M){
 let ex=p.similar||[];if(!ex.length)return {adj:p.matchup_adj,list:[]};
 let scored=ex.map(x=>{
  let share=p.pos==='RB'?M.touch_share:M.target_share,rz=p.pos==='RB'?M.rz_carries_pg:M.rz_targets_pg;
  let dif=[Math.abs(M.targets_pg-x.tgt)/Math.max(3,x.tgt,1),Math.abs(M.rec_pg-x.rec)/Math.max(2,x.rec,1),Math.abs(M.snap_pct-x.snap)/35,Math.abs(share-x.share)/20,Math.abs(rz-x.rz)/2];
  let sim=Math.max(.15,Math.exp(-.8*dif.reduce((a,b)=>a+b,0)/dif.length));
  return {...x,csim:sim,delta:(x.actual/Math.max(x.normal,3)-1)*100};
 }).sort((a,b)=>b.csim-a.csim);
 let use=scored.filter(x=>x.csim>=.28),den=use.reduce((s,x)=>s+x.csim*x.csim,0);
 if(!den)return {adj:0,list:scored.slice(0,4)};
 let eff=use.reduce((s,x)=>s+x.delta*x.csim*x.csim,0)/den,maturity=Math.min(1,META.week/8),evidence=Math.min(1,use.length/6),adj=Math.max(-6,Math.min(6,eff/25*6))*maturity*(.45+.55*evidence);
 return {adj:p.pos==='TE'?adj*.35:adj,list:scored.slice(0,4)};
}

function calc(p,idxs){
 let L=idxs.map(i=>p.logs[i]).filter(Boolean);if(!L.length)return null;
 let M={};
 M.ppr_pg=avg(L,'ppr');M.targets_pg=avg(L,'tgt');M.rec_pg=avg(L,'rec');M.rec_yards_pg=avg(L,'ry');M.carries_pg=avg(L,'car');M.rush_yards_pg=avg(L,'ruy');M.scrim_pg=M.rec_yards_pg+M.rush_yards_pg;M.td_pg=avg(L,'rtd')+avg(L,'rutd');M.snap_pct=avg(L,'snap');M.target_share=avg(L,'tshare');M.rush_share=avg(L,'rshare');M.air_share=avg(L,'ashare');M.touch_share=avg(L,'touch');M.rz_targets_pg=avg(L,'rzt');M.endzone_targets_pg=avg(L,'ez');M.rz_carries_pg=avg(L,'rzc');M.gl_carries_pg=avg(L,'gl');M.yards_per_target=sum(L,'tgt')?sum(L,'ry')/sum(L,'tgt'):0;M.catch_rate=sum(L,'tgt')?sum(L,'rec')/sum(L,'tgt')*100:0;M.yards_per_carry=sum(L,'car')?sum(L,'ruy')/sum(L,'car'):0;
 let opp=sum(L,'car')+2*sum(L,'tgt');M.fp_per_weighted_opp=opp?sum(L,'ppr')/opp:0;
 let P={};Object.keys(M).forEach(k=>P[k]=percentile(p.pos,k,M[k]));

 let prod,op,role,hv,eff;
 if(p.pos==='RB'){
  prod=.55*P.ppr_pg+.30*P.scrim_pg+.15*P.td_pg;
  op=.45*P.carries_pg+.35*P.targets_pg+.20*P.rec_pg;
  role=.60*P.touch_share+.40*P.snap_pct;
  hv=.40*P.rz_carries_pg+.35*P.gl_carries_pg+.25*P.rz_targets_pg;
  eff=.55*P.fp_per_weighted_opp+.25*P.yards_per_carry+.20*P.catch_rate;
 }else{
  prod=.55*P.ppr_pg+.30*P.rec_yards_pg+.15*P.td_pg;
  op=.50*P.targets_pg+.30*P.rec_pg+.20*P.target_share;
  role=.45*P.target_share+.30*P.air_share+.25*P.snap_pct;
  hv=.60*P.rz_targets_pg+.40*P.endzone_targets_pg;
  eff=.45*P.yards_per_target+.30*P.catch_rate+.25*P.fp_per_weighted_opp;
 }
 let W=p.pos==='RB'?[.30,.30,.20,.125,.075]:[.30,.30,.225,.10,.075];
 let raw=W[0]*prod+W[1]*op+W[2]*role+W[3]*hv+W[4]*eff,base=45+.52*raw,eg=L.reduce((s,x)=>s+(x.weight||1),0),conf=Math.min(1,eg/Math.max(META.week,1)),sh=(1-conf)*.12,mania=Math.max(0,Math.min(99.5,base*(1-sh)+72*sh));
 let recent=L.slice(-Math.min(2,L.length)),recentOpp=recent.reduce((s,x)=>s+2*x.tgt+x.car,0)/recent.length,seasonOpp=M.targets_pg*2+M.carries_pg,tr=seasonOpp?((recentOpp/seasonOpp)-1)*100:0,trend=Math.max(-2.5,Math.min(2.5,tr/20*2.5));
 if(L.length<2)trend*=.25;
 let mt=customMatch(p,M),start=Math.max(0,Math.min(100,mania+mt.adj+trend));

 return {
  mania,start,M,match:mt,
  b:p.pos==='RB'?{'Fantasy Production':prod,'Touch Volume':op,'Backfield Control':role,'Goal-Line / Red-Zone':hv,'Per-Touch Efficiency':eff}:
  {'Fantasy Production':prod,'Target Volume':op,'Team Target Role':role,'Red-Zone Threat':hv,'Per-Target Efficiency':eff}
 };
}

// ==========================================
// PLAYER PROFILE PAGE RENDERER
// ==========================================
const HELP={
 'Fantasy Production':'Position-relative scoring output and scrimmage yardage.',
 'Target Volume':'Targets, receptions, and active-game target share.',
 'Team Target Role':'Air yard command, snap participation, and target slice.',
 'Red-Zone Threat':'High-leverage targets inside the 20 and end-zone looks.',
 'Per-Target Efficiency':'Yards per target, catch conversion, and fantasy points per opportunity.',
 'Touch Volume':'Carries and targets earned per contest.',
 'Backfield Control':'Touch share command and offensive snap rate.',
 'Goal-Line / Red-Zone':'Inside-the-20 touches and high-value goal-line carries inside the 5.',
 'Per-Touch Efficiency':'Yards per carry, conversion efficiency, and point production per touch.'
};

function barColor(v){return v>=70?'#37c979':v>=45?'#e2b43b':'#ef5a5a'}
function barGrade(v){return v>=70?'g90':v>=45?'g70':'g0'}
function posAvg(pos,key){let q=DB.filter(x=>x.pos===pos&&x.m.snap>=30).map(x=>Number(x.m[key]||0)).filter(Number.isFinite);return q.length?q.reduce((a,b)=>a+b,0)/q.length:0}

function bucketRows(b){
 return Object.entries(b).map(([k,v],i)=>`
  <div class="barrow">
   <div><button class="lblBtn" data-help="bh_${i}">${k} <span class="chev">ⓘ</span></button></div>
   <div class="track"><div class="fill" style="width:${v}%;background:${barColor(v)}"></div></div>
   <div class="barValue ${barGrade(v)}">${fmt(v)}</div>
  </div>
  <div class="bucketHelp" id="bh_${i}">${HELP[k]||''} Position-relative grade: ${fmt(v)}.</div>
 `).join('');
}

function roomHTML(p){
 let weeks=[...new Set(p.room.flatMap(x=>x.weekly.map(w=>w.w)))].sort((a,b)=>a-b);
 return `<div class="tablewrap"><table class="roomtable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>W${w}</th>`).join('')}<th>Season</th></tr></thead><tbody>${p.room.map(x=>`<tr><td class="${x.id===p.id?'you':''}" data-open="${x.id}">${x.name}</td>${weeks.map(w=>{let z=x.weekly.find(q=>q.w===w);return `<td>${z?`${p.pos==='RB'?z.car+z.tgt:z.tgt} /${z.share}%`:'—'}</td>`}).join('')}<td><b>${x.share}%</b></td></tr>`).join('')}</tbody></table></div><div class="kicker" style="margin-top:10px">${p.pos==='RB'?'Carries + Targets / Touch Share':'Targets / Target Share'}. Tap any player to open profile.</div>`;
}

function matchupHTML(p,c){
 let m=c?c.match:{adj:p.matchup_adj,list:p.similar.slice(0,4)};
 return `<div class="matchHead"><div><div class="ph">WEEK ${META.next_week} MATCHUP • ${p.opp}</div><div class="kicker">Similar ${p.pos} historical performance vs ${p.opp}</div></div><div class="matchScore ${m.adj>=0?'good':'bad'}">${m.adj>=0?'+':''}${fmt(m.adj)}</div></div><div class="tablewrap"><table class="matchTable"><thead><tr><th>Similar Player</th><th>Similarity</th><th>Season PPR</th><th>PPR vs ${p.opp}</th></tr></thead><tbody>${m.list.map(x=>`<tr><td><span class="playerlink" data-open="${x.id||''}">${x.name}</span></td><td>${Math.round((x.csim||x.sim/100)*100)}%</td><td>${x.normal}</td><td>${x.actual}</td></tr>`).join('')||'<tr><td colspan="4" class="kicker">Not enough matchup comps yet.</td></tr>'}</tbody></table></div>`;
}

function usageHTML(p,c){
 let M=c.M,isRB=p.pos==='RB',pc=x=>x?'%':'';
 let items=isRB?[
  ['Scrimmage Yds / G',M.scrim_pg,'scrim','scrim_pg','Rushing plus receiving total yards',0],
  ['Red-Zone Carries / G',M.rz_carries_pg,'rzc','rz_carries_pg','Carries inside opponent 20-yard line',0],
  ['Goal-Line Carries / G',M.gl_carries_pg,'gl','gl_carries_pg','High-value attempts inside the 5-yard line',0],
  ['Red-Zone Targets / G',M.rz_targets_pg,'rzt','rz_targets_pg','Pass looks inside the 20',0],
  ['End-Zone Targets / G',M.endzone_targets_pg,'ez','endzone_targets_pg','PBP charted passes into end zone',0],
  ['Touch Share',M.touch_share,'touch','touch_share','Share of total team carries plus targets',1]
 ]:[
  ['Receiving Yards / G',M.rec_yards_pg,'recy','rec_yards_pg','Air yards + YAC combined',0],
  ['Red-Zone Targets / G',M.rz_targets_pg,'rzt','rz_targets_pg','Pass looks inside opponent 20',0],
  ['End-Zone Targets / G',M.endzone_targets_pg,'ez','endzone_targets_pg','Downfield passes into the end zone',0],
  ['Target Share',M.target_share,'tshare','target_share','Share of team pass attempts',1],
  ['Air-Yard Share',M.air_share,'ashare','air_share','Share of total team throwing air depth',1],
  ['Receptions / G',M.rec_pg,'rec','rec_pg','Per-game catch volume',0]
 ];
 return `<div class="ph">Usage Details</div><div class="usageGrid">${items.map(([l,v,mk,pk,d,isP])=>{let pr=percentile(p.pos,pk,v),col=barColor(pr);return `<div class="usageItem"><div class="uv ${barGrade(pr)}">${fmt(v)}${pc(isP)}</div><div class="ul">${l}</div><div class="ubar"><i style="width:${Math.max(5,pr)}\%;background:${col}"></i></div><div class="uavg">${p.pos} Avg: ${fmt(posAvg(p.pos,mk))}${pc(isP)}</div><div class="ud">${d}</div></div>`}).join('')}</div>`;
}

function openPlayer(p){
 showView('profile');
 let share=p.pos==='RB'?p.m.touch:p.m.tshare, shareName=p.pos==='RB'?'Touch Share':'Target Share';
 let official=calc(p,p.logs.map((_,i)=>i));
 let logs=p.logs.map((l,i)=>`<tr><td><input class="gameToggle" type="checkbox" checked data-i="${i}"></td><td>W${l.w} ${l.partial?'<span style="color:var(--yellow);font-size:10px">SHORT</span>':''}</td><td class="grade ${grade(Math.min(99,l.ppr*4))}">${l.ppr}</td><td>${l.tgt}</td><td>${l.rec}</td><td>${l.ry}</td><td>${l.car}</td><td>${l.ruy}</td><td>${l.snap}%</td></tr>`).join('');

 $('profileBody').innerHTML=`
  <div class="kicker playerlink" id="backRanks" style="margin-bottom:14px">← Rankings</div>
  <div class="playerHeader">
   <div class="pPortraitBox" id="portrait" style="--score:${p.mania};--ring:${ring(p.mania)}">${pic(p)}</div>
   <div class="pHeaderInfo">
    <h1 class="pName">${p.name}</h1>
    <div class="pMeta">${p.team} • ${p.pos} • ${p.games} games</div>
    <div class="pRankLine">#${p.rank} Overall &nbsp;•&nbsp; ${p.pos}${p.pos_rank} &nbsp;•&nbsp; #${p.team_rank} ${p.pos} on ${p.team}</div>
    <div class="pScoresRow">
     <div class="pScoreBox">
      <div id="maniaTop" class="pBigNum ${grade(p.mania)}">${p.mania}</div>
      <div class="pScoreTitle">Mania Score</div>
      <div class="pScoreSub">Overall profile • ${tier(p.mania)}</div>
     </div>
     <div class="pScoreBox">
      <div id="startTop" class="pBigNum ${grade(p.start)}">${p.start}</div>
      <div class="pScoreTitle">Week ${META.next_week} Score</div>
      <div class="pScoreSub">${p.opp==='TBD'?'Matchup TBD':'Start value vs '+p.opp}</div>
     </div>
    </div>
   </div>
  </div>

  <div class="statStrip">
   <div class="sCell"><div class="sVal">${p.m.ppr}</div><div class="sKey">PPR / Game</div></div>
   <div class="sCell"><div class="sVal">${p.m.tgt}</div><div class="sKey">Targets / G</div></div>
   <div class="sCell"><div class="sVal">${p.m.rec}</div><div class="sKey">Rec / Game</div></div>
   <div class="sCell"><div class="sVal">${p.m.scrim}</div><div class="sKey">Scrimmage / G</div></div>
   <div class="sCell"><div class="sVal">${share}%</div><div class="sKey">${shareName}</div></div>
   <div class="sCell"><div class="sVal">${p.m.snap}%</div><div class="sKey">Snap Share</div></div>
  </div>

  <div class="grid2">
   <div class="panel">
    <div class="ph">MANIA BREAKDOWN</div>
    <div id="profileBars">${bucketRows(official.b)}</div>
    <div class="legend"><span><i style="background:#37c979"></i>Strong</span><span><i style="background:#e2b43b"></i>Average</span><span><i style="background:#ef5a5a"></i>Weak</span></div>
   </div>
   <div class="panel"><div class="ph">${p.team} ${p.pos} ROOM • WEEK BY WEEK</div>${roomHTML(p)}</div>
  </div>

  <div class="grid2">
   <div class="panel" id="matchPanel">${matchupHTML(p,official)}</div>
   <div class="panel" id="usagePanel">${usageHTML(p,official)}</div>
  </div>

  <div class="panel">
   <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <div class="ph" style="margin:0">Active Game Log: Filter Included Games</div>
    <button class="reset" id="resetGames">RESET ALL</button>
   </div>
   <div class="gameControls">
    <label class="gamechip"><input id="allGames" type="checkbox" checked>ALL</label>
    ${p.logs.map((l,i)=>`<label class="gamechip"><input class="gameToggleChip" type="checkbox" checked data-i="${i}">W${l.w} •${l.ppr} PPR</label>`).join('')}
   </div>
   <div class="kicker" id="customLine" style="margin:10px 0">Official Mania Score <b>${p.mania}</b>. Toggle games to recalculate profile.</div>
   <div class="tablewrap">
    <table><thead><tr><th>Use</th><th>Game</th><th>PPR</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>Snap</th></tr></thead><tbody>${logs}</tbody></table>
   </div>
  </div>
 `;

 $('backRanks').onclick=()=>showView('rankings');
 function wireInfo(){document.querySelectorAll('.lblBtn[data-help]').forEach(b=>b.onclick=()=>{let e=$(b.dataset.help);if(e)e.classList.toggle('open')})}
 wireInfo();

 function sync(){
  let idx=[...document.querySelectorAll('.gameToggle:checked')].map(x=>Number(x.dataset.i));
  let c=calc(p,idx);
  if(!c){$('customLine').textContent='Please select at least one game.';return}$('maniaTop').textContent=fmt(c.mania);$('maniaTop').className='pBigNum '+grade(c.mania);$('startTop').textContent=fmt(c.start);$('startTop').className='pBigNum '+grade(c.start);$('portrait').style.setProperty('--score',c.mania);
  $('portrait').style.setProperty('--ring',ring(c.mania));$('profileBars').innerHTML=bucketRows(c.b);
  wireInfo();
  $('usagePanel').innerHTML=usageHTML(p,c);
  $('matchPanel').innerHTML=matchupHTML(p,c);$('customLine').innerHTML=`Official <b>${p.mania}</b> → Adjusted Mania: <b class="${grade(c.mania)}">${fmt(c.mania)}</b> • Week ${META.next_week} Score: <b class="${grade(c.start)}">${fmt(c.start)}</b> (${idx.length} games included)`;
  $('allGames').checked=idx.length===p.logs.length;
 }

 document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.onchange=e=>{
  let i=e.target.dataset.i,checked=e.target.checked;
  document.querySelectorAll(`[data-i="${i}"]`).forEach(y=>y.checked=checked);
  sync();
 });
 $('allGames').onchange=e=>{
  document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=e.target.checked);
  sync();
 };
 $('resetGames').onclick=()=>{
  document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=true);
  $('allGames').checked=true;
  sync();
 };
}

// ==========================================
// COMPARE ENGINE
// ==========================================
function chips(p,side,sel){
 return `<div class="gameControls" style="justify-content:center"><label class="gamechip"><input type="checkbox" class="cmpAll" data-side="${side}" ${sel.length===p.logs.length?'checked':''}>ALL</label>${p.logs.map((l,i)=>`<label class="gamechip"><input type="checkbox" class="cmpGame" data-side="${side}" data-i="${i}" ${sel.includes(i)?'checked':''}>W${l.w} •${l.ppr}</label>`).join('')}</div>`;
}

function renderCompare(){
 if(!A||!B)return;
 if(!selA)selA=A.logs.map((_,i)=>i);if(!selB)selB=B.logs.map((_,i)=>i);
 let ca=calc(A,selA),cb=calc(B,selB);if(!ca||!cb)return;
 let starter=ca.start>=cb.start?A:B;
 let rows=[['Mania Score',ca.mania,cb.mania],[`Week ${META.next_week} Score`,ca.start,cb.start],['PPR/G',ca.M.ppr_pg,cb.M.ppr_pg],['Targets/G',ca.M.targets_pg,cb.M.targets_pg],['Receptions/G',ca.M.rec_pg,cb.M.rec_pg],['Scrimmage Yds/G',ca.M.scrim_pg,cb.M.scrim_pg],['Snap %',ca.M.snap_pct,cb.M.snap_pct],['Target Share %',ca.M.target_share,cb.M.target_share],['Touch Share %',ca.M.touch_share,cb.M.touch_share],['Air Share %',ca.M.air_share,cb.M.air_share],['Red-Zone Targets/G',ca.M.rz_targets_pg,cb.M.rz_targets_pg],['Red-Zone Carries/G',ca.M.rz_carries_pg,cb.M.rz_carries_pg],['Matchup Adj',ca.match.adj,cb.match.adj]];

 $('compareBody').innerHTML=`
  <div class="compareHero">
   <div class="comparePlayer">
    <div class="cmpPortrait" style="--score:${ca.mania};--ring:${ring(ca.mania)}">${pic(A)}</div>
    <h2 class="playerlink" data-open="${A.id}">${A.name}</h2>
    <div class="kicker">${A.team} • ${A.pos} • vs ${A.opp}</div>
    <div class="pBigNum ${grade(ca.mania)}">${fmt(ca.mania)}</div>
    <div class="kicker" style="margin-top:6px">Mania Score</div>
    ${chips(A,'A',selA)}
   </div>
   <div class="vs">VS</div>
   <div class="comparePlayer">
    <div class="cmpPortrait" style="--score:${cb.mania};--ring:${ring(cb.mania)}">${pic(B)}</div>
    <h2 class="playerlink" data-open="${B.id}">${B.name}</h2>
    <div class="kicker">${B.team} • ${B.pos} • vs ${B.opp}</div>
    <div class="pBigNum ${grade(cb.mania)}">${fmt(cb.mania)}</div>
    <div class="kicker" style="margin-top:6px">Mania Score</div>
    ${chips(B,'B',selB)}
   </div>
  </div>
  <div class="edge">
   <div class="kicker">WEEK ${META.next_week} RECOMMENDATION</div>
   <strong>${starter.name}</strong>
   <div class="kicker" style="margin-top:4px">${fmt(ca.start)} ${A.name} &nbsp; vs &nbsp; ${fmt(cb.start)} ${B.name}</div>
  </div>
  <div class="tablewrap" style="max-width:860px;margin:24px auto 0">
   <table><thead><tr><th>${A.name}</th><th style="text-align:center">Category</th><th>${B.name}</th></tr></thead><tbody>${rows.map(([k,a,b])=>`<tr><td class="${a>b?'good':''}">${fmt(a)}</td><td style="text-align:center">${k}</td><td class="${b>a?'good':''}">${fmt(b)}</td></tr>`).join('')}</tbody></table>
  </div>
 `;

 document.querySelectorAll('.cmpGame').forEach(x=>x.onchange=()=>{
  let side=x.dataset.side,i=Number(x.dataset.i),arr=side==='A'?selA:selB;
  if(x.checked){if(!arr.includes(i))arr.push(i)}else if(arr.length>1)arr.splice(arr.indexOf(i),1);
  arr.sort((a,b)=>a-b);
  renderCompare();
 });
 document.querySelectorAll('.cmpAll').forEach(x=>x.onchange=()=>{
  let p=x.dataset.side==='A'?A:B,arr=p.logs.map((_,i)=>i);
  if(x.dataset.side==='A')selA=x.checked?arr:[arr[arr.length-1]];else selB=x.checked?arr:[arr[arr.length-1]];
  renderCompare();
 });
}

// Initialize
renderRanks();
renderTeamShare();
showView('home');
$('homeWeek').textContent=META.week;
let top=[...DB].sort((a,b)=>b.mania-a.mania).slice(0,5);
$('homeLeaders').innerHTML=`<div class="miniRow" style="font-weight:800;color:var(--muted)"><span>#</span><span>Player</span><span style="text-align:right">Mania</span><span style="text-align:right">Week</span></div>`+top.map((p,i)=>`<div class="miniRow"><span class="n">${i+1}</span><span class="pn" data-open="${p.id}">${p.name} <span class="tag">${p.team} ${p.pos}</span></span><span class="score ${grade(p.mania)}">${Number(p.mania).toFixed(1)}</span><span class="score ${grade(p.start)}">${Number(p.start).toFixed(1)}</span></div>`).join('');
</script></body></html>'''

html = html.replace("__PAYLOAD__", payload).replace("__META__", meta).replace("__REFS__", refs_json)
os.makedirs("public", exist_ok=True)
with open("public/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("7. FANTASY MANIA built successfully.")
print(f"   {len(players)} players | Week {max_week} | Target: public/index.html")
