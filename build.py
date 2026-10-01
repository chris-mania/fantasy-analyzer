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
# One-file autonomous builder for GitHub Pages
# ============================================================

SEASON = 2026
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Core tuning
QUALIFY_SNAP = 0.30
PARTIAL_RATIO = 0.55       # <55% of a player's normal snap share = likely shortened game
PARTIAL_MIN_NORMAL = 0.45  # only detect shortened games for players with a real normal role
MIN_PARTIAL_WEIGHT = 0.20
MATCHUP_CAP = 6.0          # Start Rating can move at most +/- 6 from matchup
RECENT_CAP = 2.5           # recent role trend can move at most +/- 2.5
CONFIDENCE_SHRINK = 0.12   # modest sample shrink, not a giant injury punishment

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

def pct_rank_dict(values):
    # Percentiles for a dict player_id -> value.
    if not values:
        return {}
    ids = list(values)
    arr = np.array([values[i] for i in ids], dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return {pid: float(ranks[i] / len(arr) * 100) for i, pid in enumerate(ids)}

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

# Stats we want. Missing columns become zero so nflverse schema changes don't kill the site.
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

# Snap columns
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

# ============================================================
# RED ZONE / GOAL LINE DATA
# ============================================================
print("2. Loading red-zone usage...")
rz_ok = True
try:
    want = [
        "season_type", "week", "yardline_100", "pass_attempt", "rush_attempt",
        "receiver_player_id", "rusher_player_id"
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
    rec["endzone_targets"] = (rec["yardline_100"] <= 10).astype(int)

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
    print("   WARNING: PBP red-zone data unavailable; site will still build:", e)
    rz_ok = False

for c in ["rz_targets", "endzone_targets", "rz_carries", "gl_carries"]:
    if c not in merged.columns:
        merged[c] = 0
    merged[c] = safe_num(merged[c])

# ============================================================
# TEAM-WEEK TOTALS — fixes the Nico/missed-games share problem
# ============================================================
print("3. Building active-game shares + injury-safe game weights...")

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
merged["air_share_game"] = np.where(
    merged["team_air"] > 0, merged["receiving_air_yards"] / merged["team_air"] * 100, 0
)
merged["touch_share_game"] = np.where(
    (merged["team_targets"] + merged["team_carries"]) > 0,
    (merged["targets"] + merged["carries"]) /
    (merged["team_targets"] + merged["team_carries"]) * 100, 0
)

# Infer normal participation from each player's higher-participation games.
# This is automatic: no manual injury list.
merged["normal_snap"] = 0.0
merged["participation_ratio"] = 1.0
merged["game_weight"] = 1.0
merged["partial"] = False

for pid, idx in merged.groupby("player_id").groups.items():
    ix = list(idx)
    snaps_pct = merged.loc[ix, "offense_pct"].values.astype(float)
    positive = snaps_pct[snaps_pct > 0]
    if len(positive):
        # Upper-half median is robust to one early-exit game.
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
        # Normal games = 1. Shortened games get 0.20–0.55-ish weight.
        weights = np.where(is_partial, np.clip(ratio, MIN_PARTIAL_WEIGHT, 0.60), 1.0)
        merged.loc[ix, "game_weight"] = weights

# A stat row exists only for a played game. Missing weeks are not added as zeroes.
# That's how DNPs stop hurting Mania.

# ============================================================
# PLAYER SUMMARY
# ============================================================
def summarize_player(d):
    d = d.sort_values("week").copy()
    w = d["game_weight"].values
    games = len(d)
    effective_games = float(np.sum(w))

    # Weighted per-game production: partial games contribute less.
    def wm(col):
        return weighted_mean(d[col].values, w)

    # Rate stats
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
    air_share = wm("air_share_game")
    touch_share = wm("touch_share_game")
    rz_t_pg = wm("rz_targets")
    ez_t_pg = wm("endzone_targets")
    rz_c_pg = wm("rz_carries")
    gl_c_pg = wm("gl_carries")

    scrim_pg = rec_yards_pg + rush_yards_pg
    total_td_pg = rec_td_pg + rush_td_pg

    # Efficiency
    ypt = weighted_sum(d["receiving_yards"], w) / max(weighted_sum(d["targets"], w), 1)
    catch_rate = weighted_sum(d["receptions"], w) / max(weighted_sum(d["targets"], w), 1) * 100
    ypc = weighted_sum(d["rushing_yards"], w) / max(weighted_sum(d["carries"], w), 1)
    opp = weighted_sum(d["carries"], w) + 2.0 * weighted_sum(d["targets"], w)
    fp_per_opp = weighted_sum(d["fantasy_points_ppr"], w) / max(opp, 1)

    # Recent role uses last 2 completed appearances vs full weighted role.
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
# MANIA RATING — POSITION-SPECIFIC BUCKETS
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

def avg(*xs):
    xs = [float(x) for x in xs if np.isfinite(x)]
    return sum(xs) / len(xs) if xs else 0.0

bucket_values = []
for _, r in g.iterrows():
    pos = r["position"]

    if pos == "RB":
        production = (
            .55 * r["pct_ppr_pg"] +
            .30 * r["pct_scrim_pg"] +
            .15 * r["pct_td_pg"]
        )
        # Targets deliberately worth more than carries in PPR.
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
        # WR/TE: carries have ZERO direct weight.
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

# Raw percentile blends tend to bunch in the 60s/70s.
# Transform to an intuitive Madden-like 0-100 fantasy scale while keeping ordering intact.
# 90+ remains rare; elite complete profiles can get there.
g["mania_base"] = 45 + 0.52 * g["raw_profile"]

# Confidence: modest shrink toward 72, NOT a huge punishment.
# One strong game remains capable of a mid/high-80s score but is labeled low-confidence.
season_games = max_week
g["confidence"] = np.clip(g["effective_games"] / max(season_games, 1), 0, 1)
shrink = (1 - g["confidence"]) * CONFIDENCE_SHRINK
g["mania"] = g["mania_base"] * (1 - shrink) + 72 * shrink
g["mania"] = g["mania"].clip(0, 99.5)

# Rankings
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
# SCHEDULE + UPCOMING OPPONENT
# ============================================================
print("5. Loading schedule + building current-week matchup engine...")
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
    print("   WARNING: schedule unavailable. Start Rating will equal role-adjusted Mania.")
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

# ============================================================
# SIMILAR-PLAYER DEFENSE ENGINE
# Uses 2026 prior games only; early-season dampened.
# ============================================================
# Build per-player profile vectors from summary.
profile_cols = {
    "RB": ["targets_pg", "carries_pg", "touch_share", "snap_pct", "ppr_pg", "rz_carries_pg", "gl_carries_pg"],
    "WR": ["targets_pg", "rec_pg", "target_share", "air_share", "snap_pct", "ppr_pg", "rz_targets_pg"],
    "TE": ["targets_pg", "rec_pg", "target_share", "air_share", "snap_pct", "ppr_pg", "rz_targets_pg"],
}

# Team opponent for each completed week from schedule.
completed_opp = {}
if schedule_ok and {"week", "home_team", "away_team"}.issubset(games.columns):
    for _, game in games[safe_num(games["week"]).astype(int) <= max_week].iterrows():
        wk = int(game["week"])
        h, a = str(game["home_team"]), str(game["away_team"])
        completed_opp[(h, wk)] = a
        completed_opp[(a, wk)] = h

merged["opponent"] = [completed_opp.get((str(t), int(w)), "") for t, w in zip(merged["team"], merged["week"])]

# Z-score profile vectors within position.
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
    return math.exp(-0.75 * dist)  # 1.0 identical; fades smoothly

summary_by_id = g.set_index("player_id", drop=False)

def matchup_adjustment(player_row, opponent):
    # Compare similar players normal PPR to what they did vs this opponent.
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
        # Participation and similarity both affect evidence weight.
        wt = sim ** 2 * float(game.get("game_weight", 1))
        weighted_deltas.append(delta_pct)
        weights.append(wt)
        examples.append({
            "name": str(other["player_name"]),
            "sim": round(sim * 100),
            "normal": round(normal, 1),
            "actual": round(actual, 1),
        })

    if not weights or sum(weights) <= 0:
        return 0.0, []

    effect_pct = float(np.average(weighted_deltas, weights=weights))

    # Early season: defensive evidence is deliberately weak.
    evidence_games = len(weights)
    maturity = min(1.0, max_week / 8.0)
    evidence = min(1.0, evidence_games / 6.0)
    # +/-25% opponent effect maps to +/-6 rating points at full maturity/evidence.
    adj = np.clip(effect_pct / 25.0 * MATCHUP_CAP, -MATCHUP_CAP, MATCHUP_CAP)
    adj *= maturity * (0.45 + 0.55 * evidence)

    examples = sorted(examples, key=lambda x: x["sim"], reverse=True)[:4]
    return float(adj), examples

start_vals = []
match_examples = {}
for _, r in g.iterrows():
    opp = opp_map.get(str(r["team"]), "")
    madj, ex = matchup_adjustment(r, opp)

    # Recent opportunity trend matters, but is capped and muted this early.
    trend_adj = np.clip(r["trend_pct"] / 20.0, -1, 1) * RECENT_CAP
    if r["games"] < 2:
        trend_adj *= 0.25

    # Low confidence does not crush Mania, but slightly reduces Start certainty.
    confidence_adj = {"High": 0.0, "Medium": -0.4, "Low": -0.8}[r["confidence_label"]]

    start = clamp(r["mania"] + madj + trend_adj + confidence_adj)
    start_vals.append((start, madj, trend_adj, opp))
    match_examples[r["player_id"]] = ex

g[["start_rating", "matchup_adj", "trend_adj", "opponent"]] = pd.DataFrame(start_vals, index=g.index)
g["start_pos_rank"] = g.groupby("position")["start_rating"].rank(ascending=False, method="min").astype(int)
g["start_overall_rank"] = g["start_rating"].rank(ascending=False, method="min").astype(int)

# ============================================================
# BUILD JSON
# ============================================================
print("6. Packaging player pages, team rooms and rankings...")

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
            "car": int(x.carries),
            "ruy": round(float(x.rushing_yards), 1),
            "ppr": round(float(x.fantasy_points_ppr), 1),
            "snap": round(float(x.offense_pct) * 100, 1),
            "tshare": round(float(x.target_share_game), 1),
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
    room_data = [{
        "id": rr["player_id"], "name": rr["player_name"],
        "mania": round(float(rr["mania"]), 1),
        "ppr": round(float(rr["ppr_pg"]), 1),
        "opp": round(float(rr["targets_pg"] if pos != "RB" else rr["carries_pg"] + rr["targets_pg"]), 1),
        "share": round(float(rr["target_share"] if pos != "RB" else rr["touch_share"]), 1),
    } for _, rr in room.iterrows()]

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
        "ashare": round(float(r["air_share"]), 1),
        "touch": round(float(r["touch_share"]), 1),
        "rzt": round(float(r["rz_targets_pg"]), 2),
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
        "matchup_adj": round(float(r["matchup_adj"]), 1),
        "trend_adj": round(float(r["trend_adj"]), 1),
        "trend_pct": round(float(r["trend_pct"]), 1),
        "buckets": {
            "Production": round(float(r["production"]), 1),
            "Opportunity": round(float(r["opportunity"]), 1),
            "Role": round(float(r["role_score"]), 1),
            "High Value": round(float(r["high_value"]), 1),
            "Efficiency": round(float(r["efficiency"]), 1),
        },
        "m": metrics,
        "logs": logs,
        "room": room_data,
        "similar": match_examples.get(r["player_id"], []),
    })

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
# FRONT END — FANTASY MANIA
# ============================================================
html = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Mania</title>
<style>
:root{
 --bg:#080b10;--panel:#10151d;--panel2:#151c26;--line:#263141;
 --text:#f4f7fb;--muted:#8e9aac;--blue:#4ea1ff;--cyan:#5de4ff;
 --green:#46d381;--yellow:#f4c95d;--red:#ff6673;--purple:#aa8cff;
}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#182638 0,#080b10 42%);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input{font:inherit}.shell{max-width:1240px;margin:auto;padding:0 22px 50px}
nav{height:74px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1d2632;position:sticky;top:0;background:rgba(8,11,16,.94);backdrop-filter:blur(14px);z-index:30}
.brand{font-size:22px;font-weight:950;letter-spacing:.7px}.brand span{color:var(--blue)}.navlinks{display:flex;gap:6px}
.navbtn,.pill{border:0;background:transparent;color:var(--muted);padding:10px 13px;border-radius:9px;cursor:pointer;font-weight:750}
.navbtn:hover,.navbtn.on,.pill.on{background:#182231;color:white}.hero{text-align:center;padding:58px 10px 32px}
.eyebrow{font-size:12px;letter-spacing:2px;color:var(--cyan);font-weight:900}.hero h1{font-size:52px;line-height:1;margin:10px 0 12px;letter-spacing:-2px}.hero p{color:var(--muted);font-size:16px}
.search{max-width:720px;margin:24px auto 0;position:relative}.search input{width:100%;padding:17px 20px;border-radius:13px;border:1px solid var(--line);background:#0f151e;color:white;outline:none;font-size:16px}.search input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(78,161,255,.12)}
.dd{display:none;position:absolute;left:0;right:0;top:58px;background:#111823;border:1px solid var(--line);border-radius:12px;overflow:hidden;z-index:40;text-align:left;box-shadow:0 18px 50px #0009}.ddi{padding:13px 16px;border-bottom:1px solid #202a38;cursor:pointer}.ddi:hover{background:#192333}.ddi b{display:block}.ddi small{color:var(--muted)}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:linear-gradient(180deg,#121923,#0f141c);border:1px solid var(--line);border-radius:15px;padding:19px;box-shadow:0 10px 30px #0002}.card h3{margin:0 0 13px;font-size:13px;letter-spacing:.8px;color:#b6c1d0}.leader{display:flex;align-items:center;gap:12px;padding:9px 0;border-top:1px solid #202a36;cursor:pointer}.leader:first-of-type{border-top:0}.leader:hover .lname{color:var(--blue)}.num{font-weight:900;color:#566274;width:18px}.lname{font-weight:800;flex:1}.lsub{font-size:11px;color:var(--muted);margin-top:2px}.rating{font-weight:950;font-size:18px}
.sectionhead{display:flex;justify-content:space-between;align-items:end;margin:30px 0 13px}.sectionhead h2{margin:0;font-size:24px}.sectionhead p{margin:4px 0 0;color:var(--muted)}
.filters{display:flex;gap:7px;flex-wrap:wrap}.pill{background:#111821;border:1px solid var(--line);padding:8px 12px}.tablewrap{overflow:auto;border:1px solid var(--line);border-radius:14px;background:#0e141c}
table{width:100%;border-collapse:collapse;min-width:880px}th{font-size:11px;color:#8491a3;text-transform:uppercase;letter-spacing:.6px;text-align:right;padding:13px 12px;background:#111821;position:sticky;top:0;cursor:pointer;white-space:nowrap}th:first-child,th:nth-child(2),td:first-child,td:nth-child(2){text-align:left}th:hover{color:white}td{padding:13px 12px;border-top:1px solid #202a36;text-align:right;font-size:14px}tr:hover td{background:#131c27}.playerlink{font-weight:850;cursor:pointer}.playerlink:hover{color:var(--blue)}.tag{font-size:10px;padding:3px 6px;border-radius:5px;background:#1c2735;color:#9eacc0;margin-left:5px}
.mania{font-weight:950;color:var(--cyan)}.start{font-weight:950;color:var(--green)}
.view{display:none}.view.on{display:block}.profileTop{padding:34px 0 17px;display:flex;justify-content:space-between;align-items:end;gap:20px}.back{color:var(--blue);cursor:pointer;font-weight:800;margin-bottom:8px}.profileTop h1{margin:0;font-size:38px}.meta{color:var(--muted);margin-top:6px}
.ratingHero{display:flex;gap:10px}.ratingBox{min-width:150px;background:#111923;border:1px solid var(--line);border-radius:14px;padding:14px;text-align:center}.ratingBox .big{font-size:38px;font-weight:950;line-height:1}.ratingBox small{color:var(--muted);font-weight:800}.blue{color:var(--cyan)}.green{color:var(--green)}
.alert{padding:11px 14px;border-radius:9px;background:#2a2112;border:1px solid #5d4720;color:#f5cb73;margin:8px 0 16px}.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:12px 0 20px}.stat{background:#101720;border:1px solid var(--line);border-radius:11px;padding:14px}.stat .v{font-size:23px;font-weight:900}.stat .k{font-size:10px;color:var(--muted);font-weight:800;text-transform:uppercase;margin-top:3px}
.two{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}.bars .barrow{display:grid;grid-template-columns:110px 1fr 42px;gap:10px;align-items:center;margin:12px 0}.track{height:8px;background:#080c12;border-radius:10px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:10px}
.roomrow{display:grid;grid-template-columns:1fr repeat(3,70px);gap:8px;padding:10px 0;border-top:1px solid #202a36;align-items:center}.roomrow:first-of-type{border-top:0}.roomrow b{cursor:pointer}.roomrow b:hover{color:var(--blue)}.muted{color:var(--muted)}.good{color:var(--green)}.bad{color:var(--red)}
.gameToggle{cursor:pointer;accent-color:var(--blue)}.partial{color:var(--yellow);font-size:10px;font-weight:900}.custom{margin:10px 0;padding:12px;background:#101923;border:1px solid #27374a;border-radius:10px;color:#bcd0e6}
.comparePick{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:26px}.comparePick .search{margin:0;max-width:none}.versus{display:grid;grid-template-columns:1fr 80px 1fr;align-items:center;gap:12px;margin-top:20px}.vs{text-align:center;color:var(--muted);font-weight:950}.cmpHero{text-align:center;padding:22px}.cmpHero h2{margin:0 0 4px}.cmpBig{font-size:48px;font-weight:950}.why{font-size:13px;color:var(--muted);line-height:1.5}
@media(max-width:900px){.grid4{grid-template-columns:1fr 1fr}.stats{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:1fr}.navlinks{overflow:auto}.hero h1{font-size:42px}}
@media(max-width:600px){.shell{padding:0 12px 35px}.brand{font-size:17px}.navbtn{padding:8px 7px;font-size:12px}.hero{padding-top:35px}.hero h1{font-size:36px}.grid4{grid-template-columns:1fr}.profileTop{display:block}.ratingHero{margin-top:16px}.stats{grid-template-columns:repeat(2,1fr)}.comparePick,.versus{grid-template-columns:1fr}.vs{padding:4px}}
</style>
</head>
<body>
<div class="shell">
<nav>
 <div class="brand">FANTASY <span>MANIA</span></div>
 <div class="navlinks">
  <button class="navbtn on" data-view="home">HOME</button>
  <button class="navbtn" data-view="players">PLAYERS</button>
  <button class="navbtn" data-view="rankings">RANKINGS</button>
  <button class="navbtn" data-view="compare">COMPARE</button>
 </div>
</nav>

<section id="home" class="view on">
 <div class="hero">
  <div class="eyebrow">POWERED BY MANIA RATING</div>
  <h1>Fantasy decisions,<br>built on real usage.</h1>
  <p id="seasonline"></p>
  <div class="search"><input id="homeSearch" placeholder="Search any RB, WR or TE..." autocomplete="off"><div class="dd" id="homeDD"></div></div>
 </div>
 <div class="grid4" id="dash"></div>
</section>

<section id="players" class="view">
 <div class="sectionhead"><div><h2>Players</h2><p>Search any player for the full Mania profile.</p></div></div>
 <div class="search" style="margin:0;max-width:none"><input id="playerSearch" placeholder="Search player..." autocomplete="off"><div class="dd" id="playerDD"></div></div>
 <div id="playerLanding"></div>
</section>

<section id="rankings" class="view">
 <div class="sectionhead"><div><h2 id="rankTitle">Overall Rankings</h2><p id="rankSub">Mania Rating: who has the best fantasy profile?</p></div>
  <div class="filters"><button class="pill on rankmode" data-mode="mania">OVERALL</button><button class="pill rankmode" data-mode="start">WEEK <span id="wkBtn"></span></button></div>
 </div>
 <div class="filters" style="margin-bottom:12px"><button class="pill on posf" data-pos="ALL">ALL</button><button class="pill posf" data-pos="RB">RB</button><button class="pill posf" data-pos="WR">WR</button><button class="pill posf" data-pos="TE">TE</button></div>
 <div class="tablewrap"><table><thead id="rankHead"></thead><tbody id="rankBody"></tbody></table></div>
</section>

<section id="profile" class="view"><div id="profileBody"></div></section>

<section id="compare" class="view">
 <div class="sectionhead"><div><h2>Compare Players</h2><p>Who's better overall — and who should you start this week?</p></div></div>
 <div class="comparePick">
  <div class="search"><input id="cmpA" placeholder="Player A..." autocomplete="off"><div class="dd" id="cmpADD"></div></div>
  <div class="search"><input id="cmpB" placeholder="Player B..." autocomplete="off"><div class="dd" id="cmpBDD"></div></div>
 </div>
 <div id="compareBody"></div>
</section>
</div>

<script>
const DB=__PAYLOAD__;
const META=__META__;
let rankMode="mania",posFilter="ALL",sortKey="mania",sortDir=-1,A=null,B=null;

const $=id=>document.getElementById(id);
const fmt=(v,d=1)=>Number(v||0).toFixed(d);
const tier=x=>x>=90?"Elite":x>=85?"Excellent":x>=80?"Very Strong":x>=75?"Strong":x>=70?"Solid Role":"Developing";
const byId=id=>DB.find(p=>String(p.id)===String(id));

$("seasonline").textContent=`${META.season} NFL Season • Data through Week ${META.week} • Week ${META.next_week} outlook`;
$("wkBtn").textContent=META.next_week;

function showView(v){
 document.querySelectorAll(".view").forEach(x=>x.classList.remove("on"));
 $(v).classList.add("on");
 document.querySelectorAll(".navbtn").forEach(x=>x.classList.toggle("on",x.dataset.view===v));
 window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll(".navbtn").forEach(b=>b.onclick=()=>showView(b.dataset.view));

function wireSearch(inputId,ddId,onPick){
 const inp=$(inputId),dd=$(ddId);
 inp.oninput=()=>{
  const q=inp.value.trim().toLowerCase();
  if(!q){dd.style.display="none";return}
  const m=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);
  dd.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><b>${p.name}</b><small>${p.team} • ${p.pos} • ${p.mania} Mania</small></div>`).join("");
  dd.style.display=m.length?"block":"none";
  dd.querySelectorAll(".ddi").forEach(x=>x.onclick=()=>{dd.style.display="none";inp.value=byId(x.dataset.id).name;onPick(byId(x.dataset.id))});
 };
}
wireSearch("homeSearch","homeDD",openPlayer);
wireSearch("playerSearch","playerDD",openPlayer);
wireSearch("cmpA","cmpADD",p=>{A=p;renderCompare()});
wireSearch("cmpB","cmpBDD",p=>{B=p;renderCompare()});

function leaders(arr,key,n=3){return [...arr].sort((a,b)=>b[key]-a[key]).slice(0,n)}
function leadCard(title,arr,key,label){
 return `<div class="card"><h3>${title}</h3>${arr.map((p,i)=>`<div class="leader" data-id="${p.id}"><div class="num">${i+1}</div><div class="lname">${p.name}<div class="lsub">${p.team} • ${p.pos}</div></div><div class="rating">${label(p)}</div></div>`).join("")}</div>`
}
function renderDash(){
 const risers=[...DB].sort((a,b)=>b.trend_pct-a.trend_pct).slice(0,3);
 const breakout=[...DB].filter(p=>p.mania>=72&&p.m.ppr<16).sort((a,b)=>(b.buckets.Opportunity+b.buckets.Role)-(a.buckets.Opportunity+a.buckets.Role)).slice(0,3);
 $("dash").innerHTML=
  leadCard("⭐ TOP MANIA",leaders(DB,"mania"),"mania",p=>p.mania)+
  leadCard(`🎯 TOP WEEK ${META.next_week}`,leaders(DB,"start"),"start",p=>p.start)+
  leadCard("📈 ROLE RISERS",risers,"trend_pct",p=>(p.trend_pct>0?"+":"")+p.trend_pct+"%")+
  leadCard("👀 BREAKOUT WATCH",breakout,"mania",p=>p.mania);
 document.querySelectorAll(".leader").forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));
}
renderDash();

const COLS=[
 ["rank","#"],["name","Player"],["mania","Mania"],["start","Start"],["m.ppr","PPR/G"],["m.tgt","TGT/G"],["m.rec","REC/G"],
 ["m.recy","REC YD/G"],["m.car","CAR/G"],["m.rushy","RUSH YD/G"],["m.snap","SNAP%"],["share","SHARE%"],["rz","RZ"]
];
function val(p,k){
 if(k==="rank")return rankMode==="mania"?p.rank:p.start_rank;
 if(k==="name")return p.name;
 if(k==="share")return p.pos==="RB"?p.m.touch:p.m.tshare;
 if(k==="rz")return p.pos==="RB"?p.m.rzc:p.m.rzt;
 if(k.startsWith("m."))return p.m[k.slice(2)];
 return p[k];
}
function renderRanks(){
 let arr=DB.filter(p=>posFilter==="ALL"||p.pos===posFilter);
 arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);if(typeof x==="string")return sortDir*x.localeCompare(y);return sortDir*(x-y)});
 $("rankTitle").textContent=rankMode==="mania"?"Overall Rankings":`Week ${META.next_week} Start Rankings`;
 $("rankSub").textContent=rankMode==="mania"?"Mania Rating: who has the best fantasy profile?":"Current-week rating: Mania + role trend + opponent/similar-player matchup.";
 $("rankHead").innerHTML="<tr>"+COLS.map(([k,l])=>`<th data-k="${k}">${l}${sortKey===k?(sortDir===-1?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
 $("rankBody").innerHTML=arr.map(p=>`<tr>
  <td>${val(p,"rank")}</td><td><span class="playerlink" data-id="${p.id}">${p.name}</span> <span class="tag">${p.team} ${p.pos}</span></td>
  <td class="mania">${p.mania}</td><td class="start">${p.start}</td><td>${p.m.ppr}</td><td>${p.m.tgt}</td><td>${p.m.rec}</td>
  <td>${p.m.recy}</td><td>${p.m.car}</td><td>${p.m.rushy}</td><td>${p.m.snap}%</td><td>${val(p,"share")}%</td><td>${val(p,"rz")}</td></tr>`).join("");
 $("rankHead").querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==="name"?1:-1}renderRanks()});
 $("rankBody").querySelectorAll(".playerlink").forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));
}
document.querySelectorAll(".rankmode").forEach(b=>b.onclick=()=>{
 rankMode=b.dataset.mode;sortKey=rankMode;sortDir=-1;
 document.querySelectorAll(".rankmode").forEach(x=>x.classList.toggle("on",x===b));renderRanks();
});
document.querySelectorAll(".posf").forEach(b=>b.onclick=()=>{
 posFilter=b.dataset.pos;document.querySelectorAll(".posf").forEach(x=>x.classList.toggle("on",x===b));renderRanks();
});
renderRanks();

function statCard(v,k){return `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`}
function openPlayer(p){
 showView("profile");
 const share=p.pos==="RB"?p.m.touch:p.m.tshare;
 const shareName=p.pos==="RB"?"Touch Share":"Target Share";
 const warning=p.confidence!=="High"?`<div class="alert">⚠ ${p.confidence} confidence • ${p.games} played game${p.games===1?"":"s"}${p.partial_games?` • ${p.partial_games} shortened-participation game detected`:""}. Missed games do not count as zeroes.</div>`:"";
 const roomLabel=p.pos==="RB"?`${p.team} BACKFIELD`:`${p.team} ${p.pos} ROOM`;
 const room=p.room.map(x=>`<div class="roomrow"><b data-id="${x.id}">${x.name}</b><span>${x.mania}</span><span>${x.opp}</span><span>${x.share}%</span></div>`).join("");
 const bars=Object.entries(p.buckets).map(([k,v])=>`<div class="barrow"><span class="muted">${k}</span><div class="track"><div class="fill" style="width:${v}%"></div></div><b>${v}</b></div>`).join("");
 const sims=p.similar.length?p.similar.map(x=>`<div class="roomrow"><b>${x.name}</b><span>${x.sim}%</span><span>${x.normal}</span><span>${x.actual}</span></div>`).join(""):`<p class="muted">Not enough similar-player evidence yet. Early-season matchup effects stay intentionally small.</p>`;
 const logs=p.logs.map((l,i)=>`<tr><td><input class="gameToggle" type="checkbox" checked data-i="${i}"></td><td>W${l.w} ${l.partial?'<span class="partial">SHORT</span>':""}</td><td>${l.tgt}</td><td>${l.rec}</td><td>${l.ry}</td><td>${l.car}</td><td>${l.ruy}</td><td>${l.snap}%</td><td>${l.ppr}</td></tr>`).join("");

 $("profileBody").innerHTML=`
 <div class="profileTop"><div><div class="back" id="backRanks">← Rankings</div><h1>${p.name}</h1><div class="meta">${p.team} • ${p.pos} • ${p.games} games • #${p.rank} overall • ${p.pos}${p.pos_rank}</div></div>
 <div class="ratingHero"><div class="ratingBox"><div class="big blue">${p.mania}</div><small>MANIA RATING<br>${tier(p.mania)}</small></div>
 <div class="ratingBox"><div class="big green">${p.start}</div><small>WEEK ${META.next_week} START<br>vs ${p.opp}</small></div></div></div>
 ${warning}
 <div class="stats">${statCard(p.m.ppr,"PPR / Game")}${statCard(p.m.tgt,"Targets / Game")}${statCard(p.m.rec,"Receptions / Game")}${statCard(p.m.scrim,"Scrimmage Yds / G")}${statCard(share+"%",shareName)}${statCard(p.m.snap+"%","Snap Share")}</div>
 <div class="two">
  <div class="card"><h3>MANIA PROFILE</h3><div class="bars">${bars}</div><p class="why">Mania is position-specific. WR/TE carries have no direct rating weight. RB targets and receptions receive extra PPR value. Missed games are excluded; clearly shortened games are automatically downweighted.</p></div>
  <div class="card"><h3>${roomLabel}</h3><div class="roomrow muted"><span>Player</span><span>Mania</span><span>${p.pos==="RB"?"Opp/G":"Tgt/G"}</span><span>Share</span></div>${room}</div>
 </div>
 <div class="two" style="margin-top:14px">
  <div class="card"><h3>WEEK ${META.next_week} MATCHUP • ${p.opp}</h3>
   <div style="font-size:32px;font-weight:950" class="${p.matchup_adj>=0?"good":"bad"}">${p.matchup_adj>=0?"+":""}${p.matchup_adj}</div>
   <p class="why">Matchup adjustment from similar ${p.pos}s who already faced ${p.opp}. It is deliberately dampened early in the season. Recent-role adjustment: ${p.trend_adj>=0?"+":""}${p.trend_adj}.</p>
   <div class="roomrow muted"><span>Similar player</span><span>Match</span><span>Normal</span><span>vs ${p.opp}</span></div>${sims}
  </div>
  <div class="card"><h3>ADVANCED USAGE</h3>
   ${statCard(p.pos==="RB"?p.m.car:p.m.recy,p.pos==="RB"?"Carries / G":"Receiving Yds / G")}
   ${statCard(p.pos==="RB"?p.m.rzc:p.m.rzt,p.pos==="RB"?"RZ Carries / G":"RZ Targets / G")}
   ${statCard(p.pos==="RB"?p.m.gl:p.m.ashare,p.pos==="RB"?"Goal-Line Carries / G":"Air Yard Share %")}
  </div>
 </div>
 <div class="card" style="margin-top:14px"><h3>GAME LOG • CUSTOM VIEW</h3>
 <div class="custom" id="customLine">Official Mania: <b>${p.mania}</b>. Uncheck a game to inspect the selected-game production; official rankings never change.</div>
 <div class="tablewrap"><table><thead><tr><th>Use</th><th>Game</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>Snap</th><th>PPR</th></tr></thead><tbody>${logs}</tbody></table></div></div>`;

 $("backRanks").onclick=()=>showView("rankings");
 $("profileBody").querySelectorAll(".roomrow b[data-id]").forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));
 $("profileBody").querySelectorAll(".gameToggle").forEach(x=>x.onchange=()=>customProfile(p));
}
function customProfile(p){
 const checked=[...document.querySelectorAll(".gameToggle:checked")].map(x=>p.logs[Number(x.dataset.i)]);
 if(!checked.length){$("customLine").innerHTML=`Official Mania: <b>${p.mania}</b>. Select at least one game.`;return}
 const avg=k=>checked.reduce((s,x)=>s+Number(x[k]||0),0)/checked.length;
 $("customLine").innerHTML=`Official Mania: <b>${p.mania}</b> • Selected ${checked.length} game${checked.length===1?"":"s"}: <b>${fmt(avg("ppr"))} PPR/G</b>, ${fmt(avg("tgt"))} targets/G, ${fmt(avg("car"))} carries/G, ${fmt(avg("snap"))}% snaps. <span class="muted">This sandbox does not alter official Mania.</span>`;
}

function renderCompare(){
 if(!A||!B)return;
 const better=A.mania>B.mania?A:B, starter=A.start>B.start?A:B;
 const rows=[
  ["Mania Rating",A.mania,B.mania],["Week Start",A.start,B.start],["PPR/G",A.m.ppr,B.m.ppr],
  ["Targets/G",A.m.tgt,B.m.tgt],["Receptions/G",A.m.rec,B.m.rec],["Scrimmage Yds/G",A.m.scrim,B.m.scrim],
  ["Snap %",A.m.snap,B.m.snap]
 ];
 $("compareBody").innerHTML=`<div class="versus">
 <div class="card cmpHero"><h2>${A.name}</h2><div class="muted">${A.team} • ${A.pos}</div><div class="cmpBig blue">${A.mania}</div><div>MANIA</div><div class="cmpBig green" style="font-size:30px;margin-top:10px">${A.start}</div><div>WEEK ${META.next_week} START</div></div>
 <div class="vs">VS</div>
 <div class="card cmpHero"><h2>${B.name}</h2><div class="muted">${B.team} • ${B.pos}</div><div class="cmpBig blue">${B.mania}</div><div>MANIA</div><div class="cmpBig green" style="font-size:30px;margin-top:10px">${B.start}</div><div>WEEK ${META.next_week} START</div></div></div>
 <div class="two" style="margin-top:14px"><div class="card"><h3>WHO'S BETTER OVERALL?</h3><h2>${better.name}</h2><p class="why">${better.mania} Mania Rating. This is the season profile, independent of this week's matchup.</p></div>
 <div class="card"><h3>WHO SHOULD I START — WEEK ${META.next_week}?</h3><h2>${starter.name}</h2><p class="why">${starter.start} Start Rating. Matchup: ${starter.matchup_adj>=0?"+":""}${starter.matchup_adj}; recent role: ${starter.trend_adj>=0?"+":""}${starter.trend_adj}.</p></div></div>
 <div class="card" style="margin-top:14px"><div class="tablewrap"><table><thead><tr><th>${A.name}</th><th style="text-align:center">Metric</th><th>${B.name}</th></tr></thead><tbody>
 ${rows.map(([k,a,b])=>`<tr><td class="${a>b?"good":""}">${a}</td><td style="text-align:center">${k}</td><td class="${b>a?"good":""}">${b}</td></tr>`).join("")}
 </tbody></table></div></div>`;
}
</script>
</body></html>'''

html = html.replace("__PAYLOAD__", payload).replace("__META__", meta)

os.makedirs("public", exist_ok=True)
with open("public/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("7. FANTASY MANIA built successfully.")
print(f"   {len(players)} players | Through Week {max_week} | Week {next_week} outlook")
print("   Output: public/index.html")
