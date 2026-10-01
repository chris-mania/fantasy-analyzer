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
    print("   Headshots unavailable; initials fallback will be used:", e)


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
            "rtd": round(float(x.receiving_tds), 2),
            "air": round(float(x.receiving_air_yards), 1),
            "car": int(x.carries),
            "ruy": round(float(x.rushing_yards), 1),
            "rutd": round(float(x.rushing_tds), 2),
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


# Official same-position distributions used by the browser sandbox.
# A custom selection is graded against the same league environment as official Mania.
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
# FRONT END — FANTASY MANIA
# ============================================================
html = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Mania</title>
<style>
:root{--blue:#0866c6;--nav:#0b1f33;--ink:#17212b;--muted:#667483;--line:#dbe2e8;--soft:#f4f7f9;--green:#159447;--lime:#55b83d;--yellow:#d89b12;--orange:#e56d22;--red:#c9362b}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:Arial,Helvetica,sans-serif}button,input{font:inherit}
.topbar{background:var(--nav);color:#fff}.nav{max-width:1240px;margin:auto;height:66px;padding:0 22px;display:flex;align-items:center;justify-content:space-between}
.brand{font-size:22px;font-weight:900;letter-spacing:.3px}.brand span{color:#4da3ff}.links{display:flex;height:100%}.navb{border:0;background:transparent;color:#c8d3df;font-weight:800;padding:0 16px;cursor:pointer;border-bottom:4px solid transparent}.navb.on,.navb:hover{color:#fff;border-bottom-color:#4da3ff}
.wrap{max-width:1240px;margin:auto;padding:28px 22px 60px}.view{display:none}.view.on{display:block}
.kicker{font-size:12px;color:var(--blue);font-weight:900;text-transform:uppercase;letter-spacing:1px}.title{font-size:34px;margin:5px 0 4px}.sub{color:var(--muted);margin:0 0 22px}
.search{position:relative;max-width:720px}.search input{width:100%;padding:14px 16px;border:1px solid #aeb9c4;border-radius:4px;background:white}.dd{display:none;position:absolute;top:49px;left:0;right:0;background:white;border:1px solid var(--line);box-shadow:0 8px 24px #0002;z-index:20}.ddi{padding:12px 14px;border-bottom:1px solid var(--line);cursor:pointer}.ddi:hover{background:var(--soft)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.pill{background:white;border:1px solid #b9c4cf;border-radius:4px;padding:8px 13px;font-weight:800;cursor:pointer}.pill.on{background:var(--nav);border-color:var(--nav);color:white}
.panel{border:1px solid var(--line);border-radius:5px;background:white;margin-top:18px}.ph{padding:14px 16px;border-bottom:1px solid var(--line);font-weight:900}.pad{padding:18px}
.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th{background:#f3f6f8;color:#526171;font-size:11px;text-transform:uppercase;letter-spacing:.4px;cursor:pointer}th,td{padding:12px 11px;border-bottom:1px solid var(--line);text-align:right}th:nth-child(2),td:nth-child(2){text-align:left}.playerlink{font-weight:900;color:#15283b;cursor:pointer}.playerlink:hover{color:var(--blue);text-decoration:underline}.tag{font-size:10px;background:#edf2f6;color:#617181;padding:4px 6px;border-radius:3px;margin-left:6px}
.grade{font-weight:900}.g90{color:#07823d}.g80{color:#2b9a42}.g70{color:#a07900}.g60{color:#c26717}.g0{color:#bb2f2a}
.profileHead{display:grid;grid-template-columns:150px 1fr auto;gap:24px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:22px}.portrait{width:138px;height:138px;border-radius:50%;position:relative;display:grid;place-items:center;background:#edf2f6}.portrait:before{content:"";position:absolute;inset:-6px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#e2e7eb 0);z-index:-2}.portrait:after{content:"";position:absolute;inset:-1px;border-radius:50%;background:white;z-index:-1}.portrait img{width:132px;height:132px;object-fit:cover;border-radius:50%}.initials{font-size:38px;font-weight:900;color:#748291}.identity h1{font-size:38px;margin:0 0 5px}.meta{color:var(--muted);font-weight:700}.rankline{margin-top:10px;font-weight:800}.scorebox{text-align:center;min-width:125px}.scorebig{font-size:48px;font-weight:900;line-height:1}.scorelab{font-size:11px;font-weight:900;color:#6c7884;text-transform:uppercase;margin-top:4px}
.scorepair{display:flex;gap:14px}.scorecard{border-left:1px solid var(--line);padding-left:20px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid var(--line);border-radius:5px;margin-top:18px}.stat{padding:15px;border-right:1px solid var(--line)}.stat:last-child{border:0}.sv{font-size:22px;font-weight:900}.sk{font-size:10px;color:#6d7985;text-transform:uppercase;font-weight:900;margin-top:3px}
.two{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}.sectionTitle{font-size:13px;text-transform:uppercase;letter-spacing:.7px;font-weight:900;margin-bottom:14px}
.barrow{display:grid;grid-template-columns:105px 1fr 48px;gap:12px;align-items:center;margin:12px 0}.track{height:10px;background:#edf1f4;border-radius:2px;overflow:hidden}.fill{height:100%}
.roomrow{display:grid;grid-template-columns:1.6fr .65fr .65fr .65fr;gap:8px;padding:10px 0;border-bottom:1px solid var(--line)}.roomrow b[data-id]{cursor:pointer}.roomrow b[data-id]:hover{color:var(--blue)}
.alert{background:#fff7df;border-left:4px solid var(--yellow);padding:11px 13px;margin-top:15px}.custom{background:#eef6ff;border:1px solid #bad8f6;padding:12px;margin-bottom:12px}.reset{float:right;border:1px solid #9badbd;background:white;border-radius:3px;padding:6px 9px;cursor:pointer;font-weight:800}
.good{color:var(--green);font-weight:900}.bad{color:var(--red);font-weight:900}.muted{color:var(--muted)}.partial{font-size:9px;background:#fff0d9;color:#9b5c00;padding:3px 4px;border-radius:2px}
.versus{display:grid;grid-template-columns:1fr 60px 1fr;align-items:start;gap:10px}.vs{text-align:center;font-size:20px;font-weight:900;padding-top:70px}.cmp{border:1px solid var(--line);padding:18px;border-radius:5px}.cmpTop{display:flex;align-items:center;gap:14px}.miniPic{width:74px;height:74px;border-radius:50%;object-fit:cover;background:#edf2f6}.games{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}.gamechip{border:1px solid #bcc7d1;padding:7px 9px;border-radius:3px}.gamechip input{margin-right:5px}
@media(max-width:850px){.profileHead{grid-template-columns:100px 1fr}.portrait{width:92px;height:92px}.portrait img{width:88px;height:88px}.scorepair{grid-column:1/-1}.stats{grid-template-columns:repeat(3,1fr)}.two,.versus{grid-template-columns:1fr}.vs{padding:0}.links{display:none}}
</style></head><body>
<div class="topbar"><div class="nav"><div class="brand">FANTASY <span>MANIA</span></div><div class="links">
<button class="navb" data-v="home">HOME</button><button class="navb" data-v="players">PLAYERS</button><button class="navb on" data-v="rankings">RANKINGS</button><button class="navb" data-v="compare">COMPARE</button>
</div></div></div>
<main class="wrap">
<section id="home" class="view"><div class="kicker">2026 Fantasy Football</div><h1 class="title">Fantasy Mania</h1><p class="sub">Player value, usage, role and matchup context in one rating.</p><div class="search"><input id="homeQ" placeholder="Search any RB, WR or TE"><div class="dd" id="homeDD"></div></div><div id="leaders"></div></section>
<section id="players" class="view"><h1 class="title">Players</h1><p class="sub">Search a player to open the full Mania profile.</p><div class="search"><input id="playerQ" placeholder="Search player"><div class="dd" id="playerDD"></div></div></section>
<section id="rankings" class="view on"><div class="kicker">Through Week <span id="wk"></span></div><h1 class="title" id="rankTitle">Overall Rankings</h1><p class="sub" id="rankSub">Mania Rating: complete fantasy value, not name value.</p><div class="toolbar"><button class="pill rankmode on" data-mode="mania">MANIA</button><button class="pill rankmode" data-mode="start">WEEK START</button><span style="width:10px"></span><button class="pill posf on" data-pos="ALL">ALL</button><button class="pill posf" data-pos="RB">RB</button><button class="pill posf" data-pos="WR">WR</button><button class="pill posf" data-pos="TE">TE</button></div><div class="panel tablewrap"><table><thead id="rankHead"></thead><tbody id="rankBody"></tbody></table></div></section>
<section id="profile" class="view"><div id="profileBody"></div></section>
<section id="compare" class="view"><h1 class="title">Compare Players</h1><p class="sub">Overall value and this week's start decision. Remove any game from either player to run your own scenario.</p><div class="two"><div class="search"><input id="aQ" placeholder="Player A"><div class="dd" id="aDD"></div></div><div class="search"><input id="bQ" placeholder="Player B"><div class="dd" id="bDD"></div></div></div><div id="compareBody"></div></section>
</main>
<script>
const DB=__PAYLOAD__, META=__META__, REFS=__REFS__;
const $=id=>document.getElementById(id), byId=id=>DB.find(p=>p.id===id), fmt=n=>Math.round(Number(n)*10)/10;
$("wk").textContent=META.week;
let posFilter="ALL",rankMode="mania",sortKey="mania",sortDir=-1,A=null,B=null,selA=null,selB=null;
function grade(v){return v>=90?"g90":v>=80?"g80":v>=70?"g70":v>=60?"g60":"g0"} function ring(v){return v>=85?"#159447":v>=72?"#d89b12":v>=60?"#e56d22":"#c9362b"}
function tier(v){return v>=90?"Elite":v>=82?"Excellent":v>=74?"Strong":v>=65?"Starter":"Depth"}
function showView(v){document.querySelectorAll(".view").forEach(x=>x.classList.toggle("on",x.id===v));document.querySelectorAll(".navb").forEach(x=>x.classList.toggle("on",x.dataset.v===v));window.scrollTo(0,0)}
document.querySelectorAll(".navb").forEach(b=>b.onclick=()=>showView(b.dataset.v));
function initials(p){return p.name.split(" ").slice(0,2).map(x=>x[0]).join("")}
function pic(p,cls=""){return p.headshot?`<img class="${cls}" src="${p.headshot}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><span class="initials" style="display:none">${initials(p)}</span>`:`<span class="initials">${initials(p)}</span>`}
function searchBox(inp,dd,cb){$(inp).oninput=e=>{let q=e.target.value.toLowerCase().trim(),d=$(dd);if(!q){d.style.display="none";return}let m=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);d.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><b>${p.name}</b><small>${p.team} • ${p.pos} • Mania ${p.mania}</small></div>`).join("");d.style.display=m.length?"block":"none";d.querySelectorAll(".ddi").forEach(x=>x.onclick=()=>{d.style.display="none";$(inp).value=byId(x.dataset.id).name;cb(byId(x.dataset.id))})}}
searchBox("homeQ","homeDD",p=>openPlayer(p));searchBox("playerQ","playerDD",p=>openPlayer(p));searchBox("aQ","aDD",p=>{A=p;selA=p.logs.map((_,i)=>i);renderCompare()});searchBox("bQ","bDD",p=>{B=p;selB=p.logs.map((_,i)=>i);renderCompare()});

const COLS=[["rank","#"],["name","Player"],["mania","Mania"],["start","Start"],["m.ppr","PPR/G"],["m.tgt","TGT/G"],["m.rec","REC/G"],["m.recy","REC YD/G"],["m.car","CAR/G"],["m.rushy","RUSH YD/G"],["m.snap","SNAP%"],["share","SHARE"],["rz","RZ"]];
function val(p,k){if(k==="rank")return rankMode==="mania"?p.rank:p.start_rank;if(k==="name")return p.name;if(k==="share")return p.pos==="RB"?p.m.touch:p.m.tshare;if(k==="rz")return p.pos==="RB"?p.m.rzc:p.m.rzt;if(k.startsWith("m."))return p.m[k.slice(2)];return p[k]}
function renderRanks(){let arr=DB.filter(p=>posFilter==="ALL"||p.pos===posFilter);arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);return typeof x==="string"?sortDir*x.localeCompare(y):sortDir*(x-y)});$("rankTitle").textContent=rankMode==="mania"?"Overall Rankings — Mania Rating":`Week ${META.next_week} Start Rankings`;$("rankSub").textContent=rankMode==="mania"?"Who has the strongest complete fantasy profile? Click any column to sort.":"Matchup-aware weekly rating. Season Mania remains the anchor.";$("rankHead").innerHTML="<tr>"+COLS.map(([k,l])=>`<th data-k="${k}">${l}${sortKey===k?(sortDir<0?" ↓":" ↑"):""}</th>`).join("")+"</tr>";$("rankBody").innerHTML=arr.map(p=>`<tr><td>${val(p,"rank")}</td><td><span class="playerlink" data-id="${p.id}">${p.name}</span><span class="tag">${p.team} ${p.pos}</span></td><td class="grade ${grade(p.mania)}">${p.mania}</td><td class="grade ${grade(p.start)}">${p.start}</td><td>${p.m.ppr}</td><td>${p.m.tgt}</td><td>${p.m.rec}</td><td>${p.m.recy}</td><td>${p.m.car}</td><td>${p.m.rushy}</td><td>${p.m.snap}%</td><td>${val(p,"share")}%</td><td>${val(p,"rz")}</td></tr>`).join("");$("rankHead").querySelectorAll("th").forEach(th=>th.onclick=()=>{let k=th.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==="name"?1:-1}renderRanks()});$("rankBody").querySelectorAll(".playerlink").forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)))}
document.querySelectorAll(".rankmode").forEach(b=>b.onclick=()=>{rankMode=b.dataset.mode;sortKey=rankMode;sortDir=-1;document.querySelectorAll(".rankmode").forEach(x=>x.classList.toggle("on",x===b));renderRanks()});document.querySelectorAll(".posf").forEach(b=>b.onclick=()=>{posFilter=b.dataset.pos;document.querySelectorAll(".posf").forEach(x=>x.classList.toggle("on",x===b));renderRanks()});renderRanks();

function percentile(pos,key,v){let a=(REFS[pos]||{})[key]||[];if(!a.length)return 0;let lo=0,hi=a.length;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<=v)lo=m+1;else hi=m}return Math.min(100,lo/a.length*100)}
function avg(a,k,w=true){let den=0,num=0;a.forEach(x=>{let z=w?(x.weight||1):1;num+=Number(x[k]||0)*z;den+=z});return den?num/den:0}
function sum(a,k){return a.reduce((s,x)=>s+Number(x[k]||0)*(x.weight||1),0)}
function calc(p,idxs){
 let L=idxs.map(i=>p.logs[i]).filter(Boolean);if(!L.length)return null;
 let M={};M.ppr_pg=avg(L,"ppr");M.targets_pg=avg(L,"tgt");M.rec_pg=avg(L,"rec");M.rec_yards_pg=avg(L,"ry");M.carries_pg=avg(L,"car");M.rush_yards_pg=avg(L,"ruy");M.scrim_pg=M.rec_yards_pg+M.rush_yards_pg;M.td_pg=avg(L,"rtd")+avg(L,"rutd");M.snap_pct=avg(L,"snap");M.target_share=avg(L,"tshare");M.air_share=avg(L,"ashare");M.touch_share=avg(L,"touch");M.rz_targets_pg=avg(L,"rzt");M.endzone_targets_pg=avg(L,"ez");M.rz_carries_pg=avg(L,"rzc");M.gl_carries_pg=avg(L,"gl");
 M.yards_per_target=sum(L,"tgt")?sum(L,"ry")/sum(L,"tgt"):0;M.catch_rate=sum(L,"tgt")?sum(L,"rec")/sum(L,"tgt")*100:0;M.yards_per_carry=sum(L,"car")?sum(L,"ruy")/sum(L,"car"):0;let opp=sum(L,"car")+2*sum(L,"tgt");M.fp_per_weighted_opp=opp?sum(L,"ppr")/opp:0;
 let P={};Object.keys(M).forEach(k=>P[k]=percentile(p.pos,k,M[k]));
 let prod,op,role,hv,eff;if(p.pos==="RB"){prod=.55*P.ppr_pg+.30*P.scrim_pg+.15*P.td_pg;op=.45*P.carries_pg+.35*P.targets_pg+.20*P.rec_pg;role=.60*P.touch_share+.40*P.snap_pct;hv=.40*P.rz_carries_pg+.35*P.gl_carries_pg+.25*P.rz_targets_pg;eff=.55*P.fp_per_weighted_opp+.25*P.yards_per_carry+.20*P.catch_rate}else{prod=.55*P.ppr_pg+.30*P.rec_yards_pg+.15*P.td_pg;op=.50*P.targets_pg+.30*P.rec_pg+.20*P.target_share;role=.45*P.target_share+.30*P.air_share+.25*P.snap_pct;hv=.60*P.rz_targets_pg+.40*P.endzone_targets_pg;eff=.45*P.yards_per_target+.30*P.catch_rate+.25*P.fp_per_weighted_opp}
 let W=p.pos==="RB"?[.30,.30,.20,.125,.075]:[.30,.30,.225,.10,.075],raw=W[0]*prod+W[1]*op+W[2]*role+W[3]*hv+W[4]*eff,base=45+.52*raw,eg=L.reduce((s,x)=>s+(x.weight||1),0),conf=Math.min(1,eg/Math.max(META.week,1)),sh=(1-conf)*.12,mania=Math.max(0,Math.min(99.5,base*(1-sh)+72*sh));
 let recent=L.slice(-Math.min(2,L.length)),recentOpp=recent.reduce((s,x)=>s+2*x.tgt+x.car,0)/recent.length,seasonOpp=M.targets_pg*2+M.carries_pg,tr=seasonOpp?((recentOpp/seasonOpp)-1)*100:0,trend=Math.max(-2.5,Math.min(2.5,tr/20*2.5));if(L.length<2)trend*=.25;let ctx=p.matchup_adj+trend;if(p.pos==="TE")ctx*=.35;let start=Math.max(0,Math.min(100,mania+ctx));
 return {mania,start,M,b:{Production:prod,Opportunity:op,Role:role,"High Value":hv,Efficiency:eff}};
}
function barColor(v){return v>=85?"#159447":v>=70?"#d89b12":v>=55?"#e56d22":"#c9362b"}
function stat(v,k){return `<div class="stat"><div class="sv">${v}</div><div class="sk">${k}</div></div>`}
function openPlayer(p){showView("profile");let share=p.pos==="RB"?p.m.touch:p.m.tshare,shareName=p.pos==="RB"?"Touch Share":"Target Share",room=p.room.map((x,i)=>`<div class="roomrow"><b data-id="${x.id}">${i+1}. ${x.name}</b><span class="grade ${grade(x.mania)}">${x.mania}</span><span>${x.opp}</span><span>${x.share}%</span></div>`).join(""),bars=Object.entries(p.buckets).map(([k,v])=>`<div class="barrow"><span>${k}</span><div class="track"><div class="fill" style="width:${v}%;background:${barColor(v)}"></div></div><b class="${grade(v)}">${v}</b></div>`).join(""),logs=p.logs.map((l,i)=>`<tr><td><input class="gameToggle" type="checkbox" checked data-i="${i}"></td><td>W${l.w} ${l.partial?'<span class="partial">SHORT</span>':""}</td><td>${l.tgt}</td><td>${l.rec}</td><td>${l.ry}</td><td>${l.car}</td><td>${l.ruy}</td><td>${l.snap}%</td><td>${l.ppr}</td></tr>`).join("");
$("profileBody").innerHTML=`<div class="kicker playerlink" id="backRanks">← Rankings</div><div class="profileHead"><div class="portrait" id="portrait" style="--score:${p.mania};--ring:${ring(p.mania)}">${pic(p)}</div><div class="identity"><h1>${p.name}</h1><div class="meta">${p.team} • ${p.pos} • ${p.games} games</div><div class="rankline">#${p.rank} overall &nbsp; • &nbsp; ${p.pos}${p.pos_rank} &nbsp; • &nbsp; #${p.team_rank} ${p.pos} on ${p.team}</div></div><div class="scorepair"><div class="scorebox"><div id="maniaTop" class="scorebig ${grade(p.mania)}">${p.mania}</div><div class="scorelab">Mania Rating<br>${tier(p.mania)}</div></div><div class="scorecard scorebox"><div id="startTop" class="scorebig ${grade(p.start)}">${p.start}</div><div class="scorelab">Week ${META.next_week} Start<br>vs ${p.opp}</div></div></div></div>
${p.confidence!=="High"?`<div class="alert">${p.confidence} sample confidence. Missed games are excluded and detected shortened games are downweighted automatically.</div>`:""}
<div class="stats">${stat(p.m.ppr,"PPR / Game")}${stat(p.m.tgt,"Targets / Game")}${stat(p.m.rec,"Receptions / Game")}${stat(p.m.scrim,"Scrimmage Yds / G")}${stat(share+"%",shareName)}${stat(p.m.snap+"%","Snap Share")}</div>
<div class="two"><div class="panel"><div class="ph">MANIA PROFILE</div><div class="pad" id="profileBars">${bars}<p class="sub" style="margin:14px 0 0">Position-specific. WR/TE carries have zero direct rating weight. RB receiving work gets extra PPR value.</p></div></div><div class="panel"><div class="ph">${p.team} ${p.pos} ROOM</div><div class="pad"><div class="roomrow muted"><span>Player</span><span>Mania</span><span>${p.pos==="RB"?"Opp/G":"Tgt/G"}</span><span>Share</span></div>${room}</div></div></div>
<div class="two"><div class="panel"><div class="ph">WEEK ${META.next_week} MATCHUP • ${p.opp}</div><div class="pad"><div class="${p.matchup_adj>=0?"good":"bad"}" style="font-size:28px">${p.matchup_adj>=0?"+":""}${p.matchup_adj}</div><p class="sub" style="margin:6px 0">Built from how similar ${p.pos}s performed against ${p.opp}, with early-season damping. TE matchup movement is intentionally more conservative.</p>${p.similar.map(x=>`<div class="roomrow"><b>${x.name}</b><span>${x.sim}% match</span><span>${x.normal}</span><span>${x.actual}</span></div>`).join("")||'<span class="muted">Not enough comparable-player evidence yet.</span>'}</div></div><div class="panel"><div class="ph">ADVANCED USAGE</div><div class="pad">${stat(p.pos==="RB"?p.m.car:p.m.recy,p.pos==="RB"?"Carries / G":"Receiving Yds / G")}${stat(p.pos==="RB"?p.m.rzc:p.m.rzt,p.pos==="RB"?"RZ Carries / G":"RZ Targets / G")}${stat(p.pos==="RB"?p.m.gl:p.m.ashare,p.pos==="RB"?"Goal-Line Carries / G":"Air Yard Share %")}</div></div></div>
<div class="panel"><div class="ph">GAME LOG • CUSTOM MANIA <button class="reset" id="resetGames">RESET TO OFFICIAL</button></div><div class="pad"><div class="custom" id="customLine">Official Mania <b>${p.mania}</b>. Uncheck any game and the full Mania + Start math recalculates.</div><div class="tablewrap"><table><thead><tr><th>Use</th><th>Game</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>Snap</th><th>PPR</th></tr></thead><tbody>${logs}</tbody></table></div></div></div>`;
$("backRanks").onclick=()=>showView("rankings");$("profileBody").querySelectorAll(".roomrow b[data-id]").forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));$("profileBody").querySelectorAll(".gameToggle").forEach(x=>x.onchange=()=>customProfile(p));$("resetGames").onclick=()=>{$("profileBody").querySelectorAll(".gameToggle").forEach(x=>x.checked=true);customProfile(p)}}
function customProfile(p){let idx=[...document.querySelectorAll(".gameToggle:checked")].map(x=>Number(x.dataset.i)),c=calc(p,idx);if(!c){$("customLine").innerHTML="Select at least one game.";return}$("maniaTop").textContent=fmt(c.mania);$("maniaTop").className="scorebig "+grade(c.mania);$("startTop").textContent=fmt(c.start);$("startTop").className="scorebig "+grade(c.start);$("portrait").style.setProperty("--score",c.mania);$("portrait").style.setProperty("--ring",ring(c.mania));$("profileBars").innerHTML=Object.entries(c.b).map(([k,v])=>`<div class="barrow"><span>${k}</span><div class="track"><div class="fill" style="width:${v}%;background:${barColor(v)}"></div></div><b class="${grade(v)}">${fmt(v)}</b></div>`).join("")+`<p class="sub" style="margin:14px 0 0">Custom sandbox — official rankings stay unchanged.</p>`;$("customLine").innerHTML=`Official <b>${p.mania}</b> → Custom Mania <b class="${grade(c.mania)}">${fmt(c.mania)}</b> • Custom Start <b class="${grade(c.start)}">${fmt(c.start)}</b> • ${idx.length} selected game${idx.length===1?"":"s"} • ${fmt(c.M.ppr_pg)} PPR/G • ${fmt(c.M.targets_pg)} TGT/G • ${fmt(c.M.snap_pct)}% snaps.`}

function chips(p,which,sel){return `<div class="games">${p.logs.map((l,i)=>`<label class="gamechip"><input type="checkbox" class="cmpGame" data-side="${which}" data-i="${i}" ${sel.includes(i)?"checked":""}>W${l.w}${l.partial?"*":""}</label>`).join("")}</div>`}
function renderCompare(){if(!A||!B)return;if(!selA)selA=A.logs.map((_,i)=>i);if(!selB)selB=B.logs.map((_,i)=>i);let ca=calc(A,selA),cb=calc(B,selB);if(!ca||!cb)return;let better=ca.mania>=cb.mania?A:B,starter=ca.start>=cb.start?A:B,rows=[["Mania",ca.mania,cb.mania],["Week Start",ca.start,cb.start],["PPR/G",ca.M.ppr_pg,cb.M.ppr_pg],["Targets/G",ca.M.targets_pg,cb.M.targets_pg],["Receptions/G",ca.M.rec_pg,cb.M.rec_pg],["Scrimmage Yds/G",ca.M.scrim_pg,cb.M.scrim_pg],["Snap %",ca.M.snap_pct,cb.M.snap_pct]];
$("compareBody").innerHTML=`<div class="versus" style="margin-top:22px"><div class="cmp"><div class="cmpTop">${A.headshot?`<img class="miniPic" src="${A.headshot}">`:""}<div><h2>${A.name}</h2><div class="muted">${A.team} • ${A.pos}</div></div></div><div class="scorebig ${grade(ca.mania)}">${fmt(ca.mania)}</div><b>Custom Mania</b>${chips(A,"A",selA)}</div><div class="vs">VS</div><div class="cmp"><div class="cmpTop">${B.headshot?`<img class="miniPic" src="${B.headshot}">`:""}<div><h2>${B.name}</h2><div class="muted">${B.team} • ${B.pos}</div></div></div><div class="scorebig ${grade(cb.mania)}">${fmt(cb.mania)}</div><b>Custom Mania</b>${chips(B,"B",selB)}</div></div><div class="two"><div class="panel"><div class="ph">OVERALL PROFILE</div><div class="pad"><h2>${better.name}</h2><span class="muted">Based on the selected games' complete Mania profile.</span></div></div><div class="panel"><div class="ph">WHO SHOULD I START • WEEK ${META.next_week}</div><div class="pad"><h2>${starter.name}</h2><span class="muted">Custom Mania plus matchup and recent-role context.</span></div></div></div><div class="panel tablewrap"><table><thead><tr><th>${A.name}</th><th style="text-align:center">Metric</th><th>${B.name}</th></tr></thead><tbody>${rows.map(([k,a,b])=>`<tr><td class="${a>b?"good":""}">${fmt(a)}</td><td style="text-align:center">${k}</td><td class="${b>a?"good":""}">${fmt(b)}</td></tr>`).join("")}</tbody></table></div>`;
document.querySelectorAll(".cmpGame").forEach(x=>x.onchange=()=>{let side=x.dataset.side,i=Number(x.dataset.i),arr=side==="A"?selA:selB;if(x.checked){if(!arr.includes(i))arr.push(i)}else if(arr.length>1){arr.splice(arr.indexOf(i),1)}arr.sort((a,b)=>a-b);renderCompare()})}
showView("rankings");
</script></body></html>"""

html = html.replace("__PAYLOAD__", payload).replace("__META__", meta).replace("__REFS__", refs_json)
os.makedirs("public", exist_ok=True)
with open("public/index.html","w",encoding="utf-8") as f:f.write(html)
print("7. FANTASY MANIA built successfully.")
print(f"   {len(players)} players | Through Week {max_week} | Week {next_week} outlook")
print("   Output: public/index.html")
