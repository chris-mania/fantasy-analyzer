import json
import math
import os
import subprocess
import sys
import warnings
from urllib.error import HTTPError

import numpy as np
import re
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# FANTASY MANIA — 2026
# One-file autonomous builder for GitHub Pages
# ============================================================

SEASON = 2026
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Report card: the build re-runs this script with MANIA_AS_OF_WEEK set. That run uses only the games through
# that week, prints the start ratings it would have published, and stops before writing the site.
AS_OF_WEEK = int(os.environ.get("MANIA_AS_OF_WEEK", "0"))

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

# ---- Only count a week once EVERY game that week has a final score ----
# (so a Thursday-night re-run can never publish a half-finished week as "through Week N")
try:
    _g = None
    for _u in (f"{BASE}/schedules/games.csv", f"{BASE}/schedules/games.csv.gz"):
        try:
            _g = pd.read_csv(_u, low_memory=False)
            break
        except Exception:
            pass
    if _g is not None and {"week", "home_score", "away_score"}.issubset(_g.columns):
        if "season" in _g.columns:
            _g = _g[safe_num(_g["season"]).astype(int) == SEASON]
        if "game_type" in _g.columns:
            _g = _g[_g["game_type"].astype(str).eq("REG")]
        _g = _g.copy()
        _g["week"] = safe_num(_g["week"]).astype(int)
        _last_done = 0
        for _w in sorted(_g["week"].unique()):
            _wk = _g[_g["week"] == _w]
            if _wk["home_score"].notna().all() and _wk["away_score"].notna().all():
                _last_done = int(_w)
            else:
                break
        if _last_done > 0:
            stats = stats[safe_num(stats["week"]).astype(int) <= _last_done].copy()
            snaps = snaps[safe_num(snaps["week"]).astype(int) <= _last_done].copy()
            print(f"   Schedule says Weeks 1-{_last_done} are complete.")
except Exception as e:
    print("   Could not check schedule for completed weeks; using all stats:", e)

if AS_OF_WEEK:
    stats = stats[safe_num(stats["week"]).astype(int) <= AS_OF_WEEK].copy()
    snaps = snaps[safe_num(snaps["week"]).astype(int) <= AS_OF_WEEK].copy()

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
    def _nk(x):
        x = re.sub(r"[.'\u2019`-]", "", str(x).lower())
        x = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", x)
        return re.sub(r"[^a-z]", "", x)
    sn["_nk"] = sn["snap_name"].map(_nk)
    sn = sn.drop_duplicates(["_nk", "team", "week"])
    s["_nk"] = s["player_name"].map(_nk)
    merged = s.merge(
        sn.drop(columns=["snap_name"]), on=["_nk", "team", "week"], how="left"
    )
    # Fallback for nickname mismatches (Kenny/Kenneth): same last name, team, week and position.
    def _ln(x):
        parts = re.sub(r"[.'\u2019`]", "", str(x).lower()).replace("-", " ").split()
        parts = [p for p in parts if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
        return re.sub(r"[^a-z]", "", parts[-1]) if parts else ""
    _sn2 = snaps[[snap_name, "team", "week", "position", "offense_snaps", "offense_pct"]].copy()
    _sn2["_ln"] = _sn2[snap_name].map(_ln)
    _sn2["week"] = safe_num(_sn2["week"]).astype(int)
    _cnt = _sn2.groupby(["_ln", "team", "week", "position"])[snap_name].transform("count")
    _sn2 = _sn2[_cnt == 1].set_index(["_ln", "team", "week", "position"])
    _na = merged["offense_pct"].isna()
    for _i in merged.index[_na]:
        _k = (_ln(merged.at[_i, "player_name"]), merged.at[_i, "team"], merged.at[_i, "week"], merged.at[_i, "position"])
        if _k in _sn2.index:
            merged.at[_i, "offense_pct"] = _sn2.at[_k, "offense_pct"]
            merged.at[_i, "offense_snaps"] = _sn2.at[_k, "offense_snaps"]
    _miss = merged[merged["offense_pct"].isna() & ((merged["targets"] + merged["carries"]) >= 3)]
    print(f"   Snap match: {len(_miss)} meaningful player-games still unmatched.")
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
        "receiver_player_id", "rusher_player_id", "air_yards", "two_point_attempt"
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
    pbp = pbp[(pbp["week"] <= max_week) & (pbp["yardline_100"] > 0)].copy()
    if "two_point_attempt" in pbp.columns:
        pbp = pbp[safe_num(pbp["two_point_attempt"]) != 1]

    # Receiving: every pass keeps its field position. Red-zone = snap inside the 20.
    # End-zone target = the pass was thrown to/through the goal line (air yards >= yards to goal),
    # counted from ANYWHERE on the field, so deep shots into the end zone count too.
    rec = pbp[(safe_num(pbp["pass_attempt"]) == 1) & pbp["receiver_player_id"].notna()].copy()
    rec["player_id"] = rec["receiver_player_id"]
    rec["rz_targets"] = (rec["yardline_100"] <= 20).astype(int)
    _air = safe_num(rec["air_yards"]) if "air_yards" in rec.columns else pd.Series(0, index=rec.index)
    rec["endzone_targets"] = (_air >= rec["yardline_100"]).astype(int)

    rus = pbp[(safe_num(pbp["rush_attempt"]) == 1) & pbp["rusher_player_id"].notna()].copy()
    rus["player_id"] = rus["rusher_player_id"]
    rus["rz_carries"] = (rus["yardline_100"] <= 20).astype(int)
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
merged["auto_excl"] = False

# Did play-by-play say he was hurt in that game, and how far into the game was it?
inj_game = {}   # (player_id, week) -> fraction of the game already played when he was hurt
try:
    _pb = pd.read_csv(f"{BASE}/pbp/play_by_play_{SEASON}.csv.gz", usecols=lambda c: c in ("season_type", "week", "desc", "game_seconds_remaining"), low_memory=False)
    if "season_type" in _pb.columns:
        _pb = _pb[_pb["season_type"].astype(str).eq("REG")]
    _pb = _pb[_pb["desc"].astype(str).str.contains("was injured during the play", na=False)]
    _look = {}
    for _r in merged[["player_id", "player_name", "team", "week"]].itertuples():
        _nm = re.sub(r"[.'\u2019`]", "", str(_r.player_name).lower()).replace("-", " ").split()
        _nm = [p_ for p_ in _nm if p_ not in ("jr", "sr", "ii", "iii", "iv", "v")]
        if len(_nm) >= 2:
            _look[(_r.team, int(_r.week), _nm[0][0], re.sub(r"[^a-z]", "", _nm[-1]))] = _r.player_id
    for _r in _pb.itertuples():
        for _m in re.finditer(r"([A-Z]{2,3})-\d+-([A-Za-z]+)\.([A-Za-z'\-]+) was injured during the play", str(_r.desc)):
            _k = (_m.group(1), int(_r.week), _m.group(2)[0].lower(), re.sub(r"[^a-z]", "", _m.group(3).lower()))
            if _k in _look and pd.notna(_r.game_seconds_remaining):
                _el = 1 - float(_r.game_seconds_remaining) / 3600.0
                _key = (_look[_k], int(_r.week))
                inj_game[_key] = min(inj_game.get(_key, 9), _el)
    print(f"   Found {len(inj_game)} in-game injuries from play-by-play.")
except Exception as e:
    print("   In-game injury detection unavailable:", e)

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
        # Low snaps alone is only an outlier. It is discounted fully only if play-by-play
        # confirms an in-game injury before the 4th quarter was nearly over.
        wks_ = merged.loc[ix, "week"].astype(int).values
        el_ = np.array([inj_game.get((pid, w_), -1.0) for w_ in wks_])
        hurt_early = is_partial & (el_ >= 0) & (el_ < 0.85)
        # Hurt with at least half the game still to play (or barely played): leave the game out entirely.
        major = hurt_early & ((el_ <= 0.60) | (ratio < 0.5))
        late_hurt = is_partial & (el_ >= 0.85)
        unexplained = is_partial & (el_ < 0)
        merged.loc[ix, "partial"] = hurt_early
        weights = np.where(major, 0.0, np.where(hurt_early, np.clip(ratio, MIN_PARTIAL_WEIGHT, 0.60),
                  np.where(unexplained, np.clip(0.5 + 0.5 * ratio, 0.7, 1.0),
                  np.where(late_hurt, 1.0, 1.0))))
        if weights.max() <= 0:      # never throw out every game he has
            major = np.zeros(len(weights), dtype=bool)
            weights = np.where(hurt_early, np.clip(ratio, MIN_PARTIAL_WEIGHT, 0.60), 1.0)
        merged.loc[ix, "auto_excl"] = major
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
    _dd = d[d["game_weight"] > 0]
    recent = (_dd if len(_dd) else d).tail(min(2, games))
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
# Tight ends are ranked against a small, top-heavy pool, so mid-tier TEs looked as good as
# top-tier RBs/WRs. Keep elite TEs (93.6+) as-is and spread the rest out so a rating
# means roughly the same thing across positions.
TE_PIVOT, TE_SPREAD = 93.6, 1.5
_te = g["position"] == "TE"
g.loc[_te, "mania_base"] = np.where(
    g.loc[_te, "mania_base"] >= TE_PIVOT,
    g.loc[_te, "mania_base"],
    np.maximum(35.0, TE_PIVOT - (TE_PIVOT - g.loc[_te, "mania_base"]) * TE_SPREAD),
)

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
week_has_games = False
if schedule_ok and {"week", "home_team", "away_team"}.issubset(games.columns):
    nw = games[safe_num(games["week"]).astype(int) == next_week]
    week_has_games = len(nw) > 0
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

    # Early season: defensive evidence is deliberately weak.
    evidence_games = len(weights)
    maturity = min(1.0, max_week / 8.0)
    evidence = min(1.0, evidence_games / 6.0)
    # +/-25% opponent effect maps to +/-6 rating points at full maturity/evidence.
    adj = np.clip(effect_pct / 25.0 * MATCHUP_CAP, -MATCHUP_CAP, MATCHUP_CAP)
    adj *= maturity * (0.45 + 0.55 * evidence)

    examples = sorted(examples, key=lambda x: x["sim"], reverse=True)[:12]
    return float(adj), examples

# ============================================================
# INJURIES (official NFL injury report via nflverse) -> Week rating
# ============================================================
print("5b. Applying the latest injury report...")
inj_by_pid = {}      # player_id -> info for the player himself
inj_boost = {}       # player_id -> total rating change from teammates' injuries
inj_why = {}         # player_id -> list of teammates driving that change
depth_info = {}
inj_repl = {}
inj_has_ev = set()
inj_meta = {"ok": False, "updated": "", "n": 0, "out": 0}

def _avail(status, practice):
    st = str(status).strip().lower() if pd.notna(status) else ""
    pr = str(practice).strip().lower() if pd.notna(practice) else ""
    if st == "out":
        return 0.0, "OUT"
    if st == "doubtful":
        return 0.1, "Doubtful"
    if st == "questionable":
        return 0.6, "Questionable"
    if "did not" in pr:
        return 0.7, "Missed practice"
    if "limited" in pr:
        return 0.95, "Limited in practice"
    return 1.0, ""

if not AS_OF_WEEK:
    try:
        inj = pd.read_csv(f"{BASE}/injuries/injuries_{SEASON}.csv", low_memory=False)
        inj = inj[safe_num(inj["week"]).astype(int) == next_week].copy()
        if "game_type" in inj.columns:
            inj = inj[inj["game_type"].astype(str).eq("REG")]
        inj = inj.drop_duplicates(["gsis_id"], keep="last")
        gi = g.set_index("player_id")
        # Starting QB per team = most pass attempts so far.
        qb_starter = {}
        if "attempts" in stats.columns:
            qbs = stats[stats["position"] == "QB"].copy()
            qbs["attempts"] = safe_num(qbs["attempts"])
            qbs = qbs.groupby(["team" if "team" in qbs.columns else "recent_team", "player_id"])["attempts"].sum().reset_index()
            qbs.columns = ["team", "player_id", "attempts"]
            for t_, grp in qbs.groupby("team"):
                top = grp.sort_values("attempts", ascending=False).iloc[0]
                qb_starter[t_] = top["player_id"]
        qb_avail = {}
        for _, ir in inj.iterrows():
            a, label = _avail(ir.get("report_status"), ir.get("practice_status"))
            pid = ir.get("gsis_id")
            note = ir.get("report_primary_injury") if pd.notna(ir.get("report_primary_injury")) else ir.get("practice_primary_injury")
            note = str(note) if pd.notna(note) else ""
            if ir.get("position") == "QB":
                if qb_starter.get(ir.get("team")) == pid:
                    qb_avail[ir.get("team")] = (a, label, str(ir.get("full_name")))
                continue
            if pid in gi.index and a < 1.0:
                inj_by_pid[pid] = {"avail": a, "label": label, "note": note}

        for _nm in [x.strip() for x in os.environ.get("MANIA_FORCE_OUT", "").split(",") if x.strip()]:   # test hook
            for _pid in g.loc[g["player_name"] == _nm, "player_id"]:
                inj_by_pid[_pid] = {"avail": 0.0, "label": "OUT", "note": ""}
        # ---- Roster status: injured reserve / inactive / released players are out ----
        ROSTER_LABEL = {"RES": "Injured reserve", "INA": "Inactive", "EXE": "Exempt list", "CUT": "Released", "RET": "Retired"}
        try:
            ro = pd.read_csv(f"{BASE}/weekly_rosters/roster_weekly_{SEASON}.csv", low_memory=False)
            ro["week"] = safe_num(ro["week"]).astype(int)
            wk_r = next_week if (ro["week"] == next_week).any() else int(ro["week"].max())
            ro = ro[(ro["week"] == wk_r) & ro["status"].isin(list(ROSTER_LABEL))]
            for _, rr_ in ro.iterrows():
                key_ = (rr_["gsis_id"], rr_["team"])
                for _, gr_ in g[(g["player_id"] == rr_["gsis_id"]) & (g["team"] == rr_["team"])].iterrows():
                    cur = inj_by_pid.get(gr_["player_id"])
                    if cur is None or cur["avail"] > 0.0:
                        inj_by_pid[gr_["player_id"]] = {"avail": 0.0, "label": ROSTER_LABEL[rr_["status"]],
                                                        "note": (cur or {}).get("note", "")}
        except Exception as e:
            print("   Roster status unavailable:", e)

        # ---- Depth chart: who is next up at each position ----
        depth_rank = {}
        depth_pre = {}
        try:
            dc = pd.read_csv(f"{BASE}/depth_charts/depth_charts_{SEASON}.csv", low_memory=False)
            _all = dc.copy()
            dc = dc[dc["dt"] == dc["dt"].max()]
            dc = dc[dc["pos_abb"].isin(["RB", "WR", "TE"])].copy()
            # The chart as it stood right after Week 1: shows who was a starter before injuries reshuffled it.
            _all["_d"] = _all["dt"].astype(str).str[:10]
            _after = sorted(x for x in _all["_d"].unique() if x >= f"{SEASON}-09-12")
            _ref = _after[0] if _after else sorted(_all["_d"].unique())[0]
            _p = _all[(_all["_d"] == _ref) & _all["pos_abb"].isin(["RB", "WR", "TE"])].copy()
            _p["pos_rank"] = safe_num(_p["pos_rank"]).astype(int)
            for _, d_ in _p.iterrows():
                depth_pre[(d_["gsis_id"], d_["team"])] = int(d_["pos_rank"])
            dc["pos_rank"] = safe_num(dc["pos_rank"]).astype(int)
            for _, d_ in dc.iterrows():
                depth_rank[(d_["gsis_id"], d_["team"])] = int(d_["pos_rank"])
        except Exception as e:
            print("   Depth chart unavailable; using usage only:", e)

        # Recent (last two games) work, blended with the season average.
        rec2 = (merged.sort_values("week").groupby("player_id").tail(2)
                .groupby("player_id")[["carries", "targets"]].mean())
        def blend(pid, col, season_val):
            if pid in rec2.index:
                return 0.4 * float(season_val) + 0.6 * float(rec2.at[pid, col])
            return float(season_val)

        DEPTH_W = [1.0, 0.55, 0.3, 0.15, 0.08]
        def shares(cands, rows, av, depth_ord, use_col):
            """Split of vacated work: mostly by recent usage, partly by depth-chart order."""
            if not cands:
                return {}
            floor = 0.5 if use_col == "carries" else 0.6
            us = {k: av[k] * (blend(k, use_col, rows[k][use_col + "_pg"]) + floor) ** 1.5 for k in cands}
            ordered = sorted(cands, key=lambda k: depth_ord.get(k, 99))
            ds = {k: av[k] * (DEPTH_W[i] if i < len(DEPTH_W) else 0.05) for i, k in enumerate(ordered)}
            ut, dt_ = sum(us.values()), sum(ds.values())
            if ut <= 0 and dt_ <= 0:
                return {}
            return {k: 0.65 * (us[k] / ut if ut else 0) + 0.35 * (ds[k] / dt_ if dt_ else 0) for k in cands}

        INJ_TRUST = 1.0
        QREF = {}
        for _pos in ("RB", "WR", "TE"):
            _q = g[(g["position"] == _pos) & g["qualified"]]
            QREF[_pos] = (np.sort(_q["ppr_pg"].values.astype(float)), np.sort(_q["mania"].values.astype(float)))
        # Spread what injured players leave behind to their healthy teammates.
        RET = 1.0          # share of vacated work that stays with listed players
        PPR_CARRY, PPR_TGT_RB, PPR_TGT = 0.65, 1.3, 1.55
        for team, tg in g.groupby("team"):
            rows = {r_["player_id"]: r_ for _, r_ in tg.iterrows()}
            mt = merged[merged["team"] == team]
            tw = sorted(set(int(w_) for w_ in mt["week"]))
            pw = {}
            for _r in mt.itertuples():
                pw.setdefault(_r.player_id, {})[int(_r.week)] = (float(_r.game_weight), float(_r.fantasy_points_ppr), float(_r.targets), float(_r.carries), float(_r.offense_pct))
            dppr, pairs = {}, {}
            av = {pid: inj_by_pid.get(pid, {}).get("avail", 1.0) for pid in rows}
            d_ord = {pid: depth_rank.get((pid, team), 99) for pid in rows}
            # Role check: depth-chart order vs recent usage among healthy players.
            for pos_ in ("RB", "WR", "TE"):
                healthy = [k for k, x in rows.items() if x["position"] == pos_ and av[k] > 0.1 and d_ord[k] < 99]
                if not healthy:
                    continue
                by_depth = sorted(healthy, key=lambda k: d_ord[k])
                by_use = sorted(healthy, key=lambda k: -(blend(k, "carries", rows[k]["carries_pg"]) + 1.5 * blend(k, "targets", rows[k]["targets_pg"])))
                for i, k in enumerate(by_depth):
                    depth_info[k] = {"rank": i + 1, "pos": pos_}
                if pos_ in ("RB", "TE") and len(by_depth) >= 2:
                    top = by_depth[0]
                    use_of = lambda k: blend(k, "carries", rows[k]["carries_pg"]) + 1.5 * blend(k, "targets", rows[k]["targets_pg"])
                    # Only trust the depth chart over usage when the usage gap is small.
                    if by_use[0] != top and use_of(top) >= 0.6 * use_of(by_use[0]):
                        inj_boost[top] = inj_boost.get(top, 0.0) + 1.0
                        inj_why.setdefault(top, []).append({"n": "Depth chart", "pos": pos_, "s": pos_ + "1 now", "pts": 1.0, "kind": "depth"})
                        old = by_use[0]
                        inj_boost[old] = inj_boost.get(old, 0.0) - 1.0
                        inj_why.setdefault(old, []).append({"n": "Depth chart", "pos": pos_, "s": "now behind " + str(rows[top]["player_name"]), "pts": -1.0, "kind": "depth"})
            for pid, r_ in rows.items():
                u = 1.0 - av[pid]
                pos = r_["position"]
                if u <= 0 or pos not in ("RB", "WR", "TE"):
                    continue
                pw_i = pw.get(pid, {})
                # ---- How big was his role? Healthy-game usage blended with where the depth chart had him.
                mx_i = max([v_[4] for v_ in pw_i.values()] or [0.0])
                full = [v_ for v_ in pw_i.values() if v_[0] >= 1.0 and mx_i > 0 and v_[4] >= 0.75 * mx_i]
                rk_pre = depth_pre.get((pid, team), 9)
                pri_t = {"WR": {1: 7.0, 2: 5.5, 3: 3.8, 4: 2.0, 5: 1.0}, "TE": {1: 5.0, 2: 2.0}, "RB": {1: 3.5, 2: 2.0, 3: 0.8}}[pos].get(rk_pre, 0.4)
                pri_c = {"RB": {1: 14.0, 2: 6.0, 3: 2.0}}.get(pos, {}).get(rk_pre, 0.0)
                nf = len(full)
                use_t = float(np.mean([v_[2] for v_ in full])) if full else 0.0
                use_c = float(np.mean([v_[3] for v_ in full])) if full else 0.0
                role_t = (nf * use_t + 2.0 * pri_t) / (nf + 2.0)
                role_c = (nf * use_c + 2.0 * pri_c) / (nf + 2.0)
                role_ppr = role_c * PPR_CARRY + role_t * (PPR_TGT_RB if pos == "RB" else PPR_TGT)
                if role_t + role_c * 0.45 < 1.5:
                    continue
                v_car = role_c * u
                v_tgt = role_t * u
                rbs = [k for k, x in rows.items() if x["position"] == "RB" and k != pid]
                pcs = [k for k, x in rows.items() if x["position"] in ("WR", "TE") and k != pid]
                give = []   # (receiver, carries, targets)
                if v_car > 0:
                    sh = shares(rbs, rows, av, d_ord, "carries")
                    give += [(k, v_car * sh[k], 0.0) for k in sh]
                if v_tgt > 0:
                    to_rb = 0.6 if pos == "RB" else (0.2 if pos == "WR" else 0.3)
                    sh = shares(rbs, rows, av, d_ord, "targets")
                    give += [(k, 0.0, v_tgt * to_rb * sh[k]) for k in sh]
                    sh = shares(pcs, rows, av, d_ord, "targets")
                    give += [(k, 0.0, v_tgt * (1 - to_rb) * sh[k]) for k in sh]
                star_x = 1.0 + min(max(float(r_["ppr_pg"]) - 12.0, 0.0) / 20.0, 0.35)   # a star's work is worth more than his box score
                model = {}
                for k, c_, t_ in give:
                    rp = rows[k]["position"]
                    model[k] = model.get(k, 0.0) + (c_ * PPR_CARRY + t_ * (PPR_TGT_RB if rp == "RB" else PPR_TGT)) * star_x
                # ---- With / without evidence, measured in targets and carries (not fantasy points).
                # A game counts as "without him" in proportion to the snaps he missed.
                absw = {}
                if pw_i:
                    for w_ in tw:
                        if w_ <= min(pw_i):
                            continue
                        if w_ not in pw_i:
                            absw[w_] = 1.0
                        elif mx_i > 0 and pw_i[w_][4] < 0.6 * mx_i:
                            absw[w_] = max(0.0, 1.0 - pw_i[w_][4] / mx_i)
                never_played = (not pw_i) and len(tw) >= 2
                recv = [k for k, x in rows.items() if k != pid and x["position"] in ("RB", "WR", "TE")]
                obs, nn_ = {}, {}
                for k in recv:
                    pw_k = pw.get(k, {})
                    Nw = [(w_, a_) for w_, a_ in absw.items() if a_ > 0 and w_ in pw_k and pw_k[w_][0] >= 0.5]
                    if not Nw:
                        continue
                    rpk = rows[k]["position"]
                    tv = PPR_TGT_RB if rpk == "RB" else PPR_TGT
                    val = lambda v_: v_[2] * tv + v_[3] * PPR_CARRY
                    na = sum(a_ for _, a_ in Nw)
                    mean_n = sum(a_ * val(pw_k[w_]) for w_, a_ in Nw) / na
                    Wk = [w_ for w_ in pw_k if absw.get(w_, 0) == 0 and w_ in pw_i and pw_k[w_][0] >= 0.5]
                    pool = Wk or [w_ for w_ in pw_k if absw.get(w_, 0) == 0 and pw_k[w_][0] >= 0.5]
                    if not pool:
                        continue
                    base_ = float(np.mean([val(pw_k[w_]) for w_ in pool]))
                    obs[k] = mean_n - base_
                    nn_[k] = (na, round(base_, 1), round(mean_n, 1), sum(1 for _ in Nw))
                # Teammates together cannot gain more than the injured player actually had.
                psum = sum(max(v_, 0.0) for v_ in obs.values())
                scale = min(1.0, 1.25 * role_ppr / psum) if psum > 0 else 1.0
                sig = min(1.0, role_ppr / 7.5)
                for k in recv:
                    m_d = model.get(k, 0.0)
                    ev = None
                    if k in obs:
                        o_ = float(np.clip(obs[k] * scale, -1.5, 0.6 * role_ppr + 1.0))
                        na, b_, m_n, ng = nn_[k]
                        wgt = u * sig * na / (na + 1.5)
                        d_ppr = (1 - wgt) * m_d + wgt * o_
                        ev = {"games": ng, "with": b_, "without": m_n}
                    else:
                        d_ppr = m_d * (0.4 if never_played else 1.0)
                    if rows[k]["position"] == "TE":
                        d_ppr *= 0.7
                    if abs(d_ppr) < 0.3:
                        continue
                    dppr[k] = dppr.get(k, 0.0) + d_ppr
                    if ev:
                        inj_has_ev.add(k)
                    pairs.setdefault(k, []).append((pid, str(r_["player_name"]), pos, inj_by_pid.get(pid, {}).get("label", ""), d_ppr, ev))
            # Turn the extra points per game into a rating change: where would his projected
            # PPR/G rank at his position, and what Mania Rating do players at that rank have?
            for k, dp in dppr.items():
                pos_k = rows[k]["position"]
                xs, ms = QREF[pos_k]
                if len(xs) < 5:
                    continue
                f = lambda x: float(np.interp(np.interp(x, xs, np.linspace(0, 1, len(xs))), np.linspace(0, 1, len(ms)), ms))
                cur = float(rows[k]["ppr_pg"])
                gain = INJ_TRUST * (f(max(cur + dp, 0.0)) - f(cur))
                if pos_k == "TE":
                    gain *= 0.55
                sumd = sum(x[4] for x in pairs[k])
                for pid_i, nm_, pos_i, lab_, d_, ev_ in pairs[k]:
                    pts = gain * (d_ / sumd if abs(sumd) > 1e-6 else 1.0 / len(pairs[k]))
                    inj_boost[k] = inj_boost.get(k, 0.0) + pts
                    item = {"n": nm_, "pos": pos_i, "s": lab_, "pts": round(pts, 1), "kind": "role"}
                    if ev_:
                        item["ev"] = ev_
                    inj_why.setdefault(k, []).append(item)
                    inj_repl.setdefault(pid_i, []).append({"n": str(rows[k]["player_name"]), "pos": pos_k, "pts": round(pts, 1), "g": (ev_ or {}).get("games", 0)})
            # A hurt starting QB hurts the whole passing game.
            if team in qb_avail:
                qa, qlabel, qname = qb_avail[team]
                uq = 1.0 - qa
                if uq > 0:
                    for k, x in rows.items():
                        d = -2.2 * uq if x["position"] == "WR" else (-1.6 * uq if x["position"] == "TE" else -0.5 * uq)
                        inj_boost[k] = inj_boost.get(k, 0.0) + d
                        inj_why.setdefault(k, []).append({"n": qname, "pos": "QB", "s": qlabel, "pts": round(d, 1), "kind": "qb"})
        inj_meta["ok"] = True
        inj_meta["n"] = len(inj_by_pid)
        inj_meta["out"] = sum(1 for v in inj_by_pid.values() if v["avail"] <= 0.1)
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            inj_meta["updated"] = datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %-d, %-I:%M %p ET")
        except Exception:
            inj_meta["updated"] = ""
        print(f"   {inj_meta['n']} skill players flagged, {inj_meta['out']} out/doubtful.")
    except Exception as e:
        print("   Injury report unavailable; ratings use no injury data:", e)

def inj_why_clean(pid):
    agg = {}
    for w in inj_why.get(pid, []):
        k = (w["n"], w["kind"])
        if k in agg:
            agg[k]["pts"] = round(agg[k]["pts"] + w["pts"], 1)
        else:
            agg[k] = dict(w)
    out = [w for w in agg.values() if abs(w["pts"]) >= 0.3]
    return sorted(out, key=lambda x: -abs(x["pts"]))[:3]

# ---- Injuries list for the Injuries tab ----
inj_list = []
if inj_meta.get("ok"):
    gx = g.copy()
    gx["use_"] = gx["carries_pg"] + 1.5 * gx["targets_pg"]
    gx["urank"] = gx.groupby(["team", "position"])["use_"].rank(ascending=False, method="first")
    for pid_, info_ in inj_by_pid.items():
        rr = gx[gx["player_id"] == pid_]
        if rr.empty:
            continue
        rr = rr.iloc[0]
        lim_ = {"RB": 1, "WR": 3, "TE": 1}.get(rr["position"], 0)
        if rr["urank"] <= lim_:
            role_ = "Starter"
        elif rr["use_"] >= 5:
            role_ = "Rotation"
        else:
            continue
        rep_ = sorted(inj_repl.get(pid_, []), key=lambda x: -x["pts"])
        rep_ = [x for x in rep_ if x["pts"] >= 0.3][:3]
        inj_list.append({"id": pid_, "name": str(rr["player_name"]), "team": str(rr["team"]), "pos": str(rr["position"]),
                         "label": info_["label"], "note": info_.get("note", ""), "avail": info_["avail"],
                         "role": role_, "repl": rep_})
    inj_list.sort(key=lambda x: (x["avail"], 0 if x["role"] == "Starter" else 1, x["team"]))
inj_meta["list"] = inj_list

start_vals = []
inj_totals = {}
match_examples = {}
for _, r in g.iterrows():
    opp = opp_map.get(str(r["team"]), "")
    if not opp and week_has_games:
        opp = "BYE"   # next week's schedule exists and this team is not on it
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
    own = inj_by_pid.get(r["player_id"])
    inj_own = -4.0 * (1.0 - own["avail"]) if own else 0.0
    team_b = max(-6.0, min(26.0, inj_boost.get(r["player_id"], 0.0)))
    if team_b > 0:
    # harder to climb when already near the top: boost shrinks as Mania rises
        team_b *= max(0.2, min(1.0, 1.0 - (float(r["mania"]) - 70.0) / 21.0))
    inj_total = inj_own + team_b
    start = clamp(r["mania"] + context_adj + confidence_adj + inj_total)
    if own and own["avail"] <= 0.1:
        start = 0.0
    inj_totals[r["player_id"]] = inj_total
    start_vals.append((start, madj, trend_adj, opp))
    match_examples[r["player_id"]] = ex

g[["start_rating", "matchup_adj", "trend_adj", "opponent"]] = pd.DataFrame(start_vals, index=g.index)
g["start_pos_rank"] = g.groupby("position")["start_rating"].rank(ascending=False, method="min").astype(int)
g["start_overall_rank"] = g["start_rating"].rank(ascending=False, method="min").astype(int)

if AS_OF_WEEK:
    print("AS_OF " + json.dumps({r["player_id"]: [round(float(r["start_rating"]), 1), str(r["opponent"])] for _, r in g.iterrows()}))
    sys.exit(0)

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
            "ry": round(float(x.receiving_yards), 3),
            "rtd": round(float(x.receiving_tds), 2),
            "air": round(float(x.receiving_air_yards), 1),
            "car": int(x.carries),
            "ruy": round(float(x.rushing_yards), 3),
            "rutd": round(float(x.rushing_tds), 2),
            "ppr": round(float(x.fantasy_points_ppr), 3),
            "snap": round(float(x.offense_pct) * 100, 3),
            "tshare": round(float(x.target_share_game), 3),
            "ashare": round(float(x.air_share_game), 3),
            "touch": round(float(x.touch_share_game), 3),
            "rzt": int(x.rz_targets),
            "ez": int(x.endzone_targets),
            "rzc": int(x.rz_carries),
            "gl": int(x.gl_carries),
            "partial": bool(x.partial),
            "weight": round(float(x.game_weight), 2),
            "auto": bool(x.auto_excl),
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
            "share": round(float(xx.touch_share_game if pos == "RB" else xx.target_share_game), 1),
            "ppr": round(float(xx.fantasy_points_ppr), 1),
        } for xx in rd.itertuples()]
        room_data.append({
            "id": rr["player_id"], "name": rr["player_name"],
            "mania": round(float(rr["mania"]), 1),
            "ppr": round(float(rr["ppr_pg"]), 1),
            "opp": round(float(rr["targets_pg"] if pos != "RB" else rr["carries_pg"] + rr["targets_pg"]), 1),
            "share": round(float(rr["target_share"] if pos != "RB" else rr["touch_share"]), 1),
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
        "inj": inj_by_pid.get(r["player_id"]),
        "depth": depth_info.get(r["player_id"]),
        "out": bool(inj_by_pid.get(r["player_id"]) and inj_by_pid[r["player_id"]]["avail"] <= 0.1),
        "inj_adj": round(float(inj_totals.get(r["player_id"], 0.0)), 1),
        "inj_why": inj_why_clean(r["player_id"]),
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
    for k, col in {"b_production": "production", "b_opportunity": "opportunity", "b_role": "role_score",
                   "b_high_value": "high_value", "b_efficiency": "efficiency", "start": "start_rating"}.items():
        pct_refs[pos][k] = [round(float(v), 3) for v in np.sort(gp[col].replace([np.inf, -np.inf], np.nan).dropna().values)]
refs_json = json.dumps(pct_refs, separators=(",", ":"))

# ============================================================
# REPORT CARD — what the model said about the week just played, before it was played
# ponytail: one extra full run (and its downloads) per build, and only the latest week is graded.
# Save each week's ratings to the repo instead if a season-long record is wanted.
# ============================================================
prev_calls = {}   # player_id -> [start rating, opponent] as of the week before max_week
if max_week > 1:
    print(f"   Rebuilding Week {max_week} ratings as of Week {max_week - 1} for the report card...")
    try:
        child = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env={**os.environ, "MANIA_AS_OF_WEEK": str(max_week - 1)},
            capture_output=True, text=True, timeout=1200, check=True,
        )
        prev_calls = json.loads(next(l for l in child.stdout.splitlines() if l.startswith("AS_OF "))[6:])
    except Exception as e:
        print("   WARNING: report card unavailable; site will still build:", e)

# Every team's opponent by week, past and future, for the browser views. A missing week is a bye.
opps = {}
if schedule_ok and {"week", "home_team", "away_team"}.issubset(games.columns):
    for _, game in games.iterrows():
        h, a, wk = str(game["home_team"]), str(game["away_team"]), int(game["week"])
        opps.setdefault(h, {})[wk] = a
        opps.setdefault(a, {})[wk] = h

payload = json.dumps(players, separators=(",", ":"))
meta = json.dumps({
    "season": SEASON,
    "week": max_week,
    "next_week": next_week,
    "rz": rz_ok,
    "schedule": schedule_ok,
    "injuries": inj_meta,
    "players": len(players),
    "opps": opps,
    "prev": prev_calls,
})

# ============================================================
# FRONT END — FANTASY MANIA
# ============================================================
html = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Mania</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#e9edf2;--surface:#e0e5eb;--surface2:#d9dfe7;--line:#c4ccd6;--ink:#060709;--muted:#5a616a;--blue:#2563eb;--blue2:#1d4ed8;--green:#22a45d;--yellow:#e3a21a;--orange:#d9731f;--red:#e2505b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Plus Jakarta Sans',ui-sans-serif,system-ui,-apple-system,sans-serif}button,input{font:inherit}.topbar{background:#e7edf3;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50}.nav{max-width:1240px;margin:auto;height:62px;padding:0 24px;display:flex;align-items:center;justify-content:space-between}.brand{font-family:'Plus Jakarta Sans',sans-serif;font-size:25px;font-weight:800;letter-spacing:.8px}.brand span{color:var(--blue)}.links{display:flex;height:100%}.navb{border:0;background:transparent;color:#4f5760;font-size:13px;font-weight:800;padding:0 17px;cursor:pointer;border-bottom:2px solid transparent}.navb.on,.navb:hover{color:#000000;border-bottom-color:var(--blue)}
.wrap{max-width:1280px;margin:auto;padding:42px 30px 80px}.view{display:none}.view.on{display:block}.kicker{font-size:11px;color:var(--blue2);font-weight:800;text-transform:uppercase;letter-spacing:1.2px}.title{font-family:'Plus Jakarta Sans',sans-serif;font-size:44px;font-weight:800;letter-spacing:-1px;letter-spacing:.2px;margin:4px 0}.sub{color:var(--muted);font-size:16px;margin:4px 0 20px;line-height:1.55}.divider{border-top:1px solid var(--line);margin:24px 0}.sectionTitle{font-size:11px;letter-spacing:1.2px;color:#40474f;font-weight:800;text-transform:uppercase;margin-bottom:14px}
.search{position:relative}.search input{width:100%;height:48px;padding:0 15px;border:1px solid var(--line);border-radius:5px;background:#e3e8ee;color:#000000;outline:none}.search input:focus{border-color:var(--blue)}.dd{display:none;position:absolute;top:52px;left:0;right:0;background:#e1e7ed;border:1px solid var(--line);box-shadow:0 18px 45px #0008;z-index:60}.ddi{padding:12px 14px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:16px}.ddi:hover{background:#d5dce4}.ddiName{font-weight:700}.ddiMeta{font-size:12px;color:var(--muted);margin-left:8px}.ddiRate{font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:800;color:var(--green)}
.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin:18px 0}.pill{background:transparent;border:1px solid #b2bbc6;color:#3e444b;border-radius:4px;padding:7px 12px;font-size:12px;font-weight:800;cursor:pointer}.pill.on,.pill:hover{background:var(--blue);border-color:var(--blue);color:#e7edf3}.panel{border-top:1px solid var(--line);margin-top:24px;padding-top:20px}.ph{font-size:11px;letter-spacing:1.15px;color:#40474f;font-weight:800;text-transform:uppercase;margin-bottom:14px}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th{color:#676f79;font-size:10px;text-transform:uppercase;letter-spacing:.7px;cursor:pointer}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right;font-size:13px}th:nth-child(2),td:nth-child(2){text-align:left}.playerlink{font-weight:750;color:#070a0e;cursor:pointer}.playerlink:hover{color:var(--blue2)}.tag{font-size:9px;background:#d2d9e1;color:#535d68;padding:3px 5px;border-radius:3px;margin-left:6px}.grade{font-weight:800}.g90,.g80{color:var(--green)}.g70,.g60{color:var(--yellow)}.g0{color:var(--red)}
.hero{text-align:center;padding:6px 0 25px;border-bottom:1px solid var(--line)}.portrait{width:190px;height:190px;border-radius:50%;position:relative;display:grid;place-items:center;background:#d8dee5;margin:0 auto 18px}.portrait:before{content:"";position:absolute;inset:-7px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#cdd5dd 0);z-index:-2}.portrait:after{content:"";position:absolute;inset:-2px;border-radius:50%;background:var(--bg);z-index:-1}.portrait img{width:184px;height:184px;object-fit:cover;object-position:center top;border-radius:50%}.initials{font-family:'Plus Jakarta Sans',sans-serif;font-size:48px;font-weight:800;color:#717985}.hero h1{font-family:'Plus Jakarta Sans',sans-serif;font-size:42px;font-weight:800;letter-spacing:-1px;margin:0}.meta{color:var(--muted);font-size:14px;font-weight:600;margin-top:3px}.rankline{color:#272c33;font-size:13px;margin-top:8px}.scorepair{display:flex;justify-content:center;gap:48px;margin-top:20px}.scorebox{min-width:120px}.scorebig{font-family:'Plus Jakarta Sans',sans-serif;font-size:48px;font-weight:800;line-height:.9}.scorelab{font-size:9px;font-weight:800;color:#69717a;text-transform:uppercase;letter-spacing:.8px;margin-top:6px}.stats{display:grid;grid-template-columns:repeat(6,1fr);border-bottom:1px solid var(--line)}.stat{padding:18px 12px;text-align:center}.sv{font-family:'Plus Jakarta Sans',sans-serif;font-size:25px;font-weight:800}.sk{font-size:9px;color:#69717b;text-transform:uppercase;font-weight:800;letter-spacing:.6px;margin-top:2px}
.grid2{display:grid;grid-template-columns:1.08fr .92fr;gap:36px}.barrow{display:grid;grid-template-columns:145px 1fr 48px;gap:12px;align-items:center;margin:13px 0;font-size:13px}.track{height:7px;background:#d3dae1;overflow:hidden}.fill{height:100%}.explain{font-size:12px;color:var(--muted);line-height:1.5;margin-top:14px}.roomHead{display:flex;justify-content:space-between;align-items:center}.roomtable th,.roomtable td{text-align:center}.roomtable th:first-child,.roomtable td:first-child{text-align:left}.roomtable td:first-child{font-weight:700}.roomtable .you{color:var(--blue2)}
.matchScore{font-family:'Plus Jakarta Sans',sans-serif;font-size:38px;font-weight:800}.matchHead{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}.matchTable th:nth-child(2),.matchTable td:nth-child(2){text-align:right}.muted{color:var(--muted)}.good{color:var(--green);font-weight:800}.bad{color:var(--red);font-weight:800}.partial{font-size:8px;background:#e0d7cc;color:#816030;padding:2px 4px;border-radius:2px}.custom{color:#3f464f;font-size:12px;margin-bottom:12px}.reset{border:1px solid #adb7c3;background:transparent;color:#3f464e;border-radius:3px;padding:6px 9px;cursor:pointer;font-size:10px;font-weight:800}.gameControls{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}.gamechip{border:1px solid #b2bbc6;background:#e3e8ee;color:#3f464e;padding:7px 10px;border-radius:3px;font-size:11px;font-weight:700;cursor:pointer}.gamechip:has(input:checked){border-color:var(--blue);color:#000000;background:#cdd8e5}.gamechip input{display:none}
.compareHero{display:grid;grid-template-columns:1fr 90px 1fr;align-items:center;margin-top:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:30px 0}.comparePlayer{text-align:center}.cmpPortrait{width:180px;height:180px;margin:0 auto 14px;border-radius:50%;position:relative;display:grid;place-items:center;background:#d8dee5}.cmpPortrait:before{content:"";position:absolute;inset:-5px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#cdd5dd 0);z-index:-2}.cmpPortrait:after{content:"";position:absolute;inset:-1px;border-radius:50%;background:var(--bg);z-index:-1}.cmpPortrait img{width:174px;height:174px;object-fit:cover;object-position:center top;border-radius:50%}.comparePlayer h2{font-family:'Plus Jakarta Sans',sans-serif;font-size:28px;font-weight:800;margin:3px 0}.vs{font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:800;color:#7d8691;text-align:center}.edge{text-align:center;padding:18px 0;border-bottom:1px solid var(--line)}.edge strong{font-family:'Plus Jakarta Sans',sans-serif;font-size:28px}.cmpStats{max-width:760px;margin:18px auto 0}.allchip{border-color:#8c98a5}
.homeHero{padding:42px 0 34px;border-bottom:1px solid var(--line)}.homeHero h1{font-family:'Plus Jakarta Sans',sans-serif;font-size:52px;line-height:1.02;margin:8px 0 12px;letter-spacing:-2px}.homeHero h1 span{color:var(--blue)}.homeHero .sub{max-width:720px;font-size:16px}.homeSearch{max-width:760px;margin-top:25px}.homeActions{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.cta{border:1px solid #acb7c3;background:#dfe5ec;color:#060a0e;padding:10px 15px;border-radius:4px;font-weight:800;font-size:12px;cursor:pointer}.cta.primary{background:var(--blue);border-color:var(--blue);color:#e8eef4}.homeGrid{display:grid;grid-template-columns:1.1fr .9fr;gap:42px;padding-top:32px}.how{font-size:14px;color:#383e46;line-height:1.65}.how strong{color:#000000}.miniRank{margin-top:8px}.miniRow{display:grid;grid-template-columns:32px 1fr 64px 64px;gap:10px;align-items:center;padding:11px 0;border-bottom:1px solid var(--line);font-size:13px}.miniRow .n{color:#717a86}.miniRow .pn{font-weight:750;cursor:pointer}.miniRow .pn:hover{color:var(--blue2)}.miniRow .score{text-align:right;font-weight:800}.homeNote{border-left:3px solid var(--blue);padding:3px 0 3px 15px;margin-top:20px;color:#5a616a;font-size:12px;line-height:1.55}
/* v4 readability + interactive explanations */
.homeHero{max-width:980px;padding:38px 0 34px}.homeHero h1{font-size:58px;line-height:1.02;letter-spacing:-2px;margin:8px 0 14px}.homeHero .sub{max-width:850px;font-size:18px}.homeSearch{max-width:850px}.homeGrid{gap:58px}.how{font-size:15px;line-height:1.7}.miniRow{font-size:14px;padding:12px 0}.scorepair{gap:72px}.scorebox{min-width:190px}.scorelab{font-size:11px}.stats .stat{min-height:86px;display:flex;flex-direction:column;justify-content:center}.barrow{grid-template-columns:190px 1fr 56px;font-size:14px}.infoBtn{border:0;background:transparent;color:#40474f;cursor:pointer;font-weight:800;margin-left:5px;padding:0 4px}.infoBtn:hover{color:#000000}.bucketHelp{display:none;margin:-3px 0 13px 202px;color:var(--muted);font-size:12px;line-height:1.5;max-width:560px}.bucketHelp.open{display:block}.usageGrid{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}.usageItem{background:var(--bg);padding:17px}.usageItem .uv{font-size:25px;font-weight:800}.usageItem .ul{font-size:10px;color:var(--muted);text-transform:uppercase;font-weight:800;margin-top:3px}.usageItem .ud{font-size:11px;color:#5f6771;margin-top:8px;line-height:1.45}.teamControls{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.teamSelect{background:#e3e8ee;color:#000000;border:1px solid var(--line);height:40px;padding:0 12px;border-radius:4px}.shareSection{margin-top:26px}.shareSection h2{font-size:21px;margin:0 0 10px}.shareTable td,.shareTable th{font-size:13px;text-align:center}.shareTable td:first-child,.shareTable th:first-child{text-align:left;position:sticky;left:0;background:var(--bg)}.sharePlayer{font-weight:750;cursor:pointer}.sharePlayer:hover{color:var(--blue2)}
@media(max-width:850px){.homeGrid{grid-template-columns:1fr}.homeHero h1{font-size:40px}.links{display:none}.wrap{padding:24px 16px}.grid2{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.portrait{width:150px;height:150px}.portrait img{width:144px;height:144px}.hero h1{font-size:38px}.compareHero{grid-template-columns:1fr}.cmpPortrait{width:155px;height:155px}.cmpPortrait img{width:149px;height:149px}.vs{padding:16px}.scorepair{gap:24px}.barrow{grid-template-columns:125px 1fr 42px}}

/* v5 — readability + visible Team Share navigation */
body{font-size:16px;line-height:1.5}.nav{max-width:1380px;height:70px;padding:0 32px}.brand{font-size:27px}.navb{font-size:14px;padding:0 18px}.wrap{max-width:1400px;padding:48px 38px 90px}.title{font-size:48px}.sub{font-size:17px}.sectionTitle,.ph{font-size:12px}.search input{height:52px;font-size:16px}.pill{font-size:13px;padding:9px 14px}th{font-size:11px}th,td{font-size:14px;padding:13px 11px}.tag{font-size:10px}.hero h1{font-size:46px}.meta{font-size:15px}.rankline{font-size:14px}.scorebig{font-size:52px}.scorelab{font-size:11px}.sv{font-size:27px}.sk{font-size:10px}.barrow{font-size:15px}.explain,.custom{font-size:13px}.how{font-size:16px}.homeHero{max-width:1100px}.homeHero h1{font-size:62px}.homeHero .sub{font-size:19px}.miniRow{font-size:15px}.comparePlayer h2{font-size:31px}.cmpStats{max-width:900px}.teamControls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:20px 0 28px}.teamSelect{min-width:180px;height:42px;background:#e3e8ee;color:#000000;border:1px solid #b2bbc6;border-radius:4px;padding:0 12px;font-size:14px;font-weight:700}.shareSection{margin:34px 0 46px}.shareSection h2{font-size:27px;margin:0 0 12px}.shareTable th,.shareTable td{font-size:15px;padding:15px 13px}.shareTable th:first-child,.shareTable td:first-child{text-align:left;position:sticky;left:0;background:var(--bg);z-index:2}.sharePlayer{font-weight:800;color:#060709;cursor:pointer}.sharePlayer:hover{color:var(--blue2)}
@media(max-width:900px){.nav{padding:0 14px}.brand{font-size:22px}.navb{font-size:11px;padding:0 9px}.wrap{padding:30px 18px 70px}.homeHero h1{font-size:44px}.title{font-size:38px}.grid2,.homeGrid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.scorepair{gap:28px}.scorebox{min-width:130px}.compareHero{grid-template-columns:1fr 45px 1fr}.cmpPortrait{width:150px;height:150px}.cmpPortrait img{width:144px;height:144px}.comparePlayer h2{font-size:24px}}

/* v6 — bigger, tighter, green/yellow/red only for grades */
:root{--ink:#060709;--muted:#4e555e}
html{-webkit-text-size-adjust:100%}
body{font-size:18px;line-height:1.5}
.nav{max-width:1240px;height:74px}.brand{font-size:30px}.navb{font-size:15px;padding:0 20px}
.wrap{max-width:1240px;padding:40px 32px 90px}
.kicker{color:var(--muted);font-size:13px;text-transform:none;letter-spacing:.3px;font-weight:700}
.title{font-size:48px;margin:6px 0 8px}.sub{font-size:19px;color:var(--muted)}
.sectionTitle,.ph{font-size:14px;letter-spacing:.4px;text-transform:none;color:#282d34;font-weight:800}
.search input{height:58px;font-size:18px;padding:0 18px;border-radius:8px}
.search input:focus{border-color:#666f7a}
.ddi{padding:15px 16px}.ddiName{font-size:17px}.ddiMeta{font-size:14px}.ddiRate{font-size:20px}
.pill{font-size:15px;padding:10px 16px;border-radius:6px}
.pill.on,.pill:hover{background:#060709;border-color:#060709;color:#e7edf3}
.cta{font-size:15px;padding:13px 20px;border-radius:6px}.cta.primary{background:#060709;border-color:#060709;color:#e7edf3}
th{font-size:12px}th,td{font-size:16px;padding:15px 12px}
.tag{font-size:12px;padding:3px 7px}
.playerlink,.sharePlayer,.pn,[data-open]{cursor:pointer}
.playerlink:hover,.sharePlayer:hover,.pn:hover,[data-open]:hover,.roomtable td[data-id]:hover{color:#000000;text-decoration:underline}
.roomtable td[data-id]{cursor:pointer}.roomtable .you{color:#000000;font-weight:800}
.gamechip:has(input:checked){border-color:#57606c;background:#d5dbe2}

/* home */
.homeHero{max-width:none;padding:34px 0 30px}
.homeHero h1{font-size:68px;line-height:1.02;letter-spacing:-2px}.homeHero h1 span{color:var(--ink)}
.homeHero .sub{font-size:22px;max-width:760px}.homeSearch{max-width:760px}
.homeGrid{display:block;max-width:900px;padding-top:34px}
.homeBlock{margin-bottom:46px}
.miniRow{grid-template-columns:40px 1fr 110px 110px;font-size:19px;padding:16px 0}
.miniRow .score{font-size:24px}.miniRow .n{font-size:18px}
.miniHead{color:#676f79;font-size:13px!important;padding:6px 0!important;font-weight:700}
.how3{display:grid;grid-template-columns:repeat(3,1fr);gap:28px}
.how3 div{font-size:17px;line-height:1.6;color:#333941}.how3 b{display:block;color:#000000;font-size:20px;margin-bottom:6px}

/* profile hero: big photo, both scores beside it */
.hero{display:grid;grid-template-columns:auto 1fr;gap:56px;align-items:center;text-align:left;padding:10px 0 34px}
.portrait{width:270px;height:270px;margin:0}.portrait:before{inset:-9px}.portrait img{width:262px;height:262px}.initials{font-size:72px}
.hero h1{font-size:58px;letter-spacing:-1.5px}.meta{font-size:19px}.rankline{font-size:17px;color:#272c33}
.scorepair{justify-content:flex-start;gap:64px;margin-top:26px}
.scorebox{min-width:0}.scorebig{font-size:84px;line-height:.9}
.scorelab{font-size:17px;color:#000000;text-transform:none;letter-spacing:0;font-weight:800;margin-top:10px}
.scoresub{font-size:14px;color:var(--muted);margin-top:3px}
.stats{grid-template-columns:repeat(6,1fr)}.stat{padding:24px 10px}.sv{font-size:34px}.sk{font-size:12px;text-transform:none;letter-spacing:0;color:var(--muted)}

/* breakdown bars + click-to-explain */
.barrow{grid-template-columns:230px 1fr 58px;font-size:17px;margin:15px 0}
.track{height:12px;border-radius:6px;background:#d3dae1}.fill{border-radius:6px}
.lblBtn{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:7px;font-weight:700;color:#0c0f14;font-size:17px}
.lblBtn:hover{color:#000000;text-decoration:underline}.lblBtn .chev{color:#59616b;font-size:15px}
.bucketHelp{margin:-4px 0 16px;padding:12px 14px;background:#e3e8ee;border-left:3px solid #96a0ac;font-size:15px;line-height:1.55;max-width:none;color:#363c44}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:16px;font-size:14px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.explain,.custom{font-size:15px}

/* usage details */
.usageGrid{grid-template-columns:repeat(2,1fr)}
.usageItem{padding:20px}.usageItem .uv{font-size:34px;line-height:1}.usageItem .ul{font-size:15px;text-transform:none;color:#12151a;font-weight:700;margin-top:8px}
.usageItem .ud{font-size:14px;color:var(--muted);margin-top:7px}.usageItem .uavg{font-size:14px;color:#292f36;margin-top:8px;font-weight:700}
.ubar{height:6px;background:#d3dae1;border-radius:3px;margin-top:12px;overflow:hidden}.ubar i{display:block;height:100%;border-radius:3px}

/* team share */
.teamSelect{height:50px;min-width:220px;font-size:17px;padding:0 14px}
.shareSection h2{font-size:30px}.shareTable th,.shareTable td{font-size:17px;padding:16px 14px}
.shareTable td.up{color:var(--green)}.shareTable td.down{color:var(--red)}.shareTable td.season{background:#dde3ea;font-weight:800;color:#000000}
.shareNote{color:var(--muted);font-size:15px;margin-top:6px}

.compareHero{padding:36px 0}.cmpPortrait{width:210px;height:210px}.cmpPortrait img{width:204px;height:204px}.comparePlayer h2{font-size:34px}
.cmpStats{max-width:980px}.edge strong{font-size:34px}

@media(max-width:900px){
 body{font-size:17px}.wrap{padding:26px 16px 70px}
 .homeHero h1{font-size:46px}.how3{grid-template-columns:1fr}
 .miniRow{grid-template-columns:30px 1fr 74px 74px;font-size:16px}
 .hero{grid-template-columns:1fr;text-align:center;gap:22px}.portrait{margin:0 auto;width:210px;height:210px}.portrait img{width:202px;height:202px}
 .scorepair{justify-content:center;gap:34px}.scorebig{font-size:60px}.hero h1{font-size:42px}
 .stats{grid-template-columns:repeat(2,1fr)}.barrow{grid-template-columns:1fr 70px 50px;font-size:15px}.barrow .lblBtn{font-size:15px}
 .usageGrid{grid-template-columns:1fr}.title{font-size:38px}
 .compareHero{grid-template-columns:1fr}.cmpPortrait{width:170px;height:170px}.cmpPortrait img{width:164px;height:164px}
}

/* v6b fixes */
.roomtable th,.roomtable td{padding:13px 9px;font-size:16px}
.grid2{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:48px}
.barrow{grid-template-columns:210px 1fr 54px}
.portrait{margin-left:12px}
.usageGrid{gap:12px;background:none;border:0}
.usageItem{border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.shareTable{table-layout:fixed}.shareTable th:first-child,.shareTable td:first-child{width:30%}
.navb{white-space:nowrap}
@media(max-width:900px){
 .nav{flex-direction:column;align-items:stretch;height:auto;padding:10px 14px 0;gap:4px}
 .links{display:flex!important;overflow-x:auto;height:44px;width:100%}
 .grid2{grid-template-columns:1fr;gap:20px}.portrait{margin:0 auto}
 .barrow{grid-template-columns:1fr 70px 50px}
 .shareTable{table-layout:auto}.shareTable th:first-child,.shareTable td:first-child{width:auto}
}

/* v7 — clean type, centered hero, percentiles, real mobile layout */
body{font-family:'Plus Jakarta Sans',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-feature-settings:"tnum" 1}
.brand{font-weight:800;letter-spacing:-.3px;font-size:25px}
.hero2{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);grid-template-areas:"ml ph wk" "id id id";column-gap:56px;row-gap:24px;align-items:center;padding:6px 0 34px;border-bottom:1px solid var(--line)}
.hs{min-width:0}.hs.left{grid-area:ml;text-align:right}.hs.right{grid-area:wk;text-align:left}.hp{grid-area:ph}.hid{grid-area:id;text-align:center}
.hero2 .portrait{width:250px;height:250px;margin:0}.hero2 .portrait img{width:242px;height:242px}
.hs .scorebig{font-size:80px;font-weight:800;line-height:.95}.hs .scorelab{font-size:18px;color:#000000;font-weight:800;margin-top:10px;text-transform:none;letter-spacing:0}
.hs .scoresub{font-size:14px;color:var(--muted);margin-top:3px}
.hid h1{font-size:48px;font-weight:800;letter-spacing:-1.2px;margin:0}.hid .meta{font-size:18px}.hid .rankline{font-size:16px}
.outlook{display:inline-block;margin-top:12px;padding:7px 14px;border-radius:999px;font-weight:800;font-size:15px;border:1px solid currentColor}
.outlook.g90{color:var(--green);background:rgba(55,201,121,.10)}.outlook.g70{color:var(--yellow);background:rgba(226,180,59,.10)}.outlook.g0{color:var(--red);background:rgba(239,90,90,.10)}
.olDesc{font-size:15px;color:#2b3037;margin-top:9px;line-height:1.45;max-width:320px}
.disclaimer{font-size:13px;color:var(--muted);line-height:1.6;max-width:820px;margin:18px auto 0;text-align:center}
.disclaimer.left{text-align:left;margin-left:0}
.olTag{font-weight:800;font-size:13px;white-space:nowrap}.olTag.g90{color:var(--green)}.olTag.g70{color:var(--yellow)}.olTag.g0{color:var(--red)}
th[data-k=outlook],.olCell{text-align:left!important}
.pctHead{font-size:14px;color:var(--muted);margin:-4px 0 14px}
.pctNum{font-size:20px;font-weight:800;text-align:right}
.barrow{grid-template-columns:215px minmax(0,1fr) 46px}
.legend{font-size:13px}
.uPct{font-size:15px;font-weight:800;margin-top:9px}.uPct span{color:var(--muted);font-weight:600}
.sortSel{display:none}
.rankCards{display:none}
.rcard{display:grid;grid-template-columns:30px minmax(0,1fr) 58px 58px;gap:8px;align-items:center;padding:13px 2px;border-bottom:1px solid var(--line);cursor:pointer}
.rcard .rn{color:#717a86;font-size:14px;text-align:center}.rname{font-weight:800;font-size:16px}.rsub{margin-top:3px}.rstats{font-size:12.5px;color:var(--muted);margin-top:3px}
.rs{text-align:right;font-weight:800;font-size:20px;line-height:1}.rs small{display:block;font-size:10.5px;color:var(--muted);font-weight:600;margin-top:4px}
.rcHead{display:grid;grid-template-columns:30px minmax(0,1fr) 58px 58px;gap:8px;padding:8px 2px;color:#676f79;font-size:11.5px;font-weight:700;border-bottom:1px solid var(--line)}.rcHead span:nth-child(3),.rcHead span:nth-child(4){text-align:right}
.cmpOl{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:6px}.cmpOl .outlook{margin-top:6px;font-size:14px}

/* ---------- PHONE ---------- */
@media(max-width:700px){
 body{font-size:15px}
 .wrap{padding:16px 14px 56px}
 .nav{height:auto;padding:10px 14px 0;gap:2px;align-items:stretch;flex-direction:column}
 .brand{font-size:21px}.links{display:flex!important;overflow-x:auto;width:100%;height:42px;scrollbar-width:none}.links::-webkit-scrollbar{display:none}
 .navb{font-size:12.5px;padding:0 13px;white-space:nowrap}
 .title{font-size:30px;margin:4px 0 6px}.sub{font-size:15px}.kicker{font-size:12px}
 .search input{height:50px;font-size:16px;padding:0 14px}.teamSelect{font-size:16px;height:46px;min-width:0;width:100%}
 .ddi{padding:13px 12px}.ddiName{font-size:15px}.ddiMeta{font-size:12px}
 .pill{font-size:13px;padding:9px 12px}.toolbar{gap:6px;margin:14px 0}
 .sortSel{display:block;width:100%;margin:0 0 10px}
 .rankDesk{display:none}.rankCards{display:block}
 .homeHero{padding:14px 0 22px}.homeHero h1{font-size:36px;letter-spacing:-1px}.homeHero .sub{font-size:16px}
 .cta{font-size:13px;padding:11px 14px;flex:1 1 44%}.homeActions{gap:8px}
 .homeGrid{padding-top:22px}.homeBlock{margin-bottom:30px}
 .miniRow{grid-template-columns:22px minmax(0,1fr) 58px 58px;gap:6px;font-size:15px;padding:12px 0}.miniRow .score{font-size:18px}
 .miniHead{font-size:11px!important}
 .how3{grid-template-columns:1fr;gap:18px}.how3 div{font-size:15px}.how3 b{font-size:17px}
 .hero2{grid-template-columns:1fr 1fr;grid-template-areas:"ph ph" "id id" "ml wk";column-gap:10px;row-gap:14px;padding:0 0 22px}
 .hp{justify-self:center}.hero2 .portrait{width:156px;height:156px}.hero2 .portrait img{width:150px;height:150px}.initials{font-size:44px}
 .hs.left,.hs.right{text-align:center}
 .hs .scorebig{font-size:54px}.hs .scorelab{font-size:15px;margin-top:6px}.hs .scoresub{font-size:12px}
 .outlook{font-size:13px;padding:6px 11px;margin-top:8px}.olDesc{font-size:12.5px;max-width:none;margin-top:6px}
 .hid h1{font-size:32px;letter-spacing:-.8px}.hid .meta{font-size:15px}.hid .rankline{font-size:13px}
 .disclaimer{font-size:12px;margin-top:14px;text-align:left}
 .stats{grid-template-columns:repeat(3,1fr)}.stat{padding:14px 4px;min-height:0}.sv{font-size:21px}.sk{font-size:11px}
 .grid2{grid-template-columns:minmax(0,1fr);gap:0}
 .panel{margin-top:20px;padding-top:16px}
 .barrow{grid-template-columns:minmax(0,1fr) 42px;grid-template-areas:"l n" "t t";row-gap:7px;margin:14px 0}
 .barrow>span{grid-area:l}.barrow>.track{grid-area:t}.barrow>b{grid-area:n}
 .lblBtn{font-size:15px}.pctNum{font-size:18px}.bucketHelp{font-size:13px;margin:-4px 0 14px}
 .usageGrid{grid-template-columns:1fr 1fr;gap:8px}.usageItem{padding:12px}.usageItem .uv{font-size:25px}.usageItem .ul{font-size:13px;margin-top:6px}
 .usageItem .uPct{font-size:13px;margin-top:7px}.usageItem .uavg{font-size:12px}.usageItem .ud{font-size:12px;line-height:1.4}
 th,td{font-size:13.5px;padding:10px 7px}th{font-size:10.5px}
 .roomtable th,.roomtable td{font-size:14px;padding:10px 7px}.roomtable td:first-child,.roomtable th:first-child{position:sticky;left:0;background:var(--bg);z-index:1}
 .shareTable th,.shareTable td{font-size:14px;padding:11px 8px}.shareTable{table-layout:auto}.shareTable td:first-child,.shareTable th:first-child{width:auto;min-width:118px}
 .shareSection h2{font-size:21px}.shareSection{margin:22px 0 28px}
 .matchHead{flex-direction:column;align-items:flex-start;gap:6px}.matchScore{font-size:30px}
 .gamechip{font-size:11.5px;padding:6px 8px}
 .compareHero{grid-template-columns:1fr 1fr;gap:8px;padding:18px 0}.vs{display:none}
 .cmpPortrait{width:100px;height:100px;margin-bottom:10px}.cmpPortrait img{width:94px;height:94px}
 .comparePlayer h2{font-size:17px}.comparePlayer .scorebig{font-size:38px}.comparePlayer .scorelab{font-size:12px}.comparePlayer .meta{font-size:12px}
 .edge strong{font-size:24px}.cmpStats th,.cmpStats td{font-size:13.5px;padding:10px 5px}
}

/* v8 — percentile bubbles, compact compare, nav fits phones */
.barrow{grid-template-columns:215px minmax(0,1fr) 44px;gap:14px}
.track{height:10px;border-radius:5px}.fill{border-radius:5px}
.pctNum{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;color:#e7edf3;font-size:16px;font-weight:800;line-height:1;text-align:center;justify-self:end}
.gdet{margin-top:10px}.gdet summary{cursor:pointer;color:#3f464e;font-size:14px;font-weight:700;list-style:none;padding:8px 0}.gdet summary::-webkit-details-marker{display:none}.gdet summary:after{content:" ▾";color:#676f79}.gdet[open] summary:after{content:" ▴"}
@media(max-width:700px){
 .links{justify-content:space-between}.navb{flex:0 0 auto;font-size:11.5px;padding:0 6px}
 .pctNum{width:36px;height:36px;font-size:15px}
 .barrow{grid-template-columns:minmax(0,1fr) 36px}
 .title{font-size:28px}
}
.spc{font-size:13px;font-weight:700;margin-top:4px}
@media(max-width:700px){.spc{font-size:11.5px}}

/* v9 — light blue theme, cards, clean type */
:root{--bg:#f1f6fd;--surface:#ffffff;--surface2:#f6f9fe;--line:#dbe4f1;--ink:#0f2038;--muted:#5a6b86;--blue:#2563eb;--blue2:#1d4ed8;--green:#12834a;--yellow:#a46c00;--red:#cc3340;--shadow:0 1px 2px rgba(15,32,56,.05),0 8px 24px rgba(37,99,235,.06)}
html{background:#f1f6fd}
body{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-feature-settings:normal;color:var(--ink);background:linear-gradient(180deg,#e6efff 0,#f1f6fd 380px,#f7faff 100%) no-repeat,#f7faff;min-height:100vh}
.scorebig,.sv,td,th,.pctNum,.rs,.score,.uv,.matchScore{font-variant-numeric:tabular-nums}
h1,h2,.title,.scorebig,.brand{letter-spacing:-.01em}
.title,.hid h1,.homeHero h1,.comparePlayer h2,.edge strong{font-weight:700}
.scorebig{font-weight:700}
.topbar{background:rgba(255,255,255,.9);backdrop-filter:saturate(160%) blur(10px);-webkit-backdrop-filter:saturate(160%) blur(10px);border-bottom:1px solid var(--line);box-shadow:0 1px 0 rgba(15,32,56,.02)}
.brand{color:var(--ink);font-weight:800;letter-spacing:.01em}.brand span{color:var(--blue)}
.navb{color:var(--muted);font-weight:700}.navb.on,.navb:hover{color:var(--ink);border-bottom-color:var(--blue)}
.kicker{color:var(--blue2)}
.sub,.muted,.meta,.rankline{color:var(--muted)}
.rankline{color:#33445f}
.sectionTitle,.ph{color:var(--ink);font-weight:800}
.search input{background:#fff;border:1px solid var(--line);color:var(--ink);box-shadow:var(--shadow)}
.search input::placeholder{color:#8a99b1}.search input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.15)}
.dd{background:#fff;border:1px solid var(--line);box-shadow:0 18px 44px rgba(15,32,56,.14);border-radius:10px;overflow:hidden}
.ddi{border-bottom:1px solid #edf2f9}.ddi:hover{background:#eef4ff}.ddiName{color:var(--ink)}.ddiMeta{color:var(--muted)}.ddiRate{color:var(--green)}
.pill{background:#fff;border:1px solid var(--line);color:#33445f;font-weight:700}
.pill:hover{border-color:var(--blue);color:var(--blue2);background:#fff}.pill.on{background:var(--blue);border-color:var(--blue);color:#fff}
.cta{background:#fff;border:1px solid var(--line);color:var(--ink)}.cta:hover{border-color:var(--blue)}.cta.primary{background:var(--blue);border-color:var(--blue);color:#fff}
.teamSelect{background:#fff;color:var(--ink);border:1px solid var(--line)}
.reset{background:#fff;border:1px solid var(--line);color:#33445f}
.tag{background:#eaf0fb;color:#3d4f6d}.partial{background:#fff3d6;color:#8a5a00}
.g90,.g80{color:var(--green)}.g70,.g60{color:var(--yellow)}.g0{color:var(--red)}
.good{color:var(--green)}.bad{color:var(--red)}
.playerlink,.sharePlayer,.pn,.rname{color:var(--ink)}
.playerlink:hover,.sharePlayer:hover,.pn:hover,[data-open]:hover,.roomtable td[data-id]:hover{color:var(--blue2);text-decoration:underline}

/* cards */
.panel,.stats,.hero2,.shareSection,.compareHero,.cmpStats,.edge,.miniRank,.how3>div,#rankings .tablewrap{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.panel{padding:24px;margin-top:20px}
.colL,.colR{min-width:0}
.hero2{padding:36px 28px 30px;border-bottom:1px solid var(--line);margin-bottom:0;background:linear-gradient(180deg,#fff 0,#f7faff 100%)}
.stats{margin-top:20px;border-bottom:1px solid var(--line)}
.stat+.stat{border-left:1px solid #edf2f9}
.sk{color:var(--muted)}.sv{color:var(--ink)}
th{color:#6a7b96;font-weight:700}
th,td{border-bottom:1px solid #e9eff8}tr:last-child td{border-bottom:0}
tbody tr:hover td{background:#f5f9ff}
#rankings .tablewrap{padding:6px 18px;margin-top:6px}
.shareSection{padding:22px 24px;margin:22px 0}.shareSection h2{margin-bottom:6px}
.shareTable td.season{background:#f1f6fd;color:var(--ink)}
.shareTable td.up{color:var(--green);font-weight:700}.shareTable td.down{color:var(--red);font-weight:700}
.shareTable th:first-child,.shareTable td:first-child,.roomtable td:first-child,.roomtable th:first-child{background:#fff}
.roomtable .you{color:var(--blue2)}

/* hero: scores aligned, outlook under name */
.hero2{grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);grid-template-areas:"ml ph wk" "id id id" "ol ol ol";column-gap:56px;row-gap:22px;align-items:center}
.hs.left{align-self:center}.hs.right{align-self:center}
.hs .scorebig{font-size:76px;line-height:1}
.hs .scorelab{color:var(--ink);font-size:18px;font-weight:800;margin-top:8px}
.hs .scoresub{color:var(--muted);font-size:14px;margin-top:4px}
.hs.left .scoresub,.hs.right .scoresub{white-space:nowrap}
.hol{grid-area:ol;display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;text-align:left;padding-top:4px}
.olDesc{margin:0;font-size:15px;color:#33445f;max-width:480px}
.outlook{margin-top:0;font-size:16px;padding:8px 16px;font-weight:800;border:1.5px solid currentColor}
.outlook.g90{background:#e7f7ee}.outlook.g70{background:#fdf3d9}.outlook.g0{background:#fde9eb}
.disclaimer:empty{display:none}
.portrait,.cmpPortrait{isolation:isolate}.portrait{background:#e8f0fc}.portrait:after{background:#fff}
.hero2 .portrait{width:240px;height:240px}.hero2 .portrait img{width:232px;height:232px}
.disclaimer{background:#eaf1fe;border-left:3px solid var(--blue);color:#3d4f6d;border-radius:8px;padding:12px 16px;max-width:none;margin:18px 0 0;text-align:left;font-size:13.5px}
.disclaimer.left{margin-left:0}

/* bars */
.track{background:#e8eef8}
.lblBtn{color:var(--ink)}.lblBtn:hover{color:var(--blue2)}.lblBtn .chev{color:#7b8caa}
.bucketHelp{background:#f1f6fd;border-left:3px solid var(--blue);color:#33445f}
.pctNum{background:#fff;border:3px solid currentColor;color:var(--ink);width:42px;height:42px;font-size:15px;font-weight:800}
.pctNum.g90{color:var(--green)}.pctNum.g70{color:var(--yellow)}.pctNum.g0{color:var(--red)}
.legend{color:var(--muted)}
.usageItem{background:#f8fbff;border:1px solid var(--line);border-radius:12px}.usageItem .ul{color:var(--ink)}.usageItem .uavg{color:#33445f}.usageItem .ud{color:var(--muted)}.ubar{background:#e3ebf7}
.uPct span{color:var(--muted)}
.gamechip{background:#fff;border:1px solid var(--line);color:#33445f}.gamechip:has(input:checked){border-color:var(--blue);background:#eaf1ff;color:var(--blue2)}
.gdet summary{color:var(--blue2)}

/* home + compare + rankings */
.homeHero{border-bottom:0;padding:38px 0 26px}.homeHero h1{color:var(--ink);font-weight:700;letter-spacing:-.02em}.homeHero h1 span{color:var(--blue)}
.miniRank{padding:6px 22px}.how3>div{padding:22px;color:#33445f}.how3 b{color:var(--ink)}
.miniHead{color:#6a7b96!important}
.compareHero{padding:30px 20px;margin-top:22px;border-top:1px solid var(--line)}
.cmpPortrait{background:#e8f0fc}.cmpPortrait:after{background:#fff}
.vs{color:#7b8caa}.edge{padding:22px;margin-top:18px;border-bottom:1px solid var(--line)}.cmpStats{padding:8px 20px;margin-top:18px;max-width:none}
.cmpOl .outlook{font-size:14px}
.rcard{background:#fff}.rcHead{color:#6a7b96}.rn{color:#8a99b1}.rstats{color:var(--muted)}
.disclaimer.left{margin-left:0}
@media(max-width:700px){
 .hero2{grid-template-columns:1fr 1fr;grid-template-areas:"ph ph" "id id" "ml wk" "ol ol";column-gap:10px;row-gap:14px;padding:22px 14px 20px}
 .hero2 .portrait{width:150px;height:150px}.hero2 .portrait img{width:142px;height:142px}
 .hs .scorebig{font-size:50px}.hs .scorelab{font-size:15px}.hs .scoresub{font-size:12px;white-space:normal}
 .hol{flex-direction:column;gap:8px;text-align:center}.olDesc{font-size:13.5px}
 .panel{padding:18px 16px}.shareSection{padding:16px}
 .pctNum{width:36px;height:36px;font-size:14px}
 .stat+.stat{border-left:0}
 .compareHero{padding:18px 8px}.cmpStats{padding:4px 10px}
}

/* v10 — thick ring, compare pull-downs, screenshot view */
.portrait:before{inset:-13px}.portrait:after{inset:-2px}
.cmpPortrait:before{inset:-10px}.cmpPortrait:after{inset:-1px}
.cmpPortrait{margin-bottom:20px}
.backBtn{background:#fff;border:1px solid #b2bbc6;color:#1d2530;border-radius:8px;padding:9px 16px;font-size:15px;font-weight:800;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.06)}.backBtn:hover{background:#eef2f7}#profile .topRow,#board .topRow{position:sticky;top:62px;z-index:40;background:var(--bg);padding:10px 0;margin-bottom:6px}.lnk{cursor:pointer}.lnk:hover{filter:brightness(.95);text-decoration:underline}.pctNum.lnk:hover{text-decoration:none;transform:scale(1.07)}.boardT .r{text-align:right}.boardT tr.hlRow td{background:#fff3c4;font-weight:800}.boardT{width:100%;table-layout:fixed;min-width:0}.boardT td,.boardT th{padding:9px 6px}.boardT th:nth-child(1),.boardT td:nth-child(1){width:34px}.boardT th:nth-child(3),.boardT td:nth-child(3){width:86px}.boardT th:nth-child(4),.boardT td:nth-child(4){width:62px}.boardT td:nth-child(2){overflow:hidden}#board .tablewrap{overflow-x:visible}@media(max-width:700px){.boardT th:nth-child(3),.boardT td:nth-child(3){width:70px}.boardT th:nth-child(4),.boardT td:nth-child(4){width:48px}.boardT .tag{display:block;width:fit-content;margin:2px 0 0}}@media(max-width:900px){#profile .topRow,#board .topRow{top:54px}}
.topRow{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.shotBtn{background:#fff;border:1px solid var(--line);color:#33445f;border-radius:999px;padding:8px 14px;font-size:13px;font-weight:700;cursor:pointer}
.shotBtn:hover{border-color:var(--blue);color:var(--blue2)}
.shotMark,.shotExit{display:none}
.matchBox{text-align:right;flex:0 0 auto}.mlab{font-size:12.5px;color:var(--muted);font-weight:700;margin-top:2px}
.matchWhy{background:#f1f6fd;border-left:3px solid var(--blue);border-radius:8px;padding:12px 14px;margin:10px 0 14px;font-size:15px;line-height:1.55;color:#33445f}
.matchWhy b{color:var(--ink)}
.cdet{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);margin-top:18px}
.cdet>summary{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:18px 22px;font-weight:800;font-size:17px;cursor:pointer;list-style:none}
.cdet>summary::-webkit-details-marker{display:none}
.cdet>summary i{font-style:normal;font-size:13px;font-weight:700;color:var(--blue2);white-space:nowrap}
.cdet>summary i:after{content:" \25BE"}.cdet[open]>summary i{font-size:0}.cdet[open]>summary i:after{content:"Hide \25B4";font-size:13px}
.cdBody{padding:6px 22px 22px;border-top:1px solid #e9eff8}
.cwin{text-align:center;padding:16px 0 6px}.cwin strong{display:block;font-size:30px;font-weight:700;margin:4px 0}.cwk{font-size:13px;color:var(--blue2);font-weight:800}
.cstart{width:100%;margin-top:10px}.cstart th{text-align:center;font-size:12px}.cstart td{text-align:center}.cstart .cmid{color:var(--muted);font-weight:700}
.cstart .outlook{font-size:13px;padding:5px 10px}
.cexp{font-size:14px;color:var(--muted);line-height:1.55;margin-top:14px}.cexp b{color:var(--ink)}
.cbhead{display:flex;justify-content:space-between;gap:12px;font-weight:800;font-size:16px;padding:14px 0 2px}
.cbrow{margin:14px 0}.cbl{text-align:center;font-weight:700;font-size:15px;margin-bottom:7px}
.cbbars{display:flex;align-items:center;gap:8px}
.cbbars .pctNum{flex:0 0 auto;width:40px;height:40px;font-size:14px}
.cbt{flex:1;height:13px;background:#e8eef8;border-radius:7px;display:flex;overflow:hidden}.cbt.l{justify-content:flex-end}.cbt>div{height:100%;border-radius:7px}
@media(max-width:700px){
 .portrait:before{inset:-10px}.cmpPortrait:before{inset:-8px}
 .cdet>summary{padding:15px 16px;font-size:15.5px}.cdBody{padding:4px 14px 18px}
 .cwin strong{font-size:24px}.cbbars .pctNum{width:34px;height:34px;font-size:13px}
 .shotBtn{padding:7px 12px;font-size:12px}
}
/* screenshot view: zoom out, strip the controls, add a footer mark */
body.shot{background:#fff}
body.shot .topbar .links,body.shot .shotBtn,body.shot .search,body.shot .gdet,body.shot .gamePanelTop,body.shot #backRanks,body.shot .gameControls,body.shot .reset,body.shot .custom,body.shot #compare .grid2,body.shot #compare .sub,body.shot .infoBtn .chev{display:none!important}
body.shot .topbar{position:static}
body.shot .wrap{zoom:.62;padding-top:12px}
body.shot .shotMark{display:block;text-align:center;color:var(--muted);font-size:13px;font-weight:700;margin-top:26px}
body.shot .shotExit{display:block;text-align:center;padding:18px}
@media(min-width:701px){body.shot .wrap{zoom:.85}}
.stats{grid-template-columns:repeat(7,1fr)}.gamePanelTop{padding-bottom:14px}.gamePanelTop .custom{margin:4px 0 10px}.gamePanelTop .gameControls{margin-bottom:0}@media(max-width:900px){.stats{grid-template-columns:repeat(4,1fr)}}@media(max-width:480px){.stats{grid-template-columns:repeat(3,1fr)}}
.rcard{grid-template-areas:"rn main a b" ". chips chips chips"}.rcard .rn{grid-area:rn}.rcard .rmain{grid-area:main}.rchips{grid-area:chips;display:grid;gap:6px 4px;margin-top:6px;padding-top:9px;border-top:1px dashed var(--line)}.rchips.c4{grid-template-columns:repeat(4,1fr)}.rchips.c3{grid-template-columns:repeat(3,1fr)}.rchips div{text-align:center}.rchips b{display:block;font-size:15px;font-weight:800}.rchips span{display:block;font-size:9.5px;color:var(--muted);font-weight:700;letter-spacing:.3px;margin-top:1px}.rs.sec{font-size:16px;opacity:.75}.rmain{min-width:0}
.bdgs{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.bdgs.c{justify-content:center;margin:8px 0 2px}.bdg{display:inline-block;background:#eef3fb;border:1px solid #c5d3ea;color:#1c2b4a;font-weight:800;font-size:12.5px;padding:5px 11px;border-radius:999px}.bdgD{margin-top:8px;font-size:13px;color:#4a5a73;line-height:1.4;max-width:460px}.bdgD b{color:#1c2b4a}.bdgMini{font-size:12px;color:var(--muted);font-weight:600}.bdsCell{font-size:12.5px;color:#33445f}.hol{align-items:flex-start;text-align:left}@media(max-width:700px){.hol{align-items:center;text-align:center}.hol .bdgs{justify-content:center}.hol .bdgD{max-width:none;text-align:left}}
.bi{display:inline-flex;gap:3px;margin-left:7px;vertical-align:middle}.bi i{width:22px;height:22px;border-radius:50%;background:#e6edfa;color:#27407a;display:inline-grid;place-items:center}.bi svg{width:13px;height:13px}.rsub .bi{margin-left:6px}.bdgs{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;justify-content:center}.bdgBtn{display:inline-flex;align-items:center;gap:7px;background:#eef3fb;border:1px solid #c5d3ea;border-radius:999px;padding:4px 12px 4px 4px;font:inherit;font-weight:800;font-size:12.5px;color:#1c2b4a;cursor:pointer}.bdgBtn i{width:24px;height:24px;border-radius:50%;background:#27407a;color:#fff;display:inline-grid;place-items:center}.bdgBtn svg{width:14px;height:14px}.bdgBtn.on{background:#1c2b4a;color:#fff;border-color:#1c2b4a}.bdgBtn.on i{background:#fff;color:#1c2b4a}.bdgD{margin:10px auto 0;font-size:13.5px;color:#33445f;line-height:1.4;max-width:420px;text-align:center}.bdgD[hidden]{display:none}.bdgD b{color:#1c2b4a}.hol{flex-direction:column!important;align-items:center!important;text-align:center!important;gap:0!important}.cmpBd .bdgs{margin-top:8px}.cmpBd .bdgBtn{font-size:11.5px}.cmpBd .bdgD{font-size:12.5px}.hol .outlook{margin:0}
.compareHero{align-items:start!important}.compareHero .vs{align-self:center}
.bi i{width:22px;height:25px;border-radius:0;background:none;color:inherit}.bi svg{width:22px;height:25px}.bi{gap:2px}.bdgBtn{padding:3px 12px 3px 5px}.bdgBtn i{width:24px;height:27px;border-radius:0;background:none;color:inherit}.bdgBtn svg{width:24px;height:27px}.bdgBtn.on i{background:none;color:inherit}.bdgBtn.on{background:#1c2b4a;color:#fff}.cmpBd .bdgBtn svg{width:20px;height:23px}.cmpBd .bdgBtn i{width:20px;height:23px}
.cmpStats table,.cstart{width:100%;table-layout:fixed}.cmpStats th,.cmpStats td,.cstart th,.cstart td{text-align:center!important;padding-left:6px!important;padding-right:6px!important}.cmpStats th:nth-child(2),.cmpStats td:nth-child(2),.cstart th:nth-child(2),.cstart td:nth-child(2){width:38%}.cmpStats th:first-child,.cmpStats th:last-child{white-space:normal;overflow-wrap:anywhere}.cmpStats.tablewrap{overflow-x:visible}
@media(max-width:700px){.cstart .outlook{font-size:11px;padding:4px 7px;white-space:nowrap}.cstart td,.cstart th{padding-left:3px!important;padding-right:3px!important}.cstart th:nth-child(2),.cstart td:nth-child(2){width:26%}}
.badgePick{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}.badgePick[hidden]{display:none}.bpick{display:inline-flex;align-items:center;gap:7px;background:#fff;border:1px solid #c9d6ea;border-radius:999px;padding:4px 12px 4px 5px;font:inherit;font-weight:800;font-size:13px;color:#1c2b4a;cursor:pointer}.bpick i{width:24px;height:27px;display:inline-grid;place-items:center}.bpick svg{width:24px;height:27px}.bpick em{font-style:normal;font-weight:700;font-size:11.5px;color:#6a7b96;background:#eef3fb;border-radius:999px;padding:1px 7px}.bpick.on{background:#1c2b4a;color:#fff;border-color:#1c2b4a}.bpick.on em{background:rgba(255,255,255,.18);color:#fff}@media(max-width:700px){.badgePick{gap:6px}.bpick{font-size:12px;padding:3px 10px 3px 4px}.bpick i,.bpick svg{width:20px;height:23px}}
.badgePick.sel{flex-wrap:nowrap;overflow-x:auto;padding-bottom:6px;scrollbar-width:thin}.badgePick.sel .bpick{flex:0 0 auto}
/* insights — one nav tab; its sub-tabs appear once there is more than one */
.injTag{font-size:11px;font-weight:700;color:var(--muted);background:#fff;border:1px solid var(--line);border-radius:6px;padding:1px 6px}.injItem{padding:12px 14px;border-bottom:1px solid var(--line)}.injItem:last-child{border-bottom:0}.injTop{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-weight:700}.injNote{color:var(--muted);font-size:13px;margin-top:3px}.injRep{font-size:13px;margin-top:6px;line-height:1.7}.injRep .up{color:#1a9e5a;font-weight:800}
.insNav{display:flex;flex-wrap:wrap;gap:2px 4px;border-bottom:1px solid var(--line);margin:-14px 0 26px;scrollbar-width:none}.insNav::-webkit-scrollbar{display:none}.insNav:has(.insTab:only-child){display:none}
.insTab{flex:0 0 auto;border:0;background:none;font:inherit;font-size:15px;font-weight:800;color:var(--muted);padding:12px 14px;cursor:pointer;border-bottom:2px solid transparent}.insTab.on,.insTab:hover{color:var(--ink);border-bottom-color:var(--blue)}
.insPane{display:none}.insPane.on{display:block}
/* report card */
.rcGrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0 4px}
/* schedule strength */
.sch{display:inline-block;margin:0 6px 6px 0;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:13px;font-weight:700}.sch.good,.scTable td.good{background:#e7f7ee}.sch.bad,.scTable td.bad{background:#fde9eb}.sch.muted{font-weight:600}
.scTable .c{text-align:center}.scSum{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}.scBox{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fff}.scBox b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.5px}.scBox span{font-size:17px;font-weight:800}.scBox.good b{color:var(--green)}.scBox.bad b{color:var(--red)}.scKey{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-bottom:10px}.scKeyNote{font-size:13px;color:var(--muted);margin-left:6px}.scList{border:1px solid var(--line);border-radius:10px;background:#fff;overflow:hidden}.scRow{padding:12px 14px;border-bottom:1px solid var(--line)}.scRow:last-child{border-bottom:0}.scTop{display:flex;align-items:center;gap:10px}.scRank{width:22px;color:var(--muted);font-weight:800;font-size:13px}.scTeam{width:46px;font-weight:800;font-size:17px}.scBar{position:relative;flex:1;height:10px;background:#e4e9ef;border-radius:6px;overflow:hidden}.scBar:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:2px;background:#fff}.scBar i{position:absolute;top:0;bottom:0}.scBar i.good{background:var(--green)}.scBar i.bad{background:var(--red)}.scBar i.muted{background:#9aa5b1}.scVal{width:92px;text-align:right;font-weight:800;font-size:14px}.scVal small{display:block;font-size:11px;font-weight:700;opacity:.8}.scChips{margin-top:8px}.scChips .sch small{font-size:10px;opacity:.7;margin-right:3px}.scRow .scWho{display:block;margin:2px 0 0;font-size:13px}@media(max-width:700px){.scSum{grid-template-columns:1fr}.scTeam{width:40px}.scVal{width:78px}}.scWho{margin-left:10px;font-size:13px;font-weight:600;color:var(--muted)}
/* buy low / sell high */
.mkGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px;align-items:start}.mkRow{display:grid;grid-template-columns:24px minmax(0,1fr) 64px 64px;gap:10px;align-items:center;padding:13px 0;border-top:1px solid #e9eff8}.mkRow .n{color:#8a99b1;font-size:14px}.mkRow .tag{white-space:nowrap}.mkRow .partial{margin-left:6px;padding:3px 6px;font-size:10px;font-weight:800;white-space:nowrap}.mkWhy{font-size:13.5px;color:var(--muted);line-height:1.45;margin-top:4px}.mkPct{text-align:center}.mkPct .pctNum{margin:0 auto}.mkPct small{display:block;font-size:10.5px;color:var(--muted);font-weight:700;margin-top:4px}
@media(max-width:700px){.mkGrid{grid-template-columns:minmax(0,1fr)}.mkRow{grid-template-columns:18px minmax(0,1fr) 58px 58px;gap:6px}}
@media(max-width:700px){.navb{font-size:10px!important;padding:0 4px!important;letter-spacing:0!important;flex:1 1 auto}.links{justify-content:space-between}#insights table{font-size:12.5px}#insights th,#insights td{padding:9px 4px!important;white-space:normal}#insights th{font-size:10.5px;letter-spacing:.2px}#insights .tag{display:none}#insights .tablewrap{overflow-x:visible}}
.injc{display:inline-block;margin-left:6px;padding:2px 7px;border-radius:999px;font-size:10.5px;font-weight:800;vertical-align:middle;letter-spacing:.3px}.injc.o{background:#fde9eb;color:#c0303d}.injc.q{background:#fff3d6;color:#9a6a00}.injBox{background:#fff;border:1px solid #dbe5f4;border-radius:14px;padding:14px 16px;margin:14px 0}.injHd{font-size:12px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#6a7b96;margin-bottom:6px}.injRow{font-size:14px;line-height:1.45;padding:8px 0 8px 12px;border-left:3px solid #9aa8bd;margin-top:6px;color:#33445f}.injRow b{color:#1c2b4a}.injRow.o{border-color:#e2505b}.injRow.q{border-color:#e3a21a}.injRow.up{border-color:#22a45d}
</style></head><body>
<div class="topbar"><div class="nav"><div class="brand">FANTASY <span>MANIA</span></div><div class="links"><button class="navb on" data-v="home">HOME</button><button class="navb" data-v="players">PLAYERS</button><button class="navb" data-v="rankings">RANKINGS</button><button class="navb" data-v="compare">COMPARE</button><button class="navb" data-v="insights">INSIGHTS</button></div></div></div>
<main class="wrap">
<section id="home" class="view on"><div class="homeHero"><div class="kicker">2026 Fantasy Football • Through Week <span id="homeWeek"></span></div><h1>Who's actually getting the work.</h1><p class="sub">Usage, role and matchup for every RB, WR and TE, boiled down to one rating.</p><div class="search homeSearch"><input id="homeQ" placeholder="Search any RB, WR or TE"><div class="dd" id="homeDD"></div></div><div class="homeActions"><button class="cta" id="homeRanks">VIEW RANKINGS</button><button class="cta" id="homeCompare">COMPARE PLAYERS</button><button class="cta" id="homeShare">TEAM SHARE</button><button class="cta" id="homePlayers">BROWSE PLAYERS</button><button class="cta" id="homeInsights">INSIGHTS</button></div></div><div class="homeGrid"><div class="homeBlock"><div class="sectionTitle">Top players right now</div><div id="homeLeaders" class="miniRank"></div><p class="disclaimer left" id="homeDisc"></p></div><div class="homeBlock"><div class="sectionTitle">How Mania works</div><div class="how3"><div><b>Mania Rating</b>How strong his role and production have been this season, compared with other players at his position. It blends production, volume, team role, red-zone work and efficiency.</div><div><b>Week <span class="nextWeekText"></span> Rating</b>His Mania Rating adjusted for this week: the opponent, how players with a similar role have done against that defense, and his recent usage.</div><div><b>Percentiles and colors</b>Every bar shows where he ranks among players at his position. Green is the top 30%, yellow is the middle, red is the bottom 40%.</div></div></div></div></section>
<section id="players" class="view"><h1 class="title">Players</h1><p class="sub">Search a player to open the full Mania profile.</p><div class="search"><input id="playerQ" placeholder="Search player"><div class="dd" id="playerDD"></div></div></section>
<section id="rankings" class="view"><div class="kicker">Through Week <span id="wk"></span></div><h1 class="title" id="rankTitle">Overall Rankings</h1><p class="sub" id="rankSub"></p><div class="toolbar"><button class="pill rankmode on" data-mode="mania">Mania Rating</button><button class="pill rankmode" data-mode="start">Week start rating</button><button class="pill rankmode" data-mode="badges">Badges</button><span style="width:8px"></span><button class="pill posf on" data-pos="ALL">ALL</button><button class="pill posf" data-pos="RB">RB</button><button class="pill posf" data-pos="WR">WR</button><button class="pill posf" data-pos="TE">TE</button></div><div id="badgePick" class="badgePick" hidden></div><select id="sortSel" class="teamSelect sortSel"></select><p class="disclaimer left" id="rankDisc"></p><div class="tablewrap rankDesk"><table><thead id="rankHead"></thead><tbody id="rankBody"></tbody></table></div><div class="rankCards" id="rankCards"></div></section>
<section id="profile" class="view"><div id="profileBody"></div></section><section id="board" class="view"><div id="boardBody"></div></section>
<section id="insights" class="view"><div class="insNav"><button class="insTab on" data-v="injuries">Injuries</button><button class="insTab" data-v="share">Team Share</button><button class="insTab" data-v="market">Buy Low / Sell High</button><button class="insTab" data-v="defense">Defense vs. Position</button><button class="insTab" data-v="schedule">Schedule Strength</button><button class="insTab" data-v="report">Report Card</button></div>
<div class="insPane on" id="ins-injuries"><div class="kicker">Week <span class="nextWeekText"></span> report <span id="injUpd"></span></div><h1 class="title">Injuries</h1><p class="sub">Who is hurt, whether he is a starter, and who picks up the work. The points are the change to the replacement's Week rating, and they lean on how that player actually did in games the injured player missed.</p><div class="toolbar"><button class="pill injPos on" data-v="ALL">ALL</button><button class="pill injPos" data-v="RB">RB</button><button class="pill injPos" data-v="WR">WR</button><button class="pill injPos" data-v="TE">TE</button><span style="width:8px"></span><button class="pill injRole on" data-v="Starter">Starters</button><button class="pill injRole" data-v="ALL">All</button></div><div id="injBody"></div></div>
<div class="insPane" id="ins-share"><div class="kicker">Through Week <span id="shareWeek"></span></div><h1 class="title">Team Share</h1><p class="sub">See who is actually on the field and who is earning the offense each week. Every player name opens the full Fantasy Mania profile.</p><div class="teamControls"><select id="teamPick" class="teamSelect"></select><button class="pill shareMode on" data-share="snap">SNAP SHARE</button><button class="pill shareMode" data-share="target">TARGET SHARE</button></div><div id="teamShareBody"></div></div>
<div class="insPane" id="ins-market"><div class="kicker">Through Week <span id="mkWeek"></span></div><h1 class="title">Buy Low / Sell High</h1><p class="sub">Players whose role and fantasy points disagree. Usage is how much work he gets: volume, team role and red-zone chances. Production is the points and yards he has turned it into. Both are percentiles at his position.</p><div class="toolbar"><button class="pill mkPos on" data-v="ALL">ALL</button><button class="pill mkPos" data-v="RB">RB</button><button class="pill mkPos" data-v="WR">WR</button><button class="pill mkPos" data-v="TE">TE</button></div><div class="mkGrid" id="marketBody"></div><p class="disclaimer left">This compares usage and production so far this season. It flags a player who left his last game early or missed it, but it does not see injury reports, depth-chart changes or a new quarterback, and this early a game or two can swing it. Treat it as a list of names to look into, not a trade verdict.</p></div>
<div class="insPane" id="ins-defense"><div class="kicker">Through Week <span id="dvWeek"></span></div><h1 class="title">Defense vs. Position</h1><p class="sub">What each defense has given up to running backs, receivers and tight ends. "Vs usual" compares what players scored against this defense with what those same players normally score, so a defense is not punished for a schedule full of great offenses.</p><div class="toolbar"><button class="pill dvPos on" data-v="RB">RB</button><button class="pill dvPos" data-v="WR">WR</button><button class="pill dvPos" data-v="TE">TE</button></div><div class="shareNote">Easiest matchups first. Green defenses have allowed 10%+ more than usual, red ones 10%+ less.</div><div class="panel" id="defenseBody"></div></div>
<div class="insPane" id="ins-schedule"><div class="kicker">Week <span id="scWeek"></span> onward</div><h1 class="title">Schedule Strength</h1><p class="sub">Every team's remaining opponents, graded by what those defenses have given up to the position so far. It is the "vs usual" number from Defense vs. Position, looked up for each game still to come. Easiest schedules first.</p><div class="toolbar"><button class="pill scPos on" data-v="RB">RB</button><button class="pill scPos" data-v="WR">WR</button><button class="pill scPos" data-v="TE">TE</button><span style="width:8px"></span><button class="pill scWin on" data-v="next4">Next 4 weeks</button><button class="pill scWin" data-v="all">Rest of season</button><button class="pill scWin" data-v="playoffs">Weeks 15-17</button></div><div class="shareNote">Green is a defense allowing 10%+ more than usual to the position, red is 10%+ less. This early most defenses have only a few games behind them, so expect these to move.</div><div class="panel" id="scheduleBody"></div></div>
<div class="insPane" id="ins-report"><div class="kicker">Week <span class="rcWeek"></span> results</div><h1 class="title">Report Card</h1><p class="sub">How the Week <span class="rcWeek"></span> ratings held up. These are the ratings the model gives using only the games before Week <span class="rcWeek"></span>, set against what each player then scored.</p><div class="rcGrid" id="reportTiles"></div><div class="toolbar"><button class="pill rcPos on" data-v="RB">RB</button><button class="pill rcPos" data-v="WR">WR</button><button class="pill rcPos" data-v="TE">TE</button></div><div id="reportBody"></div></div>
</section>
<section id="compare" class="view"><div class="topRow"><h1 class="title">Compare Players</h1><button class="shotBtn" onclick="toggleShot(true)">Screenshot view</button></div><p class="sub">Overall value and Week <span id="cmpWeek"></span> start decision. Change the games used for either player and the model recalculates.</p><div class="grid2"><div class="search"><input id="aQ" placeholder="Player A"><div class="dd" id="aDD"></div></div><div class="search"><input id="bQ" placeholder="Player B"><div class="dd" id="bDD"></div></div></div><div id="compareBody"></div></section>
<div class="shotMark">FANTASY MANIA &bull; 2026 &bull; Through Week <span id="shotWk"></span></div></main><div class="shotExit"><button class="pill" onclick="toggleShot(false)">Exit screenshot view</button></div><script>
const DB=__PAYLOAD__, META=__META__, REFS=__REFS__; const $=id=>document.getElementById(id),byId=id=>DB.find(p=>p.id===id),fmt=n=>Math.round(Number(n)*10)/10;$('wk').textContent=META.week;$('shotWk').textContent=META.week;$('cmpWeek').textContent=META.next_week;
const DISC=`Week ${META.next_week} Rating is an estimate built from a player's recent role, his opponent and the latest NFL injury report${META.injuries&&META.injuries.updated?' (updated '+META.injuries.updated+')':''}. Injuries shift carries and targets to teammates, but the model cannot know final inactives or depth-chart decisions, so check the injury report before you set your lineup. Badges describe how a player is used, not how good he is, and the rank is his spot at his position by Week rating.`;
function ord(n){n=Math.round(n);let v=n%100,x=['th','st','nd','rd'];return n+(x[(v-20)%10]||x[v]||x[0])}
function fs(p,v){return p.out?'OUT':Number(v).toFixed(1)}
function injChip(p){if(!p.inj||p.inj.avail>0.6)return '';let t=p.out?'OUT':p.inj.label==='Missed practice'?'DNP':'Q';return `<span class="injc ${p.out?'o':'q'}" title="${p.inj.label}${p.inj.note?' ('+p.inj.note+')':''}">${t}</span>`}
function injHTML(p){let wk=META.next_week,o=[],pm=x=>(x>0?'+':'')+x.toFixed(1);if(p.inj){let pen=(4*(1-p.inj.avail)).toFixed(1);o.push(`<div class="injRow ${p.out?'o':'q'}"><b>${p.inj.label}</b>${p.inj.note?' ('+p.inj.note+')':''}. ${p.out?'He is not expected to play this week, so there is no Week '+wk+' Rating.':'His Week '+wk+' Rating is lowered by '+pen+' for the injury risk.'}</div>`)}(p.inj_why||[]).forEach(w=>{o.push(w.kind==='depth'?`<div class="injRow ${w.pts>0?'up':'q'}"><b>Depth chart</b> (${w.s}) ${pm(w.pts)} on Week ${wk} Rating. ${w.pts>0?'He is first up at his position, ahead of players who got more work earlier.':'The depth chart now has someone else ahead of him.'}</div>`:w.kind==='qb'?`<div class="injRow ${w.pts<0?'q':'up'}"><b>${w.n}</b> (QB, ${w.s}) ${pm(w.pts)} on Week ${wk} Rating. A hurt starting quarterback hurts the whole passing game.</div>`:`<div class="injRow up"><b>${w.n}</b> (${w.pos}, ${w.s}) ${pm(w.pts)} on Week ${wk} Rating. ${w.ev?`In the ${w.ev.games} game${w.ev.games>1?'s':''} without ${w.n}, his targets and carries were worth about ${w.ev.without} points, against ${w.ev.with} when ${w.n} played. That counts, scaled to how big ${w.n}'s role was.`:'His carries and targets are expected to shift to teammates.'}</div>`)});return o.length?`<div class="injBox"><div class="injHd">Injury report${META.injuries&&META.injuries.updated?' \u2022 updated '+META.injuries.updated:''}</div>${o.join('')}</div>`:''}
function outlook(p,start){
 if(p.opp==='BYE')return {t:'On bye',d:'',c:'g0',r:999};
 if(p.out)return {t:'OUT',d:'',c:'g0',r:998};
 let a=(REFS[p.pos]||{}).start||[],r=Math.abs(start-p.start)<1e-9?p.start_pos_rank:Math.max(1,1+a.filter(x=>x>start+.0005).length-(p.start>start?1:0)),lim=p.pos==='TE'?[6,12]:[12,24];
 return {t:p.pos+r+' this week',d:'',c:r<=lim[0]?'g90':r<=lim[1]?'g70':'g0',r}
}
function badges(p,idxs){
 let L=idxs.map(i=>p.logs[i]).filter(Boolean);if(!L.length)return [];
 let c=calc(p,idxs),M=c.M,P=k=>percentile(p.pos,k,M[k]),tg=sum(L,'tgt'),rc=sum(L,'rec'),ry=sum(L,'ry'),air=sum(L,'air'),adot=tg?air/tg:0,ypr=rc?ry/rc:0,pp=L.map(x=>x.ppr),mu=pp.reduce((x,y)=>x+y,0)/pp.length,sd=Math.sqrt(pp.reduce((x,y)=>x+(y-mu)*(y-mu),0)/pp.length),cv=mu>0?sd/mu:0,tdp=sum(L,'ppr')>0?6*(sum(L,'rtd')+sum(L,'rutd'))/sum(L,'ppr'):0,tr=p.trend_pct||0,o=[],add=(n,d,s)=>o.push({n,d,s});
 let pc=P('carries_pg'),pt=P('targets_pg'),ps=P('snap_pct'),pts=P('touch_share'),pgl=P('gl_carries_pg'),prz=P('rz_targets_pg'),ptsh=P('target_share'),pcr=P('catch_rate'),pyc=P('yards_per_carry'),pair=P('air_share');
 if(p.pos==='RB'){
  if(pc>=60&&pt>=60&&ps>=70)add('Three-Down Back','Runs, catches and stays on the field.',(pc+pt+ps)/3+8);
  if(pc>=70&&pts>=70)add('Workhorse','Big carry load and a big share of the team\u2019s touches.',(pc+pts)/2+5);
  if(pt>=75&&pc<=65)add('Pass-Catcher','Gets his work through the air more than on the ground.',pt);
  if(pgl>=75)add('Goal-Line Hammer','Gets the carries inside the 5.',pgl);
  if(pyc>=80&&M.carries_pg>=5)add('Home Run Hitter','Big yards every time he carries it.',pyc-5);
  if(tdp>=.4&&sum(L,'ppr')>0)add('TD Dependent','A big part of his points come from touchdowns.',60);
  if(pts<=55&&ps<=60)add('Committee Back','Shares the backfield work.',45);
  if(ps<=30)add('Change of Pace','Used in short bursts, not every down.',35);
  if(!o.length)add('Depth Back','Part of the rotation without a standout role.',10);
 }else if(p.pos==='WR'){
  if(ptsh>=85)add('Target Hog','A huge share of the team\u2019s targets run through him.',ptsh+5);
  if(adot>=11&&pair>=60)add('Deep Threat','Gets targets downfield and a big share of the air yards.',pair);
  if(adot<=8&&pcr>=60&&pt>=50)add('Short-Game Specialist','Short targets, catches most of them, PPR-friendly.',pcr);
  if(prz>=80)add('Red-Zone Weapon','Targeted near the goal line at a high rate.',prz);
  if(ypr>=14.5&&rc>=6)add('Big-Play Threat','Piles up yards on his catches.',70);
  if(L.length>=3&&cv>=.55)add('Boom or Bust','His weekly points swing a lot.',50);
  if(tr>=20)add('Rising Role','His recent targets are well up from his season average.',55);
  if(tr<=-20)add('Fading Role','His recent targets are well down from his season average.',40);
  if(!o.length)add('Role Player','Contributes without a standout trait yet.',10);
 }else{
  if(ptsh>=85)add('Featured Tight End','A real share of the offense runs through him.',ptsh+5);
  if(adot>=9.5)add('Seam Stretcher','Runs deeper routes than most tight ends.',70);
  if(adot<=6.5&&pcr>=55&&pt>=45)add('Safety Blanket','Short, steady targets for the quarterback.',pcr);
  if(prz>=80)add('Red-Zone Target','The team looks for him near the goal line.',prz);
  if(ps>=65&&pt<=30)add('Blocking Tight End','On the field a lot, but not much in the passing game.',30);
  if(tr>=20)add('Rising Role','His recent targets are well up from his season average.',55);
  if(tr<=-20)add('Fading Role','His recent targets are well down from his season average.',40);
  if(!o.length)add('Role Player','Contributes without a standout trait yet.',10);
 }
 let n=c.mania>=90?3:c.mania>=82?2:1;return o.sort((x,y)=>y.s-x.s).slice(0,n)
}
const BI={"Workhorse": "M6 8v8M18 8v8M3 10v4M21 10v4M6 12h12", "Three-Down Back": "M4 7l8-4 8 4-8 4zM4 12l8 4 8-4M4 17l8 4 8-4", "Pass-Catcher": "M12 3v12M7 10l5 5 5-5M5 21h14", "Goal-Line Hammer": "M5 21V4M5 4h12l-2 4 2 4H5", "Home Run Hitter": "M13 2L4 14h7l-1 8 9-12h-7z", "TD Dependent": "M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0zM7 6H4v2a3 3 0 0 0 3 3M17 6h3v2a3 3 0 0 1-3 3", "Committee Back": "M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM2 20c0-3 3-5 7-5s7 2 7 5M17 11a3 3 0 1 0 0-6M18 15c3 0 4 2 4 4", "Change of Pace": "M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5", "Depth Back": "M9 12a3 3 0 1 0 6 0a3 3 0 1 0 -6 0M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0", "Role Player": "M9 12a3 3 0 1 0 6 0a3 3 0 1 0 -6 0M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0", "Target Hog": "M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0M7 12a5 5 0 1 0 10 0a5 5 0 1 0 -10 0M11 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0", "Deep Threat": "M7 17L17 7M8 7h9v9", "Short-Game Specialist": "M8 7h12M8 7l3-3M8 7l3 3M16 17H4M16 17l-3-3M16 17l-3 3", "Red-Zone Weapon": "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0M12 2v5M12 17v5M2 12h5M17 12h5", "Red-Zone Target": "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0M12 2v5M12 17v5M2 12h5M17 12h5", "Big-Play Threat": "M12 3c1 4 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3 0-6 1-9z", "Boom or Bust": "M3 12h4l3-8 4 16 3-8h4", "Rising Role": "M3 17l6-6 4 4 8-8M15 7h6v6", "Fading Role": "M3 7l6 6 4-4 8 8M15 17h6v-6", "Featured Tight End": "M12 2l3 7 7 .6-5.3 4.7 1.6 7.2L12 17.8 6.7 21.5l1.6-7.2L3 9.6 10 9z", "Seam Stretcher": "M12 20V4M6 10l6-6 6 6", "Safety Blanket": "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z", "Blocking Tight End": "M5 5h14v14H5z"};
const BC={"Workhorse": "#c2410c", "Three-Down Back": "#7c3aed", "Pass-Catcher": "#0ea5e9", "Goal-Line Hammer": "#9f1239", "Home Run Hitter": "#f59e0b", "TD Dependent": "#92400e", "Committee Back": "#64748b", "Change of Pace": "#2dd4bf", "Depth Back": "#94a3b8", "Role Player": "#a8b0bd", "Target Hog": "#e11d48", "Deep Threat": "#2563eb", "Short-Game Specialist": "#16a34a", "Red-Zone Weapon": "#ef4444", "Red-Zone Target": "#f472b6", "Big-Play Threat": "#d946ef", "Boom or Bust": "#eab308", "Rising Role": "#059669", "Fading Role": "#78716c", "Featured Tight End": "#6366f1", "Seam Stretcher": "#0891b2", "Safety Blanket": "#0f766e", "Blocking Tight End": "#475569"};
function bIcon(n){let c=BC[n]||'#94a3b8';return `<svg viewBox="0 0 32 36"><path d="M16 1.5l13 4.2v11c0 9-5.8 14.8-13 18.3C8.8 31.5 3 25.7 3 16.7v-11z" fill="${c}" stroke="rgba(0,0,0,.28)" stroke-width="1.4"/><path d="M16 3.6l11 3.6v9.5c0 7.6-4.8 12.6-11 15.8" fill="rgba(255,255,255,.16)"/><g transform="translate(7.6 7.4) scale(.7)" fill="none" stroke="#fff" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="${BI[n]||BI['Role Player']}"/></g></svg>`}
function bIcons(l){return `<span class="bi">${l.map(b=>`<i title="${b.n}">${bIcon(b.n)}</i>`).join('')}</span>`}
function badgeHTML(l){return `<div class="bdgs">${l.map(b=>`<button type="button" class="bdgBtn" data-n="${b.n}" data-d="${b.d.replace(/"/g,'&quot;')}"><i>${bIcon(b.n)}</i>${b.n}</button>`).join('')}</div><div class="bdgD" hidden></div>`}
document.addEventListener('click',e=>{let btn=e.target.closest&&e.target.closest('.bdgBtn');if(!btn)return;let box=btn.closest('.bdgs').parentElement,d=box.querySelector('.bdgD'),was=btn.classList.contains('on');box.querySelectorAll('.bdgBtn').forEach(x=>x.classList.remove('on'));if(was){d.hidden=true;d.innerHTML=''}else{btn.classList.add('on');d.hidden=false;d.innerHTML='<b>'+btn.dataset.n+'</b> '+btn.dataset.d}});
function olHTML(p,o,idx){idx=idx||p.logs.map((_,i)=>i);return `<span class="outlook ${o.c}">${o.t}</span>${badgeHTML(badges(p,idx))}`}
DB.forEach(p=>{let o=outlook(p,p.start);p.ol=o.t;p.oc=o.c;p.orank=o.r;p.bl=badges(p,dfl(p));p.bds=p.bl.map(b=>b.n).join(' \u2022 ')});
let selBadge=null,posFilter='ALL',rankMode='mania',sortKey='mania',sortDir=-1,A=null,B=null,selA=null,selB=null;
function grade(v){return v>=80?'g90':v>=60?'g70':'g0'} function ring(v){return v>=80?'#22a45d':v>=60?'#e3a21a':'#e2505b'} function tier(v){return v>=90?'Elite':v>=82?'Excellent':v>=74?'Strong':v>=65?'Starter':'Depth'}

var STATF={ppr_pg:['ppr','',1],targets_pg:['tgt','',1],rec_pg:['rec','',1],scrim_pg:['scrim','',1],touch_share:['touch','%',1],target_share:['tshare','%',1],snap_pct:['snap','%',0],td_pg:['td','',2]};
function openBoard(spec){if(skipPush)skipPush=false;else pushNav();curBoard=spec;showView('board');
 let pop=DB.filter(p=>p.pos===spec.pos&&p.games>=1&&(p.m.snap>=30||p.m.ppr>=6)),f=STATF[spec.key]||['ppr','',1],val,show;
 if(spec.kind==='bucket'){let i=Number(spec.idx),cache={};pop.forEach(p=>{cache[p.id]=Object.values(calc(p,dfl(p)).b)[i]});val=p=>cache[p.id];show=p=>Math.round(percentile(p.pos,BK[i],cache[p.id]))}
 else{val=p=>p.m[f[0]];show=p=>fmt2(p.m[f[0]],f[2])+f[1]}
 let rows=[...pop].sort((a,b)=>val(b)-val(a)),hl=spec.hl;
 $('boardBody').innerHTML=`<div class="topRow"><button class="backBtn" onclick="goBack()">← Back</button></div><div class="kicker">Ranked among ${rows.length} ${spec.pos}s</div><h1 class="title">${spec.title}</h1><p class="sub">${spec.kind==='bucket'?'The number in the circle is the percentile among '+spec.pos+'s. ':''}Tap any name to open his profile. Your player is highlighted.</p><div class="panel"><div class="tablewrap"><table class="boardT"><thead><tr><th>#</th><th>Player</th><th class="r">${spec.kind==='bucket'?'Pctile':'Value'}</th><th class="r">Mania</th></tr></thead><tbody>${rows.map((p,i)=>`<tr class="${p.id===hl?'hlRow':''}" ${p.id===hl?'id="hlRow"':''}><td>${i+1}</td><td><span class="playerlink" data-open="${p.id}">${p.name}</span><span class="tag">${p.team}</span></td><td class="r"><b>${show(p)}</b></td><td class="r grade ${grade(p.mania)}">${p.mania}</td></tr>`).join('')}</tbody></table></div></div>`;
 let e=$('hlRow');if(e)setTimeout(()=>e.scrollIntoView({block:'center'}),30)}
function fmt2(v,d){return Number(v).toFixed(d)}
document.addEventListener('click',e=>{let t=e.target.closest('[data-board]');if(!t)return;e.stopPropagation();let v=document.querySelector('.view.on'),onP=v&&v.id==='profile'&&curPlayer;openBoard({kind:t.dataset.board,key:t.dataset.key,idx:t.dataset.idx,title:t.dataset.title+' \u2022 '+t.dataset.pos+'s',pos:t.dataset.pos,hl:onP?curPlayer.id:null})});
function showView(v){document.body.classList.remove('shot');document.querySelectorAll('.view').forEach(x=>x.classList.toggle('on',x.id===v));document.querySelectorAll('.navb').forEach(x=>x.classList.toggle('on',x.dataset.v===v));window.scrollTo(0,0)}document.querySelectorAll('.navb').forEach(b=>b.onclick=()=>showView(b.dataset.v));document.addEventListener('click',e=>{let t=e.target.closest('[data-open]');if(t&&t.dataset.open){let q=DB.find(x=>x.id===t.dataset.open);if(q)openPlayer(q)}});
function initials(p){return p.name.split(' ').slice(0,2).map(x=>x[0]).join('')} function pic(p){return p.headshot?`<img src="${p.headshot}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><span class="initials" style="display:none">${initials(p)}</span>`:`<span class="initials">${initials(p)}</span>`}
function searchBox(inp,dd,cb){$(inp).oninput=e=>{let q=e.target.value.toLowerCase().trim(),d=$(dd);if(!q){d.style.display='none';return}let m=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);d.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><div><span class="ddiName">${p.name}</span><span class="ddiMeta">${p.team} • ${p.pos}</span></div><span class="ddiRate">${p.mania}</span></div>`).join('');d.style.display=m.length?'block':'none';d.querySelectorAll('.ddi').forEach(x=>x.onclick=()=>{let p=byId(x.dataset.id);d.style.display='none';$(inp).value=p.name;cb(p)})}}
searchBox('homeQ','homeDD',openPlayer);searchBox('playerQ','playerDD',openPlayer);searchBox('aQ','aDD',p=>{A=p;selA=dfl(p);renderCompare()});searchBox('bQ','bDD',p=>{B=p;selB=dfl(p);renderCompare()});
const COLS_M=[['rank','#'],['name','Player'],['mania','Mania'],['start','Wk'],['m.ppr','PPR/G'],['m.tgt','TGT/G'],['m.rec','REC/G'],['m.recy','REC YD/G'],['m.car','CAR/G'],['m.rushy','RUSH YD/G'],['m.snap','SNAP%'],['share','SHARE'],['rz','RZ']];
const COLS_S=[['rank','#'],['name','Player'],['start','Wk'],['opp','Opp'],['matchup_adj','Matchup boost'],['trend_adj','Role trend'],['outlook','Pos rank'],['mania','Mania'],['m.ppr','PPR/G'],['m.snap','SNAP%']];
function COLS(){return rankMode!=='start'?COLS_M:COLS_S}
function val(p,k){if(k==='rank')return rankMode!=='start'?p.rank:p.start_rank;if(k==='name')return p.name;if(k==='outlook')return -p.orank;if(k==='share')return p.pos==='RB'?p.m.touch:p.m.tshare;if(k==='rz')return p.pos==='RB'?p.m.rzc:p.m.rzt;if(k.startsWith('m.'))return p.m[k.slice(2)];return p[k]}
let shareMode='snap';function renderTeamShare(){let teams=[...new Set(DB.map(p=>p.team))].sort(),pick=$('teamPick');if(!pick.options.length)pick.innerHTML=teams.map(t=>`<option>${t}</option>`).join('');let team=pick.value||teams[0];if(!pick.value)pick.value=team;$('shareWeek').textContent=META.week;let isSnap=shareMode==='snap',ps=DB.filter(p=>p.team===team),weeks=[...new Set(ps.flatMap(p=>p.logs.map(l=>l.w)))].sort((a,b)=>a-b);
 let seasonOf=p=>isSnap?p.m.snap:(p.pos==='RB'?p.m.tshare:p.m.tshare);
 $('teamShareBody').innerHTML=`<div class="shareNote">${isSnap?'Share of the offense snaps each player was on the field.':'Share of the team passes thrown to each player.'} Green or red numbers moved 5+ points from the week before.</div>`+['RB','WR','TE'].map(pos=>{let q=ps.filter(p=>p.pos===pos&&(p.m.snap>=8||p.m.tshare>=3));if(!q.length)return'';q.sort((a,b)=>seasonOf(b)-seasonOf(a));return `<div class="shareSection"><h2>${pos==='RB'?'Running backs':pos==='WR'?'Wide receivers':'Tight ends'}</h2><div class="tablewrap"><table class="shareTable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>Wk ${w}</th>`).join('')}<th>Season</th></tr></thead><tbody>${q.map(p=>{let prev=null,cells=weeks.map(w=>{let l=p.logs.find(x=>x.w===w);if(!l){return '<td>—</td>'}let v=isSnap?l.snap:l.tshare,cls=prev==null?'':(v-prev>=5?'up':v-prev<=-5?'down':'');prev=v;return `<td class="${cls}">${fmt(v)}%</td>`}).join('');return `<tr><td><span class="sharePlayer" data-id="${p.id}">${p.name}</span></td>${cells}<td class="season">${fmt(seasonOf(p))}%</td></tr>`}).join('')}</tbody></table></div></div>`}).join('');
 $('teamShareBody').querySelectorAll('.sharePlayer').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)))}
function renderHome(){if($('homeWeek'))$('homeWeek').textContent=META.week;document.querySelectorAll('.nextWeekText').forEach(x=>x.textContent=META.next_week);let top=[...DB].sort((a,b)=>b.mania-a.mania).slice(0,5);$('homeLeaders').innerHTML=`<div class="miniRow miniHead"><span></span><span>Player</span><span style="text-align:right">Mania Rating</span><span style="text-align:right">Week ${META.next_week} Rating</span></div>`+top.map((p,i)=>`<div class="miniRow"><span class="n">${i+1}</span><span class="pn" data-id="${p.id}">${p.name} <span class="tag">${p.team} ${p.pos}</span></span><span class="score ${grade(p.mania)}">${Number(p.mania).toFixed(1)}</span><span class="score ${grade(p.start)}">${fs(p,p.start)}</span></div>`).join('');$('homeLeaders').querySelectorAll('.pn').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));$('homeDisc').textContent=DISC;$('homeRanks').onclick=()=>showView('rankings');$('homeCompare').onclick=()=>showView('compare');$('homeShare').onclick=()=>{showView('insights');showInsight('share')};$('homePlayers').onclick=()=>showView('players')}
function sg(v){v=Number(v);return (v>0?'+':'')+v.toFixed(1)}
function renderRanks(){let CL=COLS();if(!CL.some(c=>c[0]===sortKey))sortKey=rankMode;let bp=$('badgePick'),inB=rankMode==='badges';bp.hidden=!inB;let pool=DB.filter(p=>posFilter==='ALL'||p.pos===posFilter);
 if(inB){let cnt={},ds={};pool.forEach(p=>p.bl.forEach(b=>{cnt[b.n]=(cnt[b.n]||0)+1;ds[b.n]=b.d}));let gen=n=>n==='Role Player'||n==='Depth Back'?1:0,names=Object.keys(cnt).sort((a,b)=>gen(a)-gen(b)||cnt[b]-cnt[a]);if(selBadge&&!cnt[selBadge])selBadge=null;bp.classList.toggle('sel',!!selBadge);bp.innerHTML=names.map(n=>`<button type="button" class="bpick${n===selBadge?' on':''}" data-n="${n}"><i>${bIcon(n)}</i><span>${n}</span><em>${cnt[n]}</em></button>`).join('');bp.querySelectorAll('.bpick').forEach(x=>x.onclick=()=>{selBadge=selBadge===x.dataset.n?null:x.dataset.n;renderRanks()});window._bd=ds;let on=bp.querySelector('.on');if(on)bp.scrollLeft=Math.max(0,on.offsetLeft-bp.offsetLeft-12)}
 let arr=inB?(selBadge?pool.filter(p=>p.bl.some(b=>b.n===selBadge)):[]):pool;arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);return typeof x==='string'?sortDir*x.localeCompare(y):sortDir*(x-y)});$('rankTitle').textContent=inB?(selBadge?selBadge:'Rankings by badge'):rankMode!=='start'?'Overall Rankings: Mania Rating':`Week ${META.next_week} Rating Rankings`;$('rankSub').textContent=inB?(selBadge?arr.length+' players. '+(window._bd[selBadge]||'')+' Tap a heading to sort.':'Pick a badge to see every player who has it.'):rankMode!=='start'?'How good each player has been so far this season. Tap a heading to sort.':'Who to start this week: Mania Rating plus the matchup and recent role. Tap a heading to sort.';$('rankDisc').textContent=rankMode==='start'?DISC:'';
 let sel=$('sortSel');sel.innerHTML=CL.filter(c=>c[0]!=='rank').map(([k,l])=>`<option value="${k}">Sort by: ${k==='start'?'Week '+META.next_week+' Rating':k==='mania'?'Mania Rating':l}</option>`).join('');sel.value=sortKey;sel.onchange=()=>{sortKey=sel.value;sortDir=sortKey==='name'||sortKey==='opp'?1:-1;renderRanks()};
 $('rankHead').innerHTML='<tr>'+CL.map(([k,l])=>`<th data-k="${k}">${k==='start'?'Wk '+META.next_week:l}${sortKey===k?(sortDir<0?' ↓':' ↑'):''}</th>`).join('')+'</tr>';
 let tg=p=>p.opp==='BYE'?'<span class="tag">BYE</span>':'';
 let cell={rank:(p,i)=>`<td>${i+1}</td>`,name:p=>`<td><span class="playerlink" data-id="${p.id}">${p.name}</span><span class="tag">${p.team} ${p.pos}</span>${tg(p)}${injChip(p)}${bIcons(p.bl)}</td>`,mania:p=>`<td class="grade ${grade(p.mania)}">${p.mania}</td>`,start:p=>`<td class="grade ${grade(p.start)}">${p.out?'OUT':p.start}</td>`,opp:p=>`<td>${p.opp}</td>`,matchup_adj:p=>`<td class="${p.matchup_adj>0.4?'good':p.matchup_adj<-0.4?'bad':'muted'}">${p.opp==='BYE'?'—':sg(p.matchup_adj)}</td>`,trend_adj:p=>`<td class="${p.trend_adj>0.4?'good':p.trend_adj<-0.4?'bad':'muted'}">${sg(p.trend_adj)}</td>`,outlook:p=>`<td class="olCell"><span class="olTag ${p.oc}">${p.ol.replace(' this week','')}</span></td>`,bds:p=>`<td class="bdsCell">${p.bds}</td>`};
 $('rankBody').innerHTML=arr.map((p,i)=>'<tr>'+CL.map(([k])=>cell[k]?cell[k](p,i):`<td>${k==='share'?val(p,'share')+'%':k==='rz'?val(p,'rz'):k==='m.snap'?p.m.snap+'%':val(p,k)}</td>`).join('')+'</tr>').join('');
 let first=rankMode!=='start'?['mania','Mania','start','Wk '+META.next_week]:['start','Wk '+META.next_week,'mania','Mania'];
 let chips=p=>{let c=p.pos==='RB'?[[p.m.ppr,'PPR/G'],[p.m.car,'CAR/G'],[p.m.tgt,'TGT/G'],[p.m.rec,'REC/G'],[p.m.rushy,'RUSH YD/G'],[p.m.snap+'%','SNAP'],[p.m.touch+'%','TOUCH SH'],[p.m.rzc,'RZ CAR/G']]:[[p.m.ppr,'PPR/G'],[p.m.tgt,'TGT/G'],[p.m.rec,'REC/G'],[p.m.recy,'REC YD/G'],[p.m.snap+'%','SNAP'],[p.m.tshare+'%','TGT SH'],[p.m.rzt,'RZ TGT/G'],[p.m.ez,'EZ TGT/G']];
  if(rankMode==='start')c=[[p.opp==='BYE'?'BYE':p.opp,'OPP'],[p.opp==='BYE'?'—':sg(p.matchup_adj),'MATCHUP'],[sg(p.trend_adj),'ROLE TREND'],[p.m.ppr,'PPR/G']];
  return `<div class="rchips c${c.length>4?4:c.length}">${c.map(([v,l])=>`<div><b>${v}</b><span>${l}</span></div>`).join('')}</div>`};
 $('rankCards').innerHTML=`<div class="rcHead"><span>#</span><span>Player</span><span>${first[1]}</span><span>${first[3]}</span></div>`+arr.map((p,i)=>`<div class="rcard" data-id="${p.id}"><div class="rn">${i+1}</div><div class="rmain"><div class="rname">${p.name} <span class="tag">${p.team} ${p.pos}</span>${injChip(p)}</div><div class="rsub"><span class="olTag ${rankMode!=='start'?'g90':p.oc}">${rankMode!=='start'?p.pos+p.pos_rank:p.ol.replace(' this week','')}</span>${bIcons(p.bl)}</div></div><div class="rs ${grade(p[first[0]])}">${p.out&&first[0]==='start'?'OUT':p[first[0]]}</div><div class="rs sec ${grade(p[first[2]])}">${p.out&&first[2]==='start'?'OUT':p[first[2]]}</div>${chips(p)}</div>`).join('');
 $('rankHead').querySelectorAll('th').forEach(th=>th.onclick=()=>{let k=th.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==='name'||k==='opp'?1:-1}renderRanks()});$('rankBody').querySelectorAll('.playerlink').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));$('rankCards').querySelectorAll('.rcard').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)))}
document.querySelectorAll('.rankmode').forEach(b=>b.onclick=()=>{rankMode=b.dataset.mode;sortKey=rankMode==='start'?'start':'mania';sortDir=-1;document.querySelectorAll('.rankmode').forEach(x=>x.classList.toggle('on',x===b));renderRanks()});document.querySelectorAll('.posf').forEach(b=>b.onclick=()=>{posFilter=b.dataset.pos;document.querySelectorAll('.posf').forEach(x=>x.classList.toggle('on',x===b));renderRanks()});renderRanks();
function percentile(pos,key,v){let a=(REFS[pos]||{})[key]||[];if(!a.length)return 0;let lo=0,hi=a.length;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<=v+1e-5)lo=m+1;else hi=m}return Math.min(100,lo/a.length*100)}function avg(a,k,w=true){let den=0,num=0;a.forEach(x=>{let z=w?(x.weight||1):1;num+=Number(x[k]||0)*z;den+=z});return den?num/den:0}function sum(a,k){return a.reduce((s,x)=>s+Number(x[k]||0)*(x.weight||1),0)}
function customMatch(p,M){let ex=p.similar||[];if(!ex.length)return {adj:p.matchup_adj,list:[]};let scored=ex.map(x=>{let share=p.pos==='RB'?M.touch_share:M.target_share,rz=p.pos==='RB'?M.rz_carries_pg:M.rz_targets_pg;let dif=[Math.abs(M.targets_pg-x.tgt)/Math.max(3,x.tgt,1),Math.abs(M.rec_pg-x.rec)/Math.max(2,x.rec,1),Math.abs(M.snap_pct-x.snap)/35,Math.abs(share-x.share)/20,Math.abs(rz-x.rz)/2];let sim=Math.max(.15,Math.exp(-.8*dif.reduce((a,b)=>a+b,0)/dif.length));return {...x,csim:sim,delta:(x.actual/Math.max(x.normal,3)-1)*100}}).sort((a,b)=>b.csim-a.csim);let use=scored.filter(x=>x.csim>=.28),den=use.reduce((s,x)=>s+x.csim*x.csim,0);if(!den)return {adj:0,list:scored.slice(0,4)};let eff=use.reduce((s,x)=>s+x.delta*x.csim*x.csim,0)/den,maturity=Math.min(1,META.week/8),evidence=Math.min(1,use.length/6),adj=Math.max(-6,Math.min(6,eff/25*6))*maturity*(.45+.55*evidence);return {adj:p.pos==='TE'?adj*.35:adj,list:scored.slice(0,4)}}
function dfl(p){let a=p.logs.map((l,i)=>l.auto?-1:i).filter(i=>i>=0);return a.length?a:p.logs.map((_,i)=>i)}
function sameSet(a,b){return a.length===b.length&&a.every(x=>b.includes(x))}
function calc(p,idxs){let L=idxs.map(i=>p.logs[i]).filter(Boolean);if(!L.length)return null;let M={};M.ppr_pg=avg(L,'ppr');M.targets_pg=avg(L,'tgt');M.rec_pg=avg(L,'rec');M.rec_yards_pg=avg(L,'ry');M.carries_pg=avg(L,'car');M.rush_yards_pg=avg(L,'ruy');M.scrim_pg=M.rec_yards_pg+M.rush_yards_pg;M.td_pg=avg(L,'rtd')+avg(L,'rutd');M.snap_pct=avg(L,'snap');M.target_share=avg(L,'tshare');M.air_share=avg(L,'ashare');M.touch_share=avg(L,'touch');M.rz_targets_pg=avg(L,'rzt');M.endzone_targets_pg=avg(L,'ez');M.rz_carries_pg=avg(L,'rzc');M.gl_carries_pg=avg(L,'gl');M.yards_per_target=sum(L,'tgt')?sum(L,'ry')/sum(L,'tgt'):0;M.catch_rate=sum(L,'tgt')?sum(L,'rec')/sum(L,'tgt')*100:0;M.yards_per_carry=sum(L,'car')?sum(L,'ruy')/sum(L,'car'):0;let opp=sum(L,'car')+2*sum(L,'tgt');M.fp_per_weighted_opp=opp?sum(L,'ppr')/opp:0;let P={};Object.keys(M).forEach(k=>P[k]=percentile(p.pos,k,M[k]));let prod,op,role,hv,eff;if(p.pos==='RB'){prod=.55*P.ppr_pg+.30*P.scrim_pg+.15*P.td_pg;op=.45*P.carries_pg+.35*P.targets_pg+.20*P.rec_pg;role=.60*P.touch_share+.40*P.snap_pct;hv=.40*P.rz_carries_pg+.35*P.gl_carries_pg+.25*P.rz_targets_pg;eff=.55*P.fp_per_weighted_opp+.25*P.yards_per_carry+.20*P.catch_rate}else{prod=.55*P.ppr_pg+.30*P.rec_yards_pg+.15*P.td_pg;op=.50*P.targets_pg+.30*P.rec_pg+.20*P.target_share;role=.45*P.target_share+.30*P.air_share+.25*P.snap_pct;hv=.60*P.rz_targets_pg+.40*P.endzone_targets_pg;eff=.45*P.yards_per_target+.30*P.catch_rate+.25*P.fp_per_weighted_opp}let W=p.pos==='RB'?[.30,.30,.20,.125,.075]:[.30,.30,.225,.10,.075],raw=W[0]*prod+W[1]*op+W[2]*role+W[3]*hv+W[4]*eff,base0=45+.52*raw,base=(p.pos==='TE'&&base0<93.6)?Math.max(35,93.6-(93.6-base0)*1.5):base0,eg=L.reduce((s,x)=>s+(x.auto?1:(x.weight==null?1:x.weight)),0),conf=Math.min(1,eg/Math.max(META.week,1)),sh=(1-conf)*.12,mania=Math.max(0,Math.min(99.5,base*(1-sh)+72*sh));let recent=L.slice(-Math.min(2,L.length)),recentOpp=recent.reduce((s,x)=>s+2*x.tgt+x.car,0)/recent.length,seasonOpp=M.targets_pg*2+M.carries_pg,tr=seasonOpp?((recentOpp/seasonOpp)-1)*100:0,trend=Math.max(-2.5,Math.min(2.5,tr/20*2.5));if(L.length<2)trend*=.25;let mt=customMatch(p,M),start=Math.max(0,Math.min(100,mania+mt.adj+trend+(p.inj_adj||0)));if(p.out)start=0;if(sameSet(idxs,dfl(p))){mania=p.mania;start=p.start}return {mania,start,M,match:mt,b:p.pos==='RB'?{'Fantasy Production':prod,'Touch Volume':op,'Backfield Control':role,'Goal-Line / Red-Zone':hv,'Per-Touch Efficiency':eff}:{'Fantasy Production':prod,'Target Volume':op,'Team Target Role':role,'Red-Zone Threat':hv,'Per-Target Efficiency':eff}}}
function barColor(v){return v>=70?'#22a45d':v>=40?'#e3a21a':'#e2505b'}function barGrade(v){return v>=70?'g90':v>=40?'g70':'g0'}function stat(v,k,pos,key,raw){let pc=key?Math.max(1,Math.min(99,Math.round(percentile(pos,key,raw)))):null;return `<div class="stat"><div class="sv">${v}</div><div class="sk">${k}</div>${pc!==null?`<div class="spc ${barGrade(pc)} lnk" data-board="stat" data-key="${key}" data-title="${k}" data-pos="${pos}" title="See everyone ranked">${ord(pc)} percentile</div>`:''}</div>`}
function roomHTML(p){let weeks=[...new Set(p.room.flatMap(x=>x.weekly.map(w=>w.w)))].sort((a,b)=>a-b);return `<div class="tablewrap"><table class="roomtable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>W${w}</th>`).join('')}<th>Season</th></tr></thead><tbody>${p.room.map(x=>`<tr><td class="${x.id===p.id?'you':''}" data-id="${x.id}">${x.name}</td>${weeks.map(w=>{let z=x.weekly.find(q=>q.w===w);return `<td>${z?`${p.pos==='RB'?z.car+z.tgt:z.tgt} / ${z.share}%`:'—'}</td>`}).join('')}<td><b>${x.share}%</b></td></tr>`).join('')}</tbody></table></div><div class="explain">Each week shows ${p.pos==='RB'?'carries + targets':'targets'} / share of the team ${p.pos==='RB'?'touches':'targets'}. Click a name to open that player.</div>`}
function matchupHTML(p,c){let m=c?c.match:{adj:p.matchup_adj,list:p.similar.slice(0,4)},wk=META.next_week,L=m.list||[],n=L.length,avgN=n?L.reduce((s,x)=>s+Number(x.normal),0)/n:0,avgA=n?L.reduce((s,x)=>s+Number(x.actual),0)/n:0,adj=m.adj,sg=adj>=0?'+':'',why;
 if(p.opp==='BYE')why='He is on bye in Week '+wk+', so there is no matchup to adjust for.';
 else if(!n)why='There are not enough games against '+p.opp+' yet to adjust for the matchup, so this stays at 0 for now.';
 else if(adj>=0.5)why=`Players with a similar role averaged <b>${fmt(avgA)}</b> PPR against ${p.opp}, compared with <b>${fmt(avgN)}</b> in their usual games. That makes it a good matchup, so his Week ${wk} Rating gets a <b>${sg}${fmt(adj)}</b> boost.`;
 else if(adj<=-0.5)why=`Players with a similar role averaged <b>${fmt(avgA)}</b> PPR against ${p.opp}, compared with <b>${fmt(avgN)}</b> in their usual games. That makes it a tough matchup, so his Week ${wk} Rating drops by <b>${fmt(Math.abs(adj))}</b>.`;
 else why=`The matchup only nudges his Week ${wk} Rating (${sg}${fmt(adj)}). There is not much data on ${p.opp} this early in the season, so it counts for very little.`;
 return `<div class="matchHead"><div><div class="ph">Week ${wk} matchup${p.opp==='BYE'?': bye week':' vs '+p.opp}</div><div class="sub">${p.opp==='BYE'?'':'How players with a similar role have done against '+p.opp+'.'}</div></div><div class="matchBox"><div class="matchScore ${adj>=0?'good':'bad'}">${sg}${fmt(adj)}</div><div class="mlab">Matchup boost</div></div></div><div class="matchWhy">${why}</div><div class="tablewrap"><table class="matchTable"><thead><tr><th>Similar Player</th><th>Similarity</th><th>Usual PPR/G</th><th>PPR vs ${p.opp}</th></tr></thead><tbody>${L.map(x=>`<tr><td><span data-open="${x.id||''}">${x.name}</span></td><td>${Math.round((x.csim||x.sim/100)*100)}%</td><td>${x.normal}</td><td>${x.actual}</td></tr>`).join('')||'<tr><td colspan="4" class="muted">Not enough matchup evidence yet.</td></tr>'}</tbody></table></div>`}
const HELP={
'Fantasy Production':'How many fantasy points and total yards he is actually producing, compared with players at his position.',
'Target Volume':'How often he is targeted: targets per game, receptions and his share of the team targets.',
'Team Target Role':'How big a slice of his team passing game belongs to him: target share, air-yard share and snaps.',
'Red-Zone Threat':'How often the offense uses him near the goal line: red-zone targets and end-zone targets.',
'Per-Target Efficiency':'What he produces with each chance instead of just rewarding volume: yards per target, catch rate and fantasy points per opportunity.',
'Touch Volume':'How many carries and targets he gets per game compared with other running backs.',
'Backfield Control':'How much of his team backfield he owns: share of touches plus snap share.',
'Goal-Line / Red-Zone':'Scoring-position work: red-zone carries, goal-line carries (inside the 5) and red-zone targets.',
'Per-Touch Efficiency':'What he produces with each touch instead of just rewarding volume: fantasy points per touch, yards per carry and catch rate.'};
const BK=['b_production','b_opportunity','b_role','b_high_value','b_efficiency'];
function bucketRows(b,p){return Object.entries(b).map(([k,v],i)=>{let pc=Math.round(percentile(p.pos,BK[i],v));return `<div class="barrow"><span><button class="infoBtn lblBtn" data-help="h${i}" title="What does this mean?">${k}<span class="chev">ⓘ</span></button></span><div class="track"><div class="fill" style="width:${Math.max(3,pc)}%;background:${barColor(pc)}"></div></div><b class="pctNum lnk ${barGrade(pc)}" data-board="bucket" data-idx="${i}" data-title="${k}" data-pos="${p.pos}" style="border-color:${barColor(pc)};background:${barColor(pc)}1f" title="See everyone ranked">${pc}</b></div><div class="bucketHelp" id="h${i}">${HELP[k]||''} <b>Ranks better than ${pc}% of ${p.pos}s.</b></div>`}).join('')}
function posAvg(pos,key){let q=DB.filter(x=>x.pos===pos&&x.m.snap>=30).map(x=>Number(x.m[key]||0)).filter(Number.isFinite);return q.length?q.reduce((a,b)=>a+b,0)/q.length:0}
function usageHTML(p,c){let M=c.M,isRB=p.pos==='RB',pc=x=>x?'%':'';
 let items=isRB?[
  ['Scrimmage Yards / Game',M.scrim_pg,'scrim','scrim_pg','Rushing plus receiving yards.',0],
  ['Red-Zone Carries / Game',M.rz_carries_pg,'rzc','rz_carries_pg','Carries inside the opponent 20.',0],
  ['Goal-Line Carries / Game',M.gl_carries_pg,'gl','gl_carries_pg','Carries inside the 5, where most RB touchdowns come from.',0],
  ['Red-Zone Targets / Game',M.rz_targets_pg,'rzt','rz_targets_pg','Passes thrown his way inside the 20.',0],
  ['End-Zone Targets / Game',M.endzone_targets_pg,'ez','endzone_targets_pg','Passes thrown into the end zone, counted from anywhere on the field.',0],
  ['Share of Team Touches',M.touch_share,'touch','touch_share','His slice of the team carries plus targets.',1]
 ]:[
  ['Receiving Yards / Game',M.rec_yards_pg,'recy','rec_yards_pg','Yards after the catch included.',0],
  ['Red-Zone Targets / Game',M.rz_targets_pg,'rzt','rz_targets_pg','Passes thrown his way inside the 20.',0],
  ['End-Zone Targets / Game',M.endzone_targets_pg,'ez','endzone_targets_pg','Passes thrown into the end zone, counted from anywhere on the field.',0],
  ['Share of Team Targets',M.target_share,'tshare','target_share','His slice of all the team targets.',1],
  ['Share of Team Air Yards',M.air_share,'ashare','air_share','How much of the downfield passing runs through him.',1],
  ['Receptions / Game',M.rec_pg,'rec','rec_pg','Catches per game, a PPR points driver.',0]
 ];
 if(!isRB&&M.rz_carries_pg>0)items.push(['Red-Zone Carries / Game',M.rz_carries_pg,'rzc','rz_carries_pg','Gadget runs near the goal line. Not part of his score.',0]);
 return `<div class="ph">Usage details <button class="infoBtn" id="usageInfo">ⓘ</button></div><div class="explain" id="usageExplain" style="display:none;margin-bottom:14px">These are the raw numbers behind his score. Each one shows where he ranks among players at his position (100th is the best) and the position average. Green is the top 30%, yellow the middle, red the bottom 40%.</div><div class="usageGrid">${items.map(([l,v,mk,pk,d,isP])=>{let pr=percentile(p.pos,pk,v),col=barColor(pr);return `<div class="usageItem"><div class="uv ${barGrade(pr)}">${fmt(v)}${pc(isP)}</div><div class="ul">${l}</div><div class="ubar"><i style="width:${Math.max(4,pr)}%;background:${col}"></i></div><div class="uPct ${barGrade(pr)}">${ord(pr)} percentile <span>among ${p.pos}s</span></div><div class="uavg">${p.pos} average: ${fmt(posAvg(p.pos,mk))}${pc(isP)}</div><div class="ud">${d}</div></div>`}).join('')}</div>`}
function wireInfo(){document.querySelectorAll('.infoBtn[data-help]').forEach(b=>b.onclick=()=>{let e=$(b.dataset.help);if(e)e.classList.toggle('open')})}
function wireUsage(){let b=$('usageInfo'),e=$('usageExplain');if(b&&e)b.onclick=()=>e.style.display=e.style.display==='none'?'block':'none'}
var navStack=[],curPlayer=null,curBoard=null,skipPush=false;
function pushNav(){let v=document.querySelector('.view.on');if(!v)return;let id=v.id;if(id==='profile'&&curPlayer){let q=curPlayer;navStack.push(()=>{skipPush=true;openPlayer(q)})}else if(id==='board'&&curBoard){let q=curBoard;navStack.push(()=>{skipPush=true;openBoard(q)})}else navStack.push(()=>showView(id))}
function goBack(){let f=navStack.pop();if(f)f();else showView('rankings')}
function openPlayer(p){if(skipPush)skipPush=false;else pushNav();curPlayer=p;showView('profile');let share=p.pos==='RB'?p.m.touch:p.m.tshare,shareName=p.pos==='RB'?'Touch Share':'Target Share',official=calc(p,dfl(p)),bars=bucketRows(official.b,p),logs=p.logs.map((l,i)=>`<tr><td><input class="gameToggle" type="checkbox" ${l.auto?'':'checked'} data-i="${i}"></td><td>W${l.w} ${l.auto?'<span class="partial">HURT, LEFT OUT</span>':l.partial?'<span class="partial">SHORT</span>':''}</td><td class="grade ${grade(Math.min(99,l.ppr*4))}">${l.ppr}</td><td>${l.tgt}</td><td>${l.rec}</td><td>${l.ry}</td><td>${l.car}</td><td>${l.ruy}</td><td>${(l.rtd||0)+(l.rutd||0)}</td><td>${l.snap}%</td></tr>`).join('');$('profileBody').innerHTML=`<div class="topRow"><button class="backBtn" id="backRanks">← Back</button><button class="shotBtn" onclick="toggleShot(true)">Screenshot view</button></div><div class="hero2"><div class="hs left"><div id="maniaTop" class="scorebig ${grade(p.mania)}">${Number(p.mania).toFixed(1)}</div><div class="scorelab">Mania Rating</div><div class="scoresub">Overall profile • ${tier(p.mania)}</div></div><div class="hp"><div class="portrait" id="portrait" style="--score:${p.mania};--ring:${ring(p.mania)}">${pic(p)}</div></div><div class="hs right"><div id="startTop" class="scorebig ${grade(p.start)}">${fs(p,p.start)}</div><div class="scorelab">Week ${META.next_week} Rating</div><div class="scoresub">${p.opp==='BYE'?'Bye week':p.opp==='TBD'?'Matchup TBD':'Start value vs '+p.opp}</div></div><div class="hid"><h1>${p.name}</h1><div class="meta">${p.team} • ${p.pos} • ${p.games} games</div><div class="rankline">#${p.rank} Overall &nbsp;•&nbsp; ${p.pos}${p.pos_rank} &nbsp;•&nbsp; #${p.team_rank} ${p.pos} on ${p.team}${p.depth?' &nbsp;\u2022&nbsp; Depth chart '+p.pos+p.depth.rank:''}</div></div><div class="hol" id="olBox">${olHTML(p,outlook(p,p.start))}</div></div><div class="panel gamePanelTop"><div style="display:flex;justify-content:space-between;align-items:center"><div class="ph">Games counted</div><button class="reset" id="resetGames">RESET</button></div><div class="custom" id="customLine">${p.logs.some(l=>l.auto)?'A game where he got hurt and missed most of it is already left out (tick it to bring it back). ':''}Untick or tick any game and Mania Rating, matchup and Week ${META.next_week} Rating all recalculate. Official Mania Rating <b>${p.mania}</b>.</div><div class="gameControls"><label class="gamechip allchip"><input id="allGames" type="checkbox" ${p.logs.some(l=>l.auto)?'':'checked'}>ALL</label>${p.logs.map((l,i)=>`<label class="gamechip"><input class="gameToggleChip" type="checkbox" ${l.auto?'':'checked'} data-i="${i}">W${l.w} • ${l.ppr} PPR${(l.rtd+l.rutd)?' • '+(l.rtd+l.rutd)+' TD':''}</label>`).join('')}</div></div>${injHTML(p)}<div class="disclaimer">${DISC}</div><div class="stats">${stat(p.m.ppr,'PPR / Game',p.pos,'ppr_pg',p.m.ppr)}${stat(p.m.tgt,'Targets / Game',p.pos,'targets_pg',p.m.tgt)}${stat(p.m.rec,'Receptions / Game',p.pos,'rec_pg',p.m.rec)}${stat(p.m.scrim,'Scrimmage Yds / G',p.pos,'scrim_pg',p.m.scrim)}${stat(share+'%',shareName,p.pos,p.pos==='RB'?'touch_share':'target_share',share)}${stat(p.m.snap+'%','Snap Share',p.pos,'snap_pct',p.m.snap)}${stat(p.m.td,'TDs / Game',p.pos,'td_pg',p.m.td)}</div><div class="grid2"><div class="colL"><div class="panel"><div class="ph">Mania breakdown</div><div class="pctHead">Percentile among ${p.pos}s. 100 is the best, 50 is average.</div><div id="profileBars">${bars}</div><div class="legend"><span><i style="background:#22a45d"></i>Top 30%</span><span><i style="background:#e3a21a"></i>Middle</span><span><i style="background:#e2505b"></i>Bottom 40%</span><span>Tap a category to see what it means.</span></div></div><div class="panel" id="matchPanel">${matchupHTML(p,official)}</div></div><div class="colR"><div class="panel" id="usagePanel">${usageHTML(p,official)}</div></div></div><div class="panel"><div class="ph">${p.team} ${p.pos} room, week by week</div>${roomHTML(p)}</div>${schedHTML(p)}<div class="panel"><div class="ph">Game log</div><div class="tablewrap"><table><thead><tr><th>Use</th><th>Game</th><th>PPR</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>TD</th><th>Snap</th></tr></thead><tbody>${logs}</tbody></table></div></div>`;wireInfo();wireUsage();$('backRanks').onclick=goBack;$('profileBody').querySelectorAll('.roomtable td[data-id]').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));let sync=()=>customProfile(p);document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.onchange=e=>{let i=e.target.dataset.i,checked=e.target.checked;document.querySelectorAll(`[data-i="${i}"]`).forEach(y=>y.checked=checked);sync()});$('allGames').onchange=e=>{document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=e.target.checked);sync()};$('resetGames').onclick=()=>{document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=!p.logs[Number(x.dataset.i)].auto);$('allGames').checked=!p.logs.some(l=>l.auto);sync()}}
function customProfile(p){let idx=[...document.querySelectorAll('.gameToggle:checked')].map(x=>Number(x.dataset.i));let c=calc(p,idx);if(!c){$('customLine').textContent='Select at least one game.';return}$('maniaTop').textContent=c.mania.toFixed(1);$('maniaTop').className='scorebig '+grade(c.mania);$('startTop').textContent=fs(p,c.start);$('olBox').innerHTML=olHTML(p,outlook(p,c.start),idx);$('startTop').className='scorebig '+grade(c.start);$('portrait').style.setProperty('--score',c.mania);$('portrait').style.setProperty('--ring',ring(c.mania));$('profileBars').innerHTML=bucketRows(c.b,p);wireInfo();$('usagePanel').innerHTML=usageHTML(p,c);wireUsage();$('matchPanel').innerHTML=matchupHTML(p,c);$('customLine').innerHTML=`Official <b>${p.mania}</b> → Mania Rating <b class="${grade(c.mania)}">${fmt(c.mania)}</b> • Week ${META.next_week} Rating <b class="${grade(c.start)}">${fmt(c.start)}</b> • ${idx.length} game${idx.length===1?'':'s'} used.`;$('allGames').checked=idx.length===p.logs.length}
var chipsOpen={A:false,B:false};
function chips(p,side,sel){return `<details class="gdet" data-side="${side}" ${chipsOpen[side]?'open':''}><summary>Choose games (${sel.length} of ${p.logs.length})</summary><div class="gameControls" style="justify-content:center"><label class="gamechip allchip"><input type="checkbox" class="cmpAll" data-side="${side}" ${sel.length===p.logs.length?'checked':''}>ALL</label>${p.logs.map((l,i)=>`<label class="gamechip"><input type="checkbox" class="cmpGame" data-side="${side}" data-i="${i}" ${sel.includes(i)?'checked':''}>W${l.w} • ${l.ppr} PPR</label>`).join('')}</div></details>`}
var cmpOpen={start:false,brk:false};
function bdgPills(p,idx){return `<div class="cmpBd">${badgeHTML(badges(p,idx))}</div>`}
function pctBub(v){return `<b class="pctNum ${barGrade(v)}" style="border-color:${barColor(v)};background:${barColor(v)}1f">${v}</b>`}
function bkCompare(A,B,ca,cb){let la=Object.keys(ca.b),lb=Object.keys(cb.b),gen=['Production','Volume','Team role','High-value usage','Efficiency'];
 return `<div class="cbhead"><span>${A.name}</span><span>${B.name}</span></div>`+la.map((k,i)=>{let pa=Math.round(percentile(A.pos,BK[i],ca.b[k])),pb=Math.round(percentile(B.pos,BK[i],cb.b[lb[i]])),lab=la[i]===lb[i]?la[i]:gen[i];
  return `<div class="cbrow"><div class="cbl">${lab}</div><div class="cbbars">${pctBub(pa)}<div class="cbt l"><div style="width:${Math.max(3,pa)}%;background:${barColor(pa)}"></div></div><div class="cbt r"><div style="width:${Math.max(3,pb)}%;background:${barColor(pb)}"></div></div>${pctBub(pb)}</div></div>`}).join('')+`<div class="cexp">Each bubble is a percentile among players at that position. 100 is the best, 50 is average.</div>`}
function renderCompare(){if(!A||!B)return;if(!selA)selA=dfl(A);if(!selB)selB=dfl(B);let ca=calc(A,selA),cb=calc(B,selB);if(!ca||!cb)return;
 let wk=META.next_week,oa=outlook(A,ca.start),ob=outlook(B,cb.start),aB=A.opp==='BYE'||A.out,bB=B.opp==='BYE'||B.out,starter=aB&&!bB?B:bB&&!aB?A:ca.start>=cb.start?A:B,
 rows=[['Mania Rating',ca.mania,cb.mania],['PPR/G',ca.M.ppr_pg,cb.M.ppr_pg],['Targets/G',ca.M.targets_pg,cb.M.targets_pg],['Receptions/G',ca.M.rec_pg,cb.M.rec_pg],['Scrimmage Yds/G',ca.M.scrim_pg,cb.M.scrim_pg],['Snap %',ca.M.snap_pct,cb.M.snap_pct],['Target Share %',ca.M.target_share,cb.M.target_share],['Touch Share %',ca.M.touch_share,cb.M.touch_share],['Air-Yard Share %',ca.M.air_share,cb.M.air_share],['Red-Zone Targets/G',ca.M.rz_targets_pg,cb.M.rz_targets_pg],['End-Zone Targets/G',ca.M.endzone_targets_pg,cb.M.endzone_targets_pg],['Red-Zone Carries/G',ca.M.rz_carries_pg,cb.M.rz_carries_pg],['Goal-Line Carries/G',ca.M.gl_carries_pg,cb.M.gl_carries_pg]],
 sg=v=>(v>=0?'+':'')+v.toFixed(1),oppT=q=>q.opp==='BYE'?'Bye':q.opp==='TBD'?'TBD':q.opp,
 srows=[['Week '+wk+' Rating',fs(A,ca.start),fs(B,cb.start),A.out?-1:ca.start,B.out?-1:cb.start],['Opponent',oppT(A),oppT(B),0,0],['Injury impact',A.out?'OUT':A.inj_adj===0?'None':sg(A.inj_adj),B.out?'OUT':B.inj_adj===0?'None':sg(B.inj_adj),A.out?-99:A.inj_adj,B.out?-99:B.inj_adj],['Matchup boost',sg(ca.match.adj),sg(cb.match.adj),ca.match.adj,cb.match.adj],['Rank',`<span class="outlook ${oa.c}">${oa.t}</span>`,`<span class="outlook ${ob.c}">${ob.t}</span>`,0,0]],
 startHTML=`<details class="cdet" data-k="start" ${cmpOpen.start?'open':''}><summary><span>Who should you start in Week ${wk}?</span><i>Tap to see</i></summary><div class="cdBody"><div class="cwin"><div class="cwk">Our pick</div><strong>${starter.name}</strong><div class="sub">${fs(A,ca.start)} ${A.name} &nbsp;vs&nbsp; ${fs(B,cb.start)} ${B.name}</div></div><table class="cstart"><thead><tr><th>${A.name}</th><th></th><th>${B.name}</th></tr></thead><tbody>${srows.map(([k,a,b,x,y])=>`<tr><td class="${x>y?'good':''}">${a}</td><td class="cmid">${k}</td><td class="${y>x?'good':''}">${b}</td></tr>`).join('')}</tbody></table><div class="cexp"><b>Matchup boost</b> is how much the opponent moves a player's Week ${wk} Rating. Positive means similar players have done well against that defense. Negative means they struggled.</div><div class="disclaimer">${DISC}</div></div></details>`,
 brkHTML=`<details class="cdet" data-k="brk" ${cmpOpen.brk?'open':''}><summary><span>Compare the Mania breakdown</span><i>Tap to see</i></summary><div class="cdBody">${bkCompare(A,B,ca,cb)}</div></details>`;
 $('compareBody').innerHTML=`<div class="compareHero"><div class="comparePlayer"><div class="cmpPortrait" style="--score:${ca.mania};--ring:${ring(ca.mania)}">${pic(A)}</div><h2 class="playerlink cmpOpen" data-id="${A.id}">${A.name}</h2><div class="meta">${A.team} • ${A.pos} • vs ${A.opp}</div><div class="scorebig ${grade(ca.mania)}">${ca.mania.toFixed(1)}</div><div class="scorelab">Mania Rating</div>${bdgPills(A,selA)}${chips(A,'A',selA)}</div><div class="vs">VS</div><div class="comparePlayer"><div class="cmpPortrait" style="--score:${cb.mania};--ring:${ring(cb.mania)}">${pic(B)}</div><h2 class="playerlink cmpOpen" data-id="${B.id}">${B.name}</h2><div class="meta">${B.team} • ${B.pos} • vs ${B.opp}</div><div class="scorebig ${grade(cb.mania)}">${cb.mania.toFixed(1)}</div><div class="scorelab">Mania Rating</div>${bdgPills(B,selB)}${chips(B,'B',selB)}</div></div>${startHTML}${brkHTML}<div class="cmpStats tablewrap"><table><thead><tr><th>${A.name}</th><th style="text-align:center">Metric</th><th>${B.name}</th></tr></thead><tbody>${rows.map(([k,a,b])=>`<tr><td class="${a>b?'good':''}">${/Rating/.test(k)?Number(a).toFixed(1):fmt(a)}</td><td style="text-align:center">${k}</td><td class="${b>a?'good':''}">${/Rating/.test(k)?Number(b).toFixed(1):fmt(b)}</td></tr>`).join('')}</tbody></table></div>`;
 document.querySelectorAll('.cdet').forEach(d=>d.ontoggle=()=>{cmpOpen[d.dataset.k]=d.open});
 document.querySelectorAll('.gdet').forEach(d=>d.ontoggle=()=>{chipsOpen[d.dataset.side]=d.open});document.querySelectorAll('.cmpOpen').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));document.querySelectorAll('.cmpGame').forEach(x=>x.onchange=()=>{let side=x.dataset.side,i=Number(x.dataset.i),arr=side==='A'?selA:selB;if(x.checked){if(!arr.includes(i))arr.push(i)}else if(arr.length>1)arr.splice(arr.indexOf(i),1);arr.sort((a,b)=>a-b);renderCompare()});document.querySelectorAll('.cmpAll').forEach(x=>x.onchange=()=>{let p=x.dataset.side==='A'?A:B,arr=p.logs.map((_,i)=>i);if(x.dataset.side==='A')selA=x.checked?arr:[arr[arr.length-1]];else selB=x.checked?arr:[arr[arr.length-1]];renderCompare()})}
function toggleShot(on){document.body.classList.toggle('shot',on);window.scrollTo(0,0)}
$('teamPick').onchange=renderTeamShare;document.querySelectorAll('.shareMode').forEach(b=>b.onclick=()=>{shareMode=b.dataset.share;document.querySelectorAll('.shareMode').forEach(x=>x.classList.toggle('on',x===b));renderTeamShare()});renderTeamShare();renderHome();showView('home');
/* Insights: one nav tab holding the extra views. Each sub-tab shows its own pane. */

let injPos='ALL',injRole='Starter';
function renderInjuries(){let I=META.injuries||{},L=I.list||[];if($('injUpd'))$('injUpd').textContent=I.updated?'\u2022 updated '+I.updated:'';
 let r=L.filter(x=>(injPos==='ALL'||x.pos===injPos)&&(injRole==='ALL'||x.role==='Starter'));
 if(!r.length){$('injBody').innerHTML='<div class="shareNote">No '+(injRole==='Starter'?'starters ':'players ')+'on the injury report for this filter.</div>';return}
 $('injBody').innerHTML='<div class="panel">'+r.map(x=>`<div class="injItem"><div class="injTop"><span class="pn" data-open="${x.id}">${x.name}</span> <span class="injTag">${x.team} ${x.pos}</span> <span class="injc ${x.avail<=0.1?'o':'q'}">${x.label}</span> <span class="injTag">${x.role}</span></div>${x.note?`<div class="injNote">${x.note}</div>`:''}<div class="injRep">${x.repl.length?'<b>Picking up the work:</b> '+x.repl.map(y=>`<span class="pn" data-open="${(DB.find(d=>d.name===y.n&&d.team===x.team)||{}).id||''}">${y.n}</span> <span class="up">+${y.pts.toFixed(1)}</span>${y.g?` <span class="injTag">${y.g} game${y.g>1?'s':''} without him</span>`:''}`).join(' &nbsp;\u2022&nbsp; '):'No clear beneficiary; the work spreads across the offense.'}</div></div>`).join('')+'</div>'}
document.querySelectorAll('.injPos').forEach(b=>b.onclick=()=>{injPos=b.dataset.v;document.querySelectorAll('.injPos').forEach(x=>x.classList.toggle('on',x===b));renderInjuries()});
document.querySelectorAll('.injRole').forEach(b=>b.onclick=()=>{injRole=b.dataset.v;document.querySelectorAll('.injRole').forEach(x=>x.classList.toggle('on',x===b));renderInjuries()});
renderInjuries();
function showInsight(k){document.querySelectorAll('.insPane').forEach(x=>x.classList.toggle('on',x.id==='ins-'+k));document.querySelectorAll('.insTab').forEach(x=>x.classList.toggle('on',x.dataset.v===k))}
document.querySelectorAll('.insTab').forEach(b=>b.onclick=()=>showInsight(b.dataset.v));$('homeInsights').onclick=()=>showView('insights');
function pillGroup(c,set,render){document.querySelectorAll('.'+c).forEach(b=>b.onclick=()=>{set(b.dataset.v);document.querySelectorAll('.'+c).forEach(x=>x.classList.toggle('on',x===b));render()})}
/* Buy Low / Sell High: usage percentile against production percentile at his position.
   Usage blends the volume, team-role and red-zone buckets with the same weights as BUCKET_W in the Python above (keep them in sync).
   A player is listed when the two are 15+ percentile points apart and the bigger one is above 55, so there is something worth buying or selling.
   The page has no injury report, so "hurt" is what the game logs show: he left the latest week early, or his team played it without him. */
const GAP_MIN=15,MARKET_ROWS=10;let mkPos='ALL';
function market(pos){let wk=META.week,teamsPlayed=new Set(DB.filter(p=>p.logs.some(l=>l.w===wk)).map(p=>p.team)),
 hurt=p=>{if(p.inj&&p.inj.avail<1)return p.inj.label;let last=p.logs[p.logs.length-1];return last.w===wk?(last.partial?`Left Wk ${wk} early`:''):teamsPlayed.has(p.team)?`Missed Wk ${wk}`:''},
 vol=p=>p.pos==='RB'?p.m.car+p.m.tgt:p.m.tgt,
 heavy=p=>p.pos==='RB'?vol(p)>=12&&p.m.snap>=45:p.pos==='WR'?vol(p)>=6&&p.m.snap>=60:vol(p)>=4.5&&p.m.snap>=55,
 real=p=>(p.pos==='RB'?vol(p)>=8:p.pos==='WR'?vol(p)>=4:vol(p)>=3.5)&&p.m.snap>=40&&p.m.ppr>=8,
 rows=DB.filter(p=>(pos==='ALL'||p.pos===pos)&&p.games>1&&!p.out&&!(p.inj&&p.inj.avail<=0.1)).map(p=>{let b=p.buckets,P=(k,v)=>percentile(p.pos,k,v),prod=P('b_production',b.Production),eff=P('b_efficiency',b.Efficiency),w=p.pos==='RB'?[.30,.20,.125]:[.30,.225,.10],usage=(w[0]*P('b_opportunity',b.Opportunity)+w[1]*P('b_role',b.Role)+w[2]*P('b_high_value',b['High Value']))/(w[0]+w[1]+w[2]),td=p.logs.reduce((s,l)=>s+l.rtd+l.rutd,0),pts=p.logs.reduce((s,l)=>s+l.ppr,0);return {p,usage,prod,eff,gap:usage-prod,td,rz:p.logs.reduce((s,l)=>s+l.rzt+l.rzc,0),tdShare:pts>0?6*td/pts:0,hurt:hurt(p)}});
 return {buy:rows.filter(x=>x.gap>=GAP_MIN&&x.usage>=70&&heavy(x.p)).sort((a,b)=>(.55*b.usage+.45*b.gap)-(.55*a.usage+.45*a.gap)).slice(0,MARKET_ROWS),
 sell:rows.filter(x=>-x.gap>=GAP_MIN&&x.prod>=65&&real(x.p)).sort((a,b)=>(-b.gap+30*b.tdShare)-(-a.gap+30*a.tdShare)).slice(0,MARKET_ROWS)}}
function marketWhy(x,buy){let pc=v=>ord(Math.max(1,Math.min(99,v))),first=x.hurt?'Check his status ('+x.hurt+') before you make a move. ':'',m=x.p.m,work=x.p.pos==='RB'?`${fmt(m.car)} carries and ${fmt(m.tgt)} targets a game`:`${fmt(m.tgt)} targets a game`;
 if(buy)return first+(x.td===0&&x.rz>=3?`${work}, and no touchdowns yet on ${x.rz} red-zone looks.`:x.eff<=30?`${work} (${pc(x.usage)} percentile usage), but he is only ${pc(x.eff)} percentile in efficiency. Volume tends to outlast a cold stretch.`:`${work}. His role is bigger than his box scores so far.`);
 return first+(x.tdShare>=.4?`${Math.round(x.tdShare*100)}% of his points have come from touchdowns, on ${work}.`:x.eff>=85?`${pc(x.eff)} percentile efficiency on ${pc(x.usage)} percentile usage. That is hard to keep up.`:`He is scoring more than ${work} usually supports.`)}
function renderMarket(){let m=market(mkPos),row=buy=>(x,i)=>`<div class="mkRow"><span class="n">${i+1}</span><div><span class="playerlink" data-open="${x.p.id}">${x.p.name}</span><span class="tag">${x.p.team} ${x.p.pos}</span>${x.hurt?`<span class="partial">${x.hurt.toUpperCase()}</span>`:''}<div class="mkWhy">${marketWhy(x,buy)}</div></div><div class="mkPct">${pctBub(Math.round(x.usage))}<small>Usage</small></div><div class="mkPct">${pctBub(Math.round(x.prod))}<small>Production</small></div></div>`,none='<div class="muted">Nobody stands out here yet. This needs at least two games from a player.</div>';
 $('marketBody').innerHTML=`<div class="panel"><div class="ph">Buy low</div><div class="pctHead">Heavy-usage players whose points are lagging behind the work they get. Hurt players are left out.</div>${m.buy.map(row(true)).join('')||none}</div><div class="panel"><div class="ph">Sell high</div><div class="pctHead">Regular starters scoring more than their role usually supports.</div>${m.sell.map(row(false)).join('')||none}</div>`}
$('mkWeek').textContent=META.week;pillGroup('mkPos',v=>mkPos=v,renderMarket);renderMarket();
/* Defense vs Position: what each defense has allowed to a position, per game and against what those players usually score.
   "Vs usual" follows the matchup engine: players under 3 PPR a game are left out and shortened games count less. */
const DEF_EDGE=10;let dvPos='RB';
function oppOf(team,wk){return ((META.opps||{})[team]||{})[wk]||''}
function defenseVs(pos){let d={};DB.filter(p=>p.pos===pos).forEach(p=>p.logs.forEach(l=>{let o=oppOf(p.team,l.w),w=l.weight||1;if(!o)return;let x=d[o]||(d[o]={team:o,pts:0,weeks:new Set(),act:0,usual:0});x.pts+=l.ppr;x.weeks.add(l.w);if(p.m.ppr>=3){x.act+=l.ppr*w;x.usual+=p.m.ppr*w}}));return Object.values(d).map(x=>({team:x.team,games:x.weeks.size,pg:x.pts/x.weeks.size,vs:x.usual?(x.act/x.usual-1)*100:0})).sort((a,b)=>b.vs-a.vs)}
function edgeClass(v){return v>=DEF_EDGE?'good':v<=-DEF_EDGE?'bad':'muted'}
function renderDefense(){let rows=defenseVs(dvPos),nw=META.next_week;$('defenseBody').innerHTML=rows.length?`<div class="tablewrap"><table><thead><tr><th>#</th><th>Defense</th><th>Vs usual</th><th>PPR/G allowed</th><th>Games</th><th>Wk ${nw} opponent</th></tr></thead><tbody>${rows.map((x,i)=>`<tr><td>${i+1}</td><td><b>${x.team}</b></td><td class="${edgeClass(x.vs)}">${sg(x.vs)}%</td><td>${fmt(x.pg)}</td><td>${x.games}</td><td>${oppOf(x.team,nw)||'Bye'}</td></tr>`).join('')}</tbody></table></div>`:'<div class="muted">No completed games with a schedule to compare yet.</div>'}
$('dvWeek').textContent=META.week;pillGroup('dvPos',v=>dvPos=v,renderDefense);renderDefense();
/* Schedule Strength: each team's games still to come, graded by that defense's "vs usual" number for the position.
   A defense with no games yet counts as neutral, and bye weeks are left out of the average. */
let scPos='RB',scWin='next4';
function scheduleWeeks(win){let all=[...new Set(Object.values(META.opps||{}).flatMap(o=>Object.keys(o).map(Number)))].filter(w=>w>=META.next_week).sort((a,b)=>a-b);return win==='next4'?all.slice(0,4):win==='playoffs'?all.filter(w=>w>=15&&w<=17):all}
function schedule(pos,win){let vs=Object.fromEntries(defenseVs(pos).map(x=>[x.team,x.vs])),weeks=scheduleWeeks(win);return Object.keys(META.opps||{}).map(team=>{let games=weeks.map(w=>{let opp=oppOf(team,w);return {w,opp,vs:opp?(vs[opp]||0):null}}),played=games.filter(x=>x.opp);return {team,games,avg:played.length?played.reduce((s,x)=>s+x.vs,0)/played.length:0}}).sort((a,b)=>b.avg-a.avg)}
function renderSchedule(){let rows=schedule(scPos,scWin),weeks=scheduleWeeks(scWin),who=t=>DB.filter(p=>p.team===t&&p.pos===scPos).sort((a,b)=>b.mania-a.mania).slice(0,2).map(p=>`<span class="playerlink" data-open="${p.id}">${p.name}</span>`).join(', '),
 word=v=>v>=10?'Great':v>=4?'Good':v>-4?'Neutral':v>-10?'Tough':'Brutal',cls=v=>v>=4?'good':v<=-4?'bad':'muted',mx=Math.max(15,...rows.map(x=>Math.abs(x.avg)));
 if(!(rows.length&&weeks.length)){$('scheduleBody').innerHTML='<div class="muted">No games left on the schedule for this window.</div>';return}
 let best=rows.slice(0,5).map(x=>x.team).join(', '),worst=rows.slice(-5).reverse().map(x=>x.team).join(', ');
 $('scheduleBody').innerHTML=`<div class="scSum"><div class="scBox good"><b>Easiest schedules</b><span>${best}</span></div><div class="scBox bad"><b>Toughest schedules</b><span>${worst}</span></div></div>
 <div class="scKey"><span class="sch good">Easy matchup</span><span class="sch muted">Average</span><span class="sch bad">Tough matchup</span><span class="scKeyNote">Each chip is that week's opponent. Rank 1 is the best schedule for ${scPos}s.</span></div>
 <div class="scList">${rows.map((x,i)=>{let w=Math.min(100,Math.abs(x.avg)/mx*100);return `<div class="scRow"><div class="scTop"><span class="scRank">${i+1}</span><span class="scTeam">${x.team}</span><div class="scBar"><i class="${cls(x.avg)}" style="${x.avg>=0?'left:50%':'right:50%'};width:${w/2}%"></i></div><span class="scVal ${cls(x.avg)}">${word(x.avg)}<small>${sg(x.avg)}%</small></span></div><div class="scChips">${x.games.map(g=>g.opp?`<span class="sch ${edgeClass(g.vs)}" title="${g.opp} has allowed ${sg(g.vs)}% vs usual to ${scPos}s"><small>Wk ${g.w}</small> ${g.opp}</span>`:`<span class="sch muted"><small>Wk ${g.w}</small> bye</span>`).join('')}</div><div class="scWho">${who(x.team)}</div></div>`}).join('')}</div>`}
function schedHTML(p){let row=schedule(p.pos,'all').find(x=>x.team===p.team);return row&&row.games.length?`<div class="panel"><div class="ph">Schedule ahead</div><div class="pctHead">Colored by what each defense has given up to ${p.pos}s so far. Green is an easier matchup, red is a tougher one.</div>${row.games.map(g=>g.opp?`<span class="sch ${edgeClass(g.vs)}" title="${sg(g.vs)}% vs usual">Wk ${g.w} ${g.opp}</span>`:`<span class="sch muted">Wk ${g.w} bye</span>`).join('')}</div>`:''}
$('scWeek').textContent=META.next_week;pillGroup('scPos',v=>scPos=v,renderSchedule);pillGroup('scWin',v=>scWin=v,renderSchedule);renderSchedule();
/* Report Card: last week's start ratings (META.prev, rebuilt by the build from the games before that week) against what happened.
   A "start" is a top-12 rating at RB and WR and top-6 at TE, the same cut as the green "this week" tag. A hit is a finish inside twice that.
   Players on bye are not ranked, and a start who did not play is left out of the hit count. */
let rcPos='RB';
function reportCard(pos){let wk=META.week,prev=META.prev||{},cut=pos==='TE'?6:12,finish={};
 DB.filter(p=>p.pos===pos&&p.logs.some(l=>l.w===wk)).map(p=>({id:p.id,ppr:p.logs.find(l=>l.w===wk).ppr})).sort((a,b)=>b.ppr-a.ppr).forEach((x,i)=>finish[x.id]={ppr:x.ppr,rank:i+1});
 let rated=DB.filter(p=>p.pos===pos&&prev[p.id]&&prev[p.id][1]!=='BYE').sort((a,b)=>prev[b.id][0]-prev[a.id][0]).map((p,i)=>({p,rank:i+1,rating:prev[p.id][0],ppr:finish[p.id]?finish[p.id].ppr:null,finish:finish[p.id]?finish[p.id].rank:null})),
 starts=rated.slice(0,cut).filter(x=>x.finish),
 missed=DB.filter(p=>p.pos===pos&&finish[p.id]&&finish[p.id].rank<=cut).map(p=>{let r=rated.find(x=>x.p===p);return {p,ppr:finish[p.id].ppr,finish:finish[p.id].rank,rank:r?r.rank:null}}).filter(x=>!x.rank||x.rank>cut*2).sort((a,b)=>a.finish-b.finish);
 return {cut,rows:rated.slice(0,cut*2),starts:starts.length,hits:starts.filter(x=>x.finish<=cut*2).length,missed}}
function renderReport(){let wk=META.week,r=reportCard(rcPos),cut=r.cut,fin=x=>x.finish?`<span class="olTag ${x.finish<=cut?'g90':x.finish<=cut*2?'g70':'g0'}">${rcPos}${x.finish}</span>`:'<span class="muted">Did not play</span>',who=x=>`<span class="playerlink" data-open="${x.p.id}">${x.p.name}</span><span class="tag">${x.p.team} ${x.p.pos}</span>`;
 if(!Object.keys(META.prev||{}).length){$('reportTiles').innerHTML='';$('reportBody').innerHTML='<div class="panel muted">The report card starts once two weeks have been played.</div>';return}
 $('reportTiles').innerHTML=['RB','WR','TE'].map(pos=>{let c=reportCard(pos),rate=c.starts?c.hits/c.starts:0;return `<div class="usageItem"><div class="uv ${rate>=.75?'g90':rate>=.5?'g70':'g0'}">${c.hits} of ${c.starts}</div><div class="ul">top-${c.cut} ${pos} starts finished top ${c.cut*2}</div></div>`}).join('');
 $('reportBody').innerHTML=`<div class="panel"><div class="ph">The ${cut*2} highest-rated ${rcPos}s going into Week ${wk}</div><div class="pctHead">Green finished top ${cut}, yellow top ${cut*2}, red lower.</div><div class="tablewrap"><table><thead><tr><th>#</th><th>Player</th><th>Wk ${wk} rating</th><th>PPR</th><th>Finish</th></tr></thead><tbody>${r.rows.map(x=>`<tr><td>${x.rank}</td><td>${who(x)}</td><td class="grade ${grade(x.rating)}">${x.rating}</td><td>${x.ppr===null?'—':fmt(x.ppr)}</td><td>${fin(x)}</td></tr>`).join('')}</tbody></table></div></div><div class="panel"><div class="ph">Top-${cut} finishes the ratings missed</div><div class="pctHead">Finished top ${cut} at ${rcPos} while rated outside the top ${cut*2}, or not rated yet.</div>${r.missed.length?`<div class="tablewrap"><table><thead><tr><th>#</th><th>Player</th><th>Rated</th><th>PPR</th><th>Finish</th></tr></thead><tbody>${r.missed.map((x,i)=>`<tr><td>${i+1}</td><td>${who(x)}</td><td>${x.rank?rcPos+x.rank:'Not rated'}</td><td>${fmt(x.ppr)}</td><td>${fin(x)}</td></tr>`).join('')}</tbody></table></div>`:'<div class="muted">None this week.</div>'}</div>`}
document.querySelectorAll('.rcWeek').forEach(x=>x.textContent=META.week);pillGroup('rcPos',v=>rcPos=v,renderReport);renderReport();
</script></body></html>'''

html = html.replace("__PAYLOAD__", payload).replace("__META__", meta).replace("__REFS__", refs_json)
os.makedirs("public", exist_ok=True)
with open("public/index.html","w",encoding="utf-8") as f:f.write(html)
print("7. FANTASY MANIA v3 built successfully.")
print(f"   {len(players)} players | Through Week {max_week} | Week {next_week} outlook")
print("   Output: public/index.html")
