import json
import math
import os
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

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
PLAYED_ROWS = {}
PLAYED_QB = {}
PLAYED_TEAMS = set()
EARLY_TEAMS = set()   # teams whose already-finished game has BOTH stats and snap counts -> folded into ratings
EARLY_WK = 0
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
                # Only call a week done once the player stats AND snap counts for (nearly) every team that played are posted,
                # so the site never rolls to the next week on half-published data.
                _teams_w = set(_wk["home_team"]) | set(_wk["away_team"])
                _tc0 = "team" if "team" in stats.columns else "recent_team"
                _st_t = set(stats.loc[safe_num(stats["week"]).astype(int) == int(_w), _tc0])
                _sn_t = set(snaps.loc[safe_num(snaps["week"]).astype(int) == int(_w), "team"])
                if len(_teams_w & _st_t) >= 0.9 * len(_teams_w) and len(_teams_w & _sn_t) >= 0.9 * len(_teams_w):
                    _last_done = int(_w)
                else:
                    print(f"   Week {_w} has final scores but stats/snaps are not fully posted yet; keeping it as the current week.")
                    break
            else:
                break
        if _last_done > 0 and not AS_OF_WEEK:
            try:
                # Games already finished in the week still in progress (e.g. Thursday night).
                _nxt = _last_done + 1
                _wkg = _g[_g["week"] == _nxt]
                _fin = _wkg[_wkg["home_score"].notna() & _wkg["away_score"].notna()]
                if len(_fin):
                    _sc = {}
                    for _, _gm in _fin.iterrows():
                        _sc[_gm["home_team"]] = (int(_gm["home_score"]), int(_gm["away_score"]), _gm["away_team"])
                        _sc[_gm["away_team"]] = (int(_gm["away_score"]), int(_gm["home_score"]), _gm["home_team"])
                    _tc = "team" if "team" in stats.columns else "recent_team"
                    _nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
                    _sx = stats[(safe_num(stats["week"]).astype(int) == _nxt) & stats[_tc].isin(list(_sc)) & stats["position"].isin(["RB", "WR", "TE"])]
                    _sn = snaps[safe_num(snaps["week"]).astype(int) == _nxt]
                    _snc = "player" if "player" in _sn.columns else "player_name"
                    _key = lambda x: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", str(x).lower().replace(".", "")))
                    _snmap = {(_key(r_[_snc]), r_["team"]): float(r_["offense_pct"]) for _, r_ in _sn.iterrows()} if len(_sn) else {}
                    for _, r_ in _sx.iterrows():
                        pf, pa, op = _sc[r_[_tc]]
                        _n = lambda c: float(r_[c]) if c in _sx.columns and pd.notna(r_[c]) else 0.0
                        PLAYED_ROWS[r_["player_id"]] = {
                            "wk": _nxt, "opp": str(op), "score": f"{r_[_tc]} {pf}, {op} {pa}",
                            "res": "W" if pf > pa else ("L" if pf < pa else "T"),
                            "ppr": round(_n("fantasy_points_ppr"), 1), "tgt": int(_n("targets")), "rec": int(_n("receptions")),
                            "ry": int(_n("receiving_yards")), "car": int(_n("carries")), "ruy": int(_n("rushing_yards")),
                            "td": int(_n("receiving_tds") + _n("rushing_tds")),
                            "snap": (round(_snmap[(_key(r_[_nc]), r_[_tc])] * 100) if (_key(r_[_nc]), r_[_tc]) in _snmap else None)}
                    PLAYED_TEAMS = set(_sc)
                    try:
                        for _, r_ in stats[(safe_num(stats["week"]).astype(int) == _nxt) & stats[_tc].isin(list(_sc)) & stats["position"].eq("QB")].iterrows():
                            _n = lambda c: float(r_[c]) if c in stats.columns and pd.notna(r_[c]) else 0.0
                            if _n("attempts") < 10:
                                continue
                            pf, pa, op = _sc[r_[_tc]]
                            PLAYED_QB[r_["player_id"]] = {
                                "wk": _nxt, "team": str(r_[_tc]), "opp": str(op), "name": str(r_[_nc]), "res": "W" if pf > pa else ("L" if pf < pa else "T"),
                                "score": f"{r_[_tc]} {pf}, {op} {pa}", "ppr": round(_n("fantasy_points_ppr"), 1), "pyd": int(_n("passing_yards")),
                                "ptd": int(_n("passing_tds")), "int": int(_n("passing_interceptions")), "ruy": int(_n("rushing_yards")),
                                "rtd": int(_n("rushing_tds")), "att": int(_n("attempts")), "comp": int(_n("completions"))}
                    except Exception as e:
                        print("   Could not read played QBs:", e)
                    print(f"   {len(PLAYED_ROWS)} players already played Week {_nxt} (finished games shown separately).")
                    # Fold a finished game into ratings only when its stats AND snap counts are both in nflverse.
                    if len(_sn) and len(_sx):
                        _tm_snap = set(_sn["team"].unique())
                        EARLY_TEAMS = {t_ for t_ in _sc if t_ in _tm_snap and t_ in set(_sx[_tc].unique())}
                        EARLY_WK = _nxt if EARLY_TEAMS else 0
                        print(f"   Folding Week {_nxt} into ratings for: {sorted(EARLY_TEAMS)}")
                        for _pid_, _row_ in PLAYED_ROWS.items():
                            _row_["counted"] = _row_["score"].split(" ")[0] in EARLY_TEAMS
            except Exception as e:
                print("   Could not read already-played games:", e)
        if _last_done > 0:
            _tcol = "team" if "team" in stats.columns else "recent_team"
            _keep_s = (safe_num(stats["week"]).astype(int) <= _last_done) | ((safe_num(stats["week"]).astype(int) == EARLY_WK) & stats[_tcol].isin(list(EARLY_TEAMS)))
            _keep_n = (safe_num(snaps["week"]).astype(int) <= _last_done) | ((safe_num(snaps["week"]).astype(int) == EARLY_WK) & snaps["team"].isin(list(EARLY_TEAMS)))
            stats = stats[_keep_s].copy()
            snaps = snaps[_keep_n].copy()
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
if EARLY_WK and max_week == EARLY_WK:
    max_week = EARLY_WK - 1   # the folded games are extra rows; the league-wide week stays at the last full week
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
    pbp = pbp[(pbp["week"] <= max(max_week, EARLY_WK)) & (pbp["yardline_100"] > 0)].copy()
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
    for _, game in games[(safe_num(games["week"]).astype(int) <= max_week) | (safe_num(games["week"]).astype(int) == EARLY_WK)].iterrows():
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
try:
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    BUILT_AT = _dt.now(_ZI("America/New_York")).strftime("%a %b %-d, %-I:%M %p ET")
except Exception:
    BUILT_AT = ""
inj_meta = {"ok": False, "updated": "", "n": 0, "out": 0}

def _avail(status, practice):
    st = str(status).strip().lower() if pd.notna(status) else ""
    pr = str(practice).strip().lower() if pd.notna(practice) else ""
    if st == "out":
        return 0.0, "OUT"
    if st == "doubtful":
        return 0.02, "Doubtful"
    if st == "questionable":
        return (0.45 if "did not" in pr else 0.63 if "limited" in pr else 0.70), "Questionable"
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
    if r["player_id"] in PLAYED_ROWS:
        start = 0.0          # his game is already played this week
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
# VEGAS PLAYER PROPS (free SportsGameOdds key in the SGO_KEY secret).
# Every failure is silent: no key, no lines, an error or a spent quota just leaves the model's own numbers.
# Free plan = 2,500 "objects" a month, one object per game pulled. This build runs every 15 minutes, so it NEVER calls the service
# on every run: it pulls ONCE a morning (first run after 5 AM ET, Tue-Sat; a second try ten hours later only if Tue-Thu came back empty), one request of ~16 games,
# keeps a running monthly count in the cache and stops at PROPS_CAP_MONTH.
# ============================================================
SGO_KEY = os.environ.get("SGO_KEY", "").strip()
SGO_TEST = os.environ.get("SGO_TEST_JSON", "").strip()
SITE_URL = "https://chris-mania.github.io/fantasy-analyzer"
PROPS_CAP_MONTH = 1500
PROPS_MIN_HOURS = 20
PROPS_MAX_AGE_H = 72
PROPS_W = {"rec": .65, "ryd": .65, "ruy": .65, "td": .55, "pyd": .65, "ptd": .55}
PROPS_PTS = {"rec": 1.0, "ryd": .1, "ruy": .1, "td": 6.0, "pyd": .04, "ptd": 4.0}
PROPS_STAT = {"receiving_receptions": "rec", "receiving_yards": "ryd", "rushing_yards": "ruy", "receptions": "rec", "passing_yards": "pyd", "passing_touchdowns": "ptd"}

def _pn(s):
    t = re.sub(r"[^a-z0-9 ]", "", str(s).lower().replace("-", " ").replace(".", ""))
    return " ".join(w for w in t.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))

def _pid_name(pid):
    t = [w for w in str(pid).split("_") if w and not w.isdigit() and w.upper() != "NFL"]
    return " ".join(w.title() for w in t)

def _am_prob(x):
    try:
        v = float(str(x).replace("+", ""))
    except Exception:
        return None
    return 100.0 / (v + 100.0) if v > 0 else (-v) / (-v + 100.0)

def _fnum(o, *ks):
    for k in ks:
        try:
            v = o.get(k)
            if v is not None and str(v) != "":
                return float(v)
        except Exception:
            pass
    return None

def props_parse(events):
    out = {}
    for ev in events or []:
        for oid, o in (ev.get("odds") or {}).items():
            if not isinstance(o, dict):
                continue
            ent = o.get("playerID") or (o.get("statEntityID") if o.get("statEntityID") not in (None, "home", "away", "all") else None)
            if not ent or o.get("periodID", "game") != "game":
                continue
            stat, bt, side = str(o.get("statID", "")), str(o.get("betTypeID", "")), str(o.get("sideID", ""))
            d = out.setdefault(_pn(_pid_name(ent)), {"name": _pid_name(ent)})
            if bt == "ou" and side == "over":
                line = _fnum(o, "fairOverUnder", "bookOverUnder")
                if line is None:
                    continue
                if stat in PROPS_STAT:
                    d[PROPS_STAT[stat]] = round(line, 2)
                elif stat in ("touchdowns", "anytime_touchdowns") and abs(line - 0.5) < 0.01:
                    p = _am_prob(o.get("fairOdds") or o.get("bookOdds"))
                    if p:
                        d["tdp"] = round(min(p / (1.0 if o.get("fairOdds") else 1.08), 0.92), 3)
            elif bt == "yn" and side == "yes" and stat in ("touchdowns", "anytime_touchdowns"):
                p = _am_prob(o.get("fairOdds") or o.get("bookOdds"))
                if p:
                    d["tdp"] = round(min(p / (1.0 if o.get("fairOdds") else 1.08), 0.92), 3)
    for d in out.values():
        if "tdp" in d:
            d["td"] = round(-math.log(1 - d["tdp"]), 3)
    return {k: v for k, v in out.items() if len(v) > 1}

def props_cache_load():
    """Returns (cache, readable). The saved record of past pulls lives on the live site, because each build starts empty.
    If that record cannot be read (network hiccup, anything but a clean 'file does not exist yet'), readable is False and
    no pull is made: not knowing how many pulls were already used is the one way credits could leak."""
    for loc in ("props_cache.json", os.path.join("public", "props_cache.json")):
        try:
            return json.load(open(loc)), True
        except Exception:
            pass
    try:
        import requests
        r = requests.get(SITE_URL + "/props_cache.json", timeout=20)
        if r.status_code == 200:
            return r.json(), True
        if r.status_code == 404:
            return {}, True
    except Exception:
        pass
    return {}, False

def props_get(next_week):
    """Returns {normalized name: {rec, ryd, ruy, td, pyd, ptd}} for this week, or {}."""
    cache, readable = props_cache_load()
    now = datetime.now(timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    mon = et.strftime("%Y-%m")
    if cache.get("month") != mon:
        cache["month"], cache["used"] = mon, 0
    if cache.get("season") != SEASON or cache.get("week") != next_week:
        cache.update({"season": SEASON, "week": next_week, "players": {}, "fetched": None})
    age_h = None
    try:
        age_h = (now - datetime.fromisoformat(cache["fetched"])).total_seconds() / 3600 if cache.get("fetched") else None
    except Exception:
        age_h = None
    try:
        tried_h = (now - datetime.fromisoformat(cache["last_try"])).total_seconds() / 3600 if cache.get("last_try") else None
    except Exception:
        tried_h = None
    wd, hr = et.weekday(), et.hour
    window = wd in (1, 2, 3, 4, 5) and hr >= 5          # Tue-Sat, first run after 5 AM ET
    try:
        tried_today = bool(cache.get("last_try")) and datetime.fromisoformat(cache["last_try"]).astimezone(ZoneInfo("America/New_York")).date() == et.date()
    except Exception:
        tried_today = False
    cool = cache.get("cool")
    cooling = False
    try:
        cooling = bool(cool) and datetime.fromisoformat(cool) > now
    except Exception:
        pass
    events, status = None, ""
    if SGO_TEST:
        events = json.load(open(SGO_TEST)).get("data", [])
        status = "test file"
    elif not SGO_KEY:
        status = "no SGO_KEY secret set"
    elif not readable:
        status = "could not read the saved pull record, skipping to protect credits"
    elif not window:
        status = "outside the Tue-Sat morning pull window"
    elif cooling:
        status = "cooling down after an error"
    elif tried_today and not (not cache.get("players") and wd <= 3 and tried_h is not None and tried_h >= 10):
        status = "already pulled today (one pull per morning, Tue-Sat)"
    elif cache.get("used", 0) + 18 > PROPS_CAP_MONTH:
        status = f"monthly cap reached ({cache.get('used', 0)} of {PROPS_CAP_MONTH})"
    else:
        cache["last_try"] = now.isoformat()
        try:
            import requests
            start = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end = (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = requests.get("https://api.sportsgameodds.com/v2/events",
                             params={"apiKey": SGO_KEY, "leagueID": "NFL", "oddsAvailable": "true", "startsAfter": start, "startsBefore": end, "limit": 18},
                             timeout=40)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} {r.text[:160]}")
            js = r.json()
            events = js.get("data", [])
            cache["used"] = cache.get("used", 0) + max(len(events), 1)
            status = f"pulled {len(events)} games"
            try:
                os.makedirs("public", exist_ok=True)
                ev0 = events[0] if events else {}
                json.dump({"games": len(events), "first_keys": list(ev0.keys())[:30],
                           "odd_sample": {k: {kk: vv for kk, vv in v.items() if kk in ("statID", "betTypeID", "sideID", "periodID", "playerID", "bookOverUnder", "fairOverUnder", "bookOdds", "fairOdds")}
                                          for k, v in list((ev0.get("odds") or {}).items())[:400:20]}},
                          open(os.path.join("public", "props_debug.json"), "w"))
            except Exception:
                pass
        except Exception as e:
            cache["cool"] = (now + timedelta(hours=24)).isoformat()
            status = f"pull failed, pausing 24h ({str(e)[:120]})"
            events = None
    if events is not None:
        parsed = props_parse(events)
        if parsed:
            cache["players"], cache["fetched"] = parsed, now.isoformat()
            cache["cool"] = None
            cache["inj0"] = {}
        status += f"; {len(parsed)} players with lines"
    try:
        if cache.get("fetched") and (now - datetime.fromisoformat(cache["fetched"])).total_seconds() / 3600 > PROPS_MAX_AGE_H and not SGO_TEST:
            cache["players"] = {}
    except Exception:
        pass
    if readable:
        try:
            json.dump(cache, open("props_cache.json", "w"))
            os.makedirs("public", exist_ok=True)
            json.dump(cache, open(os.path.join("public", "props_cache.json"), "w"))
        except Exception:
            pass
    print(f"   Vegas props: {status}.")
    return cache

def props_blend(frame, names, cols, thin, hist, tag):
    """frame: DataFrame indexed by player_id (modified in place). names: {pid: display name}. cols: {key: column of our projection}.
    Vegas leads, our model adjusts it: blended = vegas + (1-w)*(ours - vegas), per stat.
    w starts at .65 (.55 for touchdowns), +.15 for players with few games, +.05 when the two are far apart (the line probably knows
    something we don't), -.10 when only one stat has a line. Missing stats keep our own number. Returns {pid: details}."""
    PL = hist.get("players") or {}
    det = {}
    if not PL or frame is None or not len(frame):
        return det
    log = {}
    for pid in frame.index:
        v = PL.get(_pn(names.get(pid, "")))
        if not v:
            continue
        got = [k for k in cols if k in v and cols[k] in frame.columns]
        if not got:
            continue
        add, vadd, comps = 0.0, 0.0, []
        for k in got:
            m, vv = float(frame.at[pid, cols[k]]), float(v[k])
            gap = abs(vv - m) / max(abs(vv), abs(m), 1.0)
            w = PROPS_W[k] + (0.15 if thin.get(pid) else 0.0) + (0.05 if gap > 0.35 else 0.0) - (0.10 if len(got) < 2 else 0.0)
            w = min(max(w, 0.35), 0.88)
            bl = vv + (1 - w) * (m - vv)
            add += PROPS_PTS[k] * (bl - m)
            vadd += PROPS_PTS[k] * (vv - m)
            comps.append([k, round(m, 2), round(vv, 2), round(w, 2)])
            frame.at[pid, cols[k]] = bl
        pre = float(frame.at[pid, "proj"])
        # The lines are a snapshot from the morning pull. Injury news since then (a teammate ruled out, a QB change) is kept:
        # whatever the model's injury effect has moved since the pull is added on top of the blend.
        if "g_injuries" in frame.columns:
            inj0 = hist.setdefault("inj0", {})
            gi, key = float(frame.at[pid, "g_injuries"]), _pn(names.get(pid, ""))
            inj0.setdefault(key, round(gi, 3))
            dl = gi - float(inj0[key])
            if abs(dl) >= 0.05:
                add += dl
        for c in ("proj", "floor", "ceil"):
            frame.at[pid, c] = max(float(frame.at[pid, c]) + add, 0.3 if c == "proj" else 0.0)
        det[pid] = {"pre": round(pre, 1), "v": round(max(pre + vadd, 0), 1), "c": comps}
        log[_pn(names.get(pid, ""))] = {"m": round(pre, 1), "v": round(max(pre + vadd, 0), 1), "b": round(float(frame.at[pid, "proj"]), 1)}
    if log:
        h = hist.setdefault("hist", {})
        h[f"{SEASON}-{hist.get('week')}-{tag}"] = log
        for k in sorted(h)[:-24]:
            h.pop(k, None)
        try:
            json.dump(hist, open("props_cache.json", "w"))
            json.dump(hist, open(os.path.join("public", "props_cache.json"), "w"))
        except Exception:
            pass
    return det



# ============================================================
# PROJECTED POINTS (separate engine file; the site still builds if it fails)
# ============================================================
PROJ = {}
QBS = []
HIST = {}
if not AS_OF_WEEK and schedule_ok and week_has_games:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import proj_engine
        _ros = g[["player_id", "position", "team", "opponent"]].rename(columns={"opponent": "opponent_team"}).copy()
        _ros["opponent_team"] = _ros["opponent_team"].astype(str)
        _ros = _ros[_ros["opponent_team"].str.len().between(2, 3) & ~_ros["opponent_team"].isin(["TBD", "BYE", "nan"])]
        _pj, _fit, _qbp, _hist = proj_engine.project(BASE, SEASON, next_week, _ros.drop_duplicates("player_id"), opp_map=dict(opp_map))
        HIST = {(a_, int(b_)): c_ for a_, b_, c_ in zip(_hist["player_id"], _hist["week"], _hist["proj"])}
        VG, VGQ = {}, {}
        try:
            _pc = props_get(next_week)
            _nmap = dict(zip(g["player_id"], g["player_name"]))
            _thin = {}
            if "games" in g.columns:
                _thin = {a_: (b_ is not None and b_ < 3) for a_, b_ in zip(g["player_id"], g["games"])}
            VG = props_blend(_pj, _nmap, {"rec": "p_receptions", "ryd": "p_receiving_yards", "ruy": "p_rushing_yards", "td": "p_tds"}, _thin, _pc, "skill")
            _qn = dict(zip(_qbp.index, _qbp["name"])) if "name" in _qbp.columns else {}
            VGQ = props_blend(_qbp, _qn, {"pyd": "p_passing_yards", "ptd": "p_passing_tds", "ruy": "p_rushing_yards"}, {}, _pc, "qb")
            print(f"   Vegas lines blended into {len(VG)} skill players and {len(VGQ)} QBs.")
        except Exception as e_:
            print("   Vegas props skipped (model numbers unchanged):", e_)
        for _pid, _r in _pj.iterrows():
            PROJ[_pid] = {
                "pts": round(float(_r["proj"]), 1), "lo": round(float(_r["floor"]), 1), "hi": round(float(_r["ceil"]), 1),
                "rec": round(float(_r["p_receptions"]), 1), "ryd": round(float(_r["p_receiving_yards"])),
                "ruy": round(float(_r["p_rushing_yards"])), "td": round(float(_r["p_tds"]), 2),
                "tgt": round(float(_r["p_targets"]), 1), "car": round(float(_r["p_carries"]), 1),
                "fx": {"script": round(float(_r["g_script"]), 1), "matchup": round(float(_r["g_matchup"]), 1), "inj": round(float(_r["g_injuries"]), 1)},
            }
            if _pid in VG:
                PROJ[_pid]["vg"] = VG[_pid]
        for _pid_, _row_ in PLAYED_ROWS.items():
            _row_["proj"] = PROJ[_pid_]["pts"] if _pid_ in PROJ else None
        print(f"   Projected points for {len(PROJ)} players (Week {next_week}).")
        QBS = []
        try:
            _qs = stats[stats["position"].eq("QB")].copy()
            for c_ in ("attempts", "passing_yards", "passing_tds", "passing_interceptions", "fantasy_points_ppr"):
                _qs[c_] = safe_num(_qs[c_]) if c_ in _qs.columns else 0.0
            for c_ in ("completions", "rushing_yards", "rushing_tds", "carries"):
                _qs[c_] = safe_num(_qs[c_]) if c_ in _qs.columns else 0.0
            _qs["week"] = safe_num(_qs["week"]).astype(int)
            _qs = _qs[_qs["attempts"] >= 10]
            _qa = globals().get("qb_avail", {}) or {}
            # ---- QB Mania Rating: same 0-100 scale as RB/WR/TE, graded against other QBs ----
            _qg = _qs.groupby("player_id").agg(n=("week", "nunique"), att=("attempts", "sum"), comp=("completions", "sum"), ppr=("fantasy_points_ppr", "sum"),
                                               pyd=("passing_yards", "sum"), ptd=("passing_tds", "sum"), ints=("passing_interceptions", "sum"),
                                               ruy=("rushing_yards", "sum"), rtd=("rushing_tds", "sum"), car=("carries", "sum"))
            _qg["ppr_pg"] = _qg.ppr / _qg.n; _qg["pyd_pg"] = _qg.pyd / _qg.n; _qg["ptd_pg"] = _qg.ptd / _qg.n; _qg["att_pg"] = _qg.att / _qg.n
            _qg["ruy_pg"] = _qg.ruy / _qg.n; _qg["rtd_pg"] = _qg.rtd / _qg.n; _qg["ypa"] = _qg.pyd / _qg.att.clip(lower=1)
            _qg["td_rate"] = _qg.ptd / _qg.att.clip(lower=1); _qg["int_rate"] = -_qg.ints / _qg.att.clip(lower=1); _qg["cmp"] = _qg.comp / _qg.att.clip(lower=1)
            _qg["qual"] = (_qg.n >= 2) & (_qg.att_pg >= 15)
            if _qg.qual.sum() < 8:
                _qg["qual"] = _qg.n >= 1
            def _qpct(col):
                ref_ = np.sort(_qg.loc[_qg.qual, col].values.astype(float))
                return pd.Series(np.clip(np.searchsorted(ref_, _qg[col].values.astype(float), side="right") / max(len(ref_), 1) * 100, 0, 100), index=_qg.index)
            _P = {c_: _qpct(c_) for c_ in ("ppr_pg", "pyd_pg", "att_pg", "ptd_pg", "ruy_pg", "rtd_pg", "ypa", "td_rate", "int_rate", "cmp")}
            _qg["b_prod"] = _P["ppr_pg"]
            _qg["b_pass"] = .6 * _P["pyd_pg"] + .4 * _P["att_pg"]
            _qg["b_rush"] = .65 * _P["ruy_pg"] + .35 * _P["rtd_pg"]
            _qg["b_score"] = .6 * _P["ptd_pg"] + .4 * _P["td_rate"]
            _qg["b_eff"] = .4 * _P["ypa"] + .3 * _P["int_rate"] + .3 * _P["cmp"]
            _qg["raw"] = .50 * _qg.b_prod + .13 * _qg.b_pass + .12 * _qg.b_rush + .13 * _qg.b_score + .12 * _qg.b_eff
            _conf = np.clip(_qg.n / max(max_week, 1), 0, 1)
            # A QB's stat categories are less lined up than a WR's (a runner has few passing yards, a pocket passer no rushing), so a raw
            # blend can never reach the 90s. Instead, place each QB on the same ladder as the other positions: his rank among qualified QBs
            # is matched to the same rank among qualified RB/WR/TE ratings. QB1 is then as elite as WR1, a starter-level QB is a starter-level WR.
            _sk = np.sort(g[g["position"].isin(["RB", "WR", "TE"]) & g["qualified"]]["mania"].values.astype(float))
            _qq = _qg.loc[_qg.qual, "raw"].values.astype(float)
            _pc = np.array([((_qq < v).sum() + 0.5 * (_qq == v).sum()) / max(len(_qq), 1) for v in _qg["raw"].values.astype(float)])
            _base = np.quantile(_sk, np.clip(_pc, 0, 1)) if len(_sk) > 10 else 45 + 0.52 * _qg.raw.values
            _qg["mania"] = np.clip(_base * (1 - (1 - _conf) * CONFIDENCE_SHRINK) + 72 * (1 - _conf) * CONFIDENCE_SHRINK, 0, 99.5)
            _qg.loc[~_qg.qual, "mania"] = 0.5 * _qg.loc[~_qg.qual, "mania"] + 36.0   # thin sample: pulled halfway to average
            _qg["pos_rank"] = _qg.mania.where(_qg.qual).rank(ascending=False, method="min")
            _qref_m = (np.sort(_qg.loc[_qg.qual, "ppr_pg"].values.astype(float)), np.sort(_qg.loc[_qg.qual, "mania"].values.astype(float)))
            _qgrade = lambda pts: round(float(np.interp(pts, _qref_m[0], _qref_m[1])), 1)
            _qlogs = {}
            for pid2_, d2_ in _qs.sort_values("week").groupby("player_id"):
                _qlogs[pid2_] = [{"w": int(r2_.week), "opp": str(getattr(r2_, "opponent_team", "")), "comp": int(r2_.completions), "att": int(r2_.attempts), "pyd": int(r2_.passing_yards),
                                  "ptd": int(r2_.passing_tds), "int": int(r2_.passing_interceptions), "ruy": int(r2_.rushing_yards), "rtd": int(r2_.rushing_tds),
                                  "ppr": round(float(r2_.fantasy_points_ppr), 1), "pj": (round(float(HIST[(pid2_, int(r2_.week))]), 1) if (pid2_, int(r2_.week)) in HIST else None)} for r2_ in d2_.itertuples()]
            _qpr = {k_: sorted(round(float(v_), 4) for v_ in _qg.loc[_qg.qual, c_].values) for k_, c_ in (("mania", "mania"),)}
            def _qrow(pid, name, team):
                d_ = _qs[_qs["player_id"] == pid]
                n_ = max(len(d_), 1)
                out_ = {"id": pid, "name": name, "team": team, "pos": "QB", "headshot": headshot_map.get(pid, ""), "games": int(len(d_)),
                        "ppr_pg": round(float(d_["fantasy_points_ppr"].sum()) / n_, 1), "pyd_pg": round(float(d_["passing_yards"].sum()) / n_),
                        "ptd_pg": round(float(d_["passing_tds"].sum()) / n_, 1), "int_pg": round(float(d_["passing_interceptions"].sum()) / n_, 1),
                        "mania": None, "start": None, "logs": _qlogs.get(pid, [])}
                if pid in _qg.index:
                    q2_ = _qg.loc[pid]
                    out_["mania"] = round(float(q2_["mania"]), 1)
                    out_["pos_rank"] = int(q2_["pos_rank"]) if pd.notna(q2_["pos_rank"]) else None
                    out_["buckets"] = {"Production": round(float(q2_["b_prod"]), 1), "Passing": round(float(q2_["b_pass"]), 1), "Rushing": round(float(q2_["b_rush"]), 1),
                                       "Scoring": round(float(q2_["b_score"]), 1), "Efficiency": round(float(q2_["b_eff"]), 1)}
                    out_["m"] = {"ppr": round(float(q2_["ppr_pg"]), 1), "pyd": round(float(q2_["pyd_pg"])), "ptd": round(float(q2_["ptd_pg"]), 1), "int": round(float(-q2_["int_rate"] * q2_["att_pg"]), 1),
                                 "att": round(float(q2_["att_pg"]), 1), "cmp": round(float(q2_["cmp"]) * 100, 1), "ypa": round(float(q2_["ypa"]), 1), "ruy": round(float(q2_["ruy_pg"])),
                                 "rtd": round(float(q2_["rtd_pg"]), 2), "car": round(float(q2_["car"] / q2_["n"]), 1)}
                    out_["qual"] = bool(q2_["qual"])
                return out_
            for pid_, r_ in _qbp.iterrows():
                x_ = _qrow(pid_, r_["name"], r_["team"])
                x_["opp"] = r_["opp"]
                x_["sub"] = bool(r_["sub"])
                x_["proj"] = {"pts": round(float(r_["proj"]), 1), "lo": round(float(r_["floor"]), 1), "hi": round(float(r_["ceil"]), 1),
                              "pyd": round(float(r_["p_passing_yards"])), "ptd": round(float(r_["p_passing_tds"]), 2), "int": round(float(r_["p_passing_interceptions"]), 2),
                              "ruy": round(float(r_["p_rushing_yards"])), "fx": {"script": round(float(r_["g_script"]), 1), "matchup": round(float(r_["g_matchup"]), 1), "inj": round(float(r_["g_injuries"]), 1)}}
                if pid_ in VGQ:
                    x_["proj"]["vg"] = VGQ[pid_]
                qa_ = _qa.get(r_["team"])
                x_["inj"] = {"label": qa_[1], "avail": qa_[0], "who": qa_[2]} if qa_ and qa_[0] < 1.0 else None
                x_["played"] = PLAYED_QB.get(pid_)
                if x_.get("mania") is not None:
                    x_["start"] = _qgrade(x_["proj"]["pts"])
                    if x_["played"]:
                        x_["played"]["grade"] = _qgrade(x_["played"]["ppr"])
                if x_["played"] and x_["proj"]:
                    x_["played"]["proj"] = x_["proj"]["pts"]
                QBS.append(x_)
            _have = {q_["id"] for q_ in QBS}
            for pid_, q_ in PLAYED_QB.items():
                if pid_ in _have:
                    continue
                x_ = _qrow(pid_, q_["name"], q_["team"])
                x_.update({"opp": q_["opp"], "sub": False, "proj": None, "inj": None, "played": q_})
                if x_.get("mania") is not None:
                    q_["grade"] = _qgrade(q_["ppr"])
                QBS.append(x_)
            # Every other QB with real snaps (bye-week teams, backups) stays searchable and comparable, just without a Week projection.
            _have = {q_["id"] for q_ in QBS}
            _nmq = _qs.drop_duplicates("player_id", keep="last").set_index("player_id")
            _tcq = "team" if "team" in _nmq.columns else "recent_team"
            _dnq = "player_display_name" if "player_display_name" in _nmq.columns else "player_name"
            for pid_ in _qg.index:
                if pid_ in _have:
                    continue
                _tm_ = str(_nmq.loc[pid_, _tcq])
                x_ = _qrow(pid_, str(_nmq.loc[pid_, _dnq]), _tm_)
                x_.update({"opp": str(dict(opp_map).get(_tm_, "BYE")), "sub": False, "proj": None, "inj": None, "played": PLAYED_QB.get(pid_)})
                QBS.append(x_)
            _qr = sorted([q_["mania"] for q_ in QBS if q_.get("mania") is not None and q_.get("qual")], reverse=True)
            _qs_start = sorted([q_["start"] for q_ in QBS if q_.get("start") is not None and not q_.get("played")], reverse=True)
            for q_ in QBS:
                q_["start_rank"] = (1 + sum(1 for v_ in _qs_start if v_ > q_["start"] + 1e-9)) if (q_.get("start") is not None and not q_.get("played")) else None
            META_QB_REF = {"mania": _qr, "start": _qs_start}
            print(f"   {len(QBS)} quarterbacks in the projection list ({len(_qr)} rated).")
        except Exception as e:
            print("   QB list unavailable:", e)
        _top = sorted(PROJ.items(), key=lambda kv: -kv[1]["pts"])[:12]
        _nm = dict(zip(g["player_id"], g["player_name"]))
        print("   Top projections:", ", ".join(f"{_nm.get(k, k)} {v['pts']}" for k, v in _top))
    except Exception as e:
        print("   Projections unavailable (ratings unaffected):", e)

# ============================================================
# BUILD JSON
# ============================================================
print("6. Packaging player pages, team rooms and rankings...")

# One scale for everything: projected points, final points and the Week Rating all use the SAME points -> rating curve
# (qualified RB/WR/TE season points per game lined up against their Mania Ratings). So the Week Rating order always matches
# the projection order, and a finished game's grade matches the order of the points scored.
_gq = g[g["position"].isin(["RB", "WR", "TE"]) & g["qualified"]]
_gref = (np.sort(_gq["ppr_pg"].values.astype(float)), np.sort(_gq["mania"].values.astype(float))) if len(_gq) > 10 else None
def _gscale(pts, ref):
    # points -> rating curve; above the best qualified player it keeps climbing slowly (cap 99.5) so huge games still order correctly
    top_p, top_m = float(ref[0][-1]), float(ref[1][-1])
    return float(np.interp(pts, ref[0], ref[1])) if pts <= top_p else min(99.5, top_m + 0.12 * (pts - top_p))
if _gref is not None and PROJ:
    _wk_new = {}
    for _i, _pid_ in zip(g.index, g["player_id"]):
        if _pid_ in PROJ:
            _o = inj_by_pid.get(_pid_)
            _wk_new[_i] = 0.0 if (_o and _o["avail"] <= 0.1) else round(_gscale(PROJ[_pid_]["pts"], _gref), 1)
    for _i, _v in _wk_new.items():
        g.at[_i, "start_rating"] = _v
    g["start_pos_rank"] = g.groupby("position")["start_rating"].rank(ascending=False, method="min").astype(int)
    g["start_overall_rank"] = g["start_rating"].rank(ascending=False, method="min").astype(int)
    print(f"   Week Rating now follows the projection for {len(_wk_new)} players.")
_gpos = dict(zip(g["player_id"], g["position"]))
for _pid_, _row_ in PLAYED_ROWS.items():
    _row_["grade"] = round(_gscale(_row_["ppr"], _gref), 1) if _gref is not None else None

# ============================================================
# ROLE-CHANGE FLAGS (QB change / top teammate out) -> advisory banners only
# ============================================================
print("7b. Detecting QB changes and top teammates who are out...")
role_flags = {}
try:
    _tc = "team" if "team" in stats.columns else "recent_team"
    _wks_done = sorted(int(w) for w in merged["week"].unique())
    # ---- QB per team-week (QB with the most attempts) ----
    _q = stats[stats["position"].astype(str).eq("QB")].copy()
    _q["attempts"] = safe_num(_q["attempts"])
    _q = _q.sort_values("attempts", ascending=False).groupby([_tc, "week"]).head(1)
    _q = _q[_q["attempts"] >= 8]
    _qbw = {(str(r_[_tc]), int(r_["week"])): (r_["player_id"], str(r_.get("player_display_name", r_.get("player_name", "")))) for _, r_ in _q.iterrows()}
    _first = lambda nm: [x_ for x_ in str(nm).split(" ") if x_.lower().strip(".") not in ("jr", "sr", "ii", "iii", "iv")][-1]
    for _t in sorted(set(t_ for t_, _ in _qbw)):
        _seq = [(w_, _qbw[(_t, w_)]) for w_ in _wks_done if (_t, w_) in _qbw]
        if len(_seq) < 3:
            continue
        _lastq = _seq[-1][1][0]
        _streak = 0
        for w_, qq_ in reversed(_seq):
            if qq_[0] == _lastq:
                _streak += 1
            else:
                break
        if _streak < 2 or _streak >= len(_seq):
            continue
        _new = [w_ for w_, qq_ in _seq if qq_[0] == _lastq]
        _old = [w_ for w_, qq_ in _seq if qq_[0] != _lastq]
        _newname = _first(_seq[-1][1][1])
        _oldname = _first([qq_ for w_, qq_ in _seq if qq_[0] != _lastq][-1][1])
        _rows = merged[(merged["team"] == _t) & merged["position"].isin(["WR", "TE"]) & (merged["offense_pct"] >= 0.30)]
        for pid_, d_ in _rows.groupby("player_id"):
            pw_ = set(int(x) for x in d_["week"])
            if (pw_ & set(_old)) and (pw_ & set(_new)) and d_["targets"].sum() >= 8:
                role_flags.setdefault(pid_, []).append({
                    "k": "qb", "tag": "QB change",
                    "txt": f"{_newname} has been the starting QB since Week {min(_new)}. Earlier games were with {_oldname}.",
                    "use": sorted(pw_ & set(_new)), "btn": f"Count only games with {_newname}"})
    # ---- Top WR / RB / TE (when healthy) who is out now -> ranks 2-3 (TE: rank 2) ----
    _thr = {"WR": 0.45, "TE": 0.45, "RB": 0.25}
    _act = merged[merged.apply(lambda r_: r_["offense_pct"] >= _thr.get(r_["position"], 0.45), axis=1)].copy()
    for _t in sorted(_act["team"].dropna().unique()):
        for _pos, _share, _lim in (("WR", "target_share_game", 2), ("RB", "touch_share_game", 2), ("TE", "target_share_game", 1)):
            _d = _act[(_act["team"] == _t) & (_act["position"] == _pos)]
            if _d.empty:
                continue
            _rk = _d.groupby("player_id")[_share].agg(["mean", "count"]).sort_values("mean", ascending=False)
            if len(_rk) < 2:
                continue
            _topv = float(_rk["mean"].iloc[0])
            # "key" players: the leader plus anyone within 10% of his share (a near-tie for WR1 counts as WR1)
            _key = [i_ for i_, v_ in _rk["mean"].items() if v_ >= 0.9 * _topv][:(3 if _pos == "WR" else 2)]
            for _top in _key:
                _ti = inj_by_pid.get(_top)
                if not _ti or _ti["avail"] > 0.3:
                    continue
                _tact = set(int(w_) for w_ in _act[_act["player_id"] == _top]["week"])
                _tname = str(g[g["player_id"] == _top]["player_name"].iloc[0]) if (g["player_id"] == _top).any() else "Top player"
                _tl = [x_ for x_ in _tname.split(" ") if x_.lower().strip(".") not in ("jr", "sr", "ii", "iii", "iv")][-1]
                _rest = [i_ for i_ in _rk.index if i_ != _top][:_lim]
                for pid_ in _rest:
                    d_ = merged[(merged["player_id"] == pid_) & (merged["team"] == _t) & (merged["offense_pct"] >= 0.20)]
                    pw_ = set(int(x) for x in d_["week"])
                    _wo = sorted(pw_ - _tact)
                    if not _wo:
                        continue
                    _with = sorted(pw_ & _tact)
                    role_flags.setdefault(pid_, []).append({
                        "k": "out", "tag": f"{_pos}1 out",
                        "txt": f"{_tname} was a top {_pos} on {_t} when healthy and is {('out' if _ti['avail'] <= 0.1 else 'not expected to play')} now."
                               + (f" He played in W{', W'.join(str(x) for x in _with)}." if _with else ""),
                        "use": _wo, "btn": f"Count only games without {_tl}"})
    print(f"   {sum(len(v) for v in role_flags.values())} flags on {len(role_flags)} players.")
except Exception as e:
    print("   Role flags unavailable:", e)
    role_flags = {}

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
            "pj": (round(float(HIST[(r["player_id"], int(x.week))]), 1) if (r["player_id"], int(x.week)) in HIST else (PROJ[r["player_id"]]["pts"] if (int(x.week) == next_week and r["player_id"] in PROJ) else None)),
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
            "ry": round(float(getattr(xx, "receiving_yards", 0) or 0)),
            "ruy": round(float(getattr(xx, "rushing_yards", 0) or 0)),
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
        "played": PLAYED_ROWS.get(r["player_id"]),
        "proj": (None if (inj_by_pid.get(r["player_id"]) and inj_by_pid[r["player_id"]]["avail"] <= 0.1) else PROJ.get(r["player_id"])),
        "depth": depth_info.get(r["player_id"]),
        "out": bool(inj_by_pid.get(r["player_id"]) and inj_by_pid[r["player_id"]]["avail"] <= 0.1),
        "flags": role_flags.get(r["player_id"], []),
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
    for c in ("rush_yards_pg",):
        if c in gp.columns and c not in pct_refs[pos]:
            pct_refs[pos][c] = [round(float(v), 4) for v in np.sort(gp[c].replace([np.inf, -np.inf], np.nan).dropna().values)]
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
    "built": BUILT_AT,
    "players": len(players),
    "opps": opps,
    "prev": prev_calls,
    "qbs": QBS,
})

# ============================================================
# FRONT END — FANTASY MANIA
# ============================================================
html = r'''<!doctype html>
<html lang="en"><head><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,500;0,600;0,700;0,800;1,700;1,800&display=swap" rel="stylesheet">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Mania | The Most Accurate Fantasy Football System in the World</title><meta name="description" content="Fantasy Mania: the most accurate fantasy football system in the world. Free ratings, weekly projections, team compare and trade analyzer, PPR.">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..900&display=swap');
:root{--bg:#e9edf2;--surface:#e0e5eb;--surface2:#d9dfe7;--line:#c4ccd6;--ink:#060709;--muted:#5a616a;--blue:#2563eb;--blue2:#1d4ed8;--green:#22a45d;--yellow:#e3a21a;--orange:#d9731f;--red:#e2505b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',ui-sans-serif,system-ui,-apple-system,sans-serif}button,input{font:inherit}.topbar{background:#e7edf3;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50}.nav{max-width:1240px;margin:auto;height:62px;padding:0 24px;display:flex;align-items:center;justify-content:space-between}.brand{font-family:'Inter',sans-serif;font-size:25px;font-weight:800;letter-spacing:.8px}.brand span{color:var(--blue)}.links{display:flex;height:100%}.navb{border:0;background:transparent;color:#4f5760;font-size:13px;font-weight:800;padding:0 17px;cursor:pointer;border-bottom:2px solid transparent}.navb.on,.navb:hover{color:#000000;border-bottom-color:var(--blue)}
.wrap{max-width:1280px;margin:auto;padding:42px 30px 80px}.view{display:none}.view.on{display:block}.kicker{font-size:11px;color:var(--blue2);font-weight:800;text-transform:uppercase;letter-spacing:1.2px}.title{font-family:'Inter',sans-serif;font-size:44px;font-weight:800;letter-spacing:-1px;letter-spacing:.2px;margin:4px 0}.sub{color:var(--muted);font-size:16px;margin:4px 0 20px;line-height:1.55}.divider{border-top:1px solid var(--line);margin:24px 0}.sectionTitle{font-size:11px;letter-spacing:1.2px;color:#40474f;font-weight:800;text-transform:uppercase;margin-bottom:14px}
.search{position:relative}.search input{width:100%;height:48px;padding:0 15px;border:1px solid var(--line);border-radius:5px;background:#e3e8ee;color:#000000;outline:none}.search input:focus{border-color:var(--blue)}.dd{display:none;position:absolute;top:52px;left:0;right:0;background:#e1e7ed;border:1px solid var(--line);box-shadow:0 18px 45px #0008;z-index:60}.ddi{padding:12px 14px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:16px}.ddi:hover{background:#d5dce4}.ddiName{font-weight:700}.ddiMeta{font-size:12px;color:var(--muted);margin-left:8px}.ddiRate{font-family:'Inter',sans-serif;font-size:18px;font-weight:800;color:var(--green)}
.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin:18px 0}.pill{background:transparent;border:1px solid #b2bbc6;color:#3e444b;border-radius:4px;padding:7px 12px;font-size:12px;font-weight:800;cursor:pointer}.pill.on,.pill:hover{background:var(--blue);border-color:var(--blue);color:#e7edf3}.panel{border-top:1px solid var(--line);margin-top:24px;padding-top:20px}.ph{font-size:11px;letter-spacing:1.15px;color:#40474f;font-weight:800;text-transform:uppercase;margin-bottom:14px}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th{color:#676f79;font-size:10px;text-transform:uppercase;letter-spacing:.7px;cursor:pointer}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right;font-size:13px}th:nth-child(2),td:nth-child(2){text-align:left}.playerlink{font-weight:750;color:#070a0e;cursor:pointer}.playerlink:hover{color:var(--blue2)}.tag{font-size:9px;background:#d2d9e1;color:#535d68;padding:3px 5px;border-radius:3px;margin-left:6px}.grade{font-weight:800}.g90,.g80{color:var(--green)}.g70,.g60{color:var(--yellow)}.g0{color:var(--red)}
.hero{text-align:center;padding:6px 0 25px;border-bottom:1px solid var(--line)}.portrait{width:190px;height:190px;border-radius:50%;position:relative;display:grid;place-items:center;background:#d8dee5;margin:0 auto 18px}.portrait:before{content:"";position:absolute;inset:-7px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#cdd5dd 0);z-index:-2}.portrait:after{content:"";position:absolute;inset:-2px;border-radius:50%;background:var(--bg);z-index:-1}.portrait img{width:184px;height:184px;object-fit:cover;object-position:center top;border-radius:50%}.initials{font-family:'Inter',sans-serif;font-size:48px;font-weight:800;color:#717985}.hero h1{font-family:'Inter',sans-serif;font-size:42px;font-weight:800;letter-spacing:-1px;margin:0}.meta{color:var(--muted);font-size:14px;font-weight:600;margin-top:3px}.rankline{color:#272c33;font-size:13px;margin-top:8px}.scorepair{display:flex;justify-content:center;gap:48px;margin-top:20px}.scorebox{min-width:120px}.scorebig{font-family:'Inter',sans-serif;font-size:48px;font-weight:800;line-height:.9}.scorelab{font-size:9px;font-weight:800;color:#69717a;text-transform:uppercase;letter-spacing:.8px;margin-top:6px}.stats{display:grid;grid-template-columns:repeat(6,1fr);border-bottom:1px solid var(--line)}.stat{padding:18px 12px;text-align:center}.sv{font-family:'Inter',sans-serif;font-size:25px;font-weight:800}.sk{font-size:9px;color:#69717b;text-transform:uppercase;font-weight:800;letter-spacing:.6px;margin-top:2px}
.grid2{display:grid;grid-template-columns:1.08fr .92fr;gap:36px}.barrow{display:grid;grid-template-columns:145px 1fr 48px;gap:12px;align-items:center;margin:13px 0;font-size:13px}.track{height:7px;background:#d3dae1;overflow:hidden}.fill{height:100%}.explain{font-size:12px;color:var(--muted);line-height:1.5;margin-top:14px}.roomHead{display:flex;justify-content:space-between;align-items:center}.roomtable th,.roomtable td{text-align:center}.roomtable th:first-child,.roomtable td:first-child{text-align:left}.roomtable td:first-child{font-weight:700}.roomtable .you{color:var(--blue2)}
.matchScore{font-family:'Inter',sans-serif;font-size:38px;font-weight:800}.matchHead{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}.matchTable th:nth-child(2),.matchTable td:nth-child(2){text-align:right}.muted{color:var(--muted)}.good{color:var(--green);font-weight:800}.bad{color:var(--red);font-weight:800}.partial{font-size:8px;background:#e0d7cc;color:#816030;padding:2px 4px;border-radius:2px}.custom{color:#3f464f;font-size:12px;margin-bottom:12px}.reset{border:1px solid #adb7c3;background:transparent;color:#3f464e;border-radius:3px;padding:6px 9px;cursor:pointer;font-size:10px;font-weight:800}.gameControls{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}.gamechip{border:1px solid #b2bbc6;background:#e3e8ee;color:#3f464e;padding:7px 10px;border-radius:3px;font-size:11px;font-weight:700;cursor:pointer}.gamechip:has(input:checked){border-color:var(--blue);color:#000000;background:#cdd8e5}.gamechip input{display:none}
.compareHero{display:grid;grid-template-columns:1fr 90px 1fr;align-items:center;margin-top:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:30px 0}.comparePlayer{text-align:center}.cmpPortrait{width:180px;height:180px;margin:0 auto 14px;border-radius:50%;position:relative;display:grid;place-items:center;background:#d8dee5}.cmpPortrait:before{content:"";position:absolute;inset:-5px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--score)*1%),#cdd5dd 0);z-index:-2}.cmpPortrait:after{content:"";position:absolute;inset:-1px;border-radius:50%;background:var(--bg);z-index:-1}.cmpPortrait img{width:174px;height:174px;object-fit:cover;object-position:center top;border-radius:50%}.comparePlayer h2{font-family:'Inter',sans-serif;font-size:28px;font-weight:800;margin:3px 0}.vs{font-family:'Inter',sans-serif;font-size:20px;font-weight:800;color:#7d8691;text-align:center}.edge{text-align:center;padding:18px 0;border-bottom:1px solid var(--line)}.edge strong{font-family:'Inter',sans-serif;font-size:28px}.cmpStats{max-width:760px;margin:18px auto 0}.allchip{border-color:#8c98a5}
.homeHero{padding:42px 0 34px;border-bottom:1px solid var(--line)}.homeHero h1{font-family:'Inter',sans-serif;font-size:52px;line-height:1.02;margin:8px 0 12px;letter-spacing:-2px}.homeHero h1 span{color:var(--blue)}.homeHero .sub{max-width:720px;font-size:16px}.homeSearch{max-width:760px;margin-top:25px}.homeActions{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.cta{border:1px solid #acb7c3;background:#dfe5ec;color:#060a0e;padding:10px 15px;border-radius:4px;font-weight:800;font-size:12px;cursor:pointer}.cta.primary{background:var(--blue);border-color:var(--blue);color:#e8eef4}.homeGrid{display:grid;grid-template-columns:1.1fr .9fr;gap:42px;padding-top:32px}.how{font-size:14px;color:#383e46;line-height:1.65}.how strong{color:#000000}.miniRank{margin-top:8px}.miniRow{display:grid;grid-template-columns:32px 1fr 64px 64px;gap:10px;align-items:center;padding:11px 0;border-bottom:1px solid var(--line);font-size:13px}.miniRow .n{color:#717a86}.miniRow .pn{font-weight:750;cursor:pointer}.miniRow .pn:hover{color:var(--blue2)}.miniRow .score{text-align:right;font-weight:800}.homeNote{border-left:3px solid var(--blue);padding:3px 0 3px 15px;margin-top:20px;color:#5a616a;font-size:12px;line-height:1.55}
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
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-feature-settings:"tnum" 1}
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
body{font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-feature-settings:normal;color:var(--ink);background:linear-gradient(180deg,#e6efff 0,#f1f6fd 380px,#f7faff 100%) no-repeat,#f7faff;min-height:100vh}
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
.pdone{font-size:10px;color:#6a7b96;margin-left:3px;font-weight:900}.playedTag{background:#e6ebf3!important;color:#4e5d78!important;font-weight:800}.playedPill{display:inline-block;margin-left:8px;padding:2px 9px;border-radius:999px;background:#e6ebf3;color:#4e5d78;font-size:11px;font-weight:900;letter-spacing:.4px;vertical-align:middle}.heroPj{margin-top:8px;font-size:14px;color:var(--muted)}.heroPj b{color:var(--ink);font-size:16px}.heroPj b.good{color:var(--green)}.heroPj b.bad{color:var(--red)}.how{margin-top:10px}.how summary{cursor:pointer;font-size:12.5px;font-weight:800;color:var(--muted)}.qbRow{cursor:pointer}.qbDetail td,.pjt td{white-space:normal}.pjt .pjv{white-space:nowrap}.qbDetail td{background:#f6f9ff}.qbCard{margin-top:6px}.backBtn{background:#fff;border:1px solid #b2bbc6;color:#1d2530;border-radius:8px;padding:9px 16px;font-size:15px;font-weight:800;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.06)}.backBtn:hover{background:#eef2f7}#profile .topRow,#board .topRow{position:sticky;top:62px;z-index:40;background:var(--bg);padding:10px 0;margin-bottom:6px}.lnk{cursor:pointer}.lnk:hover{filter:brightness(.95);text-decoration:underline}.pctNum.lnk:hover{text-decoration:none;transform:scale(1.07)}.boardT .r{text-align:right}.boardT tr.hlRow td{background:#fff3c4;font-weight:800}.boardT{width:100%;table-layout:fixed;min-width:0}.boardT td,.boardT th{padding:9px 6px}.boardT th:nth-child(1),.boardT td:nth-child(1){width:34px}.boardT th:nth-child(3),.boardT td:nth-child(3){width:86px}.boardT th:nth-child(4),.boardT td:nth-child(4){width:62px}.boardT td:nth-child(2){overflow:hidden}#board .tablewrap{overflow-x:visible}@media(max-width:700px){.boardT th:nth-child(3),.boardT td:nth-child(3){width:70px}.boardT th:nth-child(4),.boardT td:nth-child(4){width:48px}.boardT .tag{display:block;width:fit-content;margin:2px 0 0}}@media(max-width:900px){#profile .topRow,#board .topRow{top:54px}}
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
body.shot .maniaPanel.shotTop{margin-top:16px;padding-top:16px}
body.shot .wrap{zoom:.62;padding-top:12px}
body.shot .shotMark{display:block;text-align:center;color:var(--muted);font-size:13px;font-weight:700;margin-top:26px}
body.shot .shotExit{display:none!important}
.shotExitTop{display:none}
body.shot .shotExitTop{display:block;text-align:center;padding:10px;background:#fff;border-bottom:1px solid var(--line)}
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
.mkGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px;align-items:start}.mkRow{display:grid;grid-template-columns:24px minmax(0,1fr) 64px 64px;gap:10px;align-items:center;padding:13px 0;border-top:1px solid #e9eff8}.mkRow .n{color:#8a99b1;font-size:14px}.mkRow .tag{white-space:nowrap}.mkRow .partial{margin-left:6px;padding:3px 6px;font-size:10px;font-weight:800;white-space:nowrap}.mkNum{font-size:17px;font-weight:800;display:block;text-align:center}.mkNum.up{color:var(--green)}.mkNum.dn{color:#cc3340}.mkWhy>div+div{margin-top:3px}.mkWhy{font-size:13.5px;color:var(--muted);line-height:1.45;margin-top:4px}.mkPct{text-align:center}.mkPct .pctNum{margin:0 auto}.mkPct small{display:block;font-size:10.5px;color:var(--muted);font-weight:700;margin-top:4px}
@media(max-width:700px){.mkGrid{grid-template-columns:minmax(0,1fr)}.mkRow{grid-template-columns:18px minmax(0,1fr) 58px 58px;gap:6px}}
@media(max-width:700px){.navb{font-size:10px!important;padding:0 4px!important;letter-spacing:0!important;flex:1 1 auto}.links{justify-content:space-between}#insights table{font-size:12.5px}#insights th,#insights td{padding:9px 4px!important;white-space:normal}#insights th{font-size:10.5px;letter-spacing:.2px}#insights .tag{display:none}#insights .tablewrap{overflow-x:visible}}
.projBox{background:#fff;border:1px solid #dbe5f4;border-radius:14px;padding:14px 16px;margin:14px 0}.projTop{display:flex;align-items:center;gap:16px;margin:4px 0 12px}.projNum{font-size:46px;font-weight:900;line-height:1;color:var(--ink);min-width:92px}.projTxt b{display:block;font-size:15px;color:var(--ink)}.projTxt span{display:block;font-size:13px;color:var(--muted);margin-top:3px}.projBar{position:relative;height:12px;border-radius:99px;background:#eaf0f9;margin:6px 0 4px}.projBar i{position:absolute;top:0;bottom:0;border-radius:99px;background:#bcd3f6}.projBar u{position:absolute;top:-3px;width:4px;height:18px;border-radius:3px;background:var(--blue);margin-left:-2px;text-decoration:none}.projRng{display:flex;justify-content:space-between;gap:6px;font-size:11.5px;color:var(--muted);font-weight:700}.projRng span:nth-child(2){font-weight:600;text-align:center}.pfx{padding:7px 9px}.pfx b{font-size:18px}.projNote{font-size:13.5px;line-height:1.45;color:#33445f;margin-top:10px}.projFx{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}.pfx{border:1px solid #e3ebf7;border-radius:10px;padding:8px 10px}.pfx b{display:block;font-size:20px;font-weight:900}.pfx span{display:block;font-size:12px;font-weight:800;color:var(--ink)}.pfx em{display:block;font-style:normal;font-size:11px;color:var(--muted);margin-top:2px;line-height:1.3}.projCell{font-weight:800;color:var(--ink)}
.injc{display:inline-block;margin-left:6px;padding:2px 7px;border-radius:999px;font-size:10.5px;font-weight:800;vertical-align:middle;letter-spacing:.3px}.injc.o{background:#fde9eb;color:#c0303d}.injc.q{background:#fff3d6;color:#9a6a00}.injBox{background:#fff;border:1px solid #dbe5f4;border-radius:14px;padding:14px 16px;margin:14px 0}.injHd{font-size:12px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#6a7b96;margin-bottom:6px}.injRow{font-size:14px;line-height:1.45;padding:8px 0 8px 12px;border-left:3px solid #9aa8bd;margin-top:6px;color:#33445f}.injRow b{color:#1c2b4a}.injRow.o{border-color:#e2505b}.injRow.q{border-color:#e3a21a}.injRow.up{border-color:#22a45d}
.rcard{grid-template-areas:"rn main a b"!important;padding:9px 2px!important}.rcard .rmain{min-width:0}.rname{font-size:15px!important}.rsub2{font-size:12px;color:var(--muted);margin-top:2px;display:flex;flex-wrap:wrap;align-items:center;gap:2px 0}.rs{font-size:18px!important}.rcard .rs small{display:inline}
#rankings.projMode .rankDesk{display:block}#rankings.projMode .rankCards{display:none}
.pjt{table-layout:fixed;width:100%}.pjt td,.pjt th{padding:9px 5px}.pjt .rn{width:28px;text-align:center;color:#8a99b1}.pjp{line-height:1.25}.pjp .playerlink{font-weight:800;font-size:15px}.pjs{font-size:12px;color:var(--muted);margin-top:2px;display:block}.pjs .bi{display:inline-flex;vertical-align:middle}.pjp{overflow:hidden}.pjv{text-align:right;font-weight:800}.pjv small{display:block;font-size:10.5px;font-weight:700;color:var(--muted)}.dlt{margin-top:1px}.pjt td:nth-child(n+3),.pjt th:nth-child(n+3){text-align:right}
@media(max-width:700px){#rankings.projMode .tablewrap{overflow-x:visible}.dOnly{display:none}.pjt{font-size:13px}.pjt td,.pjt th{padding:8px 3px}.pjt th{font-size:10px;letter-spacing:0;padding-left:1px;padding-right:1px}.pjp .playerlink{font-size:14px}.pjs{font-size:11px}.pjt th:nth-child(1){width:22px}.pjt th:nth-child(3),.pjt th:nth-child(4),.pjt th:nth-child(5){width:38px}.pjt th:nth-child(6){width:46px}.pjt td:nth-child(3),.pjt td:nth-child(4){font-size:13px}.pjt .bi{display:flex;margin:3px 0 0}.pjt .bi i,.pjt .bi svg{width:16px;height:18px}}
@media(min-width:701px){.pjt th:nth-child(1){width:44px}.pjt th:nth-child(3),.pjt th:nth-child(4),.pjt th:nth-child(5),.pjt th:nth-child(6),.pjt th:nth-child(7){width:84px}}
.seg .pill{font-size:12.5px}.seg[hidden]{display:none}.bi i{cursor:pointer}
.bPop{position:fixed;left:12px;right:12px;bottom:14px;max-width:440px;margin:0 auto;background:#fff;border:1px solid #cfdaf0;border-radius:14px;box-shadow:0 10px 34px rgba(15,32,56,.28);padding:14px 16px;z-index:60}.bPop[hidden]{display:none}.bpT{display:flex;align-items:center;gap:10px}.bpT b{flex:1;font-size:16px}.bpI svg{width:26px;height:30px;display:block}.bpT button{border:0;background:#eef2f9;border-radius:50%;width:28px;height:28px;font-size:14px;cursor:pointer}.bpD{font-size:14px;line-height:1.45;color:#33445f;margin-top:8px}
.fltPanel{background:#fff;border:1px solid #dbe5f4;border-radius:14px;padding:10px 14px;margin:0 0 12px}.fltPanel[hidden]{display:none}.fltRow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0}.fl{font-size:11px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;color:var(--muted);min-width:68px}.fltNote{font-size:13px;line-height:1.4;margin:2px 0 4px}#fltBtn.active{border-color:var(--blue);color:var(--blue)}.how2{margin:0 0 12px}.how2 summary{cursor:pointer;font-size:13px;font-weight:800;color:var(--muted)}.moreRow{margin:0 0 8px;text-align:right}.moreRow[hidden]{display:none}.tabs2 .rankmode{font-size:14px;padding:9px 18px}
.rcard,.rcHead{grid-template-columns:24px minmax(0,1fr) 40px 52px 40px!important;gap:4px!important}.rcard{grid-template-areas:"rn main a b c"!important}.rcHead span:nth-child(n+3){text-align:right}.rch{cursor:pointer}.rcard .rs{font-size:16px!important}

.posRow{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 8px}.posRow .pill{font-size:13px;padding:7px 14px}.ctlRow{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:0 0 10px}.pill.sm{font-size:12px;padding:6px 11px}.viewSeg{display:inline-flex;border:1px solid #c5d3ea;border-radius:999px;overflow:hidden;background:#fff}.viewSeg[hidden]{display:none}.vbtn{border:0;background:transparent;font:inherit;font-size:12.5px;font-weight:800;color:#4a5a73;padding:7px 13px;cursor:pointer}.vbtn.on{background:var(--blue);color:#fff}.rankNote{font-size:12px;color:var(--muted);margin:0 0 8px;line-height:1.4}
#rankings.wideView .rankDesk{display:block}#rankings.wideView .rankCards{display:none}#rankings.wideView .tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
#rankings.wideView table{table-layout:auto;width:max-content;min-width:100%;border-collapse:separate;border-spacing:0}#rankings.wideView td,#rankings.wideView th{white-space:nowrap;text-align:right}#rankings.wideView td:nth-child(2),#rankings.wideView th:nth-child(2){text-align:left}#rankings.wideView td:nth-child(1),#rankings.wideView th:nth-child(1){text-align:center}
#rankings.wideView th:nth-child(1),#rankings.wideView td:nth-child(1){position:sticky;left:0;z-index:2;background:#fff;min-width:34px}#rankings.wideView th:nth-child(2),#rankings.wideView td:nth-child(2){position:sticky;left:34px;z-index:2;background:#fff;box-shadow:2px 0 0 #e9eff8}
#rankings.wideView td .bi{display:none}#rankings.wideView td:nth-child(2) .tag{display:block;width:fit-content;margin:2px 0 0;font-size:10.5px}#rankings.wideView .rname2{font-weight:800}
.injMini{background:#fff;border:1px solid #dbe5f4;border-radius:12px;margin:10px 0;padding:0}.injMini summary{cursor:pointer;list-style:none;font-size:13px;padding:8px 12px;color:#33445f;display:flex;align-items:center;gap:8px}.injMini summary::-webkit-details-marker{display:none}.injMini .injDot{width:9px;height:9px;border-radius:50%;flex:none;background:#9aa8bd}.injMini .injDot.o{background:#e2505b}.injMini .injDot.q{background:#e3a21a}.injMini .injDot.up{background:#22a45d}.injMini .injMore{margin-left:auto;color:var(--muted);font-size:12px;font-weight:700;white-space:nowrap}.injMini[open] .injMore{display:none}.injMini .injBody{padding:0 12px 10px}.injMini .injRow{font-size:13px;padding:5px 0 5px 10px;margin-top:4px}.injMini .injUp{font-size:11px;color:var(--muted);margin-top:6px}
@media(max-width:700px){#rankings{zoom:.93}#rankings.wideView .tablewrap{padding:0!important;border-radius:12px}#rankings.wideView td,#rankings.wideView th{padding:6px 9px!important;font-size:13px}#rankings.wideView th{font-size:10.5px}#rankings.wideView th:nth-child(2),#rankings.wideView td:nth-child(2){min-width:128px;max-width:150px;overflow:hidden;text-overflow:ellipsis}.posRow .pill{flex:1;padding:8px 4px;text-align:center}.vbtn{padding:7px 10px}}
.rcard .rs small.act{display:block;font-size:11px;font-weight:800;opacity:1;margin-top:1px;color:var(--muted)}.rcard .rs small.act.good{color:var(--green)}.rcard .rs small.act.bad{color:var(--red)}

/* ===== elite polish: glow rings, gradients, soft depth ===== */
.brand span{background:linear-gradient(90deg,#2563eb,#7c3aed);-webkit-background-clip:text;background-clip:text;color:transparent}
.portrait:before{background:conic-gradient(from -90deg,var(--ring) 0,var(--ring) calc(var(--score)*1%),rgba(196,206,222,.75) 0)!important;filter:drop-shadow(0 0 5px color-mix(in srgb,var(--ring) 35%,transparent))}
.portrait{box-shadow:0 10px 26px color-mix(in srgb,var(--ring) 12%,transparent)}
.scorebig,.grade,.rs{text-shadow:none!important}
.pill.on{background:linear-gradient(135deg,#1d4ed8,#3b82f6);box-shadow:0 4px 14px rgba(37,99,235,.28);border-color:transparent}
.vbtn.on{background:linear-gradient(135deg,#1d4ed8,#3b82f6)}
.panel,.hero2,.stats,.miniRank,.how3>div,#rankings .tablewrap{box-shadow:0 1px 0 rgba(255,255,255,.9) inset,0 12px 34px rgba(28,43,74,.075);border-color:#dde6f6}
.fill{background-image:linear-gradient(90deg,rgba(255,255,255,.28),rgba(255,255,255,0));box-shadow:0 0 12px -2px currentColor}
.pctNum{box-shadow:0 0 16px -2px currentColor,0 0 0 3px rgba(255,255,255,.7) inset}
.homeHero{position:relative}.homeHero:before{content:"";position:absolute;left:-40px;right:-40px;top:-30px;height:340px;background:radial-gradient(60% 70% at 25% 0%,rgba(59,130,246,.16),transparent 70%),radial-gradient(50% 60% at 85% 10%,rgba(124,58,237,.12),transparent 70%);pointer-events:none;z-index:-1}
.proof{margin-top:22px;background:linear-gradient(180deg,#ffffff,#f4f8ff);border:1px solid #dbe5f6;border-radius:18px;padding:16px 18px;box-shadow:0 14px 36px rgba(28,43,74,.08);max-width:980px}
.proofHd{display:flex;align-items:center;gap:9px;font-size:12px;font-weight:800;letter-spacing:.9px;text-transform:uppercase;color:#2f4a86}.proofHd i{width:9px;height:9px;border-radius:50%;background:#22a45d;box-shadow:0 0 0 4px rgba(34,164,93,.18),0 0 14px rgba(34,164,93,.6)}
.proofGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:12px}.proofGrid div{padding:12px 14px;border-radius:14px;background:#fff;border:1px solid #e3ebf8}.proofGrid b{display:block;font-family:'Inter',sans-serif;font-size:30px;letter-spacing:-1px;background:linear-gradient(90deg,#1d4ed8,#7c3aed);-webkit-background-clip:text;background-clip:text;color:transparent}.proofGrid span{display:block;font-size:13px;line-height:1.4;color:#4a5a73;margin-top:2px}
.proofNote{font-size:12px;color:var(--muted);margin:12px 0 0}
@media(max-width:700px){.proofGrid{grid-template-columns:1fr;gap:9px}.proofGrid div{display:flex;align-items:center;gap:12px}.proofGrid b{font-size:24px;min-width:92px}.proofGrid span{margin:0}}
.proofSub{font-size:15px;color:#33445f;margin:6px 0 2px;font-weight:600}.tinyDisc{font-size:10.5px;line-height:1.4;color:#8a96ab;max-width:760px;margin:26px 0 6px}
/* ===== v5 redesign ===== */
:root{--bg:#edf0f4;--surface:#fff;--surface2:#f4f6f9;--line:#dde2ea;--ink:#0b1220;--muted:#5d6776;--green:#139a55;--shadow:0 0 0 1px rgba(20,35,70,.06),0 24px 40px -26px rgba(20,35,70,.3)}
html body{background:var(--bg)!important;font-optical-sizing:auto;-webkit-font-smoothing:antialiased;color:var(--ink)}
.topbar{background:rgba(237,240,244,.88)!important;border-bottom:1px solid var(--line)!important;box-shadow:none!important}
.nav{max-width:1360px;height:66px;padding:0 40px}
.brand{font-size:21px;font-weight:800;letter-spacing:-.045em;color:var(--ink)}
.navb{font-size:14px;font-weight:600;color:#44505f;padding:0 16px;height:66px;border-bottom:2px solid transparent;letter-spacing:0}
.navb.on,.navb:hover{color:var(--ink);border-bottom-color:var(--ink)}
.wrap{max-width:1360px}
.title{font-size:54px;font-weight:800;letter-spacing:-.05em;line-height:1;margin:6px 0 10px}
.sub{color:var(--muted)}
.kicker{color:var(--blue);font-weight:700;font-size:13px}
.pill{border-radius:10px;background:#fff;border:0;box-shadow:0 0 0 1px rgba(20,35,70,.1);font-weight:700;color:#2a3546}
.pill.on{background:var(--ink)!important;color:#fff!important;box-shadow:0 6px 14px -6px rgba(10,20,40,.5)!important}
.vbtn.on{background:var(--ink)!important}.viewSeg{border-color:#d3d9e2}
.cta{border-radius:10px;background:#fff;border:0;box-shadow:inset 0 0 0 1.5px #c9d0da;color:var(--ink);font-weight:700;padding:13px 20px;font-size:15px}
.cta.primary{background:var(--ink);color:#fff;box-shadow:none}
.search input{border-radius:14px;border:0;background:#fff;height:58px;box-shadow:0 0 0 1px rgba(20,35,70,.08),0 12px 28px -14px rgba(20,35,70,.25)}
.search input:focus{box-shadow:0 0 0 2px var(--ink),0 12px 28px -14px rgba(20,35,70,.25)}
.panel,.stats,.hero2,.miniRank,.how3>div,#rankings .tablewrap,.cdet,.cmpStats,.proof,.fltPanel,.injMini{border:0!important;border-radius:20px!important;box-shadow:var(--shadow)!important;background:#fff}
.panel{padding:26px 28px;margin-top:20px}
.ph{font-size:19px;letter-spacing:-.025em;text-transform:none;font-weight:800;color:var(--ink)}
.stats{margin-top:20px;overflow:hidden}.stat{border-right:1px solid #edf0f4}.stat:last-child{border:0}
.sv{letter-spacing:-.04em}
/* glowing rings and bars */
.portrait,.cmpPortrait{isolation:isolate;position:relative;border-radius:50%;background:#e1e6ee;display:grid;place-items:center;box-shadow:none!important;margin:0 auto}
.portrait:before,.cmpPortrait:before,.portrait:after,.cmpPortrait:after{display:none!important}
.portrait img,.cmpPortrait img{width:100%!important;height:100%!important;border-radius:50%;object-fit:cover;object-position:center top}
.portrait .initials,.cmpPortrait .initials{width:100%;height:100%;border-radius:50%;place-items:center;background:linear-gradient(160deg,#dfe6f1,#c5cfdf);color:#5d6776}
.rsv{position:absolute;inset:-9.5%;width:119%;height:119%;transform:rotate(-90deg);overflow:visible;pointer-events:none;z-index:2;--g1:rgba(34,190,100,.6);--g2:rgba(34,190,100,.26);filter:drop-shadow(0 0 5px var(--g1)) drop-shadow(0 0 14px var(--g2))}
.rsv.Y{--g1:rgba(240,170,20,.6);--g2:rgba(240,170,20,.26)}.rsv.R{--g1:rgba(235,60,80,.55);--g2:rgba(235,60,80,.24)}
.rsv .rt{fill:none;stroke:#e1e6ee;stroke-width:4}.rsv .rp{fill:none;stroke-width:4;stroke-linecap:round}
.track{height:12px;border-radius:12px;background:#e4e9f0;box-shadow:inset 0 1px 2px rgba(10,20,40,.18);overflow:visible}
.fill{border-radius:12px;position:relative}
.fill[style*="22a45d"]{background:linear-gradient(90deg,#0f9a52,#3fe08b)!important;box-shadow:0 0 6px 1px rgba(34,190,100,.65),0 0 16px 3px rgba(34,190,100,.28),0 4px 10px -2px rgba(34,190,100,.65)!important}
.fill[style*="e3a21a"]{background:linear-gradient(90deg,#e09a0a,#ffd04a)!important;box-shadow:0 0 6px 1px rgba(240,170,20,.65),0 0 16px 3px rgba(240,170,20,.28),0 4px 10px -2px rgba(240,170,20,.65)!important}
.fill[style*="e2505b"]{background:linear-gradient(90deg,#d8283a,#ff6f7c)!important;box-shadow:0 0 6px 1px rgba(235,60,80,.6),0 0 16px 3px rgba(235,60,80,.25),0 4px 10px -2px rgba(235,60,80,.6)!important}
.fill:before{content:"";position:absolute;left:6px;right:8px;top:3px;height:3px;border-radius:3px;background:rgba(255,255,255,.7)}
.fill:after{content:"";position:absolute;right:-2px;top:50%;width:12px;height:12px;margin-top:-6px;border-radius:50%;background:radial-gradient(circle,#fff 0,rgba(255,255,255,.4) 55%,rgba(255,255,255,0) 100%)}
.cbt{height:12px;background:#e4e9f0;box-shadow:inset 0 1px 2px rgba(10,20,40,.18);overflow:visible}
.cbt>div{position:relative;border-radius:12px}
.cbt>div[style*="22a45d"]{background:linear-gradient(90deg,#0f9a52,#3fe08b)!important;box-shadow:0 0 6px 1px rgba(34,190,100,.65),0 0 14px 2px rgba(34,190,100,.26)!important}
.cbt>div[style*="e3a21a"]{background:linear-gradient(90deg,#e09a0a,#ffd04a)!important;box-shadow:0 0 6px 1px rgba(240,170,20,.65),0 0 14px 2px rgba(240,170,20,.26)!important}
.cbt>div[style*="e2505b"]{background:linear-gradient(90deg,#d8283a,#ff6f7c)!important;box-shadow:0 0 6px 1px rgba(235,60,80,.6),0 0 14px 2px rgba(235,60,80,.24)!important}
.pctNum{--c1:#0f9a52;--c2:#3fe08b;--g1:rgba(34,190,100,.55);color:#0b8a46;width:46px!important;height:46px!important;border-radius:50%;position:relative;display:grid!important;place-items:center;text-align:center!important;font-weight:800;font-size:15px!important;background:transparent!important;border:0!important;box-shadow:none!important;isolation:isolate;justify-self:end}
.pctNum.g70{--c1:#e09a0a;--c2:#ffd04a;--g1:rgba(240,170,20,.55);color:#a86f00}
.pctNum.g0{--c1:#d8283a;--c2:#ff6f7c;--g1:rgba(235,60,80,.5);color:#bf2434}
.pctNum:before{content:"";position:absolute;inset:0;border-radius:50%;z-index:-2;background:conic-gradient(from 0deg,var(--c1),var(--c2) calc(var(--s)*3.6deg*.97),#e1e6ee 0);filter:drop-shadow(0 0 4px var(--g1)) drop-shadow(0 0 9px var(--g1))}
.pctNum:after{content:"";position:absolute;inset:5px;border-radius:50%;background:#fff;z-index:-1;box-shadow:inset 0 1px 3px rgba(10,20,40,.15)}
.cbbars .pctNum,.mkPct .pctNum{width:40px!important;height:40px!important;font-size:13px!important}
.barrow{grid-template-columns:215px minmax(0,1fr) 50px;gap:14px;margin:18px 0}
.legend i{box-shadow:0 0 6px currentColor}
/* badges */
.bi{gap:5px;margin-left:8px}.bi i{width:30px;height:auto;border-radius:0;background:none;display:inline-block;line-height:0}
.bi i svg,.bsv{width:100%;height:auto;display:block;filter:drop-shadow(0 2px 2px rgba(10,20,40,.38))}
.bsv.bdia{filter:drop-shadow(0 0 4px rgba(70,190,255,.95)) drop-shadow(0 2px 2px rgba(10,20,40,.35))}
.bdgBtn{border-radius:12px;background:#fff;border:0;box-shadow:0 0 0 1px rgba(20,35,70,.1);padding:5px 14px 5px 6px;color:var(--ink)}
.bdgBtn i{width:26px;height:auto;border-radius:0;background:none;line-height:0}.bdgBtn svg{width:100%;height:auto}
.bdgBtn.on{background:var(--ink);color:#fff}
.bpI svg{width:36px;height:auto}.bPop{border:0;border-radius:18px}
/* profile hero */
.hero2{padding:38px 40px;margin-top:14px;border:0;display:grid;grid-template-columns:330px minmax(0,1fr) minmax(0,1fr);grid-template-areas:"ph id id" "ph ml wk" "ph ol ol";column-gap:44px;row-gap:12px;align-items:center;text-align:left}
.hp{grid-area:ph;display:grid;place-items:center;padding:42px;background:linear-gradient(160deg,#fff,#f0f3f7);border-radius:30px;box-shadow:0 40px 60px -28px rgba(20,35,70,.42),0 0 0 1px rgba(20,35,70,.06);transform:perspective(1100px) rotateY(-13deg) rotateX(6deg)}
.hero2 .portrait{width:230px;height:230px;margin:0}
.hid{grid-area:id;text-align:left;align-self:end}
.hid h1{font-size:64px;letter-spacing:-.05em;line-height:1}
.hid .meta{font-size:17px;margin-top:6px}
.hs.left,.hs.right{text-align:left}.hs.left{grid-area:ml}.hs.right{grid-area:wk}
.hs .scorebig{font-size:84px;letter-spacing:-.06em;line-height:.92}
.hs .scorelab{font-size:15px;margin-top:10px}
.hol{grid-area:ol;align-self:start;text-align:left;display:flex;flex-direction:column;align-items:flex-start}.hol .bdgs{justify-content:flex-start}.hid .heroPj,.hid>div{text-align:left}.hid{display:block}
.scorebig,.grade,.rs,.sv{text-shadow:none!important}
.hero2 .heroPj,.heroPj{margin-top:10px}
@media(max-width:900px){.nav{padding:10px 14px 0;height:auto}.navb{height:44px;font-size:13px;padding:0 11px}.title{font-size:38px}
.hero2{grid-template-columns:1fr 1fr;grid-template-areas:"ph ph" "id id" "ml wk" "ol ol";padding:26px 16px;text-align:center;row-gap:16px;column-gap:10px}
.hp{padding:26px;transform:perspective(900px) rotateY(-9deg) rotateX(5deg)}.hero2 .portrait{width:176px;height:176px}
.hid{text-align:center}.hid h1{font-size:38px}.hs.left,.hs.right{text-align:center}.hs .scorebig{font-size:54px}.hs .scorelab{font-size:14px}.hol{text-align:center;align-items:center}.hol .bdgs{justify-content:center}
.barrow{grid-template-columns:minmax(0,1fr) 50px;grid-template-areas:"l n" "t t";row-gap:8px;margin:16px 0}}
/* home */
.homeHero{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(0,1fr);gap:0 24px;align-items:center;max-width:none;border:0;padding:20px 0 20px}
.homeHero:before{display:none}
.hhL h1{font-size:76px;line-height:.95;letter-spacing:-.052em;margin:10px 0 18px}
.hhL .sub{font-size:19px;max-width:520px}
.homeHero .proof{grid-column:1/-1;max-width:none;margin-top:10px}
.homeSearch{max-width:560px}
.stage{perspective:1500px;height:540px;position:relative}
.stack{position:absolute;inset:0;transform-style:preserve-3d;transform:rotateY(-22deg) rotateX(9deg) rotateZ(2deg)}
.hc{position:absolute;left:calc(50% - 150px);top:50px;width:300px;height:396px;background:#fff;border-radius:22px;box-shadow:0 40px 70px -24px rgba(20,35,70,.38),0 0 0 1px rgba(20,35,70,.06);padding:20px 18px 18px;display:flex;flex-direction:column;align-items:center;cursor:pointer;transition:transform .6s cubic-bezier(.2,.8,.2,1),opacity .45s;transform:translate3d(calc(var(--o)*-78px),calc(var(--o)*22px),calc(var(--o)*-70px));opacity:calc(1 - var(--o)*.12)}
.hc[data-front="0"] .hst,.hc[data-front="0"] .hopen{visibility:hidden}
.hrk{position:absolute;left:16px;top:14px;font-weight:800;font-size:22px;letter-spacing:-.04em;background:var(--ink);color:#fff;border-radius:10px;padding:3px 11px;z-index:3}
.hpf{margin-top:30px;width:150px;height:150px;position:relative}
.hmini{width:150px;height:150px;position:relative}
.hc h3{font-size:25px;letter-spacing:-.045em;font-weight:800;margin:30px 0 4px;text-align:center;line-height:1.1}
.htag{font-size:13px;color:var(--muted);font-weight:600}
.hst{display:flex;justify-content:space-between;width:100%;margin-top:auto;padding-top:14px;border-top:1px solid #e6eaef;font-size:12px;color:var(--muted);font-weight:600}
.hst b{display:block;font-size:21px;color:var(--ink);letter-spacing:-.04em;font-weight:800}.hst div:last-child{text-align:right}.hst b.g90{color:var(--green)}.hst b.g70{color:#a46c00}.hst b.g0{color:#cc3340}
.hopen{display:none;margin-top:12px;border:0;background:var(--ink);color:#fff;border-radius:10px;padding:9px 16px;font-weight:700;font-size:13px;cursor:pointer}
.hc[data-front="1"] .hopen{display:inline-block}
.stackFoot{position:absolute;right:14px;bottom:20px;display:flex;gap:14px;align-items:center;font-size:12px;font-weight:700;color:var(--muted)}
.sdots{display:flex;gap:6px}.sdots i{width:8px;height:8px;border-radius:8px;background:#c3cad4;cursor:pointer;transition:width .3s,background .3s}.sdots i.on{width:22px;background:var(--ink)}
.homeBlock .sectionTitle{font-size:19px;letter-spacing:-.025em;text-transform:none}
@media(max-width:900px){.homeHero{grid-template-columns:1fr}.hhL h1{font-size:44px;letter-spacing:-.05em}.hhL .sub{font-size:16px}.stage{height:450px;margin:6px 0 34px}.stackFoot{z-index:40}.stack{transform:rotateY(-18deg) rotateX(8deg) scale(.8);transform-origin:50% 40%}.hc{left:calc(50% - 128px);transform:translate3d(calc(var(--o)*-40px),calc(var(--o)*20px),calc(var(--o)*-70px))}.stackFoot{right:auto;left:0;bottom:2px}.homeActions .cta{flex:1 1 44%}.homeActions .primary{flex-basis:100%}}
/* rankings */
.thumb{--c1:#0f9a52;--c2:#3fe08b;--g1:rgba(34,190,100,.5);width:44px;height:44px;flex:none;border-radius:50%;position:relative;display:grid;place-items:center;background:conic-gradient(var(--c1),var(--c2) calc(var(--s)*3.6deg*.97),#dfe4ec 0);filter:drop-shadow(0 0 4px var(--g1))}
.thumb.g70{--c1:#e09a0a;--c2:#ffd04a;--g1:rgba(240,170,20,.5)}.thumb.g0{--c1:#d8283a;--c2:#ff6f7c;--g1:rgba(235,60,80,.45)}
.thumb:before{content:"";position:absolute;inset:3px;border-radius:50%;background:#fff}
.thumb img,.thumb .initials{position:relative;width:calc(100% - 9px);height:calc(100% - 9px);border-radius:50%;object-fit:cover;object-position:center top}
.thumb .initials{place-items:center;background:linear-gradient(160deg,#dfe6f1,#c5cfdf);font-size:13px;color:#5d6776;font-weight:800}
.nmw{display:flex;align-items:center;gap:12px}.nmt{min-width:0}
#rankings table{border-collapse:separate;border-spacing:0}
#rankings th{font-size:12px;font-weight:700;color:#7a8493;text-transform:none;letter-spacing:0;padding:14px 10px}
#rankings td{border-bottom:1px solid #edf0f4}
@media(min-width:701px){#rankings td{padding:11px 10px}#rankings td.grade,#rankings td.projCell{font-size:18px;letter-spacing:-.03em}}
#rankings tbody tr{cursor:pointer;transition:background .2s}#rankings tbody tr:hover td{background:#f6f8fb}
#rankings tbody tr.open td{background:#f6f8fb}
#rankings tr.xr td{padding:0!important;background:#f6f8fb;border-bottom:1px solid #edf0f4}
.xpIn{display:grid;grid-template-columns:1.5fr 1fr;gap:40px;padding:2px 28px 24px 66px;white-space:normal}
.xpIn .barrow{grid-template-columns:170px minmax(0,1fr) 50px;grid-template-areas:none!important;margin:12px 0;font-size:14px;font-weight:600}.xpIn .barrow>*{grid-area:auto!important}
.xpT{font-size:12px;color:var(--muted);font-weight:700}.spk{width:100%;height:56px;margin-top:6px;display:block;overflow:visible}
.xpOpen{margin-top:14px;border:0;background:var(--ink);color:#fff;border-radius:10px;padding:10px 16px;font-weight:700;font-size:13px;cursor:pointer}
.rcard{border-radius:14px;padding:12px 6px}.rcard.open{background:#f6f8fb}
.rmain{display:flex;align-items:center;gap:9px;min-width:0}.rtxt{min-width:0}
.rcard .thumb{width:40px;height:40px}
.rxp{background:#f6f8fb;border-radius:0 0 14px 14px;margin:-6px 0 6px;padding:4px 12px 16px}.rxp .xpIn{grid-template-columns:1fr;gap:6px;padding:0}.rxp .xpIn .barrow{grid-template-columns:104px minmax(0,1fr) 46px;font-size:12.5px}
#rankings.wideView td:nth-child(2) .thumb{display:none}
@media(max-width:700px){#rankings.wideView td:nth-child(2) .nmw{display:block}}
/* compare */
.compareHero{display:grid;grid-template-columns:1fr 1fr;gap:24px;perspective:1600px;position:relative;border:0;padding:22px 8px;margin-top:10px}
.comparePlayer{background:#fff;border-radius:24px;box-shadow:var(--shadow);padding:30px 18px 22px;position:relative}
.comparePlayer:first-child{transform:rotateY(8deg);transform-origin:right center}.comparePlayer:last-child{transform:rotateY(-8deg);transform-origin:left center}
.cmpPortrait{width:170px;height:170px;margin:12px auto 26px}
.comparePlayer h2{font-size:30px;letter-spacing:-.045em}
.comparePlayer .scorebig{font-size:68px;letter-spacing:-.06em}
.compareHero .vs{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:54px;height:54px;border-radius:50%;background:var(--ink);color:#fff;display:grid;place-items:center;font-weight:800;z-index:5;font-size:14px}
.cdet>summary{font-size:18px;letter-spacing:-.025em;padding:20px 26px}
.cdet>summary i{color:var(--blue)}
.cwin{background:linear-gradient(135deg,#e8f7ef,#f4fbf7);box-shadow:inset 0 0 0 1px rgba(34,164,93,.28);border-radius:16px;padding:14px 18px}
.cwin strong{font-size:30px;letter-spacing:-.05em}.cwin .cwk{color:#14803f;font-weight:800;font-size:12px}
.cstart td,.cmpStats td{font-size:17px;font-weight:700;letter-spacing:-.02em;color:#8a94a3;text-align:center}
.cstart td.good,.cmpStats td.good{color:var(--green);font-weight:800}
.cstart td.cmid,.cmpStats td:nth-child(2){font-size:13px;color:var(--muted);font-weight:600;letter-spacing:0}
.cstart th,.cmpStats th{font-size:13px;color:var(--ink);font-weight:800;text-transform:none;letter-spacing:-.01em}
.cmpStats{padding:8px 24px}
@media(max-width:700px){.compareHero{grid-template-columns:1fr 1fr;gap:10px;padding:14px 0}.comparePlayer{padding:18px 8px 16px;border-radius:20px}.comparePlayer:first-child{transform:rotateY(9deg)}.comparePlayer:last-child{transform:rotateY(-9deg)}.cmpPortrait{width:92px;height:92px;margin:6px auto 16px}.comparePlayer h2{font-size:16px}.comparePlayer .scorebig{font-size:38px}.compareHero .vs{width:38px;height:38px;font-size:12px}.cstart td,.cmpStats td{font-size:15px}.cwin strong{font-size:24px}}
/* insights */
.insNav{border:0;gap:6px;margin:0 0 22px}
.insTab{border:0;border-radius:10px;background:#fff;box-shadow:0 0 0 1px rgba(20,35,70,.1);padding:9px 14px;font-size:14px;color:#2a3546}
.insTab.on,.insTab:hover{background:var(--ink);color:#fff;border:0}
.mkRow{grid-template-columns:20px 44px minmax(0,1fr) 70px 46px 46px;gap:10px 12px;border-top:1px solid #edf0f4;padding:14px 0}
.mkMain{min-width:0}.mkArrow{font-size:11px;margin-right:6px;vertical-align:1px}.mkBuy .mkArrow{color:var(--green)}.mkSell .mkArrow{color:#cc3340}
.mkSp{width:70px;height:30px;margin:0}
.mkPct small{font-size:10px;color:var(--muted);display:block;text-align:center;margin-top:2px}
@media(max-width:900px){#marketBody{grid-template-columns:1fr}.mkRow{grid-template-columns:18px 40px minmax(0,1fr) 46px 46px}.mkSp{display:none}}
.tinyDisc{color:#8a94a3}
/* ===== v6 refinements ===== */
/* compare */
.cbt.l>div[style*="22a45d"]{background:linear-gradient(270deg,#0f9a52,#3fe08b)!important}
.cbt.l>div[style*="e3a21a"]{background:linear-gradient(270deg,#e09a0a,#ffd04a)!important}
.cbt.l>div[style*="e2505b"]{background:linear-gradient(270deg,#d8283a,#ff6f7c)!important}
.compareHero{background:transparent!important;box-shadow:none!important;border:0!important;align-items:stretch;padding:22px 14px!important;gap:28px!important}
.comparePlayer{display:flex!important;flex-direction:column;align-items:center;text-align:center}
.comparePlayer:first-child{transform:rotateY(6deg)!important}.comparePlayer:last-child{transform:rotateY(-6deg)!important}
.comparePlayer .cmpBd{min-height:38px;margin-top:12px;display:flex;justify-content:center;width:100%}
.comparePlayer .cmpBd .bi{margin:0;flex-wrap:wrap;justify-content:center;gap:5px}.comparePlayer .cmpBd .bi i{width:30px;cursor:pointer}
.comparePlayer .gdet{margin-top:auto;width:100%;padding-top:14px}
.compareHero .vs{top:149px!important}
@media(max-width:700px){.compareHero{padding:12px 4px!important;gap:10px!important}.comparePlayer:first-child{transform:rotateY(5deg)!important}.comparePlayer:last-child{transform:rotateY(-5deg)!important}.comparePlayer .cmpBd{min-height:30px;margin-top:8px}.comparePlayer .cmpBd .bi i{width:24px}.compareHero .vs{top:82px!important}}
/* home */
.homeActions{flex-wrap:nowrap;gap:8px}.homeActions .cta{white-space:nowrap}
.homeSearch{position:relative;z-index:30}.hhR{position:relative;z-index:1}
.homeGrid{max-width:none!important;padding-top:26px}
.homeCap{margin:-6px 0 14px;font-size:14px;color:var(--muted)}
.posBest{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.pbc{position:relative;background:#fff;border-radius:20px;box-shadow:var(--shadow);padding:22px 12px 18px;text-align:center;cursor:pointer;transition:transform .25s,box-shadow .25s}
.pbc:hover{transform:translateY(-4px)}
.pbc .thumb{width:76px;height:76px;margin:0 auto}
.pbBadge{position:absolute;top:12px;right:12px;width:30px}.pbBadge svg{width:100%;height:auto;display:block;filter:drop-shadow(0 0 4px rgba(70,190,255,.9))}
.pbPos{font-size:11px;font-weight:800;letter-spacing:.9px;text-transform:uppercase;color:#2f8fd0;margin-top:12px}
.pbName{font-size:19px;font-weight:800;letter-spacing:-.04em;margin-top:4px;line-height:1.1}
.pbTag{font-size:12.5px;color:var(--muted);font-weight:600;margin-top:2px}
.pbScore{font-size:34px;font-weight:800;letter-spacing:-.05em;margin-top:8px;line-height:1}
.pbLab{font-size:11px;color:var(--muted);font-weight:700;margin-top:3px}
@media(max-width:700px){.posBest{grid-template-columns:1fr 1fr;gap:10px}.pbName{font-size:16px}.pbScore{font-size:28px}.pbc .thumb{width:64px;height:64px}}
/* fanned stack */
.stage{touch-action:pan-y}
.stack{transform:rotateY(-14deg) rotateX(6deg) rotateZ(1deg)!important}
.hc{left:calc(100% - 316px)!important;top:46px;opacity:1!important;overflow:visible;transform:translate3d(calc(var(--o)*-48px + var(--dx,0px)),calc(var(--o)*-7px),calc(var(--o)*-34px)) rotateZ(calc(var(--o)*-2.2deg))!important}
.hc.drag{transition:none!important}
.hc:after{content:"";position:absolute;inset:0;border-radius:22px;background:#e8ecf3;opacity:0;pointer-events:none;transition:opacity .5s}
.hc[data-o="1"]:after{opacity:.42}.hc[data-o="2"]:after{opacity:.6}.hc[data-o="3"]:after{opacity:.74}.hc[data-o="4"]:after{opacity:.85}
.hrk{left:10px!important;top:12px!important;font-size:15px!important;padding:2px 8px!important;border-radius:8px!important}
.hc[data-front="1"] .hrk{font-size:20px!important;padding:3px 11px!important;left:16px!important;top:14px!important}
.hvn{position:absolute;left:15px;top:56px;writing-mode:vertical-rl;transform:rotate(180deg);font-weight:800;font-size:13px;letter-spacing:-.01em;white-space:nowrap;color:var(--ink);z-index:3;display:none}
.hvn b{margin-top:6px;font-size:13px}
.hc:not([data-front="1"]) .hvn{display:block}
.hc[data-front="0"] .hpf,.hc[data-front="0"] h3,.hc[data-front="0"] .htag{opacity:.35}
@media(max-width:900px){.homeHero{display:flex!important;flex-direction:column;align-items:stretch!important}.hhR,.hhL .homeSearch{width:100%}.stap{white-space:nowrap}.hhL{display:contents}.hhL .kicker{order:1}.hhL h1{order:2}.hhR{order:3}.hhL .sub{order:4}.hhL .homeSearch{order:5;width:100%}.hhL .homeActions{order:6;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;width:100%;margin-top:16px}.homeActions .primary{grid-column:1/-1;flex-basis:auto}.homeActions .cta{padding:11px 4px;font-size:12px}
 .stage{height:392px!important;margin:2px 0 22px!important}.stack{transform:rotateY(-12deg) rotateX(5deg) scale(.8)!important;transform-origin:100% 40%!important}.hc{left:calc(100% - 300px)!important;transform:translate3d(calc(var(--o)*-36px + var(--dx,0px)),calc(var(--o)*-6px),calc(var(--o)*-30px)) rotateZ(calc(var(--o)*-2deg))!important}
 .stackFoot{left:0!important;right:auto!important;bottom:0!important}.hhL .sub{margin-top:0}}
/* profile hero */
.hero2{grid-template-columns:minmax(0,270px) minmax(0,270px)!important;grid-template-areas:"ph ph" "id id" "ml wk" "pj pj" "ol ol"!important;justify-content:center;justify-items:center;text-align:center!important;column-gap:0!important;row-gap:10px!important;padding:40px 28px 34px!important}
.hp{grid-area:ph;margin-bottom:14px}
.hid{grid-area:id;text-align:center!important;align-self:start!important}
.hid h1{font-size:50px!important;letter-spacing:-.05em;line-height:1.02;text-align:center!important}
.hid .meta{font-size:16px!important;color:var(--muted);margin-top:8px;text-align:center!important}
.hid .rankline{font-size:14px;color:#3a4352;font-weight:600;margin-top:6px;text-align:center!important}
.hs.left,.hs.right{text-align:center!important;width:100%;padding:20px 10px 2px}
.hs.right{border-left:1px solid #e6eaef}
.hs .scorebig{font-size:68px!important}
.hs .scorelab{font-size:12px!important;letter-spacing:.9px;margin-top:10px}
.hs .scoresub{font-size:12.5px;color:var(--muted);margin-top:4px;line-height:1.35;font-weight:600}
.hpjW{grid-area:pj;margin-top:6px;text-align:center}.hpjW .heroPj{margin:0!important;font-size:15px;text-align:center!important}
.hol{grid-area:ol!important;text-align:center!important;align-items:center!important;align-self:start;margin-top:8px}.hol .bdgs{justify-content:center!important}
@media(max-width:900px){.hero2{grid-template-columns:1fr 1fr!important;padding:26px 14px 24px!important}.hid h1{font-size:36px!important}.hs .scorebig{font-size:52px!important}.hs .scoresub{font-size:11.5px}.hpjW .heroPj{font-size:14px}}
/* by the numbers */
.usageGrid{grid-template-columns:1fr 1fr!important;gap:10px!important;background:none!important;border:0!important}
.usageItem{border:0!important;border-radius:14px!important;background:#f6f8fb!important;padding:14px 16px!important;box-shadow:inset 0 0 0 1px rgba(20,35,70,.06)}
.usageItem .ul{font-size:12.5px!important;color:var(--muted)!important;font-weight:700;text-transform:none!important;margin:0 0 6px!important}
.usageItem .uv{font-size:30px!important;letter-spacing:-.04em;line-height:1}
.usageItem .cbt{margin-top:10px;height:9px;flex:none;display:block}
.uMeta{display:flex;justify-content:space-between;gap:6px;font-size:11.5px;font-weight:700;margin-top:8px;color:var(--muted)}
.usageItem:last-child:nth-child(odd){grid-column:1/-1}
@media(max-width:700px){.usageGrid{gap:8px!important}.usageItem{padding:12px!important}.usageItem .uv{font-size:24px!important}.uMeta{flex-direction:column;gap:2px}}
/* matchup */
.muHd{display:flex;justify-content:space-between;align-items:center;gap:14px}
.muK{font-size:11px;font-weight:800;letter-spacing:.9px;color:var(--muted);text-transform:uppercase}
.muOpp{font-size:30px;font-weight:800;letter-spacing:-.045em;line-height:1.05;margin-top:2px}
.muBoost{text-align:center;min-width:96px;padding:10px 14px;border-radius:16px;background:#f6f8fb;box-shadow:inset 0 0 0 1px rgba(20,35,70,.07)}
.muBoost b{display:block;font-size:34px;letter-spacing:-.05em;line-height:1}.muBoost span{font-size:11px;font-weight:700;color:var(--muted)}
.muBoost.good b{color:var(--green)}.muBoost.bad b{color:#cc3340}.muBoost.mid b{color:#6a7482}
.muVerdict{display:inline-block;margin:12px 0 4px;padding:5px 12px;border-radius:20px;font-weight:800;font-size:13px}
.muVerdict.good{background:#e4f6ec;color:#14803f;box-shadow:0 0 12px rgba(34,190,100,.28)}
.muVerdict.bad{background:#fde9eb;color:#c42a3a;box-shadow:0 0 12px rgba(235,60,80,.25)}
.muVerdict.mid{background:#eef1f5;color:#566070}
.muFacts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}
.muFacts>div{background:#f6f8fb;border-radius:14px;padding:12px 8px;text-align:center;box-shadow:inset 0 0 0 1px rgba(20,35,70,.06)}
.muFacts b{display:block;font-size:26px;letter-spacing:-.04em;line-height:1.05}
.muFacts span{display:block;font-size:11.5px;color:var(--muted);font-weight:700;margin-top:4px;line-height:1.3}
.muFacts em{font-style:normal;display:block;font-size:11.5px;font-weight:800;margin-top:3px}
.muDet{margin-top:10px}.muDet summary{cursor:pointer;font-weight:700;font-size:13px;color:var(--blue)}
@media(max-width:700px){.muFacts{gap:6px}.muFacts>div{padding:10px 4px}.muFacts b{font-size:21px}.muFacts span{font-size:10.5px}.muOpp{font-size:26px}.muBoost b{font-size:28px}}
/* rankings: photo in every view, bars, average row */
#rankings.wideView td:nth-child(2) .thumb{display:grid!important;width:34px;height:34px}
#rankings.wideView .nmw{display:flex!important;align-items:center;gap:9px}
#rankings.wideView td:nth-child(2),#rankings.wideView th:nth-child(2){min-width:180px}
.bkc .bkw{display:flex;align-items:center;gap:8px;justify-content:flex-end}.bkw .cbt{width:78px;flex:none;height:10px}.bkw b{min-width:26px;text-align:right;font-size:15px}
.stc{text-align:right}.stc .mbw{width:54px;height:5px!important;margin:4px 0 0 auto;flex:none;display:block}
#rankings tr.avgRow td{background:#eaf1ff!important;font-weight:800;color:var(--ink);border-bottom:1px solid #d6e2f7}
#rankings tr.avgRow .tag{display:block;font-size:10.5px;color:var(--muted);font-weight:700;background:none;padding:0;margin:1px 0 0}
#rankings tr.avgRow{cursor:default}
/* ===== v7 ===== */
.title{font-weight:780!important;letter-spacing:-.035em!important;font-size:46px}
@media(max-width:700px){.title{font-size:34px!important}}
#rankings.wideView td .bi{display:flex!important;margin:4px 0 0!important;gap:3px!important}
#rankings.wideView td .bi i{width:20px!important}
.pjp .nmw{display:flex;align-items:center;gap:10px}.pjp .nmt2{min-width:0}.pjp .thumb{width:40px;height:40px}
.pjp .pjs{display:flex;flex-wrap:wrap;align-items:center;gap:2px 0}.pjp .pjs .bi{margin-left:8px}
@media(max-width:700px){#rankings.projMode .rankDesk{display:none}#rankings.projMode .rankCards{display:block}
 .rcard .rs small{display:block!important;font-size:10px;margin-top:3px}.rcHead span{white-space:nowrap}}
@media(max-width:900px){
 .hhL .sub{order:3}.hhL .homeSearch{order:4;margin:4px 0 14px}.hhR{order:5}.hhL .homeActions{order:6;margin-top:6px}
 .stage{height:452px!important;margin:0!important}.stack{bottom:58px!important}.stackFoot{left:0!important;right:0!important;bottom:6px!important;justify-content:center}
}
@media(max-width:700px){.rcard .bi i,.pjp .bi i{width:22px!important}.rcard .bi{gap:3px!important;margin-left:6px!important}}
@media(max-width:700px){.rcard,.rcHead{grid-template-columns:22px minmax(0,1fr) 42px 46px 46px!important;gap:4px!important}.rcHead span{white-space:normal!important;line-height:1.15;font-size:11px}.rcard .rs{font-size:15.5px!important}}
.roomCtl{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:2px 0 12px}
.roomNote{font-size:14px;color:#3a4352}.roomNote b{color:var(--ink)}
.rmSeg{display:inline-flex;background:#fff;border-radius:12px;box-shadow:0 0 0 1px rgba(20,35,70,.1);padding:3px}
.rmb{border:0;background:none;padding:8px 14px;border-radius:9px;font-weight:700;font-size:13px;color:#2a3546;cursor:pointer}.rmb.on{background:var(--ink);color:#fff}
.roomtable td.rmc{line-height:1.2;white-space:nowrap}.roomtable td.rmc b{display:block;font-size:17px;letter-spacing:-.02em}.roomtable td.rmc small{display:block;font-size:11px;color:var(--muted);font-weight:600;margin-top:2px}
.roomtable td.rmS{background:#f6f8fb}
@media(max-width:700px){.roomtable th,.roomtable td{padding:8px 8px!important;font-size:13px!important}.roomtable td.rmc b{font-size:15px}.roomtable td.rmc small{font-size:10px}.rmb{padding:8px 10px;font-size:12px}.rmSeg{width:100%}.rmb{flex:1}}
/* ===== v8 team compare ===== */
.cmpModes,.tseg{display:inline-flex;background:#fff;border-radius:14px;box-shadow:0 0 0 1px rgba(20,35,70,.1),0 6px 18px -8px rgba(20,35,70,.2);padding:4px;margin:4px 0 14px}
.cmpMode,.tsb{border:0;background:none;padding:10px 20px;border-radius:10px;font-weight:750;font-size:14px;color:#2a3546;cursor:pointer;white-space:nowrap}
.cmpMode.on,.tsb.on{background:var(--ink);color:#fff;box-shadow:0 6px 14px -6px rgba(10,20,40,.6)}
.tctl{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:2px 0 6px}.tctl .tseg{margin:0}
.tsample{border:0;background:none;color:var(--blue);font-weight:750;font-size:13px;cursor:pointer;padding:8px 6px}
.tnote{font-size:13.5px;color:var(--muted);margin:6px 0 18px;max-width:860px;line-height:1.5}
.tgrid{display:grid;grid-template-columns:1fr 1fr;gap:34px;padding:0 6px;position:relative;perspective:1600px;margin-bottom:22px;align-items:start}
.tcard{background:#fff;border-radius:24px;box-shadow:var(--shadow);padding:24px 20px 18px;position:relative;transition:box-shadow .3s}
.tgrid .tcard:first-child{transform:rotateY(2.5deg);transform-origin:right center}.tgrid .tcard:last-child{transform:rotateY(-2.5deg);transform-origin:left center}
.tcard.lead{box-shadow:0 0 0 2px rgba(34,190,100,.55),0 0 28px rgba(34,190,100,.28),var(--shadow)}
.tlead{position:absolute;top:16px;right:16px;background:#e4f6ec;color:#14803f;font-size:11px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;padding:4px 10px;border-radius:20px;box-shadow:0 0 12px rgba(34,190,100,.3)}
.tname{width:100%;border:0;background:none;text-align:center;font:800 24px Inter,sans-serif;letter-spacing:-.04em;color:var(--ink);outline:none;border-bottom:1px dashed transparent;padding:2px 0}.tname:hover,.tname:focus{border-bottom-color:#c4cdda}
.tscoreW{display:flex;flex-direction:column;align-items:center;margin:10px 0 8px}
.tring{position:relative;width:128px;height:128px}.tring .rsv{position:absolute;inset:0;width:100%;height:100%}
.tbig{position:absolute;inset:0;display:grid;place-items:center}.tbig b{font-size:34px;font-weight:800;letter-spacing:-.05em}
.tcap{text-align:center;font-size:12.5px;color:var(--muted);font-weight:700;margin-top:8px;line-height:1.35}.tcap span{font-weight:600;color:#8a94a3}
.tsec{font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin:16px 0 6px}
.tslot{display:grid;grid-template-columns:34px 44px minmax(0,1fr) auto auto auto;gap:9px;align-items:center;padding:9px 8px;border-radius:14px;background:#f6f8fb;margin-bottom:7px;box-shadow:inset 0 0 0 1px rgba(20,35,70,.05);width:100%;text-align:left;border:0}
.tslot.empty{border:1.5px dashed #c9d2de;background:transparent;box-shadow:none;cursor:pointer;grid-template-columns:34px 1fr;min-height:62px}.tslot.empty:hover{background:#f6f8fb}
.tslot.open{grid-template-columns:34px minmax(0,1fr) auto;position:relative;z-index:40;background:#fff;box-shadow:0 0 0 2px var(--blue)}
.tpos{font-size:10.5px;font-weight:800;color:#fff;background:var(--ink);border-radius:8px;padding:4px 0;text-align:center;letter-spacing:.3px}
.tadd{font-weight:750;color:var(--muted);font-size:14px}
.tnm{min-width:0}.tnm b{display:block;font-size:15px;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer}.tnm small{display:block;font-size:11.5px;color:var(--muted);font-weight:600;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ttag{font-style:normal;color:#cc3340;font-weight:800}
.tv{font-size:20px;font-weight:800;letter-spacing:-.04em;min-width:48px;text-align:right}
.tgm{border:0;background:#fff;border-radius:9px;box-shadow:0 0 0 1px rgba(20,35,70,.12);font-size:11px;font-weight:750;padding:5px 8px;cursor:pointer;color:#2a3546;white-space:nowrap}.tgm.ph0{background:none;box-shadow:none;border:0}.tgm.on{background:#e8f0ff;color:var(--blue);box-shadow:0 0 0 1px #9db8f5}
.tx{border:0;background:none;color:#97a2b3;font-size:14px;cursor:pointer;padding:4px 6px;border-radius:8px}.tx:hover{background:#fde9eb;color:#cc3340}
.tsearch{position:relative;min-width:0}.tsearch input{width:100%;height:38px;border:0;background:#f1f4f8;border-radius:10px;padding:0 12px;font-size:14px;outline:none}
.tsearch .dd{top:42px;z-index:60;max-height:300px;overflow:auto}
.tgames{display:flex;flex-wrap:wrap;gap:6px;margin:-2px 0 9px;padding:10px;background:#f6f8fb;border-radius:12px}
.tfoot{margin-top:10px;text-align:center}.tclr{border:0;background:none;color:#97a2b3;font-weight:700;font-size:12px;cursor:pointer}.tclr:hover{color:#cc3340}
.tvs{position:absolute;left:50%;top:108px;transform:translateX(-50%);width:54px;height:54px;border-radius:50%;background:var(--ink);color:#fff;display:grid;place-items:center;font-weight:800;font-size:14px;z-index:5;box-shadow:0 10px 24px -8px rgba(10,20,40,.6)}
.tverdict{text-align:center;margin:6px 0 18px;padding:20px}.tverdict strong{display:block;font-size:34px;letter-spacing:-.05em;margin:4px 0}.tprob{margin-top:12px;display:inline-flex;align-items:baseline;gap:10px;background:#fff;border-radius:14px;padding:10px 18px;box-shadow:0 0 0 1px rgba(34,164,93,.25)}.tprob b{font-size:30px;letter-spacing:-.05em;color:var(--green)}.tprob span{font-size:13px;color:#3a4352;font-weight:650}
.tempty{text-align:center;background:#fff;border-radius:20px;box-shadow:var(--shadow);padding:34px 20px;margin-bottom:18px}.tempty b{display:block;font-size:20px;letter-spacing:-.03em}.tempty span{color:var(--muted);font-size:14px}
.tbars{padding:22px 24px}.tbh{display:flex;justify-content:space-between;font-weight:800;font-size:15px;margin:2px 0 6px}
.tnum{flex:0 0 58px;text-align:center;font-size:17px;letter-spacing:-.03em;font-weight:800}.tnum.dim{color:#b3bcc9}
@media(max-width:800px){
 .tgrid{grid-template-columns:1fr;gap:0;perspective:none}.tgrid .tcard:first-child,.tgrid .tcard:last-child{transform:none}
 .tvs{position:relative;left:auto;top:auto;transform:none;margin:-14px auto;width:46px;height:46px;font-size:12px}
 .tcard{padding:20px 12px 14px}.tring{width:112px;height:112px}.tbig b{font-size:30px}
 .tslot{grid-template-columns:30px 38px minmax(0,1fr) auto auto 18px;gap:6px;padding:8px 6px}.tslot .thumb{width:38px;height:38px}.tgm span{display:none}.tv{font-size:18px;min-width:42px}.tnm b{font-size:14px}
 .tslot .tx{padding:2px 2px;font-size:13px}.tgm.ph0{visibility:hidden;padding:5px 0}
 .tctl{gap:8px}.tseg{width:100%}.tsb,.cmpMode{padding:10px 8px;font-size:13px;flex:1}.cmpModes{width:100%}.tbars{padding:16px 12px}.tverdict strong{font-size:28px}.tnum{flex:0 0 46px;font-size:15px}
}
.tmulti .tsb.on{background:var(--ink)}
.tpill{display:inline-block;margin:6px 0 2px;background:#eef3ff;color:#1d4ed8;font-weight:700;font-size:12px;border-radius:20px;padding:4px 12px}.tpill b{font-size:14px;letter-spacing:-.02em}
.tv2{display:flex;flex-direction:column;align-items:flex-end;line-height:1.05}.tv2 b{font-size:19px;letter-spacing:-.04em}.tv2 small{font-size:11px;color:var(--muted);font-weight:700;margin-top:2px}
.tslot:has(.tv2){grid-template-columns:34px 44px minmax(0,1fr) auto auto auto}
.tset{background:#fff;border-radius:18px;box-shadow:var(--shadow);margin:0 0 14px}.tset>summary{display:flex;justify-content:space-between;align-items:center;padding:16px 22px;cursor:pointer;font-weight:800;font-size:17px;letter-spacing:-.02em;list-style:none}.tset>summary::-webkit-details-marker{display:none}.tset>summary i{font-style:normal;color:var(--blue);font-size:13px;font-weight:700}
.tsetB{padding:0 22px 18px}.tpre{display:inline-flex;background:#f1f4f8;border-radius:12px;padding:3px;margin-bottom:14px}.tpre .tsb{padding:8px 16px;font-size:13px}
.tsteps{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}.tst{display:flex;justify-content:space-between;align-items:center;background:#f6f8fb;border-radius:12px;padding:8px 10px 8px 14px;font-size:13.5px;font-weight:700}.tst div{display:flex;align-items:center;gap:10px}.tst b{min-width:16px;text-align:center;font-size:17px}
.tstb{width:30px;height:30px;border-radius:9px;border:0;background:#fff;box-shadow:0 0 0 1px rgba(20,35,70,.14);font-size:18px;font-weight:700;cursor:pointer;color:var(--ink)}.tstb:active{background:var(--ink);color:#fff}
.tverdict.split strong{font-size:26px}.tsplit{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px auto 0;max-width:560px}.tsplit>div{background:#fff;border-radius:14px;padding:12px;box-shadow:0 0 0 1px rgba(20,35,70,.08)}.tsplit span{display:block;font-size:11.5px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}.tsplit b{display:block;font-size:20px;letter-spacing:-.04em;margin-top:4px}.tsplit small{color:var(--muted);font-weight:700;font-size:12px}
.tbt{margin:14px 0 2px!important;font-size:15px!important}
@media(max-width:800px){.tsteps{grid-template-columns:1fr 1fr}.tset>summary{padding:14px 14px}.tsetB{padding:0 14px 14px}.tst{font-size:12.5px;padding:7px 8px 7px 10px}.tst div{gap:6px}.tsplit{grid-template-columns:1fr}.tslot:has(.tv2){grid-template-columns:30px 38px minmax(0,1fr) auto auto 18px}.tv2 b{font-size:16px}}
.tmania{background:#0b0b0d;color:#fff;border:1px solid #0b0b0d;border-radius:999px;padding:8px 16px;font-size:13px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;box-shadow:0 6px 16px rgba(0,0,0,.25)}
.tmania:hover{background:#d4202c;border-color:#d4202c}
body.mvLock{overflow:hidden}
#mv{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;font-family:'Barlow Condensed','DejaVu Sans Condensed','Arial Narrow',sans-serif;color:#fff;text-transform:uppercase}
#mv[hidden]{display:none}
#mvClose{position:absolute;right:8px;top:8px;z-index:600;background:#1a1a1d;color:#fff;border:1px solid #555;border-radius:999px;padding:5px 11px;font-size:11px;font-weight:700;line-height:1;letter-spacing:.14em;text-transform:uppercase;cursor:pointer;font-family:inherit}
#mvHint{position:absolute;left:0;right:0;bottom:6px;text-align:center;font-size:10px;letter-spacing:.2em;color:#555;z-index:50;pointer-events:none}
#mvTog{position:absolute;left:8px;top:8px;z-index:600;display:flex;background:#141417;border:1px solid #444;border-radius:999px;padding:3px}
#mvTog button{background:none;border:0;color:#aaa;font-family:inherit;font-weight:700;font-size:11px;letter-spacing:.12em;text-transform:uppercase;padding:5px 10px;border-radius:999px;cursor:pointer}
#mvTog button.on{background:#fff;color:#000}
#mvStage{position:relative;overflow:hidden;background:#000}
.mvFig{position:absolute}
.mvTag{position:absolute;z-index:20;text-align:center;white-space:nowrap;line-height:1.15;text-shadow:0 2px 8px #000,0 0 3px #000}
.mvTag i{font-style:normal;color:#e63946;letter-spacing:.12em;font-size:.82em;margin-right:5px;font-weight:700}
.mvTag b{font-weight:700;letter-spacing:.04em}
.mvTag u{text-decoration:none}.v1{margin-left:6px;opacity:.95}.v2{margin-left:5px;opacity:.55;font-size:.85em}
.mvPlate{position:absolute;left:0;text-align:center;background:linear-gradient(transparent,rgba(0,0,0,.88) 45%);padding-top:12px;line-height:1.1;text-shadow:0 1px 4px #000;z-index:5}
.mvPlate u{display:none}.mvPlate b{display:block;font-weight:700;letter-spacing:.05em}
.mvPlate span{display:block;color:#ddd}.mvPlate i{font-style:normal;color:#e63946;font-weight:700;letter-spacing:.1em;margin-right:3px}
.mvLine{position:absolute;left:50%;top:20%;height:66%;width:2px;background:linear-gradient(#d4202c,transparent);z-index:350;transform:translateX(-50%)}
.mvVig{position:absolute;inset:0;z-index:250;pointer-events:none;background:radial-gradient(ellipse at 50% 58%,transparent 55%,rgba(0,0,0,.85) 100%),linear-gradient(transparent 80%,#000 97%)}
.mvVS{position:absolute;left:50%;top:45%;transform:translate(-50%,0);z-index:500;font-size:clamp(20px,2.7vw,34px);font-weight:700;letter-spacing:.1em;background:#000;padding:4px 14px;border:2px solid #d4202c}
#mvStage.mob .mvVS{top:21%;font-size:18px;padding:1px 9px;transform:translate(-50%,-50%)}
.mvTop{position:absolute;top:3%;left:0;right:0;z-index:400;display:flex;justify-content:center;gap:7%}
.mvTop.d .mvSc{width:31%}
.mvTop:not(.d){gap:0;padding:0 1%;top:9%}
.mvTop:not(.d) .mvSc{width:50%;padding:0 3%}
.mvName{display:block;width:100%;color:#d8d8de;font-size:clamp(13px,1.35vw,19px);font-weight:600;line-height:1.4;letter-spacing:.3em;text-transform:uppercase;padding:0 0 3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mvBig{font-size:clamp(54px,6.8vw,96px);font-weight:700;line-height:.95;margin-top:4px}
#mvStage.mob .mvBig{font-size:54px}
#mvStage.fc .mvBig{font-style:italic;font-weight:800}
.mvLab{font-size:clamp(10px,1.15vw,15px);letter-spacing:.2em;color:#9a9aa2;line-height:1.35}.mvLab b{color:#fff}
.mvBrand{position:absolute;left:0;right:0;bottom:13.5%;text-align:center;font-size:10px;letter-spacing:.4em;color:#666;z-index:400}
#mvStage.mob .mvBrand{bottom:10.6%}
.mvStrip{position:absolute;left:0;right:0;bottom:0;height:13%;z-index:400;display:flex;align-items:center;justify-content:space-evenly;border-top:1px solid #2a2a2e;background:#000}
#mvStage.mob .mvStrip{height:10%}
.mvStrip div{text-align:center}.mvStrip em{display:block;font-style:normal;font-size:clamp(9px,1.05vw,13px);letter-spacing:.25em;color:#e63946}
.mvStrip b{font-size:clamp(12px,1.9vw,24px);font-weight:700}.mvStrip s{text-decoration:none;color:#555;margin:0 2px}.mvStrip u{text-decoration:none}

.tname{border-bottom:1px dashed #b8c3d6!important;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238a97ad' stroke-width='2.2'%3E%3Cpath d='M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z'/%3E%3C/svg%3E") no-repeat right 6px center;padding-right:22px!important}
.tname:focus{border-bottom-color:var(--blue)!important}
.tnameHint{font-size:12.5px;color:#5b6b85;margin:2px 0 10px;font-weight:600}
.tgm{min-width:28px;text-align:center}
@media(max-width:800px){
body.shot .cmpModes,body.shot #cmpTeams .tset,body.shot #cmpTeams .tctl,body.shot #cmpTeams .tnote,body.shot #cmpTeams .tfoot,body.shot #cmpTeams .tx,body.shot #cmpTeams .tgm,body.shot #cmpTeams .tnm small,body.shot #cmpTeams .tnameHint,body.shot #cmpTeams .tlead{display:none!important}
body.shot #cmpTeams .tgrid{grid-template-columns:1fr 1fr;gap:8px;padding:0 2px;perspective:none}
body.shot #cmpTeams .tgrid .tcard:first-child,body.shot #cmpTeams .tgrid .tcard:last-child{transform:none}
body.shot #cmpTeams .tvs{position:absolute;left:50%;top:150px;transform:translate(-50%,-50%);z-index:5;width:34px;height:34px;font-size:11px;margin:0}
body.shot #cmpTeams .tcard{padding:14px 6px 10px;border-radius:18px}
body.shot #cmpTeams .tname{font-size:17px;padding-right:0!important;background:none;border-bottom-color:transparent!important}
body.shot #cmpTeams .tring{width:92px;height:92px}body.shot #cmpTeams .tbig b{font-size:26px}
body.shot #cmpTeams .tcap{font-size:10.5px}
body.shot #cmpTeams .tslot{grid-template-columns:22px 28px minmax(0,1fr) auto;gap:4px;padding:5px 4px;margin-bottom:5px;border-radius:10px}
body.shot #cmpTeams .tslot .thumb{width:28px;height:28px}
body.shot #cmpTeams .tpos{font-size:8.5px;padding:3px 0}
body.shot #cmpTeams .tnm b{font-size:11px;line-height:1.15;white-space:normal}
body.shot #cmpTeams .tv{font-size:13px;min-width:0}
}

.rflag{display:flex;align-items:flex-start;gap:10px;background:#fff7e6;border:1px solid #f1d49a;border-radius:14px;padding:10px 12px;margin:10px 0;flex-wrap:wrap}
.rflag.on{background:#eaf7ef;border-color:#a9dcbc}
.rfi{flex:0 0 auto;width:20px;height:20px;border-radius:50%;background:#e39b12;color:#fff;font-weight:800;font-size:12px;display:grid;place-items:center;margin-top:1px}.rflag.on .rfi{background:#22a45d}
.rft{flex:1 1 220px;font-size:13px;line-height:1.4;color:#3a3220}.rft b{color:#8a5a00;margin-right:3px}.rflag.on .rft b{color:#17794a}
.rfBtn{flex:0 0 auto;border:0;background:#1b2740;color:#fff;border-radius:999px;padding:7px 13px;font-size:12px;font-weight:700;cursor:pointer}
.ttag.fl{background:#fff1d6;color:#8a5a00;border-radius:999px;padding:1px 7px;font-style:normal;font-weight:700;cursor:pointer}
.tgames .rflag{flex:1 1 100%;margin:0 0 6px}
body.shot .rflag,body.shot .ttag.fl,#mv .rflag{display:none!important}
.mvRow{position:absolute;background:linear-gradient(180deg,#1b1b20,#0b0b0d);border-bottom:2px solid #d4202c}
.mvTx{position:absolute;display:flex;flex-direction:column;justify-content:center;line-height:1.1;text-shadow:0 1px 4px #000;overflow:visible}
.mvTx b{font-weight:700;letter-spacing:.04em;white-space:nowrap}
.mvTx span{font-size:11.5px;color:#ddd;white-space:nowrap}.mvTx i{font-style:normal;color:#e63946;font-weight:700;letter-spacing:.1em;margin-right:3px}
.mvTx.c{text-align:center;display:block}.mvTx.c span{display:block}
.mvTx u,.mvHd u{display:none}
.mvHd{position:absolute;filter:drop-shadow(0 6px 8px rgba(0,0,0,.9))}
.mvTile{position:absolute;background:linear-gradient(180deg,#1b1b20,#0b0b0d);border-bottom:2px solid #d4202c;border-radius:6px}
/* Trade analyzer */
#trade .trCols{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}#trade .trCols.one{grid-template-columns:minmax(0,560px)}
#trade .trCols>div{min-width:0}
.trCard{background:#fff;border-radius:18px;box-shadow:var(--shadow);padding:18px 18px 14px;margin:0 0 14px;min-width:0}
.trCard .ph{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.trCount{font-size:11.5px;font-weight:700;color:var(--muted);letter-spacing:0;text-transform:none}
.trS{margin:0 0 12px}.trS input{height:44px;border-radius:12px;background:#f1f4f9;border:1px solid #dfe6f1}
.trRow{display:grid;grid-template-columns:44px minmax(0,1fr) auto auto auto;gap:9px;align-items:center;padding:9px 8px;border-radius:14px;background:#f6f8fb;margin-bottom:7px;box-shadow:inset 0 0 0 1px rgba(20,35,70,.05)}
.trTh .thumb{width:44px;height:44px}
.trNm{min-width:0}.trNm b{display:block;font-size:15px;letter-spacing:-.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.trNm small{display:block;font-size:12px;color:var(--muted);margin-top:2px;line-height:1.35}
.trVal{text-align:center;min-width:44px}.trVal b{display:block;font-size:17px;font-weight:800;letter-spacing:-.02em;color:var(--ink)}.trVal small{font-size:10px;color:var(--muted);font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.trTag{font-style:normal;font-size:10px;font-weight:800;padding:2px 6px;border-radius:6px;letter-spacing:.3px;text-transform:uppercase;vertical-align:1px}
.trTag.bad{background:#fde8ea;color:#c0303c}.trTag.up{background:#e1f6ea;color:#16803f}.trTag.dn{background:#fdebd9;color:#b3590c}
.trEmpty{font-size:13.5px;color:var(--muted);padding:6px 4px 10px;line-height:1.45}
.trSend{border:0;border-radius:10px;padding:8px 11px;font-weight:800;font-size:12px;background:#fff;box-shadow:0 0 0 1px rgba(20,35,70,.16);color:var(--ink);cursor:pointer}
.trSend.on{background:var(--ink);color:#fff;box-shadow:0 6px 14px -6px rgba(10,20,40,.6)}
.trRow .tx{width:28px;height:28px}
.trVerdict{text-align:center;border-radius:20px;padding:22px 20px 18px;margin:2px 0 14px;background:#fff;box-shadow:var(--shadow);border-top:4px solid #8a99b1}
.trVerdict.win{border-top-color:#22a45d;box-shadow:var(--shadow),0 0 30px -12px rgba(34,164,93,.5)}.trVerdict.lose{border-top-color:#e2505b;box-shadow:var(--shadow),0 0 30px -12px rgba(226,80,91,.45)}.trVerdict.even{border-top-color:#3a7bd5}
.trvK{font-size:11px;letter-spacing:1.2px;font-weight:800;color:var(--muted);text-transform:uppercase}
.trVerdict strong{display:block;font-size:30px;letter-spacing:-.04em;margin:4px 0 6px}.trVerdict.win strong{color:#16803f}.trVerdict.lose strong{color:#c0303c}
.trvS{font-size:14.5px;color:#3d4f6d;line-height:1.5;max-width:560px;margin:0 auto}.trvS b.up,.trvS .up{color:#16803f}.trvS b.dn,.trvS .dn{color:#c0303c}
.trBar{display:flex;height:12px;border-radius:99px;overflow:hidden;background:#e9eff8;max-width:460px;margin:16px auto 6px}.trbG{background:#e2505b}.trbR{background:#22a45d}
.trBarL{display:flex;justify-content:space-between;max-width:460px;margin:0 auto;font-size:12.5px;color:var(--muted);font-weight:700}.trBarL b{color:var(--ink)}
.trHints{list-style:none;padding:0;margin:14px auto 0;max-width:560px;text-align:left}.trHints:empty{display:none}.trHints li{font-size:13.5px;color:var(--muted);line-height:1.45;padding:6px 0 6px 18px;position:relative;border-top:1px solid #edf0f4}.trHints li:before{content:"";position:absolute;left:4px;top:13px;width:6px;height:6px;border-radius:50%;background:#8a99b1}
.trFoot{display:flex;gap:10px;justify-content:center;margin:4px 0 12px}.trFoot.l{justify-content:flex-start;margin:8px 0 2px}
.trNeeds{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:0 0 12px}
.trNd{border-radius:14px;padding:10px 12px;background:#f6f8fb;box-shadow:inset 0 0 0 1px rgba(20,35,70,.05);display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px}.trNd b{font-size:15px}.trNd small{flex-basis:100%;font-size:11.5px;color:var(--muted)}
.trNd.up{background:#eaf8f0}.trNd.dn{background:#fdecee}.trFrom{font-size:11px;color:var(--muted);font-weight:700}
.trGr{font-style:normal;font-size:10px;font-weight:800;padding:3px 7px;border-radius:7px;letter-spacing:.4px;text-transform:uppercase}.trGr.up{background:#cdeedb;color:#136b36}.trGr.dn{background:#fbd3d7;color:#a82633}.trGr.ok{background:#e4eaf3;color:#4a5b78}
.trLead{font-size:14.5px;line-height:1.5;margin:0 0 12px;color:#3d4f6d}
.trSlot{display:inline-block;min-width:30px;font-size:10.5px;font-weight:800;color:var(--muted);letter-spacing:.5px}.trNm .trSlot{display:block;margin-bottom:1px}
.trNew{display:inline;font-size:10px;color:#fff;background:#22a45d;padding:2px 6px;border-radius:6px;letter-spacing:.4px;white-space:normal}
.trFit{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;padding:10px 0;border-top:1px solid #edf0f4;font-size:13.5px}.trFit span{color:var(--muted)}
.trAsk{border:0;border-radius:10px;padding:8px 11px;background:#eef4ff;color:#2457b8;font-weight:750;font-size:12.5px;cursor:pointer}.trAsk i{font-style:normal;color:var(--muted);font-weight:700;margin-left:4px}
#trade .tsec{font-size:11px;letter-spacing:1.1px;font-weight:800;color:var(--muted);text-transform:uppercase;margin:14px 2px 8px}
@media(max-width:800px){#trade .trCols{grid-template-columns:minmax(0,1fr)}.trVerdict strong{font-size:25px}.trRow{grid-template-columns:38px minmax(0,1fr) auto auto auto;gap:7px;padding:8px 7px}.trTh .thumb{width:38px;height:38px}.trNm b{font-size:14px}.trCard{padding:14px 12px 10px}.trSend{padding:7px 9px}}
/* Trade analyzer v2 */
.trD{background:#fff;border-radius:16px;box-shadow:var(--shadow);margin:0 0 12px}
.trD>summary{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:15px 18px;cursor:pointer;font-weight:800;font-size:16px;letter-spacing:-.02em;list-style:none}
.trD>summary::-webkit-details-marker{display:none}.trD>summary i{font-style:normal;color:var(--blue);font-size:12.5px;font-weight:700;text-align:right}
.trD>summary:after{content:"▾";color:#8a99b1;font-size:13px;margin-left:4px}.trD[open]>summary:after{content:"▴"}
.trDB{padding:0 18px 16px}.trDB .trCols{gap:16px}
.trWarn{display:block;margin-top:3px;font-size:11.5px;color:#b3590c;line-height:1.35}
.trRoster .trTh .thumb{width:34px;height:34px}.trRoster .trRow{grid-template-columns:34px minmax(0,1fr) auto auto auto;padding:7px 8px;margin-bottom:5px}.trRoster .trNm b{font-size:14px}.trRoster .trVal b{font-size:15px}
.trRecRow{margin:-4px 0 10px}.trRec{font-size:12.5px;font-weight:700;color:var(--muted)}.trRec input{width:44px;height:30px;text-align:center;border:1px solid #dfe6f1;border-radius:8px;background:#f1f4f9;font-weight:800;color:var(--ink);margin:0 2px}
.trMeta{font-size:12px;color:var(--muted);line-height:1.45;margin:8px 2px 0}
.trChips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:10px 0 4px}.trChip{padding:7px 12px;border-radius:99px;font-size:12.5px;font-weight:700}.trChip b{margin-right:4px}.trChip.up{background:#dff4e7;color:#136b36}.trChip.dn{background:#fbd9dd;color:#a82633}.trChip.ok{background:#e4eaf3;color:#4a5b78}
.trVerdict.left{text-align:left}.trVerdict.left .trChips{justify-content:flex-start}.trVerdict.left .trHints{margin:10px 0 0;max-width:none}
.trDeal{border-top:1px solid #edf0f4;padding:12px 0}.trDeal:first-of-type{border-top:0}.trDealT{display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:14px;line-height:1.4}.trDealT b{display:block;font-size:10.5px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted)}.trDealT i{font-style:normal;color:var(--muted);font-size:12px}
.trDealM{font-size:12.5px;color:var(--muted);margin:6px 0 8px;line-height:1.4}.trDealM b.up{color:#16803f}.trDealM b.dn{color:#c0303c}
.trFix{font-size:13px;color:#3d4f6d;margin:12px auto 0;max-width:560px;line-height:1.5}.trFix span{display:inline-block;background:#eef4ff;border-radius:8px;padding:3px 8px;margin:3px 4px 0 0;font-weight:700}.trFix span i{font-style:normal;color:var(--muted);font-weight:600}
.trBye{padding:8px 0;border-top:1px solid #edf0f4;font-size:13.5px;line-height:1.45}.trBye:first-child{border-top:0}
.trNd small+small{margin-top:-2px}
#plModes{display:flex;margin:8px 0 16px}
/* trade v3: split verdict, suggestion, lineups */
.trSug{display:flex;align-items:center;justify-content:space-between;gap:14px;background:linear-gradient(135deg,#eef4ff,#f4f8ff);border-radius:18px;padding:16px 18px;margin:0 0 14px;box-shadow:inset 0 0 0 1px rgba(58,123,213,.18)}
.trSug b{display:block;font-size:16px;letter-spacing:-.02em}.trSug span{display:block;font-size:13.5px;color:#3d4f6d;margin-top:3px;line-height:1.4}
.trSug .trAsk{flex:none;margin:0;padding:12px 18px;font-size:14px}
.trSplit{margin:0 0 14px}
.trSB{background:#fff;border-radius:18px;box-shadow:var(--shadow);padding:16px 18px 12px;border-top:4px solid #8a99b1;min-width:0}
.trSB.good{border-top-color:#22a45d}.trSB.bad{border-top-color:#e2505b}.trSB.ok{border-top-color:#3a7bd5}
.trSBh span{display:block;font-size:11px;letter-spacing:1.2px;font-weight:800;color:var(--muted);text-transform:uppercase}
.trSBh strong{display:block;font-size:22px;letter-spacing:-.03em;margin:3px 0 12px}.trSB.good strong{color:#16803f}.trSB.bad strong{color:#c0303c}
.trSBg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:8px;margin-bottom:10px}
.trSBg>div{background:#f6f8fb;border-radius:12px;padding:9px 6px;text-align:center}.trSBg small{display:block;font-size:10.5px;font-weight:700;color:var(--muted);letter-spacing:.3px;text-transform:uppercase}
.trSBg b{display:block;font-size:19px;font-weight:800;letter-spacing:-.02em;margin-top:2px}.trSBg b.up{color:#16803f}.trSBg b.dn{color:#c0303c}
.trSBl{font-size:13px;color:var(--muted);line-height:1.5}.trSBl b{color:var(--ink)}
.trSBn{font-size:13.5px;color:#3d4f6d;line-height:1.45;margin:8px 0 0}
.trSB .trHints{margin:10px 0 0;max-width:none}
.trKn{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px}.trKn .tpre{margin-bottom:0}
.trLnW{display:flex;flex-direction:column;gap:18px}
.trLnT{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #edf0f4}.trLnT b{font-size:16px;letter-spacing:-.02em}.trLnT span{font-size:12.5px;color:var(--blue);font-weight:700}
.trLn{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px;align-items:start}.trLn>div{min-width:0}
.trLn .trRow{grid-template-columns:34px minmax(0,1fr) auto;padding:7px 8px;margin-bottom:5px}.trLn .trTh .thumb{width:34px;height:34px}.trLn .trNm b{font-size:14px}.trLn .trVal b{font-size:15px}
.trSugBox .trDeal:first-of-type{border-top:0}
@media(max-width:800px){.trSug{flex-direction:column;align-items:stretch}.trSug .trAsk{width:100%}.trLn{grid-template-columns:minmax(0,1fr)}.trSB strong{font-size:20px}}
.trStars{display:block;font-size:15px;letter-spacing:1px;line-height:1.1;background:linear-gradient(90deg,#f2a516 calc(var(--s)/5*100%),#d6dce8 0);-webkit-background-clip:text;background-clip:text;color:transparent;white-space:nowrap}
.trVal{min-width:74px}.trLn .trVal,.trRoster .trVal{min-width:74px}.trVal .trStars+small{margin-top:3px;display:block}
.trSeasonT{display:flex;align-items:baseline;gap:10px;margin:2px 0 8px}.trSeasonT>b{font-size:26px;letter-spacing:-.03em;color:var(--muted)}.trSeasonT>b.up{color:#16803f}.trSeasonT>b.dn{color:#c0303c}.trSeasonT span{font-size:13px;color:var(--muted);line-height:1.4}.trSeasonT span b{color:var(--ink)}
.trChart{width:100%;height:auto;max-height:150px;display:block;margin:4px 0 8px;background:#f8fafd;border-radius:12px;padding:4px 2px}
.tnote .tlink,.trHints .tlink,.trSBn .tlink,.trDealT .tlink{font-weight:700;color:var(--ink)}
@media(max-width:800px){.trDB{padding:0 12px 14px}.trD>summary{padding:14px 14px;font-size:15px}.trDealT{grid-template-columns:1fr}.trRoster .trRow{grid-template-columns:34px minmax(0,1fr) auto auto auto}}

body.shot #trade{display:none}
@media(max-width:700px){#mv .mvStrip{height:13%}#mv .mvStrip div{flex:1;min-width:0}#mv .mvStrip b{display:flex;flex-direction:column;align-items:center;font-size:13px;line-height:1.3}#mv .mvStrip s{display:none}#mv .mvStrip u{text-decoration:none;display:block}#mv .mvStrip em{letter-spacing:.1em;font-size:9px}}
@media(max-width:700px){#cmpPlayers .grid2{gap:10px}}
.tnoteD{margin:0 0 12px}.tnoteD>summary{cursor:pointer;font-size:13px;font-weight:700;color:var(--blue);list-style:none}.tnoteD>summary::-webkit-details-marker{display:none}.tnoteD .tnote{margin:8px 0 0}
.trNm b.trNew{display:inline-block;margin-left:2px}
@media(max-width:800px){.trMine>div:first-child{order:2}}
.trRL{box-shadow:none;background:none;margin:0}.trRL>summary{padding:4px 2px 8px;font-size:12px;color:var(--muted)}.trRL .trDB{padding:0}
.topBtns{display:inline-flex;gap:8px}
</style></head><body>
<div id="mv" hidden><div id="mvTog"></div><button id="mvClose" onclick="toggleMV(false)">Close</button><div id="mvStage"></div></div><div id="bPop" class="bPop" hidden></div><div class="shotExitTop"><button class="pill" onclick="toggleShot(false)">Exit screenshot view</button></div><div class="topbar"><div class="nav"><div class="brand" id="brandHome" title="Home" style="cursor:pointer">Fantasy Mania</div><div class="links"><button class="navb on" data-v="home">Home</button><button class="navb" data-v="rankings">Rankings</button><button class="navb" data-v="players">Players</button><button class="navb" data-v="compare">Compare</button><button class="navb" data-v="trade">Trade</button><button class="navb" data-v="insights">Insights</button></div></div></div>
<main class="wrap">
<section id="home" class="view on"><div class="homeHero"><div class="hhL"><div class="kicker">2026 season • Through Week <span id="homeWeek"></span><span id="homeWkPlus"></span></div><h1>The most accurate fantasy football system in the world.</h1><p class="sub">Every player gets a 0–100 rating from how they actually play, then a projection for this week. One accurate system for rankings, lineups and trades. Free.</p><div class="search homeSearch"><input id="homeQ" placeholder="Search any QB, RB, WR or TE"><div class="dd" id="homeDD"></div></div><div class="homeActions"><button class="cta primary" id="homeRanks">See the rankings</button><button class="cta" id="homeCompare">Compare players</button><button class="cta" id="homeTrade">Trade analyzer</button><button class="cta" id="homeShare">Team share</button><button class="cta" id="homeInsights">Insights</button></div></div><div class="hhR"><div class="stage" id="heroStage"><div class="stack" id="heroStack"></div><div class="stackFoot"><div class="sdots" id="heroDots"></div><span class="stap">Tap to shuffle</span></div></div></div></div><div class="homeGrid"><div class="homeBlock"><div class="sectionTitle">Best at every position</div><p class="homeCap">The top-rated player at each position by Mania Rating. Each one wears the Diamond badge.</p><div class="posBest" id="posBest"></div></div><div class="homeBlock"><div class="sectionTitle">How Mania works</div><div class="how3"><div><b>Mania Rating</b>How strong his role and production have been this season, compared with other players at his position. It blends production, volume, team role, red-zone work and efficiency.</div><div><b>Week <span class="nextWeekText"></span> Rating</b>His Mania Rating adjusted for this week: the opponent, how players with a similar role have done against that defense, and his recent usage.</div><div><b>Percentiles and colors</b>Every bar shows where he ranks among players at his position. Green is the top 30%, yellow is the middle, red is the bottom 40%.</div></div><p class="disclaimer left" id="homeDisc"></p></div></div><p class="tinyDisc">Calling Fantasy Mania the best rating system in the world is our own opinion, not an official ranking or an independent audit. Accuracy numbers come from our own tests on past seasons.</p></section>
<section id="players" class="view"><h1 class="title">Players</h1><div class="tseg" id="plModes"><button type="button" class="tsb on" data-v="find">Find a player</button><button type="button" class="tsb" data-v="mine">My team analyzer</button></div><div id="plFind"><p class="sub">Search a player to open the full Mania profile.</p><div class="search"><input id="playerQ" placeholder="Search player"><div class="dd" id="playerDD"></div></div></div><div id="plMine" hidden><p class="sub">Enter your full roster, starters, bench and injured players. We find your best lineup, your needs, bye-week holes, handcuffs, waiver upgrades and fair trade ideas.</p><div id="plMineSet"></div><div id="plMineBody"></div></div></section>
<section id="rankings" class="view"><div class="kicker">Through Week <span id="wk"></span><span id="wkPlus"></span></div><h1 class="title" id="rankTitle">Overall Rankings</h1><p class="sub" id="rankSub"></p><div class="toolbar tabs2"><button class="pill rankmode on" data-mode="mania">Ratings</button><button class="pill rankmode" data-mode="proj">Projections</button></div><div class="posRow"><button class="pill posf on" data-pos="ALL">ALL</button><button class="pill posf" data-pos="RB">RB</button><button class="pill posf" data-pos="WR">WR</button><button class="pill posf" data-pos="TE">TE</button><button class="pill posf" data-pos="QB">QB</button></div><div class="ctlRow"><span class="viewSeg" id="viewSeg"><button class="vbtn on" data-view="ov">Overview</button><button class="vbtn" data-view="st">Stats</button><button class="vbtn" data-view="bd">Breakdown</button></span><button class="pill sm" id="fltBtn">Filters <span id="fltN"></span> &#9662;</button></div><div class="fltPanel" id="fltPanel" hidden><div class="fltRow"><span class="fl">Game</span><button class="pill pst on" data-st="all">All</button><button class="pill pst" data-st="todo">Not played yet</button><button class="pill pst" data-st="done">Played &#10003;</button></div><div class="fltRow"><span class="fl">Badge</span><select id="badgeSel" class="teamSelect"></select></div><div class="muted fltNote" id="badgeNote"></div><div class="fltRow"><button class="reset" id="fltReset">RESET FILTERS</button></div></div><p class="rankNote" id="rankNote"></p><details class="how2"><summary>How to read this</summary><p class="disclaimer left" id="rankDisc"></p></details><div class="tablewrap rankDesk"><table id="rankTable"><thead id="rankHead"></thead><tbody id="rankBody"></tbody></table></div><div class="rankCards" id="rankCards"></div></section>
<section id="profile" class="view"><div id="profileBody"></div></section><section id="board" class="view"><div id="boardBody"></div></section>
<section id="insights" class="view"><div class="insNav"><button class="insTab on" data-v="injuries">Injuries</button><button class="insTab" data-v="share">Team Share</button><button class="insTab" data-v="defense">Defense vs. Position</button><button class="insTab" data-v="schedule">Schedule Strength</button><button class="insTab" data-v="report">Report Card</button></div>
<div class="insPane on" id="ins-injuries"><div class="kicker">Week <span class="nextWeekText"></span> report <span id="injUpd"></span></div><h1 class="title">Injuries</h1><p class="sub">Who is hurt, whether he is a starter, and who picks up the work. The points are the change to the replacement's Week rating, and they lean on how that player actually did in games the injured player missed.</p><div class="toolbar"><button class="pill injPos on" data-v="ALL">ALL</button><button class="pill injPos" data-v="RB">RB</button><button class="pill injPos" data-v="WR">WR</button><button class="pill injPos" data-v="TE">TE</button><span style="width:8px"></span><button class="pill injRole on" data-v="Starter">Starters</button><button class="pill injRole" data-v="ALL">All</button></div><div id="injBody"></div></div>
<div class="insPane" id="ins-share"><div class="kicker">Through Week <span id="shareWeek"></span></div><h1 class="title">Team Share</h1><p class="sub">See who is actually on the field and who is earning the offense each week. Every player name opens the full Fantasy Mania profile.</p><div class="teamControls"><select id="teamPick" class="teamSelect"></select><button class="pill shareMode on" data-share="snap">SNAP SHARE</button><button class="pill shareMode" data-share="target">TARGET SHARE</button></div><div id="teamShareBody"></div></div>
<div class="insPane" id="ins-defense"><div class="kicker">Through Week <span id="dvWeek"></span></div><h1 class="title">Defense vs. Position</h1><p class="sub">What each defense has given up to running backs, receivers and tight ends. "Vs usual" compares what players scored against this defense with what those same players normally score, so a defense is not punished for a schedule full of great offenses.</p><div class="toolbar"><button class="pill dvPos on" data-v="RB">RB</button><button class="pill dvPos" data-v="WR">WR</button><button class="pill dvPos" data-v="TE">TE</button></div><div class="shareNote">Easiest matchups first. Green defenses have allowed 10%+ more than usual, red ones 10%+ less.</div><div class="panel" id="defenseBody"></div></div>
<div class="insPane" id="ins-schedule"><div class="kicker">Week <span id="scWeek"></span> onward</div><h1 class="title">Schedule Strength</h1><p class="sub">Every team's remaining opponents, graded by what those defenses have given up to the position so far. It is the "vs usual" number from Defense vs. Position, looked up for each game still to come. Easiest schedules first.</p><div class="toolbar"><button class="pill scPos on" data-v="RB">RB</button><button class="pill scPos" data-v="WR">WR</button><button class="pill scPos" data-v="TE">TE</button><span style="width:8px"></span><button class="pill scWin on" data-v="next4">Next 4 weeks</button><button class="pill scWin" data-v="all">Rest of season</button><button class="pill scWin" data-v="playoffs">Weeks 15-17</button></div><div class="shareNote">Green is a defense allowing 10%+ more than usual to the position, red is 10%+ less. This early most defenses have only a few games behind them, so expect these to move.</div><div class="panel" id="scheduleBody"></div></div>
<div class="insPane" id="ins-report"><div class="kicker">Week <span class="rcWeek"></span> results</div><h1 class="title">Report Card</h1><p class="sub">How the Week <span class="rcWeek"></span> ratings held up. These are the ratings the model gives using only the games before Week <span class="rcWeek"></span>, set against what each player then scored.</p><div class="rcGrid" id="reportTiles"></div><div class="toolbar"><button class="pill rcPos on" data-v="RB">RB</button><button class="pill rcPos" data-v="WR">WR</button><button class="pill rcPos" data-v="TE">TE</button></div><div id="reportBody"></div></div>
</section>
<section id="compare" class="view"><div class="topRow"><h1 class="title">Compare</h1><button class="shotBtn" onclick="toggleShot(true)">Screenshot view</button></div><div class="cmpModes"><button class="cmpMode on" data-cm="players">Players</button><button class="cmpMode" data-cm="teams">Teams</button></div><div id="cmpPlayers"><p class="sub">Overall value and Week <span id="cmpWeek"></span> start decision. Change the games used for either player and the model recalculates.</p><div class="grid2"><div class="search"><input id="aQ" placeholder="Player A"><div class="dd" id="aDD"></div></div><div class="search"><input id="bQ" placeholder="Player B"><div class="dd" id="bDD"></div></div></div><div id="compareBody"></div></div><div id="cmpTeams" hidden><p class="sub">Build two fantasy teams and see who is stronger. <b>Tap a team name to rename it.</b> Compare just the starters, just the bench, or everyone together.</p><div id="teamBody"></div></div></section>
<section id="trade" class="view"><div class="topRow"><h1 class="title">Trade Analyzer</h1></div><p class="sub">Value is rest-of-season points above a replacement player, PPR, for your league size. Check a single trade or test two full teams. Your own team analyzer is in the Players tab.</p><div class="tseg" id="trModes" style="display:flex"><button type="button" class="tsb on" data-v="quick">Quick trade</button><button type="button" class="tsb" data-v="full">Full-team trade</button></div><div id="trSet" style="margin-top:14px"></div><div id="trBody"></div></section>
<p class="disclaimer foot">Fantasy Mania is built to be the most accurate fantasy football system we can make, and we test it against the real results every week. But no system can predict a single game. Ratings, projections and trade values are estimates, not guarantees, and injuries and game flow can change anything. Use them as a guide for your own decisions.</p><div class="shotMark">FANTASY MANIA &bull; 2026 &bull; Through Week <span id="shotWk"></span></div></main><div class="shotExit"><button class="pill" onclick="toggleShot(false)">Exit screenshot view</button></div><script>
const DB=__PAYLOAD__, META=__META__, REFS=__REFS__; const $=id=>document.getElementById(id),byId=id=>DB.find(p=>p.id===id)||(META.qbs||[]).find(p=>p.id===id),fmt=n=>Math.round(Number(n)*10)/10;$('wk').textContent=META.week;$('shotWk').textContent=META.week;$('cmpWeek').textContent=META.next_week;
const DISC=`Week ${META.next_week} Rating = recent role + opponent + the latest injury report${META.injuries&&META.injuries.updated?' (updated '+META.injuries.updated+')':''}. Check final inactives before you set your lineup.`;
function ord(n){n=Math.round(n);let v=n%100,x=['th','st','nd','rd'];return n+(x[(v-20)%10]||x[v]||x[0])}
function fs(p,v){return p.played?(p.played.grade!=null?Number(p.played.grade).toFixed(1):'PLAYED'):p.out?'OUT':v==null?'\u2014':Number(v).toFixed(1)}
function sval(p){return p.played&&p.played.grade!=null?p.played.grade:p.start==null?-1:p.start}
function sv(p){return p.played&&p.played.grade!=null?`<span class="${grade(p.played.grade)}">${Number(p.played.grade).toFixed(1)}</span><span class="pdone">\u2713</span>`:p.out?'OUT':p.start==null?'\u2014':p.start}
function injChip(p){if(!p.inj||p.inj.avail>0.6)return '';let t=p.out?'OUT':p.inj.label==='Missed practice'?'DNP':'Q';return `<span class="injc ${p.out?'o':'q'}" title="${p.inj.label}${p.inj.note?' ('+p.inj.note+')':''}">${t}</span>`}
function pjT(p){return p.out?'OUT':p.played?'PLAYED':p.proj?p.proj.pts.toFixed(1):'\u2014'}
function vgNote(p,q){let v=q.vg;if(!v||!v.c)return '';let L={rec:['catches',1,.7],ryd:['receiving yards',.1,8],ruy:['rushing yards',.1,8],td:['touchdowns',6,.08],pyd:['passing yards',.04,15],ptd:['passing TDs',4,.15]},
 d=v.c.map(c=>{let l=L[c[0]];return l?{t:l[0],m:c[1],v:c[2],pts:Math.abs((c[2]-c[1])*l[1]),big:Math.abs(c[2]-c[1])>=l[2],f:c[0]==='td'||c[0]==='ptd'?2:0}:null}).filter(x=>x&&x.big).sort((a,b)=>b.pts-a.pts).slice(0,2),
 why=d.length?d.map(x=>`${x.t} ${x.v.toFixed(x.f)} on the sportsbooks vs ${x.m.toFixed(x.f)} from usage`).join('; '):'in line with the sportsbooks';
 return `<div class="projNote vgn"><b>Vegas lines included.</b> From usage alone: ${v.pre.toFixed(1)}. Sportsbook lines imply ${v.v.toFixed(1)}. Blended: <b>${q.pts.toFixed(1)}</b>. ${d.length?'Biggest gaps: ':'Right now he is '}${why}.</div>`}
function projBox(p,q,line,title){let top=Math.max(q.hi*1.06,20),L=q.lo/top*100,W=(q.hi-q.lo)/top*100,M=q.pts/top*100,av=p.inj?p.inj.avail:1,risk=av<1&&p.pos!=='QB'?`<div class="projNote">${p.inj.label}: about ${Math.round(av*100)}% to play. Weighed for injury risk: <b>${(q.pts*av).toFixed(1)}</b>.</div>`:'',fx=(k,l)=>{let v=q.fx[k];return `<div class="pfx"><b class="${v>0.4?'good':v<-0.4?'bad':'muted'}">${Math.abs(v)<0.05?'0.0':sg(v)}</b><span>${l}</span></div>`};return `<div class="panel projBox"><div class="ph">${title} <span class="muted">vs ${p.opp} \u2022 PPR</span></div><div class="projTop"><div class="projNum">${q.pts.toFixed(1)}</div><div class="projTxt"><b>projected points</b><span>${line}</span></div></div><div class="projBar"><i style="left:${L}%;width:${W}%"></i><u style="left:${M}%"></u></div><div class="projRng"><span>Floor ${q.lo.toFixed(1)}</span><span>4 of 5 games land here</span><span>Ceiling ${q.hi.toFixed(1)}</span></div>${risk}${vgNote(p,q)}<div class="projFx">${fx('script','Game script')}${fx('matchup','Matchup')}${fx('inj','Injuries')}</div><details class="how"><summary>How this is built</summary><div class="explain">Recent and last-season usage, the Vegas spread and total, how this defense treats his position, and the injury report (teammates out${p.pos==='QB'?', receivers out':', QB out'}). The three boxes show how many points each moved it. When sportsbook player lines are posted (Wednesday to Sunday) they are blended in, leaning on them most for new starters and players with few games. PPR scoring only${p.pos==='QB'?'; QBs get 4 per passing TD, 1 per 25 yards, -2 per INT':''}. Tested on past seasons it is about ${p.pos==='QB'?'10':'5'}% closer than a plain season average.</div></details></div>`}
function projHTML(p){let q=p.proj;if(!q||p.out||p.played||p.opp==='BYE')return '';return projBox(p,q,`${q.rec.toFixed(1)} catches, ${q.ryd+q.ruy} yards, ${q.td.toFixed(2)} TD • ${p.pos==='RB'||q.car>=1?q.car+' carries, ':''}${q.tgt} targets`,'Week '+META.next_week+' projection')}
function qbHTML(p){let out='';if(p.played){let q=p.played,d=p.proj?q.ppr-p.proj.pts:null;out+=`<div class="injBox playedBox qbCard"><div class="injHd">\u2713 Final \u2022 Week ${q.wk}</div><div class="injRow ${d==null||d>=0?'up':'o'}"><b>${resTxt(q)}</b> vs ${q.opp} \u2022 scored <b>${q.ppr}</b> PPR${d!=null?` (projected ${p.proj.pts.toFixed(1)}, ${d>=0?'+':''}${d.toFixed(1)})`:''}</div><div class="injRow q">${q.comp}/${q.att}, ${q.pyd} yds, ${q.ptd} TD, ${q.int} INT${q.ruy?', '+q.ruy+' rush yds':''}${q.rtd?', '+q.rtd+' rush TD':''}</div></div>`}let q=p.proj;if(!q)return out;if(!p.played)out+=projBox(p,q,`${q.pyd} pass yds, ${q.ptd.toFixed(1)} pass TD, ${q.int.toFixed(1)} INT \u2022 ${q.ruy} rush yds`,'Week '+META.next_week+' projection');return out+(p.sub?'<div class="projNote">Starter is out, so this is the backup (starter\'s volume, a little lower).</div>':'')+(p.inj?'<div class="projNote">'+p.inj.who+': '+p.inj.label+'.</div>':'')}
function pv(p){return p.out?-1:p.played?p.played.ppr:p.proj?p.proj.pts:-2}
function heroPj(p){if(p.played){let q=p.played,d=q.proj!=null?q.ppr-q.proj:null;return `<div class="heroPj">Projected <b>${q.proj!=null?q.proj.toFixed(1):'\u2014'}</b> \u2192 scored <b class="${d==null?'':d>=0?'good':'bad'}">${q.ppr}</b> PPR</div>`}if(p.out||!p.proj||p.opp==='BYE')return '';return `<div class="heroPj">Week ${META.next_week} projection <b>${p.proj.pts.toFixed(1)}</b> PPR <span class="muted">(${p.proj.lo.toFixed(0)}\u2013${p.proj.hi.toFixed(0)})</span></div>`}
function resTxt(q){let m=/(\d+), \w+ (\d+)/.exec(q.score||'');return m?`${q.res} ${m[1]}-${m[2]}`:q.res}
function playedHTML(p){let q=p.played;if(!q)return '';let d=q.proj!=null?q.ppr-q.proj:null;return `<div class="injBox playedBox"><div class="injHd">\u2713 Final \u2022 Week ${q.wk}</div><div class="injRow ${d==null||d>=0?'up':'o'}"><b>${resTxt(q)}</b> vs ${q.opp} \u2022 scored <b>${q.ppr}</b> PPR${q.proj!=null?` (projected ${q.proj.toFixed(1)}, ${d>=0?'+':''}${d.toFixed(1)})`:''}</div><div class="injRow q">${q.rec} catches on ${q.tgt} targets, ${q.ry} yds${q.car?', '+q.car+' carries, '+q.ruy+' yds':''}${q.td?', '+q.td+' TD':''}${q.snap!=null?' \u2022 '+q.snap+'% snaps':''}${q.grade!=null?' \u2022 graded '+Number(q.grade).toFixed(1):''}. ${q.counted?'Counted in his Mania Rating.':'Not in his Mania Rating yet (waiting on NFL data).'}</div></div>`}
function injHTML(p){let wk=META.next_week,o=[],pm=x=>(x>0?'+':'')+x.toFixed(1);
 (p.inj_why||[]).forEach(w=>{o.push(w.kind==='depth'?`<div class="injRow ${w.pts>0?'up':'q'}"><b>Depth chart</b> (${w.s}) ${pm(w.pts)} • ${w.pts>0?'first up at his position':'someone is now ahead of him'}</div>`:w.kind==='qb'?`<div class="injRow ${w.pts<0?'q':'up'}"><b>${w.n}</b> (QB, ${w.s}) ${pm(w.pts)} • passing game takes a hit</div>`:`<div class="injRow up"><b>${w.n}</b> (${w.pos}, ${w.s}) ${pm(w.pts)} • ${w.ev?`without him that work was worth ${w.ev.without} pts vs ${w.ev.with} with (${w.ev.games} game${w.ev.games>1?'s':''})`:'his work shifts to teammates'}</div>`)});
 if(!p.inj&&!o.length)return '';
 let pen=p.inj?(4*(1-p.inj.avail)).toFixed(1):0,head=p.inj?`<b>${p.inj.label}</b>${p.inj.note?' ('+p.inj.note+')':''} • ${p.out?'no Week '+wk+' Rating':'Rating -'+pen}`:`Teammate injuries shift his Week ${wk} Rating`,dot=`<span class="injDot ${p.out?'o':p.inj?'q':'up'}"></span>`;
 if(!o.length)return `<div class="injMini"><div class="injLine" style="display:flex;align-items:center;gap:8px;font-size:13px;padding:8px 12px;color:#33445f">${dot}<span>${head}</span></div></div>`;
 return `<details class="injMini"><summary>${dot}<span>${head}</span><span class="injMore">${o.length} teammate note${o.length>1?'s':''} ▾</span></summary><div class="injBody">${o.join('')}${META.injuries&&META.injuries.updated?'<div class="injUp">Injury report updated '+META.injuries.updated+'</div>':''}</div></details>`}
function outlook(p,start){
 if(p.played)return {t:'Played: '+p.played.ppr+' PPR',d:'',c:'g70',r:997};
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
Object.assign(BI,{"Dual Threat":"M12 21V11M12 11L6 4M12 11l6-7M6 4h4M6 4v4M18 4h-4M18 4v4","TD Machine":"M4 3v8a8 8 0 0 0 16 0V3M12 12v9M8 21h8","Big Arm":"M4 20C4 11 10 5 20 4M20 4l-5-1M20 4l-1 5","Pinpoint":"M12 2v5M12 17v5M2 12h5M17 12h5M9 12a3 3 0 1 0 6 0a3 3 0 1 0 -6 0","Ball Security":"M7 11V8a5 5 0 0 1 10 0v3M5 11h14v10H5z","Air Raid":"M3 12h14M13 6l6 6-6 6M3 6h6M3 18h6","Goal-Line Runner":"M4 20h16M7 20V9M17 20V9M7 9h10M12 9V4","Consistent":"M3 7h18M3 12h18M3 17h18"});
Object.assign(BC,{"Dual Threat":"#10b981","TD Machine":"#dc2626","Big Arm":"#2563eb","Pinpoint":"#0d9488","Ball Security":"#4f46e5","Air Raid":"#c026d3","Goal-Line Runner":"#b45309","Consistent":"#65a30d"});
const BKI={};Object.keys(BC).forEach((k,i)=>BKI[k]=i);BC['Diamond']='#45b8f2';
const B_OUT='M50 3 C62 9 78 13 94 13 L95 21 C95 62 82 92 50 116 C18 92 5 62 5 21 L6 13 C22 13 38 9 50 3 Z',B_RIV=[[30,13.8],[70,13.8],[12,19],[88,19],[9,40],[91,40],[14,66],[86,66],[26,88],[74,88],[50,7]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="1.8" fill="rgba(255,255,255,.7)"/>`).join('');
function bMix(h,t,a){let n=parseInt(h.slice(1),16),r=n>>16,g=n>>8&255,b=n&255,m=t?255:0;return '#'+[r,g,b].map(c=>Math.round(c+(m-c)*a).toString(16).padStart(2,'0')).join('')}
(function(){let d='<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><linearGradient id="bv" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".3"/><stop offset=".4" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".32"/></linearGradient>';
 Object.keys(BC).forEach(n=>{let k=BKI[n],m=BC[n],l=bMix(m,1,.5),dk=bMix(m,0,.55);d+=`<linearGradient id="bm${k}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${l}"/><stop offset=".45" stop-color="${m}"/><stop offset=".62" stop-color="${l}"/><stop offset="1" stop-color="${dk}"/></linearGradient><linearGradient id="bf${k}" x1="0" y1="0" x2="1" y2=".25"><stop offset="0" stop-color="${m}"/><stop offset=".2" stop-color="${l}"/><stop offset=".38" stop-color="${m}"/><stop offset=".72" stop-color="${dk}"/><stop offset="1" stop-color="${m}"/></linearGradient>`});
 [['G','#0f9a52','#3fe08b'],['Y','#e09a0a','#ffd04a'],['R','#d8283a','#ff6f7c']].forEach(([i,a,b])=>d+=`<linearGradient id="rg${i}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient>`);
 d+='</defs></svg>';document.body.insertAdjacentHTML('afterbegin',d)})();
const B_GEM='<g><polygon points="28,50 72,50 82,62 50,92 18,62" fill="#eaf9ff"/><polygon points="28,50 42,50 38,62 18,62" fill="#a3dcff"/><polygon points="42,50 58,50 62,62 38,62" fill="#fff"/><polygon points="58,50 72,50 82,62 62,62" fill="#6cc4f4"/><polygon points="18,62 38,62 50,92" fill="#4cabe9"/><polygon points="38,62 62,62 50,92" fill="#c4ebff"/><polygon points="62,62 82,62 50,92" fill="#2c86d2"/><path d="M28,50 L72,50 L82,62 L50,92 L18,62 Z" fill="none" stroke="#0b2a55" stroke-opacity=".5" stroke-width="1.2" stroke-linejoin="round"/></g><g fill="#fff"><path d="M80 30l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/></g>';
function bIcon(n){let dia=n==='Diamond',k=BKI[dia?'Diamond':(n in BKI?n:'Role Player')],ic=BI[n]||BI['Role Player'];
 return `<svg class="bsv${dia?' bdia':''}" viewBox="-4 0 108 122"><path d="${B_OUT}" fill="url(#bm${k})"/><path d="${B_OUT}" fill="none" stroke="rgba(0,0,0,.35)" stroke-width="1.4"/><g transform="translate(50 62) scale(.84) translate(-50 -62)"><path d="${B_OUT}" fill="url(#bf${k})"/><path d="${B_OUT}" fill="url(#bv)"/><path d="${B_OUT}" fill="none" stroke="rgba(0,0,0,.45)" stroke-width="1.8"/></g>${B_RIV}${dia?B_GEM:`<g transform="translate(27 40) scale(1.9)" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="${ic}" transform="translate(0 .7)" stroke="rgba(0,0,0,.5)" stroke-width="2.5"/><path d="${ic}" stroke="#fff" stroke-width="2.4"/></g>`}</svg>`}
function ringSvg(v){v=Number(v)||0;let id=v>=80?'G':v>=60?'Y':'R',s=Math.max(2,Math.min(99.4,v));return `<svg class="rsv ${id}" viewBox="0 0 100 100"><circle class="rt" cx="50" cy="50" r="46"/><circle class="rp" cx="50" cy="50" r="46" pathLength="100" stroke="url(#rg${id})" stroke-dasharray="${s} 100"/></svg>`}
function dd(p,l){return p&&p.dia?[{n:'Diamond',d:p.diaD}].concat(l):l}
function bIcons(l){return `<span class="bi">${l.map(b=>`<i title="${b.n}" data-bn="${b.n}" data-bd="${String(b.d||'').replace(/"/g,'&quot;')}">${bIcon(b.n)}</i>`).join('')}</span>`}
document.addEventListener('click',e=>{let pop=$('bPop'),i=e.target.closest('.bi i[data-bn]');if(i){e.stopPropagation();e.preventDefault();pop.innerHTML=`<div class="bpT"><span class="bpI">${bIcon(i.dataset.bn)}</span><b>${i.dataset.bn}</b><button type="button" id="bpX">\u2715</button></div><div class="bpD">${i.dataset.bd||''}</div>`;pop.hidden=false;return}if(!pop.hidden&&!e.target.closest('#bPop')){pop.hidden=true}else if(e.target.closest('#bpX')){pop.hidden=true}},true);
function badgeHTML(l){return `<div class="bdgs">${l.map(b=>`<button type="button" class="bdgBtn" data-n="${b.n}" data-d="${b.d.replace(/"/g,'&quot;')}"><i>${bIcon(b.n)}</i>${b.n}</button>`).join('')}</div><div class="bdgD" hidden></div>`}
document.addEventListener('click',e=>{let btn=e.target.closest&&e.target.closest('.bdgBtn');if(!btn)return;let box=btn.closest('.bdgs').parentElement,d=box.querySelector('.bdgD'),was=btn.classList.contains('on');box.querySelectorAll('.bdgBtn').forEach(x=>x.classList.remove('on'));if(was){d.hidden=true;d.innerHTML=''}else{btn.classList.add('on');d.hidden=false;d.innerHTML='<b>'+btn.dataset.n+'</b> '+btn.dataset.d}});
function olHTML(p,o,idx){idx=idx||p.logs.map((_,i)=>i);return `${p.played?'':`<span class="outlook ${o.c}">${o.t}</span>`}${badgeHTML(dd(p,badges(p,idx)))}`}
function qbBadges(q){if(q.qual===false||!q.m||q.mania==null)return [];
 let Q=qbBadges.Q||(qbBadges.Q=(META.qbs||[]).filter(x=>x.qual!==false&&x.m&&x.mania!=null)),m=q.m;
 let pc=(f,inv)=>{let v=f(q);if(!Number.isFinite(v))return 0;let a=Q.map(f).filter(Number.isFinite),lo=a.filter(x=>inv?x>v:x<v).length,eq=a.filter(x=>x===v).length;return (lo+eq/2)/Math.max(a.length,1)*100};
 let cv=x=>{let L=(x.logs||[]).map(l=>l.ppr);if(L.length<3)return NaN;let mu=L.reduce((s,y)=>s+y,0)/L.length,sd=Math.sqrt(L.reduce((s,y)=>s+(y-mu)*(y-mu),0)/L.length);return mu?sd/mu:NaN};
 let o=[],add=(n,d,s,ok)=>{if(ok&&s>=75)o.push({n,d,s})};
 add('Dual Threat','Hurts defenses with his legs as much as his arm. Among the best rushing yards per game of any QB.',pc(x=>x.m.ruy),m.ruy>=20);
 add('Goal-Line Runner','Scores with his legs near the goal line. Among the most rushing touchdowns per game of any QB.',pc(x=>x.m.rtd),m.rtd>=0.3);
 add('TD Machine','Throws touchdowns at one of the highest rates per game in the league.',pc(x=>x.m.ptd),m.ptd>=1.5);
 add('Big Arm','Gets more yards from every throw than almost every other QB.',pc(x=>x.m.ypa),m.ypa>=7.5);
 add('Pinpoint','Completes a higher share of his passes than almost every other QB.',pc(x=>x.m.cmp),m.cmp>=65);
 add('Ball Security','Rarely throws interceptions compared with other QBs.',pc(x=>x.m.int,true),(m.att||0)>=20);
 add('Air Raid','Throws a lot and piles up passing yards. A high-volume passing offense runs through him.',(pc(x=>x.m.att)+pc(x=>x.m.pyd))/2,(m.att||0)>=30);
 add('Consistent','His fantasy scores stay close to his average week to week, a safe floor.',pc(cv,true),Number.isFinite(cv(q)));
 add('Boom or Bust','His fantasy scores swing a lot from week to week.',pc(cv),Number.isFinite(cv(q))&&q.m.ppr>=12);
 if(!o.length)o.push({n:'Role Player',d:'A starter without one standout trait yet.',s:10});
 let n=q.mania>=90?3:q.mania>=82?2:1;return o.sort((x,y)=>y.s-x.s).slice(0,n)}
(META.qbs||[]).forEach(q=>{q.bl=qbBadges(q);q.bds=q.bl.map(b=>b.n).join(' \u2022 ');q.out=false;let r=q.start_rank;q.ol=q.played?'Played':r?'QB'+r+' this week':'\u2014';q.oc=r&&r<=8?'g90':r&&r<=16?'g70':'g0';q.orank=q.played?997:r||999;q.rank=q.pos_rank||999;q.start_pos_rank=r});DB.forEach(p=>{let o=outlook(p,p.start);p.ol=o.t;p.oc=o.c;p.orank=o.r;p.bl=badges(p,dfl(p));p.bds=p.bl.map(b=>b.n).join(' \u2022 ')});
['RB','WR','TE','QB'].forEach(pos=>{let l=allPlayers().filter(p=>p.pos===pos&&p.mania!=null&&p.qual!==false).sort((a,b)=>b.mania-a.mania);if(l[0]){let t=l[0];t.dia=true;t.diaD='The top-rated '+pos+' in the league by Mania Rating this season.';t.bl=[{n:'Diamond',d:t.diaD,s:999}].concat(t.bl||[]);t.bds='Diamond \u2022 '+(t.bds||'')}});
let selBadge=null,posSet=new Set(),openQB=new Set(),rankMode='mania',sortKey='mania',sortDir=-1,A=null,B=null,selA=null,selB=null;
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
function showView(v){document.body.classList.remove('shot');document.querySelectorAll('.view').forEach(x=>x.classList.toggle('on',x.id===v));document.querySelectorAll('.navb').forEach(x=>x.classList.toggle('on',x.dataset.v===v));window.scrollTo(0,0)}document.querySelectorAll('.navb').forEach(b=>b.onclick=()=>showView(b.dataset.v));$('brandHome').onclick=()=>showView('home');document.addEventListener('click',e=>{let t=e.target.closest('[data-open]');if(t&&t.dataset.open){let q=DB.find(x=>x.id===t.dataset.open);if(q)openPlayer(q)}});
function initials(p){return p.name.split(' ').slice(0,2).map(x=>x[0]).join('')} function pic(p){return p.headshot?`<img loading="lazy" src="${p.headshot}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><span class="initials" style="display:none">${initials(p)}</span>`:`<span class="initials">${initials(p)}</span>`}
function searchBox(inp,dd,cb,qbs){$(inp).oninput=e=>{let q=e.target.value.toLowerCase().trim(),d=$(dd);if(!q){d.style.display='none';return}let m=(qbs?allPlayers():DB).filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);d.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><div><span class="ddiName">${p.name}</span><span class="ddiMeta">${p.team} • ${p.pos}</span></div><span class="ddiRate">${p.mania==null?'—':p.mania}</span></div>`).join('');d.style.display=m.length?'block':'none';d.querySelectorAll('.ddi').forEach(x=>x.onclick=()=>{let p=byId(x.dataset.id);d.style.display='none';$(inp).value=p.name;cb(p)})}}
searchBox('homeQ','homeDD',openPlayer,1);searchBox('playerQ','playerDD',openPlayer,1);searchBox('aQ','aDD',p=>{A=p;selA=dfl(p);renderCompare()});searchBox('bQ','bDD',p=>{B=p;selB=dfl(p);renderCompare()});
const COLS_M=[['rank','#'],['name','Player'],['mania','Mania'],['start','Wk'],['m.ppr','PPR/G'],['m.tgt','TGT/G'],['m.rec','REC/G'],['m.recy','REC YD/G'],['m.car','CAR/G'],['m.rushy','RUSH YD/G'],['m.snap','SNAP%'],['share','SHARE'],['rz','RZ']];
const COLS_S=[['rank','#'],['name','Player'],['start','Wk'],['proj','Proj pts'],['opp','Opp'],['matchup_adj','Matchup boost'],['trend_adj','Role trend'],['outlook','Pos rank'],['mania','Mania'],['m.ppr','PPR/G'],['m.snap','SNAP%']];
const COLS_P=[['rank','#'],['name','Player'],['opp','Opp'],['projv','Proj pts'],['range','Range'],['line','Stat line']];
let viewMode='ov';
const COLS_OV=[['rank','#'],['name','Player'],['opp','Opp'],['mania','Mania'],['start','Wk'],['proj','Proj'],['pts','Score'],['outlook','Pos rank']];
const COLS_STK_Q=[['m.pyd','PASS YD/G'],['m.ptd','PASS TD/G'],['m.int','INT/G']];
const COLS_STK=[['rank','#'],['name','Player'],['mania','Mania'],['m.ppr','PPR/G'],['m.tgt','TGT/G'],['m.rec','REC/G'],['m.recy','REC YD/G'],['m.car','CAR/G'],['m.rushy','RUSH YD/G'],['m.snap','SNAP%'],['share','SHARE'],['rz','RZ'],['matchup_adj','Matchup'],['trend_adj','Role trend']];
const COLS_STQ=[['rank','#'],['name','Player'],['mania','Mania'],['m.ppr','PPR/G'],['m.att','ATT/G'],['m.cmp','CMP%'],['m.pyd','PASS YD/G'],['m.ptd','PASS TD/G'],['m.int','INT/G'],['m.ypa','YD/ATT'],['m.ruy','RUSH YD/G'],['m.rtd','RUSH TD/G']];
const COLS_BD=[['rank','#'],['name','Player'],['mania','Mania'],['bk0','Production'],['bk1','Volume'],['bk2','Role'],['bk3','Scoring spots'],['bk4','Efficiency'],['m.tshare','TGT SHARE'],['m.ashare','AIR SHARE'],['m.rzt','RZ TGT/G'],['m.ez','EZ TGT/G'],['m.touch','TOUCH SHARE'],['m.rzc','RZ CAR/G'],['m.gl','GL CAR/G']];
const COLS_BDQ=[['rank','#'],['name','Player'],['mania','Mania'],['bk0','Production'],['bk1','Passing'],['bk2','Rushing'],['bk3','Scoring'],['bk4','Efficiency']];
const PCTK=new Set(['snap','tshare','ashare','touch','cmp']);
function qbOnly(){return posSet.size===1&&posSet.has('QB')}
function COLS(){let q=qbOnly();return viewMode==='st'?(q?COLS_STQ:(!posSet.size||posSet.has('QB'))?COLS_STK.slice(0,-2).concat(COLS_STK_Q,COLS_STK.slice(-2)):COLS_STK):viewMode==='bd'?(q?COLS_BDQ:COLS_BD):COLS_OV}
function bk(p){if(!p._bk){let v=Object.values(p.buckets||{});p._bk=p.pos==='QB'?v.map(x=>Math.round(x)):v.map((x,i)=>Math.round(percentile(p.pos,BK[i],x)))}return p._bk}
const PMAP={ppr:'ppr_pg',tgt:'targets_pg',rec:'rec_pg',recy:'rec_yards_pg',car:'carries_pg',rushy:'rush_yards_pg',snap:'snap_pct',tshare:'target_share',ashare:'air_share',touch:'touch_share',rzt:'rz_targets_pg',ez:'endzone_targets_pg',rzc:'rz_carries_pg',gl:'gl_carries_pg'};
function bkCell(v,cls){return `<td class="bkc ${cls||''}"><div class="bkw"><div class="cbt"><div style="width:${Math.max(3,v)}%;background:${barColor(v)}"></div></div><b class="${barGrade(v)}">${v}</b></div></td>`}
function miniBar(t,pr){return `<td class="stc"><span>${t}</span><div class="cbt mbw"><div style="width:${Math.max(4,pr)}%;background:${barColor(pr)}"></div></div></td>`}
function avgRow(arr,CL){let cs=CL.map(([k])=>{if(k==='rank')return '<td></td>';if(k==='name')return '<td class="nmc"><b>Average</b><span class="tag">this list</span></td>';
  let vs;if(k.startsWith('bk'))vs=arr.map(p=>p.buckets?bk(p)[+k[2]]:null);else if(k==='share'||k==='rz')vs=arr.filter(p=>p.pos!=='QB').map(p=>val(p,k));else if(k==='matchup_adj'||k==='trend_adj')vs=arr.filter(p=>p.pos!=='QB'&&p.opp!=='BYE').map(p=>p[k]);else vs=arr.map(p=>val(p,k));
  vs=vs.filter(x=>x!=null&&Number.isFinite(Number(x))&&(k==='matchup_adj'||k==='trend_adj'||Number(x)>=0)).map(Number);if(!vs.length)return '<td class="muted">\u2014</td>';
  let a=vs.reduce((s,x)=>s+x,0)/vs.length;if(k.startsWith('bk'))return bkCell(Math.round(a),'avgc');
  let kk=k.startsWith('m.')?k.slice(2):k,t=(Math.round(a*10)/10)+((PCTK.has(kk)||k==='share')?'%':'');if(k==='mania'||k==='start'||k==='proj'||k==='pts')t=(Math.round(a*10)/10).toFixed(1);return `<td>${t}</td>`});
  return '<tr class="avgRow">'+cs.join('')+'</tr>'}
function genCell(p,k){if(k.startsWith('bk')){let v=p.buckets?bk(p)[+k[2]]:null;return v==null?'<td class="muted">—</td>':bkCell(v)}
 let t;if(k==='share')t=p.pos==='QB'?'—':val(p,'share')+'%';else if(k==='rz')t=p.pos==='QB'?'—':val(p,'rz');else if(k.startsWith('m.')){let kk=k.slice(2),mm=p.m||{},v=mm[kk];if(p.pos==='QB'&&kk==='rushy')v=mm.ruy;t=v==null?'—':v+(PCTK.has(kk)?'%':'');if(v!=null&&p.pos!=='QB'&&PMAP[kk]&&viewMode==='st'&&!((kk==='car'||kk==='rushy')&&p.pos!=='RB'))return miniBar(t,percentile(p.pos,PMAP[kk],Number(v)))}else t=val(p,k);if((k==='share'||k==='rz')&&p.pos!=='QB'&&viewMode==='st'){let pk=k==='share'?(p.pos==='RB'?'touch_share':'target_share'):(p.pos==='RB'?'rz_carries_pg':'rz_targets_pg');return miniBar(t,percentile(p.pos,pk,Number(val(p,k))))}return `<td>${t}</td>`}
function badgeOK(p){return !selBadge||(p.bl||[]).some(b=>b.n===selBadge)}
function updFlt(base){let cnt={all:base.length,todo:base.filter(p=>!p.played).length,done:base.filter(p=>p.played).length};document.querySelectorAll('.pst').forEach(b=>{b.classList.toggle('on',b.dataset.st===projStatus);b.textContent=({all:'All',todo:'Not played yet',done:'Played ✓'})[b.dataset.st]+' ('+cnt[b.dataset.st]+')'});let pl=allPlayers().filter(inPos),c={},ds={};pl.forEach(p=>(p.bl||[]).forEach(b=>{c[b.n]=(c[b.n]||0)+1;ds[b.n]=b.d}));let names=Object.keys(c).sort((x,y)=>c[y]-c[x]);if(selBadge&&!c[selBadge])selBadge=null;$('badgeSel').innerHTML='<option value="">All badges</option>'+names.map(n=>`<option value="${n}"${n===selBadge?' selected':''}>${n} (${c[n]})</option>`).join('');$('badgeNote').textContent=selBadge?(ds[selBadge]||''):'';let n=(projStatus!=='all'?1:0)+(selBadge?1:0);$('fltN').textContent=n?'('+n+')':'';$('fltBtn').classList.toggle('active',n>0)}
function allPlayers(){return DB.concat(META.qbs||[])}
function inPos(p){return !posSet.size||posSet.has(p.pos)}
function val(p,k){if(k==='start')return sval(p);if(k==='projv')return p.out?-1:p.proj?p.proj.pts:-2;if(k==='pts')return p.played?p.played.ppr:-1;if(k==='mania')return p.mania==null?-1:p.mania;if(k==='range')return p.proj?p.proj.hi:-1;if(k==='line')return 0;if(k==='proj')return p.proj&&!p.out?p.proj.pts:-1;if(k==='rank')return p.rank==null?999:p.rank;if(k==='name')return p.name;if(k==='outlook')return -p.orank;if(k.startsWith('bk'))return p.buckets?bk(p)[+k[2]]:-1;if(k==='share')return p.pos==='QB'?-1:p.pos==='RB'?p.m.touch:p.m.tshare;if(k==='rz')return p.pos==='QB'?-1:p.pos==='RB'?p.m.rzc:p.m.rzt;if(k.startsWith('m.')){let v=(p.m||{})[k.slice(2)];return v==null?-1:Number(v)}return p[k]==null?-1:p[k]}
let shareMode='snap';function renderTeamShare(){let teams=[...new Set(DB.map(p=>p.team))].sort(),pick=$('teamPick');if(!pick.options.length)pick.innerHTML=teams.map(t=>`<option>${t}</option>`).join('');let team=pick.value||teams[0];if(!pick.value)pick.value=team;$('shareWeek').textContent=META.week;let isSnap=shareMode==='snap',ps=DB.filter(p=>p.team===team),weeks=[...new Set(ps.flatMap(p=>p.logs.map(l=>l.w)))].sort((a,b)=>a-b);
 let seasonOf=p=>isSnap?p.m.snap:(p.pos==='RB'?p.m.tshare:p.m.tshare);
 $('teamShareBody').innerHTML=`<div class="shareNote">${isSnap?'Share of the offense snaps each player was on the field.':'Share of the team passes thrown to each player.'} Green or red numbers moved 5+ points from the week before.</div>`+['RB','WR','TE'].map(pos=>{let q=ps.filter(p=>p.pos===pos&&(p.m.snap>=8||p.m.tshare>=3));if(!q.length)return'';q.sort((a,b)=>seasonOf(b)-seasonOf(a));return `<div class="shareSection"><h2>${pos==='RB'?'Running backs':pos==='WR'?'Wide receivers':'Tight ends'}</h2><div class="tablewrap"><table class="shareTable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>Wk ${w}</th>`).join('')}<th>Season</th></tr></thead><tbody>${q.map(p=>{let prev=null,cells=weeks.map(w=>{let l=p.logs.find(x=>x.w===w);if(!l){return '<td>—</td>'}let v=isSnap?l.snap:l.tshare,cls=prev==null?'':(v-prev>=5?'up':v-prev<=-5?'down':'');prev=v;return `<td class="${cls}">${fmt(v)}%</td>`}).join('');return `<tr><td><span class="sharePlayer" data-id="${p.id}">${p.name}</span></td>${cells}<td class="season">${fmt(seasonOf(p))}%</td></tr>`}).join('')}</tbody></table></div></div>`}).join('');
 $('teamShareBody').querySelectorAll('.sharePlayer').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)))}

let heroTop=[],heroIdx=0;
function thumb(p,sz){let v=p.mania!=null?p.mania:50;return `<span class="thumb ${grade(v)}" style="--s:${v}">${pic(p)}</span>`}
function drawStack(){let n=heroTop.length,st=$('heroStack');if(!st||!n)return;
 if(!st.children.length){st.innerHTML=heroTop.map((p,i)=>`<div class="hc" data-i="${i}"><span class="hrk">#${i+1}</span><span class="hvn">${p.name} <b class="${grade(p.mania)}">${Number(p.mania).toFixed(1)}</b></span><div class="hpf"><div class="portrait hmini" style="--score:${p.mania};--ring:${ring(p.mania)}">${ringSvg(p.mania)}${pic(p)}</div></div><h3>${p.name}</h3><span class="htag">${p.team} • ${p.pos}</span><div class="hst"><div><b>${p.m&&p.m.ppr!=null?p.m.ppr:'—'}</b>PPR / game</div><div><b class="${grade(p.mania)}">${Number(p.mania).toFixed(1)}</b>Mania Rating</div></div><button type="button" class="hopen">Open profile</button></div>`).join('')}
 [...st.children].forEach((c,i)=>{let o=(i-heroIdx+n)%n;c.style.setProperty('--o',o);c.style.zIndex=20-o;c.dataset.front=o===0?1:0;c.dataset.o=o});
 let dt=$('heroDots');dt.innerHTML=heroTop.map((p,i)=>`<i class="${i===heroIdx?'on':''}" data-i="${i}"></i>`).join('');dt.querySelectorAll('i').forEach(x=>x.onclick=e=>{e.stopPropagation();heroIdx=+x.dataset.i;drawStack()})}
function heroStack(){heroTop=allPlayers().filter(p=>p.mania!=null&&p.qual!==false&&p.m).sort((a,b)=>b.mania-a.mania).slice(0,5);heroIdx=0;if($('heroStack'))$('heroStack').innerHTML='';drawStack();let st=$('heroStack'),n=heroTop.length;
 let hint=document.querySelector('#heroStage .stap');if(hint)hint.textContent=(window.matchMedia&&matchMedia('(pointer:coarse)').matches)?'Swipe to shuffle':'Tap to shuffle';
 st.onclick=e=>{let c=e.target.closest('.hc');if(!c)return;let i=+c.dataset.i;if(e.target.closest('.hopen')&&c.dataset.front==='1'){openPlayer(heroTop[i]);return}if(c.dataset.front==='1')heroIdx=(heroIdx+1)%heroTop.length;else heroIdx=i;drawStack()};
 if(st._w)return;st._w=1;let x0=null,dx=0,moved=false,front=()=>st.querySelector('[data-front="1"]');
 st.addEventListener('touchstart',e=>{x0=e.touches[0].clientX;dx=0;moved=false;let f=front();if(f)f.classList.add('drag')},{passive:true});
 st.addEventListener('touchmove',e=>{if(x0==null)return;dx=e.touches[0].clientX-x0;if(Math.abs(dx)>8)moved=true;let f=front();if(f)f.style.setProperty('--dx',(dx*.9)+'px')},{passive:true});
 let end=()=>{if(x0==null)return;let f=front();if(f){f.classList.remove('drag');f.style.setProperty('--dx','0px')}if(Math.abs(dx)>50){let k=heroTop.length;heroIdx=(heroIdx+(dx<0?1:k-1))%k;drawStack()}x0=null;if(moved){st.dataset.sw=1;setTimeout(()=>delete st.dataset.sw,350)}};
 st.addEventListener('touchend',end);st.addEventListener('touchcancel',end);
 st.addEventListener('click',e=>{if(st.dataset.sw){e.stopPropagation();e.preventDefault()}},true)}
function renderHome(){if($('homeWeek'))$('homeWeek').textContent=META.week;document.querySelectorAll('.nextWeekText').forEach(x=>x.textContent=META.next_week);
 let pl=allPlayers().filter(p=>p.played).length,plus=pl?' (plus Week '+META.next_week+' games as they finish)':'';if($('homeWkPlus'))$('homeWkPlus').textContent=plus;if($('wkPlus'))$('wkPlus').textContent=plus;
 if($('posBest')){$('posBest').innerHTML=['QB','RB','WR','TE'].map(pos=>{let p=allPlayers().find(x=>x.pos===pos&&x.dia);if(!p)return '';return `<div class="pbc" data-id="${p.id}"><span class="pbBadge">${bIcon('Diamond')}</span>${thumb(p)}<div class="pbPos">Best ${pos}</div><div class="pbName">${p.name}</div><div class="pbTag">${p.team}</div><div class="pbScore ${grade(p.mania)}">${Number(p.mania).toFixed(1)}</div><div class="pbLab">Mania Rating</div></div>`}).join('');$('posBest').querySelectorAll('.pbc').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)))}
 $('homeDisc').textContent=DISC+(META.built?' Site last refreshed '+META.built+'.':'');
 [['homeRanks',()=>showView('rankings')],['homeCompare',()=>showView('compare')],['homeTrade',()=>openTrade('quick')],['homeShare',()=>{showView('insights');showInsight('share')}],['homePlayers',()=>showView('players')],['homeInsights',()=>showView('insights')]].forEach(([i,f])=>{let e=$(i);if(e&&!e.onclick)e.onclick=f});
 heroStack()}
function sg(v){v=Number(v);return (v>0?'+':'')+v.toFixed(1)}
function lineOf(p){if(p.played){let q=p.played;return p.pos==='QB'?`${q.pyd} yd, ${q.ptd} TD, ${q.int} INT`:`${q.rec} rec, ${q.ry+q.ruy} yd, ${q.td} TD`}let q=p.proj;if(!q||p.out)return '\u2014';return p.pos==='QB'?`${q.pyd} yd, ${q.ptd.toFixed(1)} TD, ${q.int.toFixed(1)} INT`:`${q.rec.toFixed(1)} rec, ${q.ryd+q.ruy} yd, ${q.td.toFixed(1)} TD`}
let projStatus='all';
function ptsCell(p){if(!p.played)return '<span class="muted">\u2014</span>';let a=p.played.ppr,d=p.proj?a-p.proj.pts:null;return `<b class="${d==null?'':d>=2?'good':d<=-2?'bad':''}">${a.toFixed(1)}</b>${d==null?'':`<small class="dlt">${d>=0?'+':''}${d.toFixed(1)}</small>`}`}
function renderProjTab(){let CL=[['rank','#'],['name','Player'],['mania','Mania'],['start','Wk '+META.next_week],['projv','Proj'],['pts','Score'],['range','Range']];if(!CL.some(c=>c[0]===sortKey))sortKey='projv';let pool=allPlayers().filter(p=>inPos(p)&&badgeOK(p)&&(p.opp!=='BYE'||p.played));updFlt(pool);
 let arr=pool.filter(p=>projStatus==='all'||(projStatus==='done')===!!p.played);arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);return typeof x==='string'?sortDir*x.localeCompare(y):sortDir*(x-y)});
 $('rankTitle').textContent=`Week ${META.next_week} Projections`;$('rankSub').textContent='Projected PPR points before the game, and what each player scored once it is played. Tap a heading to sort.';$('rankNote').innerHTML='<b>✓</b> = already played. <b>Range</b> = the low–high score that about 8 in 10 games land inside (desktop).';$('rankDisc').textContent='PPR scoring only. QBs: 4 per passing TD, 1 per 25 yards, -2 per INT. No kickers or defenses. Tap a badge to see what it means.';
 $('rankHead').innerHTML='<tr>'+CL.map(([k,l])=>`<th data-k="${k}" class="${k==='range'?'dOnly':''}">${l}${sortKey===k?(sortDir<0?' ↓':' ↑'):''}</th>`).join('')+'</tr>';
 let rows=[];arr.forEach((p,i)=>{let oppT=p.played?`vs ${p.opp} • ✓ ${resTxt(p.played)}`:`vs ${p.opp}`,sub=`${p.team} ${p.pos} • ${oppT}${p.sub?' • backup':''}`,pr=p.out?'OUT':p.proj?p.proj.pts.toFixed(1):'—',mn=p.mania==null?'<td class="muted">—</td>':`<td class="grade ${grade(p.mania)}">${p.mania}</td>`,wk=p.played&&p.played.grade!=null?`<td class="grade ${grade(p.played.grade)}">${Number(p.played.grade).toFixed(1)}<span class="pdone">✓</span></td>`:p.start==null||p.out?`<td class="muted">${p.out?'OUT':'—'}</td>`:`<td class="grade ${grade(p.start)}">${p.start}</td>`;
  rows.push(`<tr class="pjr"><td class="rn">${i+1}</td><td class="pjp"><div class="nmw">${thumb(p)}<div class="nmt2"><span class="playerlink" data-id="${p.id}">${p.name}</span>${injChip(p)}<div class="pjs">${sub}${bIcons(p.bl||[])}</div></div></div></td>${mn}${wk}<td class="pjv">${pr}</td><td class="pjv">${ptsCell(p)}</td><td class="dOnly muted">${p.proj&&!p.out?p.proj.lo.toFixed(0)+'–'+p.proj.hi.toFixed(0):'—'}</td></tr>`)});
 $('rankBody').innerHTML=rows.join('')||`<tr><td colspan="${CL.length}" class="muted">No players for this filter.</td></tr>`;
 let nw=META.next_week,ar=k=>sortKey===k?(sortDir<0?' ↓':' ↑'):'';
 $('rankCards').innerHTML=`<div class="rcHead"><span>#</span><span>Player</span>${[['start','Wk '+nw],['projv','Proj'],['pts','Score']].map(([k,l])=>`<span class="rch" data-k="${k}">${l}${ar(k)}</span>`).join('')}</div>`+arr.map((p,i)=>{let d=p.played&&p.proj?p.played.ppr-p.proj.pts:null,wv=p.played&&p.played.grade!=null?Number(p.played.grade).toFixed(1):p.out?'OUT':p.start==null?'—':p.start,wg=p.played&&p.played.grade!=null?p.played.grade:p.start;return `<div class="rcard" data-id="${p.id}"><div class="rn">${i+1}</div><div class="rmain">${thumb(p)}<div class="rtxt"><div class="rname">${p.name}${injChip(p)}</div><div class="rsub2">${p.team} ${p.pos} • ${p.played?'vs '+p.opp+' ✓ '+resTxt(p.played):'vs '+p.opp}${bIcons(p.bl||[])}</div></div></div><div class="rs ${wg==null?'':grade(wg)}">${wv}</div><div class="rs sec">${p.out?'OUT':p.proj?p.proj.pts.toFixed(1):'—'}</div><div class="rs sec ${d==null?'':d>=2?'good':d<=-2?'bad':''}">${p.played?p.played.ppr:'—'}${d!=null?`<small>${d>=0?'+':''}${d.toFixed(1)}</small>`:''}</div></div>`}).join('');
 $('rankCards').querySelectorAll('.rch').forEach(x=>x.onclick=()=>{let k=x.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1}renderRanks()});$('rankCards').querySelectorAll('.rcard').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));
 $('rankHead').querySelectorAll('th').forEach(th=>th.onclick=()=>{let k=th.dataset.k;if(k==='rank')return;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==='name'?1:-1}renderRanks()});
 $('rankBody').querySelectorAll('.playerlink[data-id]').forEach(x=>x.onclick=e=>{e.stopPropagation();openPlayer(byId(x.dataset.id))})}

function xpHTML(p){let nm=Object.keys(p.buckets||{}),v=bk(p),lg=(p.logs||[]).map(l=>l.ppr),mx=Math.max(1,...lg),pts=lg.length>1?lg.map((y,i)=>`${(i/(lg.length-1)*158+1).toFixed(1)},${(46-y/mx*42).toFixed(1)}`).join(' '):'';
 return `<div class="xpIn"><div class="xpBars">${nm.map((k,i)=>`<div class="barrow"><span>${k}</span><div class="track"><div class="fill" style="width:${Math.max(3,v[i])}%;background:${barColor(v[i])}"></div></div><b class="pctNum ${barGrade(v[i])}" style="--s:${v[i]}">${v[i]}</b></div>`).join('')}</div><div class="xpSide">${pts?`<div class="xpT">PPR by week</div><svg viewBox="0 0 160 50" class="spk"><polyline points="${pts}" fill="none" stroke="${barColor(p.mania)}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`:''}<button type="button" class="xpOpen" data-id="${p.id}">Open full profile</button></div></div>`}
function wireXp(root){root.querySelectorAll('.xpOpen').forEach(b=>b.onclick=e=>{e.stopPropagation();openPlayer(byId(b.dataset.id))})}
function renderRanks(){if(rankMode==='proj')return renderProjTab();let CL=COLS(),nw=META.next_week,ov=viewMode==='ov';if(!CL.some(c=>c[0]===sortKey))sortKey='mania';$('rankings').classList.toggle('wideView',!ov);
 let base=allPlayers().filter(p=>(p.pos!=='QB'||p.mania!=null)&&inPos(p)&&badgeOK(p));updFlt(base);let arr=base.filter(p=>projStatus==='all'||(projStatus==='done')===!!p.played);arr.sort((a,b)=>{let x=val(a,sortKey),y=val(b,sortKey);return typeof x==='string'?sortDir*x.localeCompare(y):sortDir*(x-y)});
 $('rankTitle').textContent='Player Ratings';$('rankSub').textContent=ov?`Mania Rating is the season-long grade. Wk ${nw} is this week's projection put on the same scale, so it always lines up with Proj and Score. Tap any heading to sort.`:viewMode==='st'?'Per-game numbers behind the ratings. Tap a heading to sort, scroll sideways for more.':'Where each Mania grade comes from, 0–100 against players at the same position (100 is best), plus the raw usage numbers. Tap a heading to sort, scroll sideways for more.';
 let mixQB=viewMode==='bd'&&!qbOnly()&&arr.some(p=>p.pos==='QB');$('rankNote').innerHTML='<b>✓</b> = already played. Proj is the pre-game projection; Score is what he actually got.'+(mixQB?' For QBs: Volume = passing volume, Role = rushing, Scoring = touchdowns.':'');$('rankDisc').textContent=DISC+' Tap a badge to see what it means.';
 $('rankHead').innerHTML='<tr>'+CL.map(([k,l])=>`<th data-k="${k}">${k==='start'?'Wk '+nw:l}${sortKey===k?(sortDir<0?' ↓':' ↑'):''}</th>`).join('')+'</tr>';
 let tg=p=>p.played?'<span class="tag playedTag">PLAYED</span>':p.opp==='BYE'?'<span class="tag">BYE</span>':'';
 let cell={rank:(p,i)=>`<td>${i+1}</td>`,name:p=>`<td class="nmc"><div class="nmw">${thumb(p)}<div class="nmt"><span class="playerlink rname2" data-id="${p.id}">${p.name}</span><span class="tag">${p.team} ${p.pos}</span>${ov?tg(p):''}${injChip(p)}${bIcons(p.bl)}</div></div></td>`,mania:p=>`<td class="grade ${grade(p.mania)}">${p.mania}</td>`,start:p=>p.played&&p.played.grade!=null?`<td class="grade ${grade(p.played.grade)}">${Number(p.played.grade).toFixed(1)}<span class="pdone">✓</span></td>`:`<td class="${p.start==null?'muted':'grade '+grade(p.start)}">${p.out?'OUT':p.start==null?'—':p.start}</td>`,proj:p=>`<td class="projCell">${p.out?'OUT':!p.proj?'—':p.proj.pts.toFixed(1)}</td>`,pts:p=>`<td class="projCell">${ptsCell(p)}</td>`,opp:p=>`<td>${p.opp}${p.played?` <span class="muted">✓ ${resTxt(p.played)}</span>`:''}</td>`,matchup_adj:p=>p.pos==='QB'?'<td class="muted">—</td>':`<td class="${p.matchup_adj>0.4?'good':p.matchup_adj<-0.4?'bad':'muted'}">${p.opp==='BYE'?'—':sg(p.matchup_adj)}</td>`,trend_adj:p=>p.pos==='QB'?'<td class="muted">—</td>':`<td class="${p.trend_adj>0.4?'good':p.trend_adj<-0.4?'bad':'muted'}">${sg(p.trend_adj)}</td>`,outlook:p=>`<td class="olCell"><span class="olTag ${p.played?'g90':p.oc}">${p.played?p.pos+(p.pos_rank||''):p.ol.replace(' this week','')}</span></td>`};
 $('rankBody').innerHTML=arr.map((p,i)=>'<tr>'+CL.map(([k])=>cell[k]?cell[k](p,i):genCell(p,k)).join('')+'</tr>').join('')||`<tr><td colspan="${CL.length}" class="muted">No players for this filter.</td></tr>`;
 let sub2=p=>`${p.team} ${p.pos}${p.pos_rank?p.pos_rank:''} • ${p.played?`vs ${p.opp} ✓ ${resTxt(p.played)}`:p.opp==='BYE'?'Bye':`vs ${p.opp}`}`;
 $('rankCards').innerHTML=ov?`<div class="rcHead"><span>#</span><span>Player</span>${[['mania','Mania'],['start','Wk '+nw],['proj','Proj / Score']].map(([k,l])=>`<span class="rch" data-k="${k}">${l}${sortKey===k?(sortDir<0?' ↓':' ↑'):''}</span>`).join('')}</div>`+arr.map((p,i)=>`<div class="rcard" data-id="${p.id}"><div class="rn">${i+1}</div><div class="rmain">${thumb(p)}<div class="rtxt"><div class="rname">${p.name}${injChip(p)}</div><div class="rsub2">${sub2(p)}${bIcons(p.bl)}</div></div></div><div class="rs ${grade(p.mania)}">${p.mania}</div><div class="rs sec">${sv(p)}</div><div class="rs sec">${p.out?'OUT':p.proj?p.proj.pts.toFixed(1):'—'}${p.played?`<small class="act ${p.proj&&p.played.ppr-p.proj.pts>=2?'good':p.proj&&p.played.ppr-p.proj.pts<=-2?'bad':''}">${p.played.ppr.toFixed(1)}✓</small>`:''}</div></div>`).join(''):'';
 let srt=k=>{if(k==='rank'){sortKey='mania';sortDir=-1;return renderRanks()}if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=k==='name'||k==='opp'?1:-1}renderRanks()};
 $('rankHead').querySelectorAll('th').forEach(th=>th.onclick=()=>srt(th.dataset.k));$('rankCards').querySelectorAll('.rch').forEach(x=>x.onclick=()=>srt(x.dataset.k));$('rankBody').querySelectorAll('.playerlink').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));$('rankCards').querySelectorAll('.rcard').forEach(x=>x.onclick=e=>{if(e.target.closest('.bi'))return;let nx=x.nextElementSibling;if(nx&&nx.classList.contains('rxp')){nx.remove();x.classList.remove('open');return}x.classList.add('open');x.insertAdjacentHTML('afterend','<div class="rxp">'+xpHTML(byId(x.dataset.id))+'</div>');wireXp(x.nextElementSibling)});$('rankCards').querySelectorAll('.rname').forEach(x=>x.onclick=e=>{e.stopPropagation();openPlayer(byId(x.closest('.rcard').dataset.id))});$('rankBody').querySelectorAll('tr').forEach(tr=>{if(!ov)return;let a=tr.querySelector('.rname2');if(!a)return;tr.onclick=e=>{if(e.target.closest('.playerlink,.bi'))return;let nx=tr.nextElementSibling;if(nx&&nx.classList.contains('xr')){nx.remove();tr.classList.remove('open');return}tr.classList.add('open');tr.insertAdjacentHTML('afterend','<tr class="xr"><td colspan="'+CL.length+'">'+xpHTML(byId(a.dataset.id))+'</td></tr>');wireXp(tr.nextElementSibling)}})}
document.querySelectorAll('.rankmode').forEach(b=>b.onclick=()=>{rankMode=b.dataset.mode;sortKey=rankMode==='proj'?'projv':'mania';sortDir=-1;document.querySelectorAll('.rankmode').forEach(x=>x.classList.toggle('on',x===b));$('viewSeg').hidden=rankMode==='proj';$('rankings').classList.toggle('projMode',rankMode==='proj');if(rankMode==='proj')$('rankings').classList.remove('wideView');$('rankTable').classList.toggle('pjt',rankMode==='proj');renderRanks()});
document.querySelectorAll('.vbtn').forEach(b=>b.onclick=()=>{viewMode=b.dataset.view;document.querySelectorAll('.vbtn').forEach(x=>x.classList.toggle('on',x===b));sortKey='mania';sortDir=-1;renderRanks()});
function syncPos(){document.querySelectorAll('.posf').forEach(x=>x.classList.toggle('on',x.dataset.pos==='ALL'?!posSet.size:posSet.has(x.dataset.pos)))}
document.querySelectorAll('.pst').forEach(b=>b.onclick=()=>{projStatus=b.dataset.st;if(rankMode==='proj'){sortKey=projStatus==='done'?'pts':'projv';sortDir=-1}renderRanks()});
document.querySelectorAll('.posf').forEach(b=>b.onclick=()=>{let v=b.dataset.pos;if(v==='ALL')posSet.clear();else{posSet.has(v)?posSet.delete(v):posSet.add(v);if(['RB','WR','TE','QB'].every(x=>posSet.has(x)))posSet.clear()}syncPos();renderRanks()});
$('fltBtn').onclick=()=>{$('fltPanel').hidden=!$('fltPanel').hidden};$('badgeSel').onchange=e=>{selBadge=e.target.value||null;renderRanks()};$('fltReset').onclick=()=>{projStatus='all';selBadge=null;renderRanks()};
renderRanks();
function percentile(pos,key,v){let a=(REFS[pos]||{})[key]||[];if(!a.length)return 0;let lo=0,hi=a.length;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<=v+1e-5)lo=m+1;else hi=m}return Math.min(100,lo/a.length*100)}function avg(a,k,w=true){let den=0,num=0;a.forEach(x=>{let z=w?(x.weight||1):1;num+=Number(x[k]||0)*z;den+=z});return den?num/den:0}function sum(a,k){return a.reduce((s,x)=>s+Number(x[k]||0)*(x.weight||1),0)}
function customMatch(p,M){let ex=p.similar||[];if(!ex.length)return {adj:p.matchup_adj,list:[]};let scored=ex.map(x=>{let share=p.pos==='RB'?M.touch_share:M.target_share,rz=p.pos==='RB'?M.rz_carries_pg:M.rz_targets_pg;let dif=[Math.abs(M.targets_pg-x.tgt)/Math.max(3,x.tgt,1),Math.abs(M.rec_pg-x.rec)/Math.max(2,x.rec,1),Math.abs(M.snap_pct-x.snap)/35,Math.abs(share-x.share)/20,Math.abs(rz-x.rz)/2];let sim=Math.max(.15,Math.exp(-.8*dif.reduce((a,b)=>a+b,0)/dif.length));return {...x,csim:sim,delta:(x.actual/Math.max(x.normal,3)-1)*100}}).sort((a,b)=>b.csim-a.csim);let use=scored.filter(x=>x.csim>=.28),den=use.reduce((s,x)=>s+x.csim*x.csim,0);if(!den)return {adj:0,list:scored.slice(0,4)};let eff=use.reduce((s,x)=>s+x.delta*x.csim*x.csim,0)/den,maturity=Math.min(1,META.week/8),evidence=Math.min(1,use.length/6),adj=Math.max(-6,Math.min(6,eff/25*6))*maturity*(.45+.55*evidence);return {adj:p.pos==='TE'?adj*.35:adj,list:scored.slice(0,4)}}
function dfl(p){let a=p.logs.map((l,i)=>l.auto?-1:i).filter(i=>i>=0);return a.length?a:p.logs.map((_,i)=>i)}
function sameSet(a,b){return a.length===b.length&&a.every(x=>b.includes(x))}
function calc(p,idxs){let L=idxs.map(i=>p.logs[i]).filter(Boolean);if(!L.length)return null;let M={};M.ppr_pg=avg(L,'ppr');M.targets_pg=avg(L,'tgt');M.rec_pg=avg(L,'rec');M.rec_yards_pg=avg(L,'ry');M.carries_pg=avg(L,'car');M.rush_yards_pg=avg(L,'ruy');M.scrim_pg=M.rec_yards_pg+M.rush_yards_pg;M.td_pg=avg(L,'rtd')+avg(L,'rutd');M.snap_pct=avg(L,'snap');M.target_share=avg(L,'tshare');M.air_share=avg(L,'ashare');M.touch_share=avg(L,'touch');M.rz_targets_pg=avg(L,'rzt');M.endzone_targets_pg=avg(L,'ez');M.rz_carries_pg=avg(L,'rzc');M.gl_carries_pg=avg(L,'gl');M.yards_per_target=sum(L,'tgt')?sum(L,'ry')/sum(L,'tgt'):0;M.catch_rate=sum(L,'tgt')?sum(L,'rec')/sum(L,'tgt')*100:0;M.yards_per_carry=sum(L,'car')?sum(L,'ruy')/sum(L,'car'):0;let opp=sum(L,'car')+2*sum(L,'tgt');M.fp_per_weighted_opp=opp?sum(L,'ppr')/opp:0;let P={};Object.keys(M).forEach(k=>P[k]=percentile(p.pos,k,M[k]));let prod,op,role,hv,eff;if(p.pos==='RB'){prod=.55*P.ppr_pg+.30*P.scrim_pg+.15*P.td_pg;op=.45*P.carries_pg+.35*P.targets_pg+.20*P.rec_pg;role=.60*P.touch_share+.40*P.snap_pct;hv=.40*P.rz_carries_pg+.35*P.gl_carries_pg+.25*P.rz_targets_pg;eff=.55*P.fp_per_weighted_opp+.25*P.yards_per_carry+.20*P.catch_rate}else{prod=.55*P.ppr_pg+.30*P.rec_yards_pg+.15*P.td_pg;op=.50*P.targets_pg+.30*P.rec_pg+.20*P.target_share;role=.45*P.target_share+.30*P.air_share+.25*P.snap_pct;hv=.60*P.rz_targets_pg+.40*P.endzone_targets_pg;eff=.45*P.yards_per_target+.30*P.catch_rate+.25*P.fp_per_weighted_opp}let W=p.pos==='RB'?[.30,.30,.20,.125,.075]:[.30,.30,.225,.10,.075],raw=W[0]*prod+W[1]*op+W[2]*role+W[3]*hv+W[4]*eff,base0=45+.52*raw,base=(p.pos==='TE'&&base0<93.6)?Math.max(35,93.6-(93.6-base0)*1.5):base0,eg=L.reduce((s,x)=>s+(x.auto?1:(x.weight==null?1:x.weight)),0),conf=Math.min(1,eg/Math.max(META.week,1)),sh=(1-conf)*.12,mania=Math.max(0,Math.min(99.5,base*(1-sh)+72*sh));let recent=L.slice(-Math.min(2,L.length)),recentOpp=recent.reduce((s,x)=>s+2*x.tgt+x.car,0)/recent.length,seasonOpp=M.targets_pg*2+M.carries_pg,tr=seasonOpp?((recentOpp/seasonOpp)-1)*100:0,trend=Math.max(-2.5,Math.min(2.5,tr/20*2.5));if(L.length<2)trend*=.25;let mt=customMatch(p,M),start=Math.max(0,Math.min(100,mania+mt.adj+trend+(p.inj_adj||0)));if(p.out||p.played)start=0;if(sameSet(idxs,dfl(p))){mania=p.mania;start=p.start}return {mania,start,M,match:mt,b:p.pos==='RB'?{'Fantasy Production':prod,'Touch Volume':op,'Backfield Control':role,'Goal-Line / Red-Zone':hv,'Per-Touch Efficiency':eff}:{'Fantasy Production':prod,'Target Volume':op,'Team Target Role':role,'Red-Zone Threat':hv,'Per-Target Efficiency':eff}}}
function barColor(v){return v>=70?'#22a45d':v>=40?'#e3a21a':'#e2505b'}function barGrade(v){return v>=70?'g90':v>=40?'g70':'g0'}function stat(v,k,pos,key,raw){let pc=key?Math.max(1,Math.min(99,Math.round(percentile(pos,key,raw)))):null;return `<div class="stat"><div class="sv">${v}</div><div class="sk">${k}</div>${pc!==null?`<div class="spc ${barGrade(pc)} lnk" data-board="stat" data-key="${key}" data-title="${k}" data-pos="${pos}" title="See everyone ranked">${ord(pc)} percentile</div>`:''}</div>`}
var roomMode='tch';
function roomVal(p,z,mode){if(p.pos!=='RB')return [z.tgt,z.ry||0];if(mode==='car')return [z.car,z.ruy||0];if(mode==='tgt')return [z.tgt,z.ry||0];return [z.car+z.tgt,(z.ruy||0)+(z.ry||0)]}
function roomTable(p){let rb=p.pos==='RB',mode=rb?roomMode:'tgt',unit={tch:'touch',car:'carry',tgt:'target'}[mode],showShare=mode==='tch'||!rb,weeks=[...new Set(p.room.flatMap(x=>x.weekly.map(w=>w.w)))].sort((a,b)=>a-b);
 return `<div class="tablewrap"><table class="roomtable"><thead><tr><th>Player</th>${weeks.map(w=>`<th>W${w}</th>`).join('')}<th>Per game</th></tr></thead><tbody>${p.room.map(x=>{let tc=0,ty=0,g=x.weekly.length;x.weekly.forEach(z=>{let v=roomVal(p,z,mode);tc+=v[0];ty+=v[1]});return `<tr><td class="${x.id===p.id?'you':''}" data-id="${x.id}">${x.name}</td>${weeks.map(w=>{let z=x.weekly.find(q=>q.w===w);if(!z)return '<td>—</td>';let v=roomVal(p,z,mode);return `<td class="rmc"><b>${v[0]}</b><small>${v[1]} yds${showShare?' • '+z.share+'%':''}</small></td>`}).join('')}<td class="rmc rmS">${g?`<b>${(tc/g).toFixed(1)}</b><small>${Math.round(ty/g)} yds a game${tc?' • '+(ty/tc).toFixed(1)+' per '+unit:''}${showShare?' • '+x.share+'%':''}</small>`:'—'}</td></tr>`}).join('')}</tbody></table></div>`}
function roomHTML(p){let rb=p.pos==='RB';return `<div class="roomCtl">${rb?`<div class="roomNote"><b>Touches</b> = carries + targets. Switch to look at just carries or just targets.</div><div class="rmSeg">${[['tch','Touches'],['car','Carries only'],['tgt','Targets only']].map(([k,l])=>`<button type="button" class="rmb${roomMode===k?' on':''}" data-rm="${k}">${l}</button>`).join('')}</div>`:`<div class="roomNote">Targets and receiving yards each week, with his share of the team targets.</div>`}</div><div id="roomBody">${roomTable(p)}</div><div class="explain">${rb?'Each cell is the count that week with his yards on those '+({tch:'touches',car:'carries',tgt:'targets'}[roomMode])+(roomMode==='tch'?' and his share of the team touches':'')+'.':'Each cell is targets that week, with receiving yards and share of team targets.'} Click a name to open that player.</div>`}
function wireRoom(p){document.querySelectorAll('.rmb').forEach(b=>b.onclick=()=>{roomMode=b.dataset.rm;document.querySelectorAll('.rmb').forEach(x=>x.classList.toggle('on',x===b));$('roomBody').innerHTML=roomTable(p);document.querySelectorAll('#roomBody .roomtable td[data-id]').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));let e=$('roomBody').nextElementSibling;if(e)e.innerHTML='Each cell is the count that week with his yards on those '+({tch:'touches',car:'carries',tgt:'targets'}[roomMode])+(roomMode==='tch'?' and his share of the team touches':'')+'. Click a name to open that player.'})}
function matchupHTML(p,c){let m=c?c.match:{adj:p.matchup_adj,list:p.similar.slice(0,4)},wk=META.next_week,L=m.list||[],n=L.length,avgN=n?L.reduce((s,x)=>s+Number(x.normal),0)/n:0,avgA=n?L.reduce((s,x)=>s+Number(x.actual),0)/n:0,adj=m.adj||0,sg=adj>=0?'+':'';
 if(p.opp==='BYE')return `<div class="muHd"><div><div class="muK">Week ${wk} matchup</div><div class="muOpp">Bye week</div></div></div><div class="matchWhy">No game this week, so there is no matchup to adjust for.</div>`;
 if(p.opp==='TBD'||!p.opp)return `<div class="muHd"><div><div class="muK">Week ${wk} matchup</div><div class="muOpp">To be decided</div></div></div><div class="matchWhy">His opponent is not set yet.</div>`;
 let rows=defenseVs(p.pos).slice().sort((a,b)=>b.pg-a.pg),ix=rows.findIndex(x=>x.team===p.opp),d=ix>=0?rows[ix]:null,
 v=adj>=1?['Good matchup','good']:adj<=-1?['Tough matchup','bad']:['Neutral matchup','mid'],diff=avgN?(avgA/avgN-1)*100:0,
 why=!n?`Not enough games against ${p.opp} yet, so the boost stays close to 0 for now.`:`Players with a role like his scored <b>${fmt(avgA)}</b> PPR a game against ${p.opp}, versus <b>${fmt(avgN)}</b> in their usual games (${diff>=0?'+':''}${Math.round(diff)}%). That moves his Week ${wk} Rating by <b>${sg}${fmt(adj)}</b>.${v[1]==='mid'?' It is a small sample this early, so it counts for little.':''}`;
 return `<div class="muHd"><div><div class="muK">Week ${wk} matchup</div><div class="muOpp">vs ${p.opp}</div></div><div class="muBoost ${v[1]}"><b>${sg}${fmt(adj)}</b><span>Rating boost</span></div></div><span class="muVerdict ${v[1]}">${v[0]}</span><div class="muFacts"><div><b>${d?fmt(d.pg):'—'}</b><span>PPR a game ${p.opp} allows to ${p.pos}s</span><em class="${d?(ix<8?'good':ix>=rows.length-8?'bad':'muted'):'muted'}">${d?ord(ix+1)+' most of '+rows.length:'No games yet'}</em></div><div><b>${n?fmt(avgN):'—'}</b><span>Similar players, usual PPR a game</span></div><div><b class="${!n?'':diff>=0?'good':'bad'}">${n?fmt(avgA):'—'}</b><span>Same players against ${p.opp}</span></div></div><div class="matchWhy">${why}</div><details class="muDet"><summary>See the similar players</summary><div class="tablewrap"><table class="matchTable"><thead><tr><th>Similar player</th><th>Usual PPR/G</th><th>Vs ${p.opp}</th></tr></thead><tbody>${L.map(x=>`<tr><td><span data-open="${x.id||''}">${x.name}</span></td><td>${x.normal}</td><td>${x.actual}</td></tr>`).join('')||'<tr><td colspan="3" class="muted">Not enough matchup evidence yet.</td></tr>'}</tbody></table></div></details>`}
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
function bucketRows(b,p){return Object.entries(b).map(([k,v],i)=>{let pc=Math.round(percentile(p.pos,BK[i],v));return `<div class="barrow"><span><button class="infoBtn lblBtn" data-help="h${i}" title="What does this mean?">${k}<span class="chev">ⓘ</span></button></span><div class="track"><div class="fill" style="width:${Math.max(3,pc)}%;background:${barColor(pc)}"></div></div><b class="pctNum lnk ${barGrade(pc)}" data-board="bucket" data-idx="${i}" data-title="${k}" data-pos="${p.pos}" style="--s:${pc}" title="See everyone ranked">${pc}</b></div><div class="bucketHelp" id="h${i}">${HELP[k]||''} <b>Ranks better than ${pc}% of ${p.pos}s.</b></div>`}).join('')}
function posAvg(pos,key){let q=DB.filter(x=>x.pos===pos&&x.m.snap>=30).map(x=>Number(x.m[key]||0)).filter(Number.isFinite);return q.length?q.reduce((a,b)=>a+b,0)/q.length:0}
function usageHTML(p,c){let M=c.M,isRB=p.pos==='RB',pc=x=>x?'%':'';
 let items=isRB?[
  ['PPR / Game',M.ppr_pg,'ppr','ppr_pg','Fantasy points a game in PPR scoring.',0],
  ['Rushing Attempts / Game',M.carries_pg,'car','carries_pg','Carries per game.',0],
  ['Rush Yards / Game',M.rush_yards_pg,'rushy','rush_yards_pg','Rushing yards per game.',0],
  ['Targets / Game',M.targets_pg,'tgt','targets_pg','Passes thrown his way each game.',0],
  ['Scrimmage Yards / Game',M.scrim_pg,'scrim','scrim_pg','Rushing plus receiving yards.',0],
  ['Share of Team Touches',M.touch_share,'touch','touch_share','His slice of the team carries plus targets.',1],
  ['Red-Zone Carries / Game',M.rz_carries_pg,'rzc','rz_carries_pg','Carries inside the opponent 20.',0],
  ['Goal-Line Carries / Game',M.gl_carries_pg,'gl','gl_carries_pg','Carries inside the 5, where most RB touchdowns come from.',0],
  ['Red-Zone Targets / Game',M.rz_targets_pg,'rzt','rz_targets_pg','Passes thrown his way inside the 20.',0],
  ['End-Zone Targets / Game',M.endzone_targets_pg,'ez','endzone_targets_pg','Passes thrown into the end zone, counted from anywhere on the field.',0],
  ['Snap Share',M.snap_pct,'snap','snap_pct','Share of the team offensive snaps he is on the field.',1],
  ['TDs / Game',M.td_pg,'td','td_pg','Rushing plus receiving touchdowns per game.',0]
 ]:[
  ['PPR / Game',M.ppr_pg,'ppr','ppr_pg','Fantasy points a game in PPR scoring.',0],
  ['Targets / Game',M.targets_pg,'tgt','targets_pg','Passes thrown his way each game.',0],
  ['Receiving Yards / Game',M.rec_yards_pg,'recy','rec_yards_pg','Yards after the catch included.',0],
  ['Receptions / Game',M.rec_pg,'rec','rec_pg','Catches per game, a PPR points driver.',0],
  ['Share of Team Targets',M.target_share,'tshare','target_share','His slice of all the team targets.',1],
  ['Share of Team Air Yards',M.air_share,'ashare','air_share','How much of the downfield passing runs through him.',1],
  ['Red-Zone Targets / Game',M.rz_targets_pg,'rzt','rz_targets_pg','Passes thrown his way inside the 20.',0],
  ['End-Zone Targets / Game',M.endzone_targets_pg,'ez','endzone_targets_pg','Passes thrown into the end zone, counted from anywhere on the field.',0],
  ['Snap Share',M.snap_pct,'snap','snap_pct','Share of the team offensive snaps he is on the field.',1],
  ['TDs / Game',M.td_pg,'td','td_pg','Touchdowns per game.',0]
 ];
 if(!isRB&&M.rz_carries_pg>0)items.push(['Red-Zone Carries / Game',M.rz_carries_pg,'rzc','rz_carries_pg','Gadget runs near the goal line. Not part of his score.',0]);
 return `<div class="ph">By the numbers <button class="infoBtn" id="usageInfo">ⓘ</button></div><div class="explain" id="usageExplain" style="display:none;margin-bottom:14px">The raw numbers behind his score. Each bar shows where he ranks among players at his position (100 is the best), next to the position average. Green is the top 30%, yellow the middle, red the bottom 40%.</div><div class="usageGrid">${items.map(([l,v,mk,pk,d,isP])=>{let pr=percentile(p.pos,pk,v),col=barColor(pr);return `<div class="usageItem" title="${d}"><div class="ul">${l}</div><div class="uv ${barGrade(pr)}">${fmt(v)}${pc(isP)}</div><div class="cbt ubt"><div style="width:${Math.max(4,pr)}%;background:${col}"></div></div><div class="uMeta"><span class="${barGrade(pr)}">${ord(pr)} percentile</span><span>${p.pos} avg ${fmt(posAvg(p.pos,mk))}${pc(isP)}</span></div></div>`}).join('')}</div>`}
function wireInfo(){document.querySelectorAll('.infoBtn[data-help]').forEach(b=>b.onclick=()=>{let e=$(b.dataset.help);if(e)e.classList.toggle('open')})}
function wireUsage(){let b=$('usageInfo'),e=$('usageExplain');if(b&&e)b.onclick=()=>e.style.display=e.style.display==='none'?'block':'none'}
var navStack=[],curPlayer=null,curBoard=null,skipPush=false;
function pushNav(){let v=document.querySelector('.view.on');if(!v)return;let id=v.id;if(id==='profile'&&curPlayer){let q=curPlayer;navStack.push(()=>{skipPush=true;openPlayer(q)})}else if(id==='board'&&curBoard){let q=curBoard;navStack.push(()=>{skipPush=true;openBoard(q)})}else navStack.push(()=>showView(id))}
function goBack(){let f=navStack.pop();if(f)f();else showView('rankings')}
const QHELP={'Production':'Fantasy points per game (PPR) compared with other quarterbacks.','Passing':'Passing yards and pass attempts per game. More throws and more yards mean a safer floor.','Rushing':'Rushing yards and rushing touchdowns per game. Quarterbacks who run get a big fantasy boost.','Scoring':'Passing touchdowns per game and how often his throws turn into touchdowns.','Efficiency':'Yards per attempt, completion rate and not throwing interceptions.'};
function openQBP(p){if(skipPush)skipPush=false;else pushNav();curPlayer=p;showView('profile');let m=p.m||{},rated=p.mania!=null,wkv=p.played&&p.played.grade!=null?p.played.grade:p.start,
 bars=Object.entries(p.buckets||{}).map(([k,v],i)=>{let pc=Math.round(v);return `<div class="barrow"><span><button class="infoBtn lblBtn" data-help="qh${i}" title="What does this mean?">${k}<span class="chev">ⓘ</span></button></span><div class="track"><div class="fill" style="width:${Math.max(3,pc)}%;background:${barColor(pc)}"></div></div><b class="pctNum ${barGrade(pc)}" style="--s:${pc}">${pc}</b></div><div class="bucketHelp" id="qh${i}">${QHELP[k]||''} <b>Ranks better than ${pc}% of quarterbacks.</b></div>`}).join(''),
 logs=(p.logs||[]).map(l=>`<tr><td>W${l.w} <span class="muted">${l.opp}</span></td><td class="muted">${l.pj!=null?Number(l.pj).toFixed(1):'—'}</td><td><b>${l.ppr}</b></td><td>${l.comp}/${l.att}</td><td>${l.pyd}</td><td>${l.ptd}</td><td>${l.int}</td><td>${l.ruy}</td><td>${l.rtd}</td></tr>`).join(''),
 st=(v,k)=>`<div class="stat"><div class="sv">${v==null?'—':v}</div><div class="sk">${k}</div></div>`;
 $('profileBody').innerHTML=`<div class="topRow"><button class="backBtn" id="backRanks">← Back</button><span class="topBtns"><button class="shotBtn" id="profTrade">Trade value</button><button class="shotBtn" onclick="toggleShot(true)">Screenshot view</button></span></div><div class="hero2"><div class="hs left"><div id="maniaTop" class="scorebig ${rated?grade(p.mania):''}">${rated?Number(p.mania).toFixed(1):'—'}</div><div class="scorelab">Mania Rating</div><div class="scoresub">${rated?'Overall profile • '+tier(p.mania):'Not enough games yet'}</div></div><div class="hp"><div class="portrait" id="portrait" style="--score:${p.mania||50};--ring:${ring(p.mania||50)}">${ringSvg(p.mania||50)}${pic(p)}</div></div><div class="hs right"><div id="startTop" class="scorebig ${wkv!=null?grade(wkv):''}">${wkv!=null?Number(wkv).toFixed(1):'—'}</div><div class="scorelab">Week ${META.next_week} Rating</div><div class="scoresub">${p.played?'✓ Final • graded ('+p.played.res+' vs '+p.played.opp+')':p.opp==='BYE'?'Bye week':'Projection graded vs QBs, vs '+p.opp}</div></div><div class="hid"><h1>${p.name}</h1><div class="meta">${p.team} • QB • ${p.games} games</div><div class="rankline">${p.pos_rank?'QB'+p.pos_rank+' by Mania Rating':'Limited sample, not ranked yet'}${p.start_rank?' &nbsp;•&nbsp; QB'+p.start_rank+' this week':''}</div></div><div class="hpjW">${heroPj(p)}</div><div class="hol">${badgeHTML(p.bl||[])}</div></div>${qbHTML(p)}<div class="stats">${st(m.ppr,'PPR / Game')}${st(m.pyd,'Pass Yds / Game')}${st(m.ptd,'Pass TD / Game')}${st(m.int,'INT / Game')}${st(m.cmp!=null?m.cmp+'%':null,'Completion %')}${st(m.ypa,'Yards / Attempt')}${st(m.ruy,'Rush Yds / Game')}</div><div class="grid2"><div class="colL"><div class="panel maniaPanel"><div class="ph">Mania breakdown</div><div class="pctHead">Percentile among quarterbacks. 100 is the best, 50 is average.</div><div id="profileBars">${bars}</div><div class="legend"><span><i style="background:#22a45d"></i>Top 30%</span><span><i style="background:#e3a21a"></i>Middle</span><span><i style="background:#e2505b"></i>Bottom 40%</span><span>Tap a category to see what it means.</span></div></div></div><div class="colR"><div class="panel"><div class="ph">How a QB is rated</div><div class="explain">Same 0–100 scale as every other position, but quarterbacks are only graded against quarterbacks. Production counts most (50%), then passing volume, rushing, touchdown scoring and efficiency (about 12% each). Each QB is then placed on the same ladder as every other position: QB1 is as elite as WR1. Week rating takes this week's projected points and grades them on the same scale.${p.qual===false?' <b>Limited sample:</b> with this few games the rating is pulled halfway toward average.':''}</div></div></div></div><div class="panel"><div class="ph">Game log</div><div class="tablewrap"><table><thead><tr><th>Game</th><th>Proj</th><th>PPR</th><th>Cmp/Att</th><th>Pass Yd</th><th>TD</th><th>INT</th><th>Rush Yd</th><th>Rush TD</th></tr></thead><tbody>${logs||'<tr><td colspan="9" class="muted">No games yet.</td></tr>'}</tbody></table></div></div>`;wireInfo();if(document.body.classList.contains('shot'))shotMove(true);$('backRanks').onclick=goBack}
function openPlayer(p){if(p.pos==='QB')return openQBP(p);if(skipPush)skipPush=false;else pushNav();curPlayer=p;showView('profile');let share=p.pos==='RB'?p.m.touch:p.m.tshare,shareName=p.pos==='RB'?'Touch Share':'Target Share',official=calc(p,dfl(p)),bars=bucketRows(official.b,p),logs=p.logs.map((l,i)=>`<tr><td><input class="gameToggle" type="checkbox" ${l.auto?'':'checked'} data-i="${i}"></td><td>W${l.w} ${l.auto?'<span class="partial">HURT, LEFT OUT</span>':l.partial?'<span class="partial">SHORT</span>':''}</td><td class="muted">${l.pj!=null?Number(l.pj).toFixed(1):'\u2014'}</td><td class="grade ${grade(Math.min(99,l.ppr*4))}">${l.ppr}</td><td>${l.tgt}</td><td>${l.rec}</td><td>${l.ry}</td><td>${l.car}</td><td>${l.ruy}</td><td>${(l.rtd||0)+(l.rutd||0)}</td><td>${l.snap}%</td></tr>`).join('');$('profileBody').innerHTML=`<div class="topRow"><button class="backBtn" id="backRanks">← Back</button><span class="topBtns"><button class="shotBtn" id="profTrade">Trade value</button><button class="shotBtn" onclick="toggleShot(true)">Screenshot view</button></span></div><div class="hero2"><div class="hs left"><div id="maniaTop" class="scorebig ${grade(p.mania)}">${Number(p.mania).toFixed(1)}</div><div class="scorelab">Mania Rating</div><div class="scoresub">Overall profile • ${tier(p.mania)}</div></div><div class="hp"><div class="portrait" id="portrait" style="--score:${p.mania};--ring:${ring(p.mania)}">${ringSvg(p.mania)}${pic(p)}</div></div><div class="hs right"><div id="startTop" class="scorebig ${grade(sval(p))}">${fs(p,p.start)}</div><div class="scorelab">Week ${META.next_week} Rating</div><div class="scoresub">${p.played?'\u2713 Final \u2022 graded ('+p.played.res+' vs '+p.played.opp+')':p.opp==='BYE'?'Bye week':p.opp==='TBD'?'Matchup TBD':'Start value vs '+p.opp}</div></div><div class="hid"><h1>${p.name}</h1><div class="meta">${p.team} • ${p.pos} • ${p.games} games</div><div class="rankline">#${p.rank} Overall &nbsp;•&nbsp; ${p.pos}${p.pos_rank} &nbsp;•&nbsp; #${p.team_rank} ${p.pos} on ${p.team}${p.depth?' &nbsp;\u2022&nbsp; Depth chart '+p.pos+p.depth.rank:''}</div></div><div class="hpjW">${heroPj(p)}</div><div class="hol" id="olBox">${olHTML(p,outlook(p,p.start))}</div></div><div class="panel gamePanelTop"><div style="display:flex;justify-content:space-between;align-items:center"><div class="ph">Games counted</div><button class="reset" id="resetGames">RESET</button></div><div class="custom" id="customLine">${p.logs.some(l=>l.auto)?'A game where he got hurt and missed most of it is already left out (tick it to bring it back). ':''}Untick or tick any game and Mania Rating, matchup and Week ${META.next_week} Rating all recalculate. Official Mania Rating <b>${p.mania}</b>.</div>${flagBar(p,'prof',dfl(p))}<div class="gameControls"><label class="gamechip allchip"><input id="allGames" type="checkbox" ${p.logs.some(l=>l.auto)?'':'checked'}>ALL</label>${p.logs.map((l,i)=>`<label class="gamechip"><input class="gameToggleChip" type="checkbox" ${l.auto?'':'checked'} data-i="${i}">W${l.w} • ${l.ppr} PPR${(l.rtd+l.rutd)?' • '+(l.rtd+l.rutd)+' TD':''}</label>`).join('')}</div></div>${playedHTML(p)}${projHTML(p)}${injHTML(p)}<div class="disclaimer">${DISC}</div><div class="grid2"><div class="colL"><div class="panel maniaPanel"><div class="ph">Mania breakdown</div><div class="pctHead">Percentile among ${p.pos}s. 100 is the best, 50 is average.</div><div id="profileBars">${bars}</div><div class="legend"><span><i style="background:#22a45d"></i>Top 30%</span><span><i style="background:#e3a21a"></i>Middle</span><span><i style="background:#e2505b"></i>Bottom 40%</span><span>Tap a category to see what it means.</span></div></div><div class="panel" id="matchPanel">${matchupHTML(p,official)}</div></div><div class="colR"><div class="panel" id="usagePanel">${usageHTML(p,official)}</div></div></div><div class="panel"><div class="ph">${p.team} ${p.pos} room, week by week</div>${roomHTML(p)}</div>${schedHTML(p)}<div class="panel"><div class="ph">Game log</div><div class="tablewrap"><table><thead><tr><th>Use</th><th>Game</th><th>Proj</th><th>PPR</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>TD</th><th>Snap</th></tr></thead><tbody>${logs}</tbody></table></div></div>`;wireInfo();wireUsage();wireRoom(p);if(document.body.classList.contains('shot'))shotMove(true);$('backRanks').onclick=goBack;$('profileBody').querySelectorAll('.roomtable td[data-id]').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));let sync=()=>customProfile(p);document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.onchange=e=>{let i=e.target.dataset.i,checked=e.target.checked;document.querySelectorAll(`[data-i="${i}"]`).forEach(y=>y.checked=checked);sync()});$('allGames').onchange=e=>{document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=e.target.checked);sync()};$('resetGames').onclick=()=>{document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=!p.logs[Number(x.dataset.i)].auto);$('allGames').checked=!p.logs.some(l=>l.auto);sync()}}
function customProfile(p){let idx=[...document.querySelectorAll('.gameToggle:checked')].map(x=>Number(x.dataset.i));let c=calc(p,idx);if(!c){$('customLine').textContent='Select at least one game.';return}$('maniaTop').textContent=c.mania.toFixed(1);$('maniaTop').className='scorebig '+grade(c.mania);$('startTop').textContent=fs(p,c.start);$('olBox').innerHTML=olHTML(p,outlook(p,c.start),idx);$('startTop').className='scorebig '+(p.played?grade(sval(p)):grade(c.start));$('portrait').style.setProperty('--score',c.mania);$('portrait').style.setProperty('--ring',ring(c.mania));{let o=$('portrait').querySelector('.rsv');if(o)o.outerHTML=ringSvg(c.mania)}$('profileBars').innerHTML=bucketRows(c.b,p);wireInfo();$('usagePanel').innerHTML=usageHTML(p,c);wireUsage();$('matchPanel').innerHTML=matchupHTML(p,c);$('customLine').innerHTML=`Official <b>${p.mania}</b> → Mania Rating <b class="${grade(c.mania)}">${fmt(c.mania)}</b> • Week ${META.next_week} Rating <b class="${grade(c.start)}">${fmt(c.start)}</b> • ${idx.length} game${idx.length===1?'':'s'} used.`;$('allGames').checked=idx.length===p.logs.length}
var chipsOpen={A:false,B:false};
function chips(p,side,sel){return flagBar(p,'cmp',sel,`data-side="${side}"`)+`<details class="gdet" data-side="${side}" ${chipsOpen[side]?'open':''}><summary>Choose games (${sel.length} of ${p.logs.length})</summary><div class="gameControls" style="justify-content:center"><label class="gamechip allchip"><input type="checkbox" class="cmpAll" data-side="${side}" ${sel.length===p.logs.length?'checked':''}>ALL</label>${p.logs.map((l,i)=>`<label class="gamechip"><input type="checkbox" class="cmpGame" data-side="${side}" data-i="${i}" ${sel.includes(i)?'checked':''}>W${l.w} • ${l.ppr} PPR</label>`).join('')}</div></details>`}
var cmpOpen={start:false,brk:false};
function bdgPills(p,idx){return `<div class="cmpBd">${bIcons(p.pos==='QB'?(p.bl||[]):dd(p,badges(p,idx)))}</div>`}
function pctBub(v){return `<b class="pctNum ${barGrade(v)}" style="--s:${v}">${v}</b>`}
function bkCompare(A,B,ca,cb){let la=Object.keys(ca.b),lb=Object.keys(cb.b),gen=['Production','Volume','Team role','High-value usage','Efficiency'];
 return `<div class="cbhead"><span>${A.name}</span><span>${B.name}</span></div>`+la.map((k,i)=>{let pa=Math.round(percentile(A.pos,BK[i],ca.b[k])),pb=Math.round(percentile(B.pos,BK[i],cb.b[lb[i]])),lab=la[i]===lb[i]?la[i]:gen[i];
  return `<div class="cbrow"><div class="cbl">${lab}</div><div class="cbbars">${pctBub(pa)}<div class="cbt l"><div style="width:${Math.max(3,pa)}%;background:${barColor(pa)}"></div></div><div class="cbt r"><div style="width:${Math.max(3,pb)}%;background:${barColor(pb)}"></div></div>${pctBub(pb)}</div></div>`}).join('')+`<div class="cexp">Each bubble is a percentile among players at that position. 100 is the best, 50 is average.</div>`}
function renderCompare(){if(!A||!B)return;if(!selA)selA=dfl(A);if(!selB)selB=dfl(B);let ca=calc(A,selA),cb=calc(B,selB);if(!ca||!cb)return;
 let wk=META.next_week,oa=outlook(A,ca.start),ob=outlook(B,cb.start),aB=A.opp==='BYE'||A.out||A.played,bB=B.opp==='BYE'||B.out||B.played,starter=aB&&!bB?B:bB&&!aB?A:ca.start>=cb.start?A:B,
 rows=[['Mania Rating',ca.mania,cb.mania],['PPR/G',ca.M.ppr_pg,cb.M.ppr_pg],['Targets/G',ca.M.targets_pg,cb.M.targets_pg],['Receptions/G',ca.M.rec_pg,cb.M.rec_pg],['Rushing Attempts/G',ca.M.carries_pg,cb.M.carries_pg],['Rush Yds/G',ca.M.rush_yards_pg,cb.M.rush_yards_pg],['Scrimmage Yds/G',ca.M.scrim_pg,cb.M.scrim_pg],['Snap %',ca.M.snap_pct,cb.M.snap_pct],['Target Share %',ca.M.target_share,cb.M.target_share],['Touch Share %',ca.M.touch_share,cb.M.touch_share],['Air-Yard Share %',ca.M.air_share,cb.M.air_share],['Red-Zone Targets/G',ca.M.rz_targets_pg,cb.M.rz_targets_pg],['End-Zone Targets/G',ca.M.endzone_targets_pg,cb.M.endzone_targets_pg],['Red-Zone Carries/G',ca.M.rz_carries_pg,cb.M.rz_carries_pg],['Goal-Line Carries/G',ca.M.gl_carries_pg,cb.M.gl_carries_pg]],
 sg=v=>(v>=0?'+':'')+v.toFixed(1),oppT=q=>q.opp==='BYE'?'Bye':q.opp==='TBD'?'TBD':q.opp,
 srows=[['Week '+wk+' Rating',fs(A,ca.start),fs(B,cb.start),A.out?-1:ca.start,B.out?-1:cb.start],['Projected points',pjT(A),pjT(B),A.proj&&!A.out?A.proj.pts:-1,B.proj&&!B.out?B.proj.pts:-1],['Opponent',oppT(A),oppT(B),0,0],['Injury impact',A.out?'OUT':A.inj_adj===0?'None':sg(A.inj_adj),B.out?'OUT':B.inj_adj===0?'None':sg(B.inj_adj),A.out?-99:A.inj_adj,B.out?-99:B.inj_adj],['Matchup boost',sg(ca.match.adj),sg(cb.match.adj),ca.match.adj,cb.match.adj],['Rank',`<span class="outlook ${oa.c}">${oa.t}</span>`,`<span class="outlook ${ob.c}">${ob.t}</span>`,0,0]],
 startHTML=`<details class="cdet" data-k="start" ${cmpOpen.start?'open':''}><summary><span>Who should you start in Week ${wk}?</span><i>Tap to see</i></summary><div class="cdBody"><div class="cwin"><div class="cwk">Our pick</div><strong>${starter.name}</strong><div class="sub">${fs(A,ca.start)} ${A.name} &nbsp;vs&nbsp; ${fs(B,cb.start)} ${B.name}</div></div><table class="cstart"><thead><tr><th>${A.name}</th><th></th><th>${B.name}</th></tr></thead><tbody>${srows.map(([k,a,b,x,y])=>`<tr><td class="${x>y?'good':''}">${a}</td><td class="cmid">${k}</td><td class="${y>x?'good':''}">${b}</td></tr>`).join('')}</tbody></table><div class="cexp"><b>Matchup boost</b> is how much the opponent moves a player's Week ${wk} Rating. Positive means similar players have done well against that defense. Negative means they struggled.</div><div class="disclaimer">${DISC}</div></div></details>`,
 brkHTML=`<details class="cdet" data-k="brk" ${cmpOpen.brk?'open':''}><summary><span>Compare the Mania breakdown</span><i>Tap to see</i></summary><div class="cdBody">${bkCompare(A,B,ca,cb)}</div></details>`;
 $('compareBody').innerHTML=`<div class="compareHero"><div class="comparePlayer"><div class="cmpPortrait" style="--score:${ca.mania};--ring:${ring(ca.mania)}">${ringSvg(ca.mania)}${pic(A)}</div><h2 class="playerlink cmpOpen" data-id="${A.id}">${A.name}</h2><div class="meta">${A.team} • ${A.pos} • vs ${A.opp}</div><div class="scorebig ${grade(ca.mania)}">${ca.mania.toFixed(1)}</div><div class="scorelab">Mania Rating</div>${bdgPills(A,selA)}${chips(A,'A',selA)}</div><div class="vs">VS</div><div class="comparePlayer"><div class="cmpPortrait" style="--score:${cb.mania};--ring:${ring(cb.mania)}">${ringSvg(cb.mania)}${pic(B)}</div><h2 class="playerlink cmpOpen" data-id="${B.id}">${B.name}</h2><div class="meta">${B.team} • ${B.pos} • vs ${B.opp}</div><div class="scorebig ${grade(cb.mania)}">${cb.mania.toFixed(1)}</div><div class="scorelab">Mania Rating</div>${bdgPills(B,selB)}${chips(B,'B',selB)}</div></div>${startHTML}${brkHTML}<details class="cdet" data-k="stats" ${cmpOpen.stats===false?'':'open'}><summary><span>Stat by stat</span><i>Tap to hide</i></summary><div class="cdBody"><div class="cmpStats tablewrap"><table><thead><tr><th>${A.name}</th><th style="text-align:center">Metric</th><th>${B.name}</th></tr></thead><tbody>${rows.map(([k,a,b])=>`<tr><td class="${a>b?'good':''}">${/Rating/.test(k)?Number(a).toFixed(1):fmt(a)}</td><td style="text-align:center">${k}</td><td class="${b>a?'good':''}">${/Rating/.test(k)?Number(b).toFixed(1):fmt(b)}</td></tr>`).join('')}</tbody></table></div></div></details>`;
 document.querySelectorAll('.cdet').forEach(d=>d.ontoggle=()=>{cmpOpen[d.dataset.k]=d.open});
 document.querySelectorAll('.gdet').forEach(d=>d.ontoggle=()=>{chipsOpen[d.dataset.side]=d.open});document.querySelectorAll('.cmpOpen').forEach(x=>x.onclick=()=>openPlayer(byId(x.dataset.id)));document.querySelectorAll('.cmpGame').forEach(x=>x.onchange=()=>{let side=x.dataset.side,i=Number(x.dataset.i),arr=side==='A'?selA:selB;if(x.checked){if(!arr.includes(i))arr.push(i)}else if(arr.length>1)arr.splice(arr.indexOf(i),1);arr.sort((a,b)=>a-b);renderCompare()});document.querySelectorAll('.cmpAll').forEach(x=>x.onchange=()=>{let p=x.dataset.side==='A'?A:B,arr=p.logs.map((_,i)=>i);if(x.dataset.side==='A')selA=x.checked?arr:[arr[arr.length-1]];else selB=x.checked?arr:[arr[arr.length-1]];renderCompare()})}
/* ===== Team compare ===== */
const TTYPES=['QB','RB','WR','TE','FLEX','SFLEX','BN'];
const TLIM={QB:[0,3],RB:[0,5],WR:[0,5],TE:[0,3],FLEX:[0,4],SFLEX:[0,2],BN:[0,10]};
const TPRE={standard:{QB:1,RB:2,WR:2,TE:1,FLEX:1,SFLEX:0,BN:6},superflex:{QB:1,RB:2,WR:2,TE:1,FLEX:1,SFLEX:1,BN:6},deep:{QB:1,RB:3,WR:3,TE:1,FLEX:2,SFLEX:0,BN:7}};
var LU=Object.assign({},TPRE.standard),teamScope='starters',showM=false,showW=true,tOpen=null,tGames={};
try{let l=JSON.parse(localStorage.getItem('fmLineup3')||'null');if(l&&TTYPES.every(k=>typeof l[k]==='number'))LU=l}catch(e){}
function newTeam(n){return {name:n,s:{},sel:{}}}
function tEnsure(T){TTYPES.forEach(k=>{let a=T.s[k]||(T.s[k]=[]);while(a.length<LU[k])a.push(null);if(a.length>LU[k])a.length=LU[k]})}
var TM={A:newTeam('My team'),B:newTeam('Their team')};
try{let sv=JSON.parse(localStorage.getItem('fmTeams3')||'null');if(sv&&sv.A&&sv.B&&sv.A.s)TM=sv}catch(e){}
function saveTeams(){try{localStorage.setItem('fmTeams3',JSON.stringify(TM));localStorage.setItem('fmLineup3',JSON.stringify(LU))}catch(e){}}
function tElig(k,p){return k==='QB'?p.pos==='QB':k==='FLEX'?p.pos!=='QB':(k==='RB'||k==='WR'||k==='TE')?p.pos===k:true}
function tLab(k,p){return k==='BN'?(p?p.pos:'BN'):k==='FLEX'?'FLX':k==='SFLEX'?'SFX':k}
function tVal(p,idxs){let q=p.pos==='QB',c=null;if(!q){try{c=calc(p,idxs&&idxs.length?idxs:dfl(p))}catch(e){c=null}}
 let mania=c?c.mania:(p.mania==null?null:p.mania),tag='',pts=0,sd=0;
 if(p.opp==='BYE'){tag='BYE'}else if(p.out){tag='OUT'}
 if(p.played){pts=p.played.ppr;tag='✓ Final'}else if(!tag&&p.proj){pts=p.proj.pts;sd=Math.max(0,(p.proj.hi-p.proj.lo)/2.56)}
 return {mania,pts,sd,tag}}
function tList(T,scope){tEnsure(T);let o=[];TTYPES.forEach(k=>{if(scope==='starters'&&k==='BN')return;if(scope==='bench'&&k!=='BN')return;T.s[k].forEach((id,i)=>o.push({id,k,i}))});return o}
const GKEYS=['QB','RB','WR','TE','FLEX','SFLEX'],GLAB={QB:'Quarterback',RB:'Running backs',WR:'Wide receivers',TE:'Tight ends',FLEX:'Flex',SFLEX:'Superflex'};
function tScore(T,scope){let it=tList(T,scope).filter(x=>x.id&&byId(x.id)).map(x=>{let p=byId(x.id);return {x,p,v:tVal(p,T.sel[x.id])}});
 let sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>a.length?sum(a)/a.length:0,mv=it.filter(e=>e.v.mania!=null).map(e=>e.v.mania);
 let r={it,n:it.length,slots:tList(T,scope).length,pts:sum(it.map(e=>e.v.pts)),mania:avg(mv),sd:Math.sqrt(sum(it.map(e=>e.v.sd*e.v.sd))),gP:{},gM:{},subP:{},subM:{}};
 const gk=e=>e.x.k!=='BN'?e.x.k:(scope==='bench'?e.p.pos:null);GKEYS.forEach(pos=>{let a=it.filter(e=>gk(e)===pos);if(a.length){r.gP[pos]=sum(a.map(e=>e.v.pts));r.gM[pos]=avg(a.filter(e=>e.v.mania!=null).map(e=>e.v.mania))}});
 [['st',e=>e.x.k!=='BN'],['bn',e=>e.x.k==='BN']].forEach(([n,f])=>{let a=it.filter(f);if(a.length){r.subP[n]=sum(a.map(e=>e.v.pts));r.subM[n]=avg(a.filter(e=>e.v.mania!=null).map(e=>e.v.mania))}});
 return r}
function erf(x){let s=x<0?-1:1;x=Math.abs(x);let t=1/(1+.3275911*x),y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);return s*y}
function tSlotHTML(T,t,x){let p=x.id&&byId(x.id),lab=tLab(x.k,p);
 if(tOpen&&tOpen.t===t&&tOpen.k===x.k&&tOpen.i===x.i)return `<div class="tslot open"><span class="tpos">${tLab(x.k)}</span><div class="tsearch"><input class="tq" placeholder="Search ${x.k==='FLEX'?'RB / WR / TE':x.k==='SFLEX'?'any position':x.k==='BN'?'any player':x.k}" autocomplete="off"><div class="dd tdd"></div></div><button type="button" class="tx tcancel" title="Cancel">✕</button></div>`;
 if(!p)return `<button type="button" class="tslot empty" data-t="${t}" data-k="${x.k}" data-i="${x.i}"><span class="tpos">${tLab(x.k)}</span><span class="tadd">+ Add ${x.k==='BN'?'player':x.k==='FLEX'?'flex':x.k==='SFLEX'?'superflex':x.k}</span></button>`;
 let v=tVal(p,T.sel[x.id]),isQ=p.pos==='QB',cnt=T.sel[x.id]?T.sel[x.id].length:dfl(p).length,opp=p.opp==='BYE'?'Bye':p.opp==='TBD'?'TBD':'vs '+p.opp;
 let mTxt=v.mania==null?'—':Number(v.mania).toFixed(1),pTxt=v.pts.toFixed(1),val=showM&&showW?`<div class="tv tv2"><b class="${grade(v.mania||0)}">${mTxt}</b><small>${pTxt} pts</small></div>`:showM?`<div class="tv ${grade(v.mania||0)}">${mTxt}</div>`:`<div class="tv">${pTxt}</div>`;
 return `<div class="tslot" data-t="${t}" data-k="${x.k}" data-i="${x.i}" data-id="${p.id}"><span class="tpos">${lab}</span>${thumb(p)}<div class="tnm"><b class="tlink" data-open="${p.id}">${p.name}</b><small>${p.team} • ${p.pos} • ${opp}${v.tag?' • <i class="ttag">'+v.tag+'</i>':''}${flagTag(p,T.sel[x.id]||dfl(p))}</small></div>${val}${isQ?'<span class="tgm ph0"></span>':`<button type="button" class="tgm${T.sel[x.id]&&T.sel[x.id].length!==dfl(p).length?' on':''}" title="Turn games on or off for this player">${T.sel[x.id]&&T.sel[x.id].length!==dfl(p).length?cnt+'/'+p.logs.length:'⋯'}</button>`}<button type="button" class="tx tdel" title="Remove">✕</button></div>${(tGames[t+x.id])?`<div class="tgames" data-t="${t}" data-id="${p.id}">${flagBar(p,'team',T.sel[p.id]||dfl(p),`data-t="${t}"`)}${p.logs.map((l,i)=>`<label class="gamechip"><input type="checkbox" class="tgc" data-i="${i}" ${(T.sel[x.id]||dfl(p)).includes(i)?'checked':''}>W${l.w} • ${l.ppr} PPR</label>`).join('')}<button type="button" class="reset tgreset">RESET</button></div>`:''}`}
function tCardHTML(t,S,O,lead){let T=TM[t],both=showM&&showW,main=showM?'M':'P',
 big=main==='M'?S.mania:S.pts,rv=main==='M'?S.mania:Math.min(99.4,S.pts/Math.max(S.pts,O.pts,1)*96),
 cap=both?'Mania Rating (average)':main==='M'?'average Mania Rating':'projected points this week',
 sec=(label,f)=>{let l=tList(T,teamScope).filter(f);return l.length?`<div class="tsec">${label}</div>`+l.map(x=>tSlotHTML(T,t,x)).join(''):''};
 return `<div class="tcard${lead?' lead':''}" data-t="${t}"><input class="tname" value="${T.name.replace(/"/g,'&quot;')}" maxlength="22" aria-label="Team name"><div class="tscoreW"><div class="tring">${ringSvg(rv)}<div class="tbig"><b class="${main==='M'?grade(S.mania):''}">${big.toFixed(1)}</b></div></div><div class="tcap">${cap}${both?`<div class="tpill"><b>${S.pts.toFixed(1)}</b> projected pts</div>`:''}<br><span>${S.n} of ${S.slots} spots filled</span></div>${lead?'<span class="tlead">Leads</span>':''}</div>${sec('Starters',x=>x.k!=='BN')}${sec('Bench',x=>x.k==='BN')}<div class="tfoot"><button type="button" class="tclr" data-t="${t}">Clear team</button></div></div>`}
function tBarsFor(A,B,which){let g=which==='P'?['gP','subP']:['gM','subM'],pts=which==='P',pos=GKEYS.filter(k=>A[g[0]][k]!=null||B[g[0]][k]!=null).map(k=>[GLAB[k],A[g[0]][k],B[g[0]][k]]);
 if(teamScope==='all'){pos.push(['Starters',A[g[1]].st,B[g[1]].st]);pos.push(['Bench',A[g[1]].bn,B[g[1]].bn])}
 let mx=pts?Math.max(1,...pos.map(r=>Math.max(r[1]||0,r[2]||0))):100,col=(v,o)=>pts?(v>=o?'#22a45d':v>=o*.85?'#e3a21a':'#e2505b'):barColor(v),nm=v=>pts?v.toFixed(1):v.toFixed(0);
 return pos.map(([lab,a,b])=>{let ha=a!=null,hb=b!=null,wa=Math.max(3,(a||0)/mx*100),wb=Math.max(3,(b||0)/mx*100);return `<div class="cbrow"><div class="cbl">${lab}</div><div class="cbbars"><b class="tnum ${ha?'':'dim'}">${ha?nm(a):'—'}</b><div class="cbt l"><div style="width:${ha?wa:0}%;background:${col(a||0,b||0)}"></div></div><div class="cbt r"><div style="width:${hb?wb:0}%;background:${col(b||0,a||0)}"></div></div><b class="tnum ${hb?'':'dim'}">${hb?nm(b):'—'}</b></div></div>`}).join('')}
function tVerdict(A,B,nA,nB){let any=A.n+B.n>0;if(!any)return `<div class="tempty"><b>Build two teams to compare.</b><span>Add players to the slots, or start from a sample.</span></div>`;
 let dM=A.mania-B.mania,dP=A.pts-B.pts,tieM=Math.abs(dM)<.3,tieP=Math.abs(dP)<.5,wM=tieM?null:dM>0?'A':'B',wP=tieP?null:dP>0?'A':'B',nm=w=>w==='A'?nA:nB;
 let prob=()=>{if(!(A.sd+B.sd>0&&A.n&&B.n)||tieP)return '';let z=dP/Math.sqrt(A.sd*A.sd+B.sd*B.sd),pa=.5*(1+erf(z/Math.SQRT2)),pw=wP==='B'?1-pa:pa;return `<div class="tprob"><b>${Math.round(pw*100)}%</b><span>chance ${nm(wP)} outscores ${nm(wP==='A'?'B':'A')} this week</span></div>`};
 if(showM&&showW){if(wM&&wM===wP)return `<div class="cwin tverdict"><div class="cwk">Our pick • both agree</div><strong>${nm(wM)}</strong><div class="sub">Ahead by ${Math.abs(dM).toFixed(1)} Mania points and ${Math.abs(dP).toFixed(1)} projected points</div>${prob()}</div>`;
  return `<div class="cwin tverdict split"><div class="cwk">${wM&&wP?'Split decision':'Too close to call'}</div><strong>${wM&&wP?'Season vs this week':nA+' vs '+nB}</strong><div class="tsplit"><div><span>Mania Rating</span><b>${wM?nm(wM):'Even'}</b><small>${wM?'by '+Math.abs(dM).toFixed(1):''}</small></div><div><span>Week ${META.next_week} projection</span><b>${wP?nm(wP):'Even'}</b><small>${wP?'by '+Math.abs(dP).toFixed(1)+' pts':''}</small></div></div>${prob()}</div>`}
 let w=showM?wM:wP,d=showM?dM:dP;if(!w)return `<div class="cwin tverdict"><div class="cwk">Too close to call</div><strong>${nA} vs ${nB}</strong><div class="sub">Within ${Math.abs(d).toFixed(1)} ${showM?'rating points':'projected points'}</div></div>`;
 return `<div class="cwin tverdict"><div class="cwk">Our pick</div><strong>${nm(w)}</strong><div class="sub">Ahead by ${Math.abs(d).toFixed(1)} ${showM?'rating points':'projected points'}</div>${showW?prob():''}</div>`}
function tSettings(){let names={QB:'Quarterback',RB:'Running back',WR:'Wide receiver',TE:'Tight end',FLEX:'Flex (RB/WR/TE)',SFLEX:'Superflex (any, incl. QB)',BN:'Bench'},n=TTYPES.slice(0,6).reduce((s,k)=>s+LU[k],0);
 let cur=Object.keys(TPRE).find(k=>TTYPES.every(t=>TPRE[k][t]===LU[t]));
 return `<details class="tset" ${tOpen==='set'?'open':''}><summary><span>Lineup</span><i>${n} starters • ${LU.BN} bench${LU.SFLEX?' • superflex':''}</i></summary><div class="tsetB"><div class="tpre">${[['standard','Standard'],['superflex','Superflex'],['deep','Deep']].map(([k,l])=>`<button type="button" class="tsb tpb${cur===k?' on':''}" data-pre="${k}">${l}</button>`).join('')}</div><div class="tsteps">${TTYPES.map(k=>`<div class="tst"><span>${names[k]}</span><div><button type="button" class="tstb" data-k="${k}" data-d="-1">−</button><b>${LU[k]}</b><button type="button" class="tstb" data-k="${k}" data-d="1">+</button></div></div>`).join('')}</div><p class="tnote" style="margin:8px 0 0">Both teams use the same lineup so the comparison is fair. Shrinking a position removes the last slot of that type.</p></div></details>`}
function renderTeams(){let box=$('teamBody');if(!box)return;let SA=tScore(TM.A,teamScope),SB=tScore(TM.B,teamScope),nA=TM.A.name||'Team A',nB=TM.B.name||'Team B',any=SA.n+SB.n>0;
 let main=showM?'M':'P',d=main==='M'?SA.mania-SB.mania:SA.pts-SB.pts,tie=Math.abs(d)<(main==='M'?.3:.5),lead=!any||tie?null:d>0?'A':'B';
 let note=[showW?'Projected points add up each player’s PPR projection for the week (finished players show what they scored). Turning games off changes Mania, not projected points.':'',showM?'Mania Rating is the average season rating (0–100). Turn games off on any player and it recalculates.':''].filter(Boolean).join(' ');
 box.innerHTML=`${tSettings()}<div class="tctl"><div class="tseg" data-g="scope">${[['starters','Starters'],['bench','Bench'],['all','All together']].map(([k,l])=>`<button type="button" class="tsb${teamScope===k?' on':''}" data-v="${k}">${l}</button>`).join('')}</div><div class="tseg tmulti" data-g="basis"><button type="button" class="tsb${showM?' on':''}" data-b="M">Mania Rating</button><button type="button" class="tsb${showW?' on':''}" data-b="W">Week ${META.next_week} projection</button></div><button type="button" class="tsample">Try a sample</button><button type="button" class="tmania">Maniac View</button></div><details class="tnoteD"><summary>How this works</summary><p class="tnote">${note} Tap both to compare on both. ${teamScope==='starters'?'Bench is hidden, switch to Bench or All together to add one.':''} Each slot is compared with the same slot on the other team, so a flex is measured against a flex.</p></details>
 <div class="tgrid">${tCardHTML('A',SA,SB,lead==='A')}<div class="tvs">VS</div>${tCardHTML('B',SB,SA,lead==='B')}</div>${tVerdict(SA,SB,nA,nB)}${any?`<div class="panel tbars"><div class="tbh"><span>${nA}</span><span>${nB}</span></div>${showM?`<div class="ph tbt">Mania Rating by lineup slot</div>${tBarsFor(SA,SB,'M')}`:''}${showW?`<div class="ph tbt">Week ${META.next_week} projected points by lineup slot</div>${tBarsFor(SA,SB,'P')}`:''}</div>`:''}`;
 wireTeams()}
function wireTeams(){let box=$('teamBody');
 box.querySelector('.tset').ontoggle=e=>{tOpen=e.target.open?'set':null};
 box.querySelectorAll('.tpb').forEach(b=>b.onclick=()=>{LU=Object.assign({},TPRE[b.dataset.pre]);tOpen='set';saveTeams();renderTeams()});
 box.querySelectorAll('.tstb').forEach(b=>b.onclick=()=>{let k=b.dataset.k,v=Math.max(TLIM[k][0],Math.min(TLIM[k][1],LU[k]+(+b.dataset.d)));if(v===LU[k])return;LU[k]=v;tOpen='set';saveTeams();renderTeams()});
 box.querySelectorAll('.tseg[data-g=scope] .tsb').forEach(b=>b.onclick=()=>{teamScope=b.dataset.v;tOpen=null;renderTeams()});
 box.querySelectorAll('.tseg[data-g=basis] .tsb').forEach(b=>b.onclick=()=>{let k=b.dataset.b;if(k==='M'){if(showM&&!showW)return;showM=!showM}else{if(showW&&!showM)return;showW=!showW}tOpen=null;renderTeams()});
 box.querySelector('.tsample').onclick=fillSample;
 box.querySelectorAll('.tname').forEach(i=>{i.oninput=()=>{TM[i.closest('.tcard').dataset.t].name=i.value;saveTeams()};i.onchange=()=>renderTeams()});
 box.querySelectorAll('.tslot.empty').forEach(b=>b.onclick=()=>{tOpen={t:b.dataset.t,k:b.dataset.k,i:+b.dataset.i};renderTeams();let q=box.querySelector('.tq');if(q)q.focus()});
 box.querySelectorAll('.tcancel').forEach(b=>b.onclick=()=>{tOpen=null;renderTeams()});
 let q=box.querySelector('.tq');if(q&&tOpen&&tOpen!=='set'){let k=tOpen.k,T=TM[tOpen.t],used=new Set(TTYPES.flatMap(z=>T.s[z]||[]).filter(Boolean)),dd=box.querySelector('.tdd');
  q.oninput=()=>{let s=q.value.toLowerCase().trim();if(!s){dd.style.display='none';return}let m=allPlayers().filter(p=>(p.pos!=='QB'||p.mania!=null)&&tElig(k,p)&&p.name.toLowerCase().includes(s)).sort((a,b)=>(b.mania||0)-(a.mania||0)).slice(0,8);dd.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><div><span class="ddiName">${p.name}</span><span class="ddiMeta">${p.team} • ${p.pos}${used.has(p.id)?' • on this team (tap to move here)':''}</span></div><span class="ddiRate">${p.mania==null?'—':p.mania}</span></div>`).join('')||'<div class="ddi muted">No match</div>';dd.style.display='block';dd.querySelectorAll('.ddi[data-id]').forEach(x=>x.onclick=()=>{let id=x.dataset.id,occ=T.s[tOpen.k][tOpen.i];if(used.has(id)){TTYPES.forEach(z=>(T.s[z]||[]).forEach((v,j)=>{if(v===id&&!(z===tOpen.k&&j===tOpen.i))T.s[z][j]=(occ&&tElig(z,byId(occ)))?occ:null}))}T.s[tOpen.k][tOpen.i]=id;tOpen=null;saveTeams();renderTeams()})}}
 box.querySelectorAll('.tdel').forEach(b=>{b.onclick=()=>{let s=b.closest('.tslot'),T=TM[s.dataset.t];T.s[s.dataset.k][+s.dataset.i]=null;delete T.sel[s.dataset.id];delete tGames[s.dataset.t+s.dataset.id];saveTeams();renderTeams()}});
 box.querySelectorAll('.ttag.fl').forEach(b=>b.onclick=()=>{let s=b.closest('.tslot'),key=s.dataset.t+s.dataset.id;tGames[key]=!tGames[key];renderTeams()});
 box.querySelectorAll('.tgm:not(.ph0)').forEach(b=>b.onclick=()=>{let s=b.closest('.tslot'),key=s.dataset.t+s.dataset.id;tGames[key]=!tGames[key];renderTeams()});
 box.querySelectorAll('.tgames').forEach(g=>{let T=TM[g.dataset.t],p=byId(g.dataset.id);g.querySelectorAll('.tgc').forEach(c=>c.onchange=()=>{let cur=(T.sel[p.id]||dfl(p)).slice(),i=+c.dataset.i;if(c.checked){if(!cur.includes(i))cur.push(i)}else if(cur.length>1)cur.splice(cur.indexOf(i),1);cur.sort((a,b)=>a-b);if(sameSet(cur,dfl(p)))delete T.sel[p.id];else T.sel[p.id]=cur;saveTeams();renderTeams()});g.querySelector('.tgreset').onclick=()=>{delete T.sel[p.id];saveTeams();renderTeams()}});
 box.querySelectorAll('.tclr').forEach(b=>b.onclick=()=>{let t=b.dataset.t,nm=TM[t].name;TM[t]=newTeam(nm);tOpen=null;saveTeams();renderTeams()});
 box.querySelectorAll('.tlink').forEach(a=>a.onclick=()=>openPlayer(byId(a.dataset.open)))}
function fillSample(){let by=pos=>allPlayers().filter(p=>p.pos===pos&&p.mania!=null&&p.qual!==false&&p.opp!=='BYE'&&!p.out).sort((a,b)=>b.mania-a.mania),pool={QB:by('QB'),RB:by('RB'),WR:by('WR'),TE:by('TE')},used=new Set(),
 take=pos=>{let l=pool[pos];for(let i=0;i<l.length;i++){if(!used.has(l[i].id)){used.add(l[i].id);return l[i].id}}return null},TA=newTeam('My team'),TB=newTeam('Their team'),steps=[];
 tEnsure(TA);tEnsure(TB);
 ['QB','RB','WR','TE'].forEach(pos=>{for(let i=0;i<LU[pos];i++)steps.push([pos,i,pos])});
 ['FLEX','SFLEX'].forEach(k=>{for(let i=0;i<LU[k];i++)steps.push([k,i,k==='SFLEX'?(i===0?'QB':'WR'):(i%2?'RB':'WR')])});
 let order=['RB','WR','RB','TE','WR','QB','WR','RB','TE','WR'];for(let i=0;i<LU.BN;i++)steps.push(['BN',i,order[i%order.length]]);
 steps.forEach(([k,i,pos],n)=>{let f=n%2?[TB,TA]:[TA,TB];f[0].s[k][i]=take(pos);f[1].s[k][i]=take(pos)});
 TM={A:TA,B:TB};tGames={};tOpen=null;saveTeams();renderTeams()}
document.querySelectorAll('.cmpMode').forEach(b=>b.onclick=()=>{let m=b.dataset.cm;document.querySelectorAll('.cmpMode').forEach(x=>x.classList.toggle('on',x===b));$('cmpPlayers').hidden=m!=='players';$('cmpTeams').hidden=m!=='teams';if(m==='teams')renderTeams()});

function shotMove(on){let m=document.querySelector('#profileBody .maniaPanel'),h=document.querySelector('#profileBody .hero2'),L=document.querySelector('#profileBody .colL');if(!m||!h||!L)return;if(on){h.after(m);m.classList.add('shotTop')}else{L.prepend(m);m.classList.remove('shotTop')}}
/* ===== Maniac View ===== */
let mvOn=false,W_mv=390;
function mvSil(w,b,id){return `<svg viewBox="0 0 100 130" width="${w}" style="display:block;filter:brightness(${b})"><defs><linearGradient id="mvs${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8a8a94"/><stop offset=".55" stop-color="#3a3a42"/><stop offset="1" stop-color="#101014"/></linearGradient></defs><path d="M6 130C6 98 26 86 50 84C74 86 94 98 94 130Z" fill="url(#mvs${id})"/><ellipse cx="50" cy="48" rx="21" ry="26" fill="url(#mvs${id})"/></svg>`}
function mvOrder(T,scope){let l=tList(T,scope).filter(x=>x.id&&byId(x.id)),rank={QB:0,RB:1,WR:2,TE:3,FLEX:4,SFLEX:5,BN:6};l.sort((a,b)=>rank[a.k]-rank[b.k]||a.i-b.i);return l.map(x=>{let p=byId(x.id),v=tVal(p,T.sel[x.id]);return {p,k:x.k,v}})}
function mvLast(p){let a=p.name.split(' ');let l=a[a.length-1];if(/^(Jr\.?|Sr\.?|II|III|IV)$/i.test(l)&&a.length>2)l=a[a.length-2];return l.toUpperCase()}
function mvVals(e){let m=e.v.mania==null?'—':Number(e.v.mania).toFixed(1),p=Number(e.v.pts).toFixed(1);return showM&&showW?`<span class="v1">${m}</span><span class="v2">${p}<u> pts</u></span>`:showM?`<span class="v1">${m}</span>`:`<span class="v1">${p}<u> pts</u></span>`}
function mvImg(e,w,hh,b,mob,uid,opx){let sil=mvSil(w,b,uid),op=opx||(mob?'center top':'center bottom');
 if(!e.p.headshot)return sil;
 return `<img src="${e.p.headshot}" alt="" style="width:${w}px;height:${hh?hh+'px':'auto'};object-fit:contain;object-position:${op};display:block;filter:contrast(1.05) saturate(1.08) brightness(${b})" onerror="this.outerHTML=this.dataset.sil" data-sil='${sil.replace(/'/g,"&#39;")}'>`}
function mvFig(e,i,n,cx,top,w,hh,side,mob,fs,row){let t=n>1?i/(n-1):0,b=(1-.18*t).toFixed(2),lab=e.k==='SFLEX'?'SFX':e.k==='FLEX'?'FLX':e.k==='BN'?'BN':e.p.pos,last=mvLast(e.p),uid=i+(side>0?200:100);
 if(!mob){let sh=`drop-shadow(${-side*14}px 14px 16px rgba(0,0,0,.9))`,mask='-webkit-mask-image:linear-gradient(#000 84%,transparent 100%);mask-image:linear-gradient(#000 84%,transparent 100%);';
  return `<div class="mvFig" style="left:${cx-w/2}px;bottom:${top}px;width:${w}px;z-index:${100-i};filter:${sh}"><div class="mvTag" style="left:${w/2-110}px;width:220px;bottom:100%;font-size:${fs}px;padding-bottom:${fs*.35}px"><i>${lab}</i><b>${last}</b>${mvVals(e)}</div><div style="${mask}">${mvImg(e,w,0,b,false,uid)}</div></div>`}
 let boxH=row*(i===n-1?1.45:1.18),sh=`drop-shadow(0 -7px 9px rgba(0,0,0,.85))`;
 return `<div class="mvFig" style="left:${cx-w/2}px;top:${top}px;width:${w}px;height:${boxH}px;z-index:${100+i};filter:${sh}"><div style="height:${boxH}px;overflow:hidden;-webkit-mask-image:linear-gradient(#000 ${i===n-1?70:92}%,transparent 100%);mask-image:linear-gradient(#000 ${i===n-1?70:92}%,transparent 100%)">${mvImg(e,w,boxH,b,true,uid)}</div><div class="mvPlate" style="top:${row-fs*3.1}px;width:${w}px"><b style="font-size:${fs}px">${last}</b><span style="font-size:${fs-2.5}px"><i>${lab}</i>${mvVals(e)}</span></div></div>`}
let mvStyle='C';
function mvVal2(e){return mvVals(e)}
function mvMobileA(A,B,W,H){let n=Math.max(A.length,B.length,1),top0=H*.225,bot=H*.87,wts=Array.from({length:n},(_,i)=>i===0?1.45:1),tw=wts.reduce((a,b)=>a+b,0),unit=(bot-top0)/tw,half=W/2,out=[],y=top0;
 for(let i=0;i<n;i++){let rh=unit*wts[i],hw=Math.min(half*.55,rh*1.6),y0=y;y+=rh;
  [[-1,A],[1,B]].forEach(([side,L])=>{let e=L[i];if(!e)return;let lab=e.k==='SFLEX'?'SFX':e.k==='FLEX'?'FLX':e.k==='BN'?'BN':e.p.pos,last=mvLast(e.p),uid=i+(side>0?400:300),
   tx=half-hw-6,fs=i===0?17:13.5,x0=side<0?4:half+2,wd=half-6,
   clip=side<0?'polygon(0 0,100% 0,96% 100%,0 100%)':'polygon(4% 0,100% 0,100% 100%,0 100%)';
   out.push(`<div class="mvRow" style="left:${x0}px;top:${y0+3}px;width:${wd}px;height:${rh-5}px;clip-path:${clip};z-index:${50+i}"></div>`);
   out.push(`<div class="mvTx" style="${side<0?`left:${x0+8}px;text-align:left`:`left:${x0+wd-8-tx}px;text-align:right`};width:${tx}px;top:${y0+3}px;height:${rh-5}px;z-index:${300+i}"><b style="font-size:${fs}px">${last}</b><span><i>${lab}</i>${mvVals(e)}</span></div>`);
   let hx=side<0?half-hw-2:half+2;
   out.push(`<div class="mvHd" style="left:${hx}px;top:${y0+rh-5-hw*.95}px;width:${hw}px;height:${hw*.95}px;z-index:${100+i};"><div style="-webkit-mask-image:linear-gradient(#000 80%,transparent);mask-image:linear-gradient(#000 80%,transparent)">${mvImg(e,hw,hw*.95,1,true,uid)}</div></div>`)})}
 return out.join('')}
function mvMobileB(A,B,W,H){let half=W/2,out=[],qw=Math.min(half*.82,150),qh=qw*.85,top0=H*.225;
 [[-1,A],[1,B]].forEach(([side,L])=>{let rest=L.slice(1),e0=L[0],cx=half+side*(half/2);
  if(e0){let uid=500+(side>0?50:0),lab=e0.p.pos;
   out.push(`<div class="mvHd" style="left:${cx-qw/2}px;top:${top0}px;width:${qw}px;height:${qh}px;z-index:100"><div style="-webkit-mask-image:linear-gradient(#000 78%,transparent);mask-image:linear-gradient(#000 78%,transparent)">${mvImg(e0,qw,qh,1,true,uid)}</div></div>`);
   out.push(`<div class="mvTx c" style="left:${cx-half/2+4}px;width:${half-8}px;top:${top0+qh-4}px;z-index:200"><b style="font-size:19px">${mvLast(e0.p)}</b><span><i>${lab}</i>${mvVals(e0)}</span></div>`)}
  let tw=(half-14)/2,th=Math.min(tw*.8,64),rowH=th+36,ty0=top0+qh+44;
  rest.forEach((e,i)=>{let col=i%2,row=Math.floor(i/2),x=(side<0?4:half+6)+col*(tw+4),y=ty0+row*rowH,uid=600+i+(side>0?50:0),lab=e.k==='SFLEX'?'SFX':e.k==='FLEX'?'FLX':e.k==='BN'?'BN':e.p.pos;
   out.push(`<div class="mvTile" style="left:${x}px;top:${y}px;width:${tw}px;height:${rowH-6}px"></div><div class="mvHd" style="left:${x+tw*.1}px;top:${y+4}px;width:${tw*.8}px;height:${th}px;z-index:100"><div style="-webkit-mask-image:linear-gradient(#000 78%,transparent);mask-image:linear-gradient(#000 78%,transparent)">${mvImg(e,tw*.8,th,1,true,uid)}</div></div><div class="mvTx c" style="left:${x}px;width:${tw}px;top:${y+th+2}px;z-index:200"><b style="font-size:12.5px">${mvLast(e.p)}</b><span style="font-size:10.5px"><i>${lab}</i>${mvVals(e)}</span></div>`)})});
 return out.join('')}
function mvGroup(S,k){let v=showM?S.gM[k]:S.gP[k];return v==null||isNaN(v)?'—':Number(v).toFixed(1)}
function mvMobileC(A,B,W,H){let half=W/2,top0=H*.235,bot=H*.875,rk={QB:0,RB:1,WR:2,TE:3,FLEX:4,SFLEX:5,BN:6},big=e=>e?(showM?(e.v.mania==null?-1:Number(e.v.mania)):Number(e.v.pts)):-9,
 g=L=>{let m={};L.forEach(e=>{(m[e.k]=m[e.k]||[]).push(e)});Object.values(m).forEach(a=>a.sort((x,y)=>big(y)-big(x)));return m},GA=g(A),GB=g(B),ks=[...new Set([...Object.keys(GA),...Object.keys(GB)])].sort((a,b)=>rk[a]-rk[b]),pairs=[];
 ks.forEach(k=>{let a=GA[k]||[],b=GB[k]||[];for(let i=0;i<Math.max(a.length,b.length);i++)pairs.push([k,a[i],b[i]])});
 let n=Math.max(pairs.length,1),rh=(bot-top0)/n,sc=Math.min(1.08,Math.max(.62,rh/78)),fh=Math.min(rh*1.7,half-72),tw=Math.max(82,half-fh-14),out=[],
 lab=k=>k==='SFLEX'?'SFX':k==='FLEX'?'FLX':k,
 bolt=(y)=>{let a=20,d=[[20,0],[20+a*.55,rh*.2],[20-a*.3,rh*.36],[20+a,rh*.52],[20-a*.8,rh*.7],[20+a*.35,rh*.84],[20,rh]].map(p=>p[0].toFixed(1)+','+(p[1]).toFixed(1)).join(' ');
  return `<svg width="40" height="${rh}" viewBox="0 0 40 ${rh}" style="position:absolute;left:${half-20}px;top:${y}px;z-index:600;overflow:visible;filter:drop-shadow(0 0 3px #ff2b3d) drop-shadow(0 0 9px #e63946)"><polyline points="${d}" fill="none" stroke="#ff3b4a" stroke-width="3.4" stroke-miterlimit="10"/><polyline points="${d}" fill="none" stroke="#ffd0d4" stroke-width="1"/></svg>`},
 vs=(y)=>{let z=24*sc+4;return `<div style="position:absolute;left:${half}px;top:${y+rh*.52}px;transform:translate(-50%,-50%) rotate(45deg);width:${z}px;height:${z}px;background:#000;border:2px solid #ff3b4a;box-shadow:0 0 10px #e63946,inset 0 0 6px rgba(230,57,70,.6);z-index:700"></div><div style="position:absolute;left:${half}px;top:${y+rh*.52}px;transform:translate(-50%,-50%);font-size:${z*.5}px;font-weight:800;font-style:italic;z-index:710;color:#fff;text-shadow:0 0 6px #e63946;line-height:1">VS</div>`};
 out.push(`<div style="position:absolute;left:0;right:0;top:${top0-14}px;height:${bot-top0+24}px;background:radial-gradient(ellipse 70% 55% at 50% 45%,rgba(150,10,25,.38),transparent 72%)"></div>`);
 pairs.forEach(([k,a,b],i)=>{let y=top0+i*rh,ba=big(a),bb=big(b),uid=i*2;
  [[-1,a],[1,b]].forEach(([side,e],j)=>{
   if(!e){out.push(`<div style="position:absolute;${side<0?'left':'right'}:12px;top:${y}px;height:${rh}px;width:${tw}px;display:flex;align-items:center;justify-content:${side<0?'flex-start':'flex-end'};color:#555;font-weight:800;font-style:italic;font-size:${22*sc}px">—</div>`);return}
   let mine=side<0?ba:bb,other=side<0?bb:ba,win=mine>=other,last=mvLast(e.p),nsz=(last.length<9?13.5:last.length<11?11:9.5)*sc,al=side<0?'left':'right',
    v1=showM?(e.v.mania==null?'—':Number(e.v.mania).toFixed(1)):Number(e.v.pts).toFixed(1),v2=showM&&showW?Number(e.v.pts).toFixed(1)+' pts':'',
    gl=win?'text-shadow:0 0 14px rgba(255,59,74,.95),0 0 4px rgba(255,59,74,.8);':'';
   out.push(`<div style="position:absolute;${al}:10px;top:${y}px;height:${rh}px;width:${tw}px;display:flex;flex-direction:column;justify-content:center;text-align:${al};z-index:300;line-height:1"><div style="font-size:${nsz}px;font-weight:800;font-style:italic;letter-spacing:.03em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${last}</div><div style="font-size:${28*sc}px;font-weight:800;font-style:italic;line-height:1.03;${gl}">${v1}</div><div style="font-size:${9*sc}px;letter-spacing:.22em;color:#ff3b4a;font-weight:700;margin-top:2px;white-space:nowrap">${lab(k)}${v2?`<span style="color:#aaa;letter-spacing:.04em;margin-left:4px">${v2}</span>`:''}</div></div>`);
   out.push(`<div style="position:absolute;${side<0?'right':'left'}:${half}px;bottom:${H-(y+rh-3)}px;width:${fh}px;z-index:200;filter:drop-shadow(0 0 5px rgba(255,59,74,.55))"><div style="-webkit-mask-image:linear-gradient(#000 74%,transparent);mask-image:linear-gradient(#000 74%,transparent)">${mvImg(e,fh,0,1,true,300+uid+j)}</div></div>`)});
  out.push(bolt(y)+vs(y))});
 return out.join('')}
function renderMV(){let ov=$('mv');if(!ov||!mvOn)return;let st=$('mvStage'),vw=innerWidth,vh=innerHeight,mob=vw<760,W,H;
 if(mob){W=vw;H=vh}else{W=Math.min(vw,vh*16/9);H=W*9/16}
 st.style.width=W+'px';st.style.height=H+'px';W_mv=W;st.classList.toggle('mob',mob);st.classList.toggle('fc',mob&&mvStyle==='C');
 let A=mvOrder(TM.A,teamScope),B=mvOrder(TM.B,teamScope),SA=tScore(TM.A,teamScope),SB=tScore(TM.B,teamScope),n=Math.max(A.length,B.length,1),fig=[];
 if(!mob){let w0=W*.215,gap0=W*.012,spread=W*.285,base=H*.86,rise=H*.37,fs=Math.max(13,W*.0128);
  [[-1,A],[1,B]].forEach(([side,L])=>L.forEach((e,i)=>{let t=n>1?i/(n-1):0,w=w0*(1-.34*t),cx=W/2+side*(gap0+w/2+t*spread*(1-.0)),yb=base-t*rise;fig.push(mvFig(e,i,n,cx,H-yb,w,0,side,false,fs))}))}
 else{fig.push(mvStyle==='C'?mvMobileC(A,B,W,H):mvStyle==='B'?mvMobileB(A,B,W,H):mvMobileA(A,B,W,H))}
 let bigA=showM?SA.mania:SA.pts,bigB=showM?SB.mania:SB.pts,subA=showM&&showW?SA.pts:null,subB=showM&&showW?SB.pts:null,lab=showM?'MANIA RATING':'THIS WEEK · PROJECTED';
 let nm=(t,al)=>`<div class="mvName" style="text-align:${al}">${(TM[t].name||(t==='A'?'My team':'Their team')).replace(/</g,'&lt;')}</div>`;
 let sc=(t,big,sub,al)=>`<div class="mvSc" style="text-align:${al}">${nm(t,al)}<div class="mvBig">${(big||0).toFixed(1)}</div><div class="mvLab">${lab}${sub!=null?`<br><b>${sub.toFixed(1)}</b> PROJ PTS`:''}</div></div>`;
 let rows=GKEYS.map(k=>[k,mvGroup(SA,k),mvGroup(SB,k)]).filter(r=>r[1]!=='—'||r[2]!=='—').map(r=>[r[0]==='FLEX'?'FLX':r[0]==='SFLEX'?'SFX':r[0],r[1],r[2]]);
 let pr=0.5;if(showW){let sd=Math.sqrt(SA.sd*SA.sd+SB.sd*SB.sd)||1;pr=.5*(1+erf((SA.pts-SB.pts)/(Math.SQRT2*sd)))}
 let d=bigA-bigB,edge=showW?Math.round(pr*100)+'%':(d>=0?'+':'')+d.toFixed(1);rows.push([showW?'WIN %':'EDGE',edge,'']);
 let strip=rows.map(r=>`<div><em>${r[0]}</em><b><u>${r[1]}</u>${r[2]!==''?`<s> / </s><u>${r[2]}</u>`:''}</b></div>`).join('');
 let scores=`<div class="mvTop${mob?'':' d'}">${sc('A',bigA,subA,mob?'center':'right')}${sc('B',bigB,subB,mob?'center':'left')}</div>`;
 let fcm=mob&&mvStyle==='C';st.innerHTML=`${fcm?'':'<div class="mvLine"></div>'}${fig.join('')}${scores}${fcm?'':'<div class="mvVS">VS</div><div class="mvVig"></div>'}<div class="mvBrand">FANTASY MANIA · MANIAC VIEW · WEEK ${META.next_week}</div><div class="mvStrip">${strip}</div>`;
 $('mvTog').innerHTML=`<button type="button" data-b="M" class="${showM?'on':''}">Mania Rating</button><button type="button" data-b="W" class="${showW?'on':''}">This week</button>`;
 $('mvTog').querySelectorAll('button').forEach(b=>b.onclick=()=>{let k=b.dataset.b;if(k==='M'){if(showM&&!showW)return;showM=!showM}else{if(showW&&!showM)return;showW=!showW}try{localStorage.setItem('fmBasis',JSON.stringify([showM,showW]))}catch(e){}renderMV()});
}
function toggleMV(on){mvOn=on;let ov=$('mv');ov.hidden=!on;document.body.classList.toggle('mvLock',on);if(on)renderMV();else{$('mvStage').innerHTML='';renderTeams()}}
addEventListener('resize',()=>{if(mvOn)renderMV()});
document.addEventListener('click',e=>{let b=e.target.closest&&e.target.closest('.tmania');if(b)toggleMV(true)});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&mvOn)toggleMV(false)});

/* ===== role-change banners (advisory only; never auto-applied) ===== */
function flagIdx(p,f){return (p.logs||[]).map((l,i)=>f.use.includes(l.w)?i:-1).filter(i=>i>=0)}
function flagBar(p,ctx,cur,attrs){let fl=(p.flags||[]),out='';if(p.pos==='QB'||!fl.length)return '';
 fl.slice(0,2).forEach((f,n)=>{let idx=flagIdx(p,f);if(!idx.length||idx.length===p.logs.length)return;let on=cur&&sameSet(cur,idx);
  out+=`<div class="rflag${on?' on':''}"><span class="rfi">!</span><div class="rft"><b>${f.tag}</b> ${f.txt}</div><button type="button" class="rfBtn" data-ctx="${ctx}" data-id="${p.id}" data-f="${n}" data-on="${on?1:0}" ${attrs||''}>${on?'Count all games again':f.btn}</button></div>`});return out}
function flagTag(p,cur){let f=(p.flags||[])[0];if(!f||p.pos==='QB')return '';let idx=flagIdx(p,f);if(!idx.length||idx.length===p.logs.length||(cur&&sameSet(cur,idx)))return '';return ` • <i class="ttag fl" title="${f.txt.replace(/"/g,'&quot;')}">${f.tag}</i>`}
document.addEventListener('click',e=>{let b=e.target.closest&&e.target.closest('.rfBtn');if(!b)return;let p=byId(b.dataset.id),f=p&&(p.flags||[])[+b.dataset.f];if(!f)return;let on=b.dataset.on==='1',idx=on?dfl(p):flagIdx(p,f),ctx=b.dataset.ctx;
 if(ctx==='prof'){document.querySelectorAll('.gameToggle,.gameToggleChip').forEach(x=>x.checked=idx.includes(Number(x.dataset.i)));let al=$('allGames');if(al)al.checked=idx.length===p.logs.length;customProfile(p);b.dataset.on=on?'0':'1';b.textContent=on?f.btn:'Count all games again';b.closest('.rflag').classList.toggle('on',!on)}
 else if(ctx==='cmp'){if(b.dataset.side==='A')selA=idx;else selB=idx;renderCompare()}
 else if(ctx==='team'){TM[b.dataset.t].sel[p.id]=idx;tGames[b.dataset.t+p.id]=true;saveTeams();renderTeams()}});
function toggleShot(on){document.body.classList.toggle('shot',on);shotMove(on);window.scrollTo(0,0)}
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
/* ===== Trade analyzer =====
   Each player gets a steady points-per-game level that blends three things: this week's projection with the matchup taken out
   (it already includes the Vegas lines and our model), his Mania Rating turned into points at his position, and his shrunk season average.
   Then we play out the rest of the season week by week: byes, injuries and our estimate of when he is back, and the extra work a
   player gets while a teammate is out (which fades when that teammate returns).
   Value = points above a replacement player in a league that size (a smooth version so nobody is exactly zero), scaled 0 to 100. */
const TRK='fmTrade2',TRPRI={QB:15,RB:8,WR:8,TE:6},TRLAST=17,TRPOS=['QB','RB','WR','TE'],TRW=[1,.85,.7,.55,.45,.4],TRO={QB:0,RB:1,WR:2,TE:3},TRIDEAL={QB:1,RB:4,WR:4,TE:1};
var TR={n:12,sf:false,hz:'ros',st:{QB:1,RB:2,WR:2,TE:1,FLEX:1},mode:'quick',give:[],get:[],my:[],their:[],send:{my:[],their:[]},rec:{my:[0,0],their:[0,0]}},TRC=null,TRopen={};
try{let s=JSON.parse(localStorage.getItem(TRK)||'null');if(s&&s.st&&s.send)TR=Object.assign(TR,s,{mode:'quick'})}catch(e){}
function trSave(){try{localStorage.setItem(TRK,JSON.stringify({n:TR.n,sf:TR.sf,hz:TR.hz,st:TR.st,give:TR.give,get:TR.get,my:TR.my,their:TR.their,send:TR.send,rec:TR.rec}))}catch(e){}}
const trSoft=x=>1.4*Math.log(1+Math.exp(x/1.4)),trN=v=>v==null||isNaN(v)?'—':Math.abs(v)>=10?String(Math.round(v)):(Math.round(v*10)/10).toFixed(1).replace(/\.0$/,'');
function trGameWeeks(team){let o=(META.opps||{})[team]||{},a=[];for(let w=META.next_week;w<=TRLAST;w++)if(o[w])a.push(w);return a}
function trBye(team){let o=(META.opps||{})[team]||{};for(let w=1;w<=18;w++)if(!o[w])return w;return 0}
function trPlan(p){let i=p.inj,lab=((i&&i.label)||''),l=lab.toLowerCase(),out=!!p.out||(i&&i.avail!=null&&i.avail<=.1),R=0,gone=false,lw=0,missed=0;(p.logs||[]).forEach(x=>{if(x.w>lw&&x.snap>0)lw=x.w});if(lw>0)missed=Math.max(0,(META.week||0)-lw);
 if(out){R=/injured/.test(l)?Math.max(1,5-missed):/doubtful/.test(l)?1:Math.max(1,2-missed);if(/released|retired|inactive|exempt/.test(l)){gone=true;R=99}}
 return {out,R,gone,label:lab,missed}}
function trScale(x){let P=[[0,0],[2,5],[5,20],[10,45],[14,65],[17,77],[20,88],[25,100]];if(x<=0)return 0;for(let i=1;i<P.length;i++)if(x<=P[i][0]){let a=P[i-1],b=P[i];return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])}return 100}
var TROUT=false;
function trL(p){return `<span class="tlink" data-open="${p.id}">${p.name}</span>`}
function trCtx(){let key=[TR.n,TR.sf?1:0,JSON.stringify(TR.st),META.next_week].join('|');if(TRC&&TRC.key===key)return TRC;
 let st=TR.st,F=st.FLEX,sf=TR.sf?1:0,V={};
 allPlayers().filter(p=>TRPRI[p.pos]!=null&&(p.pos!=='QB'||p.mania!=null)).forEach(p=>{let m=p.m||{},n=p.games||0,sh=(n*(+m.ppr||0)+3*TRPRI[p.pos])/(n+3),pr=p.proj,fx=(pr&&pr.fx)||{},ok=!!pr&&!p.out&&pr.pts!=null,
  pnNo=ok?pr.pts-.75*((fx.script||0)+(fx.matchup||0))-(fx.inj||0):null,boost=ok&&fx.inj>0?fx.inj:0;
  V[p.id]={p,sh,pnNo,boost,wk:ok?pr.pts:null,gw:trGameWeeks(p.team),bye:trBye(p.team),pl:trPlan(p)}});
 let vs=Object.values(V);
 TRPOS.forEach(pos=>{let L=vs.filter(v=>v.p.pos===pos&&v.p.mania!=null&&(v.p.games||0)>=2),pp=L.map(v=>v.sh).sort((a,b)=>b-a),mm=L.map(v=>v.p.mania).sort((a,b)=>b-a);
  vs.forEach(v=>{if(v.p.pos!==pos)return;if(!pp.length||v.p.mania==null||(v.p.games||0)<2){v.mp=null;return}let q=mm.filter(x=>x>v.p.mania).length/mm.length;v.mp=pp[Math.min(pp.length-1,Math.floor(q*pp.length))]})});
 vs.forEach(v=>{let parts=[[v.pnNo,.35],[v.mp,.30],[v.sh,.35]].filter(x=>x[0]!=null),ws=parts.reduce((s,x)=>s+x[1],0);v.base=Math.max(0,parts.reduce((s,x)=>s+x[0]*x[1],0)/ws)});
 let S={QB:st.QB+sf*.85,RB:st.RB+F*.45+sf*.07,WR:st.WR+F*.45+sf*.07,TE:st.TE+F*.10+sf*.01},repl={},avg={},start={},waiv={};
 TRPOS.forEach(pos=>{let a=vs.filter(v=>v.p.pos===pos&&!v.pl.gone).map(v=>v.base).sort((x,y)=>y-x),r=Math.max(1,Math.round(TR.n*S[pos]+TR.n*(pos==='QB'?.6:.25))),k=Math.max(1,Math.round(TR.n*S[pos]));
  repl[pos]=a.length?a[Math.min(r,a.length)-1]:0;waiv[pos]=a.length?a[Math.min(a.length,trRosterCut(pos,16))-1]:0;avg[pos]=a.slice(0,k).reduce((s,x)=>s+x,0)/Math.max(1,Math.min(k,a.length));start[pos]=repl[pos]+.45*(avg[pos]-repl[pos])});
 vs.forEach(v=>{v.cause=null;let p=v.p;if(v.boost>0&&p.inj_why){let best=null;p.inj_why.forEach(e=>{if(e.pts>0&&e.kind==='role'){let t=vs.find(x=>x.p.team===p.team&&x.p.name===e.n);if(t&&t.pl.R>0&&!t.pl.gone&&(!best||e.pts>best.pts))best={t,pts:e.pts}}});
  if(best){let t=best.t,ret=.5;if(t.p.pos===p.pos&&p.depth&&p.depth.rank>=2)ret=.25;if(v.base>=t.base*.95)ret=.8;v.cause={n:t.p.name,p:t.p,R:t.pl.R,retain:ret,retW:t.gw[t.pl.R]||99}}}});
 vs.forEach(v=>{let G=v.gw.length,R=Math.min(v.pl.R,G),arr=[],rp=repl[v.p.pos],wv=waiv[v.p.pos],qbb=(v.p.pos==='QB'&&!TR.sf)?1.3:1,pt=x=>x==null?0:trSoft(x-rp)+.35*trSoft(x-wv),av=v.p.inj&&v.p.inj.avail!=null?v.p.inj.avail:1;
  v.later=v.base+(v.boost>0?(v.cause?v.boost*v.cause.retain:v.boost*.6):0);
  for(let c=0;c<G;c++){let x=null;if(c>=R&&!v.pl.gone){x=v.base+(v.boost>0?(v.cause?(c<v.cause.R?v.boost:v.boost*v.cause.retain):v.boost*.6):0);if(R>0&&c===R)x*=.85;else if(R>0&&c===R+1)x*=.93;if(R===0&&c===0&&av<1)x*=av}arr.push(x)}
  v.arr=arr;v.G=G;v.retW=v.gw[R]||99;v.raw=qbb*arr.reduce((s,x)=>s+pt(x),0);if(v.pl.R>=4&&G<=6)v.raw*=.65;v.raw4=qbb*arr.slice(0,4).reduce((s,x)=>s+pt(x),0);
  let pl=arr.filter(x=>x!=null);v.avgRest=pl.length?pl.reduce((s,x)=>s+x,0)/pl.length:0;let f4=arr.slice(0,4).filter(x=>x!=null);v.nowAvg=f4.length?f4.reduce((s,x)=>s+x,0)/f4.length:null;
  v.nowP=(v.pl.R>0||v.gw[0]!==META.next_week||v.pl.gone)?null:(v.wk!=null?v.wk:(arr[0]!=null?arr[0]:null));
  let a0=arr.length?arr.map(x=>x==null?0:x):[],ros=a0.length?a0.reduce((s,x)=>s+x,0)/a0.length:0,thisW=v.pl.gone?0:(v.pl.R>0?0:(v.wk!=null?v.wk:(arr[0]!=null?arr[0]:v.later))),lvl=v.mp!=null?v.mp:v.base,eff=v.pl.gone?0:.35*lvl+.25*thisW+.40*ros;if(v.p.pos==='QB')eff-=TR.sf?1:2.5;v.eff=eff;v.outlook=trScale(eff)});
 vs.forEach(v=>{v.tv=Math.pow(v.raw,1.3);v.tv4=Math.pow(v.raw4,1.3)});let mx=Math.max(1,...vs.map(v=>v.tv));vs.forEach(v=>{v.val=trDisp(100*v.tv/mx);v.val4=trDisp(100*v.tv4/mx)});
 return TRC={key,V,repl,waiv,avg,start,S,mx,vs}}
function trWeekPpg(v,w){let i=v.gw.indexOf(w);return i<0?null:v.arr[i]}
function trPick(ent){let c=trCtx(),s=TR.st,pool=ent.slice().sort((a,b)=>b.ppg-a.ppg),used=new Set(),st=[];
 const take=(slot,ok,fill)=>{let e=pool.find(x=>!used.has(x.v.p.id)&&ok(x.v.p.pos));if(e){used.add(e.v.p.id);st.push({slot,v:e.v,ppg:e.ppg,empty:false})}else st.push({slot,v:null,ppg:fill,empty:true})};
 for(let i=0;i<s.QB;i++)take('QB',p=>p==='QB',c.repl.QB);for(let i=0;i<s.RB;i++)take('RB',p=>p==='RB',c.repl.RB);for(let i=0;i<s.WR;i++)take('WR',p=>p==='WR',c.repl.WR);for(let i=0;i<s.TE;i++)take('TE',p=>p==='TE',c.repl.TE);
 for(let i=0;i<s.FLEX;i++)take('FLEX',p=>p!=='QB',Math.min(c.repl.RB,c.repl.WR));if(TR.sf)take('SFLEX',p=>true,c.repl.QB);
 return {st,bench:pool.filter(x=>!used.has(x.v.p.id)),tot:st.reduce((a,x)=>a+x.ppg,0)}}
function trEnt(ids,fn){let c=trCtx();return ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone).map(v=>({v,ppg:fn(v)})).filter(e=>e.ppg!=null)}
function trLineup(ids,key){let c=trCtx(),L=trPick(trEnt(ids,v=>key==='now'?v.nowP:v.later));L.off=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone&&(key==='now'?v.nowP==null:false));L.gone=ids.map(i=>c.V[i]).filter(v=>v&&v.pl.gone);return L}
function trLineupW(ids,w){return trPick(trEnt(ids,v=>trWeekPpg(v,w)))}
function trTax(){let n=TR.n;return n<=8?.13:n<=10?.13-(n-8)*.025:n<=12?.08-(n-10)*.02:Math.max(.025,.04-(n-12)*.005)}
function trSide(ids,other){let V=trCtx().V,a=ids.map(i=>V[i]).filter(Boolean).sort((x,y)=>y.tv-x.tv),t=0,t4=0,it=a.map((v,i)=>{let w=TRW[Math.min(i,TRW.length-1)];t+=w*v.tv;t4+=w*v.tv4;return {v,w}}),ex=other==null?0:Math.max(0,a.length-other),f=1-Math.min(.4,trTax()*ex);return {it,tot:t*f,tot4:t4*f,tax:ex*trTax()}}
function trNeeds(ids,key){let c=trCtx(),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),fn=v=>key==='now'?v.nowAvg:v.later;
 return TRPOS.map(pos=>{let k=Math.max(1,Math.round(c.S[pos])),a=V.filter(v=>v.p.pos===pos&&fn(v)!=null).sort((x,y)=>fn(y)-fn(x)),top=a.slice(0,k),sum=top.reduce((s,v)=>s+fn(v),0)+(k-top.length)*c.repl[pos],mine=sum/k,lg=c.avg[pos],pct=lg?(mine-lg)/lg:0,
  ideal=pos==='QB'&&TR.sf?2:TRIDEAL[pos];return {pos,k,have:V.filter(v=>v.p.pos===pos).length,startable:a.filter(v=>fn(v)>=c.start[pos]).length,ideal,mine,lg,pct,grade:pct>=.07?'Strength':pct<=-.07?'Need':'Solid'}})}
function trByes(ids){let c=trCtx(),full=trLineup(ids,'later').tot,out=[];
 for(let w=META.next_week;w<=Math.min(TRLAST,META.next_week+9);w++){let L=trLineupW(ids,w),drop=full-L.tot,weak=L.st.filter(x=>x.empty||(x.v&&x.ppg<c.start[x.v.p.pos]*.9)).map(x=>x.slot==='FLEX'||x.slot==='SFLEX'?'flex':x.slot);
  if(drop>=3.5||L.st.some(x=>x.empty))out.push({w,drop,weak:[...new Set(weak)],byes:ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone&&v.bye===w&&v.later>=c.start[v.p.pos]).map(v=>v.p.name)})}
 return out.sort((a,b)=>b.drop-a.drop).slice(0,3).sort((a,b)=>a.w-b.w)}
function trCuffs(ids){let c=trCtx(),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),res={};
 V.forEach(v=>{let p=v.p;if(p.pos==='QB'||!p.depth)return;let mates=c.vs.filter(x=>x.p.team===p.team&&x.p.pos===p.pos&&x.p.id!==p.id&&x.p.depth&&!x.pl.gone),
  ahead=mates.filter(x=>x.p.depth.rank<p.depth.rank).sort((a,b)=>b.p.depth.rank-a.p.depth.rank)[0],behind=mates.filter(x=>x.p.depth.rank>p.depth.rank).sort((a,b)=>a.p.depth.rank-b.p.depth.rank)[0];
  res[p.id]={ahead,behind,hasAhead:!!ahead&&ids.includes(ahead.p.id),hasBehind:!!behind&&ids.includes(behind.p.id)}});return res}
function trRosterCut(pos,size){let n=TR.n,sz=Math.max(size||0,TR.st.QB+TR.st.RB+TR.st.WR+TR.st.TE+TR.st.FLEX+(TR.sf?1:0)+4,12),qb=TR.sf?2.5:1.5,te=1.5,rest=Math.max(2,sz-qb-te),per={QB:qb,TE:te,RB:rest*.46,WR:rest*.54};return Math.round(n*per[pos])}
function trWaiver(ids){let c=trCtx(),own=new Set(ids),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),L=trLineup(ids,'later'),startIds=new Set(L.st.filter(x=>x.v).map(x=>x.v.p.id)),cuffs=trCuffs(ids),out=[];
 let bench=V.filter(v=>!startIds.has(v.p.id)&&!(cuffs[v.p.id]&&cuffs[v.p.id].hasAhead)).sort((a,b)=>a.later-b.later).slice(0,2);
 bench.forEach(w=>{let pos=w.p.pos,cut=trRosterCut(pos,ids.length),ranked=c.vs.filter(x=>x.p.pos===pos&&!x.pl.gone).sort((a,b)=>b.base-a.base).map(x=>x.p.id),
  cand=c.vs.filter(x=>x.p.pos===pos&&!own.has(x.p.id)&&!x.pl.gone&&x.pl.R===0&&ranked.indexOf(x.p.id)>=cut&&x.later>=w.later+.4&&x.later<=w.later+4).sort((a,b)=>b.later-a.later).slice(0,3);
  if(cand.length)out.push({w,cand})});return out}
function trIdeas(ids,needPos){let c=trCtx(),own=new Set(ids),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),L0=trLineup(ids,'later'),startIds=new Set(L0.st.filter(x=>x.v).map(x=>x.v.p.id)),cuffs=trCuffs(ids),
  pool=V.filter(v=>!startIds.has(v.p.id)||v.later<c.start[v.p.pos]*1.02).filter(v=>v.raw>0).sort((a,b)=>b.raw-a.raw).slice(0,7),subs=[];
 pool.forEach((a,i)=>{subs.push([a]);pool.slice(i+1).forEach(b=>subs.push([a,b]))});
 let cand=c.vs.filter(x=>x.p.pos===needPos&&!own.has(x.p.id)&&!x.pl.gone&&x.pl.R<=1).sort((a,b)=>b.raw-a.raw).slice(0,70),res=[];
 subs.forEach(A=>{let ga=trSide(A.map(x=>x.p.id)).tot,send=new Set(A.map(x=>x.p.id));cand.forEach(T=>{let gt=T.tv,mx=Math.max(ga,gt);if(mx<=0||Math.abs(ga-gt)/mx>.12)return;
  let after=ids.filter(i=>!send.has(i)).concat([T.p.id]),L1=trLineup(after,'later'),d=L1.tot-L0.tot;if(d>=.4){let n1=trLineup(after,'now').tot-trLineup(ids,'now').tot;if(n1>=-.5)res.push({A,T,d,n1,gap:(gt-ga)/mx})}})});
 res=res.sort((a,b)=>b.d-a.d).slice(0,10).map(x=>{let snd=new Set(x.A.map(y=>y.p.id)),aft=ids.filter(i=>!snd.has(i)).concat([x.T.p.id]);x.d=trTVd(ids,aft,'ros');x.n1=trTVd(ids,aft,'now');return x}).filter(x=>x.d>=.3&&x.n1>=-.4);
 let seen=new Set(),seenA=new Set();return res.sort((a,b)=>b.d-a.d).filter(r=>{let k=r.T.p.id,ka=r.A.map(x=>x.p.id).sort().join();if(seen.has(k)||seenA.has(ka)||r.A.some(x=>seenA.has(x.p.id)))return false;seen.add(k);seenA.add(ka);r.A.forEach(x=>seenA.add(x.p.id));return true}).slice(0,3)}
function trDeals(my,their,relax){let c=trCtx(),pool=ids=>ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone).sort((a,b)=>b.raw-a.raw).slice(0,10),subs=P=>{let o=[];P.forEach((a,i)=>{o.push([a]);P.slice(i+1).forEach(b=>o.push([a,b]))});return o},PA=subs(pool(my)),PB=subs(pool(their)),
  A0=trLineup(my,'later').tot,B0=trLineup(their,'later').tot,An=trLineup(my,'now').tot,Bn=trLineup(their,'now').tot,res=[];
 PA.forEach(A=>{let ia=A.map(x=>x.p.id),ga=trSide(ia).tot,sa=new Set(ia);PB.forEach(B=>{let ib=B.map(x=>x.p.id),gb=trSide(ib).tot,mx=Math.max(ga,gb);if(mx<=0||Math.abs(ga-gb)/mx>.14)return;let sb=new Set(ib),
  la=trLineup(my.filter(i=>!sa.has(i)).concat(ib),'later').tot-A0,lb=trLineup(their.filter(i=>!sb.has(i)).concat(ia),'later').tot-B0;
  if(relax?(la>=.4&&lb>=-.5):(la>=.25&&lb>=.25)){let na=trLineup(my.filter(i=>!sa.has(i)).concat(ib),'now').tot-An,nb=trLineup(their.filter(i=>!sb.has(i)).concat(ia),'now').tot-Bn;res.push({A,B,la,lb,na,nb,score:2*Math.min(la,lb)+la+lb+.3*(Math.min(na,0)+Math.min(nb,0)),gap:(gb-ga)/mx})}})});
 res=res.sort((a,b)=>b.score-a.score).slice(0,16).map(d=>{let ia=d.A.map(x=>x.p.id),ib=d.B.map(x=>x.p.id),sa=new Set(ia),sb=new Set(ib),am=my.filter(i=>!sa.has(i)).concat(ib),bm=their.filter(i=>!sb.has(i)).concat(ia);
  d.la=trTVd(my,am,'ros');d.lb=trTVd(their,bm,'ros');d.na=trTVd(my,am,'now');d.nb=trTVd(their,bm,'now');d.score=2*Math.min(d.la,d.lb)+d.la+d.lb+.3*(Math.min(d.na,0)+Math.min(d.nb,0));return d}).filter(d=>relax?(d.la>=.3&&d.lb>=-.4):(d.la>=.2&&d.lb>=.2));
 let seen=new Set();return res.sort((a,b)=>b.score-a.score).filter(r=>{let k2=r.A[0].p.id;if(seen.has(k2))return false;seen.add(k2);return true}).slice(0,4)}
function trStacks(ids){let L=trLineup(ids,'later'),m={},o=[];L.st.forEach(x=>{if(x.v&&x.v.bye){let k=x.v.p.pos+'|'+x.v.bye;(m[k]=m[k]||[]).push(trL(x.v.p))}});Object.keys(m).forEach(k=>{if(m[k].length>=2)o.push({pos:k.split('|')[0],w:+k.split('|')[1],names:m[k]})});return o}
function trUpgrades(before,after){let a=trLineup(before,'later'),b=trLineup(after,'later'),up=0,dn=0;a.st.forEach((x,i)=>{let d=b.st[i].ppg-x.ppg;if(d>.4)up++;else if(d<-.4)dn++});return {up,dn}}
function trCount(ids,pos){let V=trCtx().V;return ids.filter(i=>V[i]&&V[i].p.pos===pos&&!V[i].pl.gone).length}

function trWeeks(kind){let a=[];for(let w=META.next_week;w<=TRLAST;w++)a.push(w);return kind==='now'?a.slice(0,4):kind==='play'?a.filter(w=>w>=14):a}
const TRMISS={QB:.07,RB:.15,WR:.09,TE:.09};var TRTV={},TRCVmap=null;
function trIns(ids,frac){let c=trCtx(),L=trLineup(ids,'later'),used=new Set(),tot=0,st=L.st.filter(x=>x.v).sort((a,b)=>TRMISS[b.v.p.pos]*b.v.G*b.ppg-TRMISS[a.v.p.pos]*a.v.G*a.ppg);
 st.forEach(x=>{let s=x.v,m=TRMISS[s.p.pos]*s.G,best=null,bu=0;L.bench.forEach(b=>{if(used.has(b.v.p.id))return;let ok=x.slot==='FLEX'?b.v.p.pos!=='QB':x.slot==='SFLEX'?true:b.v.p.pos===x.slot;if(!ok)return;let u=b.ppg,cuff=b.v.p.team===s.p.team&&b.v.p.pos===s.p.pos&&s.p.pos!=='QB';if(cuff)u=b.ppg+Math.max(0,s.later*.9-b.ppg)*.8;if(u>bu){bu=u;best=b;best.cuff=cuff}});
  if(best){used.add(best.v.p.id);tot+=m*Math.max(0,bu-c.repl[s.p.pos])*(best.cuff&&s.p.pos==='RB'?1.8:1)}});return tot*frac}
function trTV(ids,kind){let key=kind+'|'+ids.slice().sort().join(','),h=TRTV[key];if(h!==undefined)return h;let c=trCtx(),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),W=trWeeks(kind),tot=0;
 W.forEach(w=>{tot+=trPick(V.map(v=>({v,ppg:trWeekPpg(v,w)})).filter(e=>e.ppg!=null)).tot});tot+=trIns(ids,W.length/Math.max(1,trWeeks('ros').length));return TRTV[key]=tot}
function trTVd(before,after,kind){return (trTV(after,kind)-trTV(before,kind))/Math.max(1,trWeeks(kind).length)}
function trMarg(ids,id,kind){let k=kind||TR.hz,w=trWeeks('ros').length/Math.max(1,trWeeks(k).length),m=ids.includes(id)?trTV(ids,k)-trTV(ids.filter(x=>x!==id),k):trTV(ids.concat(id),k)-trTV(ids,k);return m*w}
function trStarN(v){return Math.round(5*Math.pow(Math.max(v,0)/100,.65)*2)/2}
function trStars(v){let n=trStarN(v);return `<span class="trStars" style="--s:${n}" title="${trN(v)} out of 100" aria-label="${n} of 5 stars">★★★★★</span>`}
function trStarTxt(v){return trStarN(v)+'★'}
function trDisp(x){return 100*Math.pow(Math.max(x,0)/100,.45)}
function trCV(m,id){let v=trCtx().V[id],g=v?v.raw:0;return trDisp(100*Math.pow(.45*g+.55*Math.max(m,0),1.3)/trCtx().mx)}
function trVals(ids,extra,who){let o={};ids.concat(extra||[]).forEach(i=>{if(trCtx().V[i]&&!trCtx().V[i].pl.gone)o[i]={val:trCV(trMarg(ids,i),i),who}});return o}
function trNeedStart(){return TR.st.QB+TR.st.RB+TR.st.WR+TR.st.TE+TR.st.FLEX+(TR.sf?1:0)}
function trRecNote(ids,side){let r=TR.rec[side]||[0,0],w=r[0],l=r[1],g=w+l;if(g<2)return '';let c=trCtx(),lg=TRPOS.reduce((s,k)=>s+c.avg[k]*c.S[k],0),d=trLineup(ids,'now').tot-lg,p=.5*(1+erf(d/31.1/Math.SQRT2)),rw=w/g,pct=Math.round(p*100);
 return rw-p>=.3?`Record ${w}-${l} looks better than this roster. On points alone it is a ${pct}% team each week, so some of the wins were luck. Fix the weak spots now instead of trusting the record.`:p-rw>=.3?`Record ${w}-${l} is worse than this roster deserves (about ${pct}% a week). It has been unlucky, so hold your best players and do not panic sell.`:`Record ${w}-${l} fits the roster (about ${pct}% a week).`}
function trD(k,title,sub,body,def,cls){let o=TRopen[k]!==undefined?TRopen[k]:def;return `<details class="trD${cls?' '+cls:''}" data-k="${k}"${o?' open':''}><summary><span>${title}</span><i>${sub||''}</i></summary><div class="trDB">${body}</div></details>`}
function trWarn(v){let o=[],p=v.p;
 if(v.pl.gone)o.push(`${v.pl.label||'Not on a roster'}. No value.`);
 else if(v.pl.R>0)o.push(`${v.pl.label||'Out'} now${v.pl.missed>0?` (already missed ${v.pl.missed} week${v.pl.missed>1?'s':''})`:''}. Back around ${v.retW<=TRLAST?'Week '+v.retW:'next season'} (our estimate). Does not help until then.`);
 if(!v.pl.gone&&v.cause&&v.cause.R<v.G&&v.boost>=.8)o.push(`Getting extra work while ${trL(v.cause.p)} is out: about ${trN(v.base+v.boost)} a game now, ${trN(v.later)} after he returns (~Week ${v.cause.retW}).`);
 if(!v.pl.gone&&v.pl.R===0&&v.bye>=META.next_week&&v.bye<=META.next_week+1)o.push(`Bye in Week ${v.bye}.`);
 return o.map(t=>`<small class="trWarn">${t}</small>`).join('')}
function trPpgTxt(v){if(v.pl.gone)return 'not rostered';if(v.pl.R>0)return `${trN(v.later)} ppg when healthy`;let n=v.nowAvg;return n!=null&&Math.abs(n-v.later)>=1.2?`${trN(n)} ppg now → ${trN(v.later)} later`:`${trN(v.later)} ppg`}
function trRow(id,btns,note,tag){let v=trCtx().V[id];if(!v)return '';let cv=TROUT||TRSHOWF?null:(TRCVmap&&TRCVmap[id]),sh=TRSHOWF?TRSHOWF(v):null;let p=v.p,t=v.pl.label&&v.pl.R>0?` <i class="trTag bad">${v.pl.label}</i>`:p.inj&&p.inj.label&&p.inj.avail<1?` <i class="trTag bad">${p.inj.label}</i>`:'';
 return `<div class="trRow"><span class="trTh">${thumb(p)}</span><div class="trNm"><b class="tlink" data-open="${p.id}">${p.name}</b><small>${p.team} • ${p.pos} • ${trPpgTxt(v)}${t}${note?' • '+note:''}</small>${tag||''}${trWarn(v)}</div><div class="trVal" title="${sh?'Shown for this view':cv?'How much the other team would value him, 0 to 100':TROUT?'Outlook rating: Mania Rating, this week and the rest of the season combined':'Trade value in a typical team, 0 to 100'}">${sh?`<b>${sh[0]}</b><small>${sh[1]}</small>`:TROUT?`<b>${trN(v.outlook)}</b><small>outlook</small>`:`${trStars(cv?cv.val:v.val)}<small>${cv?cv.who:'value'}</small>`}</div>${btns||''}</div>`}
function trAttach(root,sel,pick){let inp=root.querySelector(sel+' input'),dd=root.querySelector(sel+' .dd');if(!inp)return;inp.oninput=()=>{let s=inp.value.toLowerCase().trim();if(!s){dd.style.display='none';return}let c=trCtx().V,m=Object.values(c).map(v=>v.p).filter(p=>p.name.toLowerCase().includes(s)).sort((a,b)=>c[b.id].raw-c[a.id].raw).slice(0,8);
 dd.innerHTML=m.map(p=>`<div class="ddi" data-id="${p.id}"><div><span class="ddiName">${p.name}</span><span class="ddiMeta">${p.team} • ${p.pos}</span></div><span class="ddiRate">${trStarTxt(c[p.id].val)}</span></div>`).join('')||'<div class="ddi muted">No match</div>';dd.style.display='block';dd.querySelectorAll('.ddi[data-id]').forEach(x=>x.onclick=()=>{dd.style.display='none';inp.value='';pick(x.dataset.id)})}}
function trAdd(list,id){if(!list.includes(id))list.push(id)}
function trSettings(){let n=TR.n,s=TR.st,steps=[['Teams in league','n',6,20],['RB','RB',0,4],['WR','WR',0,4],['TE','TE',0,3],['Flex (RB/WR/TE)','FLEX',0,3]];
 return `<details class="tset trSetD"><summary><span>League</span><i>${n} teams • PPR • ${TR.sf?'Superflex':'1 QB'} • ${TR.hz==='now'?'win now':TR.hz==='play'?'playoffs':'full season'} • ${s.RB}RB ${s.WR}WR ${s.TE}TE ${s.FLEX}FLEX</i></summary><div class="tsetB"><div class="tpre"><button type="button" class="tsb trsf${TR.sf?'':' on'}" data-v="0">1 QB</button><button type="button" class="tsb trsf${TR.sf?' on':''}" data-v="1">Superflex</button></div><div class="tsec" style="margin:6px 2px">What matters most</div><div class="tpre">${[['now','Next 4 weeks'],['ros','Whole season'],['play','Playoffs (Wk 14-17)']].map(([k,l])=>`<button type="button" class="tsb trhz${TR.hz===k?' on':''}" data-v="${k}">${l}</button>`).join('')}</div><div class="tsteps">${steps.map(([l,k,lo,hi])=>`<div class="tst"><span>${l}</span><div><button type="button" class="tstb trst" data-k="${k}" data-d="-1" data-lo="${lo}" data-hi="${hi}">−</button><b>${k==='n'?n:s[k]}</b><button type="button" class="tstb trst" data-k="${k}" data-d="1" data-lo="${lo}" data-hi="${hi}">+</button></div></div>`).join('')}</div><p class="tnote" style="margin:8px 0 0">Scoring is PPR. A bigger league leaves fewer good players on waivers, so every starter is worth more. Superflex makes quarterbacks much more valuable. We assume your bench is just the players you enter.</p></div></details>`}
function trSearchBox(id,ph){return `<div class="search trS" id="${id}"><input placeholder="${ph}" autocomplete="off"><div class="dd"></div></div>`}
function trSlotLab(x){return x==='SFLEX'?'SFX':x==='FLEX'?'FLX':x}
function trLineHTMLm(L,mark,map){TRCVmap=map;let h=trLineHTML(L,mark);TRCVmap=null;return h}
function trLineHTML(L,mark){return L.st.map(x=>x.v?trRow(x.v.p.id,'',mark&&mark.has(x.v.p.id)?'<b class="trNew">NEW</b>':'').replace('<div class="trNm">',`<div class="trNm"><span class="trSlot">${trSlotLab(x.slot)}</span>`):`<div class="trRow"><span class="trTh"></span><div class="trNm"><span class="trSlot">${trSlotLab(x.slot)}</span><b>Empty slot</b><small>Counted as a free agent (${trN(x.ppg)} ppg)</small></div></div>`).join('')}
function trTier(p){return p>=.2?['Elite',4]:p>=.07?['Strong',3]:p>-.07?['Solid',2]:p>-.2?['Subpar',1]:['Weak',0]}
function trNeedsDiffHTML(b,a){let n0=trNeeds(b,'later'),n1=trNeeds(a,'later');return `<div class="trNeeds">${n0.map((x,i)=>{let y=n1[i],t0=trTier(x.pct),t1=trTier(y.pct),d=t1[1]-t0[1],cls=d>0?'up':d<0?'dn':'ok';return `<div class="trNd ${cls}"><b>${x.pos}</b><i class="trGr ${cls}">${d>0?'▲ ':d<0?'▼ ':''}${d?`${t0[0]} → ${t1[0]}`:`${t1[0]} (no change)`}</i><small>Starter level ${trN(x.mine)} → ${trN(y.mine)} ppg (league starter ${trN(y.lg)})</small><small>${x.startable} → ${y.startable} startable${y.startable<y.ideal?` (aim for ${y.ideal})`:' ✓'}</small></div>`}).join('')}</div>`}
function trNeedsHTML(nd,ndNow){return `<div class="trNeeds">${nd.map((n,i)=>{let nw=ndNow&&ndNow[i],short=nw&&nw.grade==='Need'&&n.grade!=='Need';return `<div class="trNd ${n.grade==='Strength'?'up':n.grade==='Need'?'dn':'ok'}"><b>${n.pos}</b><i class="trGr ${n.grade==='Strength'?'up':n.grade==='Need'?'dn':'ok'}">${n.grade}</i><small>${trN(n.mine)} ppg vs ${trN(n.lg)} for a league starter</small><small>${n.startable} startable${n.startable<n.ideal?` (aim for ${n.ideal})`:' ✓'}${short?' • thin right now only, injured starters return':''}</small></div>`}).join('')}</div>`}
function trRecInput(side){let r=TR.rec[side]||[0,0];return `<span class="trRec">Record <input type="number" min="0" max="17" inputmode="numeric" class="trRi" data-side="${side}" data-i="0" value="${r[0]}"> – <input type="number" min="0" max="17" inputmode="numeric" class="trRi" data-side="${side}" data-i="1" value="${r[1]}"></span>`}
/* ---------- Quick trade ---------- */
function trQuick(){let h=trQuickRaw();TRCVmap=null;return h}
function trQuickRaw(){TRCVmap=(TR.my.length>=trNeedStart()&&TR.give.length&&TR.give.every(i=>TR.my.includes(i)))?trVals(TR.my,TR.get,'to you'):null;let G=trSide(TR.give,TR.get.length),R=trSide(TR.get,TR.give.length),have=G.it.length&&R.it.length,list=(S,side)=>S.it.map(x=>trRow(x.v.p.id,`<button type="button" class="tx trDel" data-side="${side}" data-id="${x.v.p.id}" title="Remove">✕</button>`,x.w<1?`counts ${Math.round(x.w*100)}%`:'')).join('')||'<div class="trEmpty">Search for a player to add.</div>',out='';
 if(have){let a=R.tot,b=G.tot,mx=Math.max(a,b,1),gap=(a-b)/mx,ag=Math.abs(gap),a4=R.tot4,b4=G.tot4,m4=Math.max(a4,b4,1),g4=(a4-b4)/m4,
  lab=ag<.08?'Fair trade':gap>0?(ag>=.25?'Big win for you':'You win this trade'):(ag>=.25?'Big loss for you':'You lose this trade'),cls=ag<.08?'even':gap>0?'win':'lose',hints=[],
  sh=g=>Math.abs(g)<.08?'about even':g>0?`you win by ${Math.round(Math.abs(g)*100)}%`:`you lose by ${Math.round(Math.abs(g)*100)}%`;
  if(Math.abs(g4-gap)>=.2||(Math.abs(g4)>=.08&&Math.abs(gap)>=.08&&g4*gap<0))hints.push(g4>gap?'This trade looks better right now than for the full season. Injured players coming back or fading roles change the picture later.':'This trade looks better for the full season than for the next month. You may wait on an injured player.');
  if(G.it.length>R.it.length)hints.push(`You send ${G.it.length} for ${R.it.length}, so you open ${G.it.length-R.it.length} roster spot${G.it.length-R.it.length>1?'s':''}. Make sure the extra spot gets a useful pickup.`);
  if(R.it.length>G.it.length)hints.push(`You get ${R.it.length} for ${G.it.length}, so you will have to drop someone. The extra players only help if they can start for you.`);
  hints.push(R.it[0].v.tv>G.it[0].v.tv?'You are getting the best player in the deal, which usually matters most.':'You are giving up the best player in the deal.');
  R.it.concat(G.it).forEach(x=>{let w=trWarn(x.v);if(w&&(x.v.pl.R>0||x.v.cause))hints.push(`<b>${trL(x.v.p)}</b>: ${x.v.pl.R>0?`${x.v.pl.label||'out'}, back ~Week ${x.v.retW}.`:`extra work now, ${trN(x.v.later)} a game after ${trL(x.v.cause.p)} returns (~Week ${x.v.cause.retW}).`}`)});
  let fix='';if(ag>=.12){let need=Math.abs(b-a)/.7,used=new Set(TR.give.concat(TR.get)),cs=trCtx().vs.filter(v=>!used.has(v.p.id)&&!v.pl.gone&&v.pl.R<=1&&v.tv>=need*.6&&v.tv<=need*1.4).sort((x,y)=>Math.abs(x.tv-need)-Math.abs(y.tv-need)).slice(0,4);
   if(cs.length)fix=`<div class="trFix"><b>${gap<0?'To even it out, ask for':'They would likely want'} about one more player worth about ${trStarTxt(trDisp(100*need/trCtx().mx))}:</b> ${cs.map(v=>`<span>${trL(v.p)} <i>${v.p.pos} ${trStarTxt(v.val)}</i></span>`).join('')}</div>`}
  let onT='';if(TR.my.length>=6&&TR.give.every(i=>TR.my.includes(i))){let aft=TR.my.filter(i=>!TR.give.includes(i)).concat(TR.get),la=trTVd(TR.my,aft,'ros'),na=trTVd(TR.my,aft,'now'),pa=trTVd(TR.my,aft,'play'),u=trUpgrades(TR.my,aft),nt=[],s0=new Set(trStacks(TR.my).map(x=>x.pos+x.w));
   nt.push(`You upgrade <b>${u.up}</b> starting spot${u.up===1?'':'s'}${u.dn?` and downgrade ${u.dn}`:''}. For your team, in points a week (byes, injuries and bench cover included): <b class="${na>=0?'up':'dn'}">${trSgn(na)}</b> over the next 4 weeks, <b class="${la>=0?'up':'dn'}">${trSgn(la)}</b> over the season${trWeeks('play').length?`, <b class="${pa>=0?'up':'dn'}">${trSgn(pa)}</b> in the playoffs`:''}.${na-la>.6?' It helps more now than later.':la-na>.6?' It helps more later than now.':''}`);
   trStacks(aft).filter(x=>!s0.has(x.pos+x.w)).forEach(x=>nt.push(`${x.names.join(' and ')} (${x.pos}) would start on the same Week ${x.w} bye.`));['RB','WR'].forEach(pos=>{let c0=trCount(TR.my,pos),c1=trCount(aft,pos),k=Math.round(trCtx().S[pos]);if(c1<=k&&c1<c0)nt.push(`You would have only ${c1} ${pos}${c1>1?'s':''} rostered, so one injury puts a free agent in the lineup.`)});
   onT=trD('qteam','On your team','from your roster in My team analyzer',nt.map(h=>`<p class="trLead" style="margin:0 0 8px">${h}</p>`).join(''),true)}
  else if(!TR.my.length)onT='<p class="tnote" style="text-align:center">Add your roster in the Players tab (My team analyzer) and this will also show how the trade changes your lineup, byes and depth.</p>';
  hints.push('');hints.pop();
  out=`<div class="trVerdict ${cls}"><div class="trvK">Verdict</div><strong>${lab}</strong><div class="trvS">Rest of season: ${sh(gap)}. Next 4 weeks: ${sh(g4)}.</div><div class="trBar"><div class="trbG" style="width:${b/(a+b||1)*100}%"></div><div class="trbR" style="width:${a/(a+b||1)*100}%"></div></div><div class="trBarL"><span>You send <b>${trN(100*b/trCtx().mx)}</b></span><span>You get <b>${trN(100*a/trCtx().mx)}</b></span></div>${fix}<ul class="trHints">${hints.map(h=>`<li>${h}</li>`).join('')}</ul></div>`+onT}
 else out=`<div class="trVerdict even"><div class="trvK">Verdict</div><strong>Add players to both sides</strong><div class="trvS">Pick who you would send and who you would get. Try the example to see how it reads.</div></div>`;
 return `<div class="trCols"><div class="trCard"><div class="ph">You send</div>${trSearchBox('trQg','Search a player to send')}${list(G,'give')}</div><div class="trCard"><div class="ph">You get</div>${trSearchBox('trQr','Search a player to get')}${list(R,'get')}</div></div>${out}<div class="trFoot"><button type="button" class="reset" id="trEx">Try an example</button><button type="button" class="reset" id="trClr">Clear</button></div><p class="tnote">Value is points above a replacement player in a ${TR.n}-team ${TR.sf?'superflex':'1-QB'} PPR league, for the rest of the season. It blends Mania Rating, this week's projection and our estimate for later, and counts byes, injuries and returns. For a side with several players the best counts in full and each next one a little less, since only so many can start.</p>`}
/* ---------- Rosters ---------- */
function trRoster(side,title,withSend){let ids0=TR[side],oth=side==='my'?TR.their:TR.my;TRCVmap=(oth.length>=trNeedStart()&&ids0.length)?trValsFor(oth,ids0,side==='my'?'to them':'to you'):null;let h=trRosterRaw(side,title,withSend);TRCVmap=null;return h}
function trRosterRaw(side,title,withSend){let ids=TR[side],V=trCtx().V,sorted=ids.map(i=>V[i]).filter(Boolean).sort((a,b)=>(a.pl.R>0)-(b.pl.R>0)||TRO[a.p.pos]-TRO[b.p.pos]||b.raw-a.raw),sd=TR.send[side],need=TR.st.QB+TR.st.RB+TR.st.WR+TR.st.TE+TR.st.FLEX+(TR.sf?1:0);
 let rows=sorted.map(v=>{let on=sd.includes(v.p.id);return trRow(v.p.id,`${withSend?`<button type="button" class="trSend${on?' on':''}" data-side="${side}" data-id="${v.p.id}">${on?'Sending':'Trade'}</button>`:''}<button type="button" class="tx trDelR" data-side="${side}" data-id="${v.p.id}" title="Remove">✕</button>`)}).join('');
 let healthy=sorted.filter(v=>!(v.pl.R>0)).length,bn=Math.max(0,healthy-need),irN=sorted.length-healthy;
 return `<div class="trCard trRoster"><div class="ph">${title} <span class="trCount">${sorted.length} players</span></div>${side==='my'||withSend?`<div class="trRecRow">${trRecInput(side)}</div>`:''}${trSearchBox('trR'+side,'Add a player to this roster')}${rows?trD('rl'+side,'Players','show or hide',rows,!(withSend&&(TR.send.my.length||TR.send.their.length)),'trRL'):''}${rows?'':'<div class="trEmpty">Add every player on the roster, injured ones too. We pick the best lineup and treat injured players as IR.</div>'}${sorted.length?`<div class="trMeta">${need} starters • ${bn} bench${irN?` • ${irN} injured`:''}. We assume the bench is what you entered${healthy<need+3?', so add your full bench for better results':''}.</div><div class="trFoot l"><button type="button" class="reset trClrR" data-side="${side}">Clear roster</button></div>`:''}</div>`}
const trSgn=v=>(v>=0?'+':'')+trN(v);
function trDealsHTML(){let ds=trDeals(TR.my,TR.their),rx=false;if(!ds.length){ds=trDeals(TR.my,TR.their,true);rx=!!ds.length}if(!ds.length)return '<p class="tnote">No clean win-win deal found between these two rosters. Try tapping Trade on a few players to test your own idea.</p>';
 return (rx?'<p class="tnote">No deal helps both lineups. These are fair on value and help you most while barely costing them, so they would mostly be accepted on value alone.</p>':'')+ds.map((d,i)=>`<div class="trDeal"><div class="trDealT"><span><b>You send</b> ${d.A.map(x=>`${trL(x.p)} <i>${x.p.pos}</i>`).join(' + ')}</span><span><b>You get</b> ${d.B.map(x=>`${trL(x.p)} <i>${x.p.pos}</i>`).join(' + ')}</span></div><div class="trDealM">Your lineup <b class="${d.la>=0?'up':'dn'}">${trSgn(d.la)}</b> a week at full strength • theirs <b class="${d.lb>=0?'up':'dn'}">${trSgn(d.lb)}</b> • value ${Math.abs(d.gap)<.06?'even':d.gap>0?'slightly in your favor':'slightly in theirs'}</div><button type="button" class="trAsk trUse" data-i="${i}">Test this trade</button></div>`).join('')}
var TRV={lv:'now',lm:'proj'},TRSHOWF=null,TRSUM='',TRSUG=false;
function trValsFor(owner,ids,who){let V=trCtx().V,o={};ids.forEach(i=>{if(V[i]&&!V[i].pl.gone)o[i]={val:trCV(trMarg(owner,i),i),who}});return o}
function trShowF(){let lv=TRV.lv,lm=TRV.lm;return v=>{if(lm==='mania')return [v.p.mania==null?'—':Math.round(v.p.mania),'mania'];if(lm==='out')return [Math.round(v.outlook),'outlook'];let x=lv==='now'?v.nowP:v.later;return [x==null?'—':trN(x),lv==='now'?'this week':'full strength']}}
function trLnKnobs(){let b=(k,v,t)=>`<button type="button" class="tsb trk${TRV[k]===v?' on':''}" data-k="${k}" data-v="${v}">${t}</button>`;return `<div class="trKn"><div class="tpre">${b('lv','now','This week')}${b('lv','later','Full strength')}</div><div class="tpre">${b('lm','proj','Projection')}${b('lm','mania','Mania Rating')}${b('lm','out','Outlook')}</div></div>`}
function trLnBlock(title,ids,mark){TRSHOWF=trShowF();let L=trLineup(ids,TRV.lv),c=trCtx(),set=new Set(L.st.filter(x=>x.v).map(x=>x.v.p.id)),bn=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone&&!set.has(v.p.id)).sort((a,b)=>(a.pl.R>0)-(b.pl.R>0)||b.outlook-a.outlook),
 h=`<div class="trLnT"><b>${title}</b><span>${trN(L.tot)} projected points ${TRV.lv==='now'?'this week':'at full strength'}</span></div><div class="trLn"><div><div class="tsec">Starters</div>${trLineHTML(L,mark)}</div><div><div class="tsec">Bench</div>${bn.length?bn.map(v=>trRow(v.p.id,'',v.pl.R>0?'':(TRV.lv==='now'&&v.nowP==null?'Bye week':''),mark&&mark.has(v.p.id)?'<b class="trNew">NEW</b>':'')).join(''):'<div class="trEmpty">No bench players</div>'}</div></div>`;TRSHOWF=null;return h}
function trLevel(d){return d>=1?['Gets better','good']:d>.25?['Gets a bit better','good']:d>=-.25?['Stays about the same','ok']:d>-1?['Gets a bit worse','bad']:['Gets worse','bad']}
function trSeason(b,a,inc,title){let wk=trWeeks('ros'),w0=wk.map(w=>trLineupW(b,w).tot),w1=wk.map(w=>trLineupW(a,w).tot),d=w1.map((x,i)=>x-w0[i]),t0=w0.reduce((s,x)=>s+x,0),t1=w1.reduce((s,x)=>s+x,0),dt=t1-t0,up=d.filter(x=>x>.05).length,dn=d.filter(x=>x<-.05).length,mxa=Math.max(1,...d.map(Math.abs)),W=wk.length*30,
 bars=d.map((x,i)=>{let h=Math.abs(x)/mxa*32,y=x>=0?50-h:50;return `<rect x="${i*30+6}" y="${y.toFixed(1)}" width="18" height="${Math.max(h,1).toFixed(1)}" rx="3" fill="${x>.05?'#22a45d':x<-.05?'#e2505b':'#c9d1de'}"><title>Week ${wk[i]}: ${trSgn(x)}</title></rect><text x="${i*30+15}" y="${x>=0?50-h-3:50+h+10}" font-size="8.5" text-anchor="middle" fill="#5a6b86">${Math.abs(x)>=.5?trSgn(x).replace(/\.0$/,''):''}</text><text x="${i*30+15}" y="104" font-size="9" text-anchor="middle" fill="#8a99b1">${wk[i]}</text>`}).join(''),
 ex=inc.filter(v=>v.pl.R>0&&wk.includes(v.retW)).map(v=>`${trL(v.p)} is back around Week ${v.retW}: your lineup gains <b>${trSgn(d[wk.indexOf(v.retW)])}</b> that week.`),
 ex2=[];let best=d.indexOf(Math.max(...d)),worst=d.indexOf(Math.min(...d));if(Math.max(...d)>.5)ex2.push(`Biggest boost: Week ${wk[best]} (${trSgn(d[best])}).`);if(Math.min(...d)<-.5)ex2.push(`Biggest dip: Week ${wk[worst]} (${trSgn(d[worst])}).`);
 return `<div><div class="tsec">${title}</div><div class="trSeasonT"><b class="${dt>=.05?'up':dt<=-.05?'dn':''}">${trSgn(dt)}</b><span>starting-lineup points over the next ${wk.length} weeks (${trN(t0)} → ${trN(t1)}), about <b>${trSgn(dt/Math.max(1,wk.length))}</b> a week</span></div><svg class="trChart" viewBox="0 0 ${W} 108" preserveAspectRatio="xMidYMid meet"><line x1="0" x2="${W}" y1="50" y2="50" stroke="#dfe6f1"/>${bars}</svg><div class="trSBl">Better in <b>${up}</b> of ${wk.length} weeks${dn?`, worse in <b>${dn}</b>`:''}. ${ex2.join(' ')}</div>${ex.length?`<ul class="trHints">${ex.map(h=>`<li>${h}</li>`).join('')}</ul>`:''}</div>`}
function trSideBox(name,d,na,la,pa,up,notes,l0,l1,nowLater){let lv=trLevel(d),cell=(t,x)=>`<div><small>${t}</small><b class="${x>=.05?'up':x<=-.05?'dn':''}">${trSgn(x)}</b></div>`;
 return `<div class="trSB ${lv[1]}"><div class="trSBh"><span>${name}</span><strong>${lv[0]}</strong></div><div class="trSBg">${cell('Next 4 weeks',na)}${cell('Whole season',la)}${trWeeks('play').length?cell('Playoffs',pa):''}</div><div class="trSBl">Points per week. Lineup this week: <b>${trN(l0)} → ${trN(l1)}</b>. Starting spots: <b>${up.up}</b> better${up.dn?`, ${up.dn} worse`:''}.</div>${nowLater?`<p class="trSBn">${nowLater}</p>`:''}<ul class="trHints">${notes.map(h=>`<li>${h}</li>`).join('')}</ul></div>`}
function trFull(){let a=trRoster('my','Your team',true),b=trRoster('their','Their team',true),res='';
 if(TR.my.length&&TR.their.length){let sa=TR.send.my.filter(i=>TR.my.includes(i)),sb=TR.send.their.filter(i=>TR.their.includes(i));
  if(!sa.length&&!sb.length){let nA=trNeeds(TR.my),nB=trNeeds(TR.their),sug=TRSUG?trDealsHTML():'';
   res=`<div class="trSug"><div><b>Not sure what to offer?</b><span>We look at both rosters, find what each team needs, and suggest an even trade that helps both.</span></div><button type="button" class="trAsk" id="trSug">${TRSUG?'Refresh suggestion':'Suggest a fair trade'}</button></div>`+(TRSUG?`<div class="trCard trSugBox"><div class="ph">Suggested trades</div>${sug}</div>`:'')+`<div class="trVerdict even"><div class="trvK">Trade check</div><strong>Pick the players who would move</strong><div class="trvS">Tap Trade on players from either team to see how it changes both lineups.</div></div>`+trD('fneeds','Where the two teams stand','',`<div class="trCols"><div><div class="tsec">You</div>${trNeedsHTML(nA,trNeeds(TR.my,'now'))}<p class="trLead">${trRecNote(TR.my,'my')}</p></div><div><div class="tsec">Them</div>${trNeedsHTML(nB,trNeeds(TR.their,'now'))}<p class="trLead">${trRecNote(TR.their,'their')}</p></div></div>`,true)}
  else{let myAfter=TR.my.filter(i=>!sa.includes(i)).concat(sb),thAfter=TR.their.filter(i=>!sb.includes(i)).concat(sa),
   A0n=trLineup(TR.my,'now'),A1n=trLineup(myAfter,'now'),B0n=trLineup(TR.their,'now'),B1n=trLineup(thAfter,'now'),
   na=trTVd(TR.my,myAfter,'now'),nb=trTVd(TR.their,thAfter,'now'),la=trTVd(TR.my,myAfter,'ros'),lb=trTVd(TR.their,thAfter,'ros'),pa=trTVd(TR.my,myAfter,'play'),pb=trTVd(TR.their,thAfter,'play'),E=.25,wN=TR.hz==='now'?.85:.3,sm=TR.hz==='play'?pa:wN*na+(1-wN)*la,st=TR.hz==='play'?pb:wN*nb+(1-wN)*lb,G=trSide(sa,sb.length),R=trSide(sb,sa.length),mx=Math.max(R.tot,G.tot,1),vgap=(R.tot-G.tot)/mx,
   gap=Math.abs(sm-st),mnD=Math.min(sm,st),lean=sm>st?'you':'them',cs=[];
   let ov,ovT,ovC;
   if(sm>E&&st>E){ov='Win-win trade';ovC='win';ovT='Both teams get better.'}
   else if(sm<=-E&&st<=-E){ov='Hurts both teams';ovC='lose';ovT='Neither lineup improves. Look for a different deal.'}
   else if(mnD>=-E){if(gap<=.5){ov='Fair trade';ovC='even';ovT='Works for both sides. Nobody is getting taken.'}else if(gap<=1.2){ov='Fair, leans toward '+lean;ovC='even';ovT=`Neither team gets worse, but ${lean==='you'?'you gain':'they gain'} a little more.`}else{ov='Favors '+lean;ovC=lean==='you'?'win':'lose';ovT=`Neither team gets worse, but ${lean==='you'?'you gain':'they gain'} a lot more than ${lean==='you'?'they do':'you do'}.`}}
   else{let hurt=sm<st?'you':'they',hd=Math.abs(mnD);if(hd<.7){ov='Close, tilts toward '+lean;ovC='even';ovT=`${hurt==='you'?'You give':'They give'} up a little, so it is close to fair but not perfectly even.`}else{ov='Favors '+lean;ovC=lean==='you'?'win':'lose';ovT=`${hurt==='you'?'You get':'They get'} worse here, so ${hurt==='you'?'you should ask for more':'they will probably want more'}.`}}
   const vOf=i=>trCtx().V[i],nd0=(ids,k)=>trNeeds(ids,k),add=(s,t)=>cs.push([s,t]);
   [[sb,'You','my'],[sa,'They','their']].forEach(([inc,who,side])=>{let ids=TR[side],n0=nd0(ids,'now'),n1=nd0(ids,'later');inc.map(vOf).filter(Boolean).forEach(v=>{
     if(v.pl.R>0)add(side,`${who==='You'?'You get':'They get'} <b>${trL(v.p)}</b>, who is ${v.pl.label||'out'} (back ~Week ${v.retW}). He will not help ${who==='You'?'you':'them'} until then.`);
     else if(v.cause&&v.boost>=.8)add(side,`<b>${trL(v.p)}</b> gets extra work while ${trL(v.cause.p)} is out. Expect about ${trN(v.base+v.boost)} a game now but ${trN(v.later)} after ~Week ${v.cause.retW}.`);
     let i=TRPOS.indexOf(v.p.pos);if(n0[i].grade==='Need'&&n1[i].grade!=='Need')add(side,`${v.p.pos} is only thin right now because injured starters come back, so <b>${trL(v.p)}</b> matters more this month than later.`)})});
   [[myAfter,TR.my,'You','my'],[thAfter,TR.their,'They','their']].forEach(([after,before,who,side])=>{let b0=new Set(trByes(before).map(x=>x.w)),nb2=trByes(after).filter(x=>!b0.has(x.w)&&x.weak.length);if(nb2.length)add(side,`Gets thin at ${nb2[0].weak.join(' and ')} in Week ${nb2[0].w} (byes).`);
     let s0=new Set(trStacks(before).map(x=>x.pos+x.w));trStacks(after).filter(x=>!s0.has(x.pos+x.w)).forEach(x=>add(side,`${x.names.join(' and ')} (${x.pos}) would start on the same Week ${x.w} bye.`));
     ['RB','WR'].forEach(pos=>{let c0=trCount(before,pos),c1=trCount(after,pos),k=Math.round(trCtx().S[pos]);if(c1<=k&&c1<c0)add(side,`Only ${c1} ${pos}${c1>1?'s':''} rostered after this, so one injury puts a free agent in the lineup.`)});
     if(after.length>before.length)add(side,`Ends up with ${after.length-before.length} more player${after.length-before.length>1?'s':''} and has to drop someone. In a ${TR.n}-team league that costs about ${Math.round(trTax()*100)}% of value per spot, already counted.`)});
   [['my'],['their']].forEach(([s])=>{let n=trRecNote(TR[s],s);if(n)add(s,n)});
   const nowLater=(d0,d1,who)=>d0-d1>.6?`This trade helps ${who} more right now than later, so it suits a push for the next few weeks.`:d1-d0>.6?`This trade pays off for ${who} later more than right now.`:'',
    ua=trUpgrades(TR.my,myAfter),ub=trUpgrades(TR.their,thAfter),mark=ids=>new Set(ids),nA=trNeeds(myAfter),nB=trNeeds(thAfter),
    names=ids=>ids.map(i=>vOf(i).p.name).join(' + ');
   TRSUM=`Fantasy Mania trade check (${TR.n}-team, PPR)\nTeam A sends: ${names(sa)}\nTeam B sends: ${names(sb)}\nResult: ${ov}. ${ovT}\nTeam A: ${trLevel(sm)[0]} (${trSgn(na)} pts/wk next 4 weeks, ${trSgn(la)} whole season)\nTeam B: ${trLevel(st)[0]} (${trSgn(nb)} pts/wk next 4 weeks, ${trSgn(lb)} whole season)`;
   res=`<div class="trVerdict ${ovC}"><div class="trvK">Trade check</div><strong>${ov}</strong><div class="trvS">${ovT}</div><div class="trvS" style="margin-top:6px;font-size:13px;color:var(--muted)">Raw value for the rest of the season: ${Math.abs(vgap)<.08?'about even':vgap>0?`you come out ahead by ${Math.round(vgap*100)}%`:`they come out ahead by ${Math.round(-vgap*100)}%`}.</div><div class="trFoot" style="margin-top:12px"><button type="button" class="reset" id="trCopy">Copy summary to share</button></div></div>`
   +`<div class="trCols trSplit">${trSideBox('Your team',sm,na,la,pa,ua,cs.filter(x=>x[0]==='my').map(x=>x[1]),A0n.tot,A1n.tot,nowLater(na,la,'you'))}${trSideBox('Their team',st,nb,lb,pb,ub,cs.filter(x=>x[0]==='their').map(x=>x[1]),B0n.tot,B1n.tot,nowLater(nb,lb,'them'))}</div>`
   +trD('fseason','Rest-of-season outlook','built week by week',`<div class="trCols">${trSeason(TR.my,myAfter,sb.map(vOf).filter(Boolean),'Your team')}${trSeason(TR.their,thAfter,sa.map(vOf).filter(Boolean),'Their team')}</div><p class="tnote">Each week we build the best starting lineup from whoever is available: bye weeks are skipped, injured players are added back in the week we expect them to return (IR counts the weeks already missed), and extra work for a backup fades when the starter returns. Bars show how much this trade changes that week's lineup.</p>`,true)
   +trD('flineups','Lineups after the trade',`you ${trN(A0n.tot)} → ${trN(A1n.tot)} • they ${trN(B0n.tot)} → ${trN(B1n.tot)} this week`,`${trLnKnobs()}<div class="trLnW">${trLnBlock('Your team',myAfter,mark(sb))}${trLnBlock('Their team',thAfter,mark(sa))}</div>`,true)
   +trD('fneeds2','Needs and depth: before → after','what each position becomes',`<div class="trCols"><div><div class="tsec">You</div>${trNeedsDiffHTML(TR.my,myAfter)}</div><div><div class="tsec">Them</div>${trNeedsDiffHTML(TR.their,thAfter)}</div></div><p class="tnote">Grades compare the starters at each position with the average starter in a ${TR.n}-team league, at full strength (healthy players, byes ignored).</p>`,true)
   +trD('fdeals','Other deals that help both teams','tap to test',trDealsHTML(),false)}}
 else res=`<div class="trVerdict even"><div class="trvK">Trade check</div><strong>Enter both teams</strong><div class="trvS">Add every player on each roster, injured players too. Then you can ask for a suggested fair trade or tap Trade on the players who would move.</div></div>`;
 return `<div class="trCols">${a}${b}</div>${res}<div class="trFoot"><button type="button" class="reset" id="trSamp2">Try a sample</button><button type="button" class="reset" id="trClrAll">Clear all</button></div>`}
/* ---------- My team (lives in the Players tab) ---------- */
function trIdeasHTML(ids,nd){let order=nd.slice().sort((a,b)=>a.pct-b.pct),pos=order.filter(n=>n.grade==='Need').map(n=>n.pos);if(!pos.length)pos=[order[0].pos];let c=trCtx(),h='';
 pos.slice(0,2).forEach(pp=>{let ds=trIdeas(ids,pp);h+=`<div class="tsec">${pp} help</div>`+(ds.length?ds.map(d=>`<div class="trDeal"><div class="trDealT"><span><b>Send</b> ${d.A.map(x=>`${trL(x.p)} <i>${x.p.pos}</i>`).join(' + ')}</span><span><b>Get</b> ${trL(d.T.p)} <i>${d.T.p.pos} ${trStarTxt(d.T.val)}</i>${d.T.pl.R>0?' <i class="trTag bad">'+(d.T.pl.label||'Out')+'</i>':''}</span></div><div class="trDealM">Fair on value (${Math.abs(d.gap)<.04?'even':d.gap>0?'slightly in your favor':'slightly in theirs'}) • lineup <b class="up">${trSgn(d.d)}</b> a week at full strength${Math.abs(d.n1)>=.2?` • ${trSgn(d.n1)} this week`:''}</div><button type="button" class="trAsk trIdea" data-a="${d.A.map(x=>x.p.id).join(',')}" data-t="${d.T.p.id}">Test in Quick trade</button></div>`).join(''):'<p class="tnote">No fair package from your spare players clearly helps here. Look at the waiver ideas, or pair a starter you can spare.</p>')});return h}
function trCuffHTML(ids){let c=trCtx(),cf=trCuffs(ids),L=trLineup(ids,'later'),sIds=new Set(L.st.filter(x=>x.v).map(x=>x.v.p.id)),V=ids.map(i=>c.V[i]).filter(v=>v&&!v.pl.gone),ins=[],noBk=[],spare=[],qbs=V.filter(v=>v.p.pos==='QB').sort((a,b)=>b.later-a.later);
 V.forEach(v=>{let k=cf[v.p.id]||{},isS=sIds.has(v.p.id);
  if(v.p.pos==='QB'){if(isS)return;let rk=c.vs.filter(x=>x.p.pos==='QB'&&!x.pl.gone&&x.base>v.base).length+1;if((!TR.sf&&TR.n<=10&&rk>6)||qbs.indexOf(v)>=(TR.sf?2:1)+1)spare.push([v,TR.sf?'Extra quarterback.':'A backup QB is mostly wasted in a league this size unless he is a top-6 QB.']);else ins.push([v,'Bye and injury insurance at QB.']);return}
  if(v.p.pos==='TE'&&!isS&&TR.n<=10){let rk=c.vs.filter(x=>x.p.pos==='TE'&&!x.pl.gone&&x.base>v.base).length+1;if(rk>6){spare.push([v,'A backup TE is wasted in a league this size unless he is a top-6 TE.']);return}}
  if(isS){if(v.p.pos==='RB'&&k.behind){if(k.hasBehind)ins.push([k.behind,`Backup to your starter ${trL(v.p)}.`]);else noBk.push([v,`${k.behind.p.name} is his backup if you want insurance.`])}return}
  if(k.ahead&&k.hasAhead&&sIds.has(k.ahead.p.id)){if(!ins.some(x=>x[0].p.id===v.p.id))ins.push([v,k.ahead.p.depth.rank===1&&k.ahead.later>=c.start.RB*1.15?`Premium handcuff: ${trL(k.ahead.p)} is a workhorse, so he becomes a starter if ${k.ahead.p.name.split(' ')[0]} gets hurt.`:`Handcuff for ${trL(k.ahead.p)}, who starts for you.`]);return}
  if(v.later>=c.start[v.p.pos])return;spare.push([v,k.ahead&&!k.hasAhead?`Backup to ${trL(k.ahead.p)}, who is not on your team.`:'Not likely to start for you.'])});
 let sec=(t,a,e)=>a.length?`<div class="tsec">${t}</div>`+a.map(([v,n])=>trRow(v.p.id,'',n)).join(''):'';
 return (sec('Insurance you already hold',ins)+sec('Starters without a backup',noBk)+sec('Spare parts you could move or cut',spare))||'<p class="tnote">Nothing to flag.</p>'}
function trMine(){TROUT=true;let h;try{h=trMineRaw()}finally{TROUT=false}TRCVmap=null;return h}
function trMineRaw(){let ids=TR.my,out=trRoster('my','Your roster',false);TRCVmap=ids.length>=trNeedStart()?trVals(ids,[],'to you'):null;
 if(!ids.length)return `<div class="trCols one">${out}</div><div class="trFoot"><button type="button" class="reset" id="trSamp">Try a sample roster</button></div>`;
 let c=trCtx(),Ln=trLineup(ids,'now'),Ll=trLineup(ids,'later'),nd=trNeeds(ids),ndN=trNeeds(ids,'now'),byes=trByes(ids),wv=trWaiver(ids),V=ids.map(i=>c.V[i]).filter(Boolean),inj=V.filter(v=>v.pl.R>0&&!v.pl.gone),gone=V.filter(v=>v.pl.gone),
  needs=nd.filter(n=>n.grade==='Need'),str=nd.filter(n=>n.grade==='Strength'),thin=nd.filter(n=>n.startable<n.ideal),rn=trRecNote(ids,'my'),bul=[];
 bul.push(needs.length?`Biggest need${needs.length>1?'s':''}: <b>${needs.map(n=>n.pos).join(', ')}</b>.`:'No position is clearly weak.');if(str.length)bul.push(`Strongest: <b>${str.map(n=>n.pos).join(', ')}</b>.`);
 if(thin.length)bul.push(`Depth: ${thin.map(n=>`${n.pos} ${n.startable} of ${n.ideal} startable`).join(', ')}. Strong teams carry about 4 startable RBs and 4 startable WRs, so injuries and byes do not sink the week.`);
 if(inj.length)bul.push(`Injured: ${inj.map(v=>`${trL(v.p)} (back ~Week ${v.retW<=TRLAST?v.retW:'—'})`).join(', ')}. They are on your IR for now.`);
 if(byes.length)bul.push(`Bye-week watch: ${byes.map(b=>`Week ${b.w}`).join(', ')}.`);if(rn)bul.push(rn);
 let offAll=V.filter(v=>!v.pl.gone&&v.nowP==null),flagged=offAll.length?`<div class="tsec">Not playing this week</div>${offAll.map(v=>trRow(v.p.id,'',v.pl.R>0?'':'Bye week')).join('')}`:'';
 let analysis=`<div class="trVerdict even left"><div class="trvK">Your team at a glance</div><div class="trChips">${nd.map(n=>`<span class="trChip ${n.grade==='Strength'?'up':n.grade==='Need'?'dn':'ok'}"><b>${n.pos}</b> ${n.grade}</span>`).join('')}</div><ul class="trHints">${bul.map(h=>`<li>${h}</li>`).join('')}</ul></div>`
 +trD('mnow','Best lineup this week',`${trN(Ln.tot)} projected points`,trLineHTML(Ln)+flagged+(Ln.bench.length?`<div class="tsec">Bench</div>${Ln.bench.map(x=>trRow(x.v.p.id,'')).join('')}`:''),true)
 +trD('mlater','Best lineup overall (everyone healthy)',`${trN(Ll.tot)} points a week`,trLineHTML(Ll),false)
 +(inj.length||gone.length?trD('minj','Injured and IR',`${inj.length+gone.length} player${inj.length+gone.length>1?'s':''}`,inj.map(v=>trRow(v.p.id,'')).join('')+gone.map(v=>trRow(v.p.id,'')).join('')+'<p class="tnote">Return weeks are our estimates from the injury status. They count as IR now and as starters again when they are back.</p>',true):'')
 +trD('mneeds','Needs and depth',needs.length?`Need: ${needs.map(n=>n.pos).join(', ')}`:'',trNeedsHTML(nd,ndN)+'<p class="tnote">Each position is compared with the average starter at that spot in a '+TR.n+'-team league. "Startable" means he scores at least a decent starter level.</p>',true)
 +(byes.length?trD('mbye','Bye weeks and future holes','',byes.map(b=>`<div class="trBye"><b>Week ${b.w}</b> costs you about ${trN(b.drop)} points${b.byes.length?` (bye: ${b.byes.join(', ')})`:''}${b.weak.length?`. Thin at ${b.weak.join(', ')}, so look for help there before then.`:'.'}</div>`).join(''),true):'')
 +trD('mcuff','Handcuffs and spare parts','',trCuffHTML(ids),false)
 +trD('mwaiv','Possible waiver upgrades',wv.length?'we cannot see your free agents':'',wv.length?wv.map(x=>`<div class="tsec">Instead of ${trL(x.w.p)}</div>`+x.cand.map(v=>trRow(v.p.id,'',`+${trN(v.later-x.w.later)} ppg`)).join('')).join('')+'<p class="tnote">These rank outside what most leagues roster. Check that they are actually free in your league.</p>':'<p class="tnote">Nobody just outside the rostered group clearly beats your weakest bench players.</p>',false)
 +trD('mideas','Realistic trade ideas','fair value, spare players',trIdeasHTML(ids,nd),true);
 return `<div class="trCols trMine"><div>${out}</div><div>${analysis}</div></div>`}
/* ---------- render ---------- */
function trBind(root){const rm=(arr,id)=>{let i=arr.indexOf(id);if(i>=0)arr.splice(i,1)};
 root.querySelectorAll('.trhz').forEach(b=>b.onclick=()=>{TR.hz=b.dataset.v;trSave();trRenderAll()});
 root.querySelectorAll('.trsf').forEach(b=>b.onclick=()=>{TR.sf=b.dataset.v==='1';trSave();trRenderAll()});
 root.querySelectorAll('.trst').forEach(b=>b.onclick=()=>{let k=b.dataset.k,cur=k==='n'?TR.n:TR.st[k],v=Math.max(+b.dataset.lo,Math.min(+b.dataset.hi,cur+(+b.dataset.d)));if(v===cur)return;if(k==='n')TR.n=v;else TR.st[k]=v;trSave();trRenderAll()});
 trAttach(root,'#trQg',id=>{rm(TR.get,id);trAdd(TR.give,id);trSave();trRenderAll()});trAttach(root,'#trQr',id=>{rm(TR.give,id);trAdd(TR.get,id);trSave();trRenderAll()});
 trAttach(root,'#trRmy',id=>{trAdd(TR.my,id);rm(TR.their,id);trSave();trRenderAll()});trAttach(root,'#trRtheir',id=>{trAdd(TR.their,id);rm(TR.my,id);trSave();trRenderAll()});
 root.querySelectorAll('.trDel').forEach(b=>b.onclick=()=>{rm(b.dataset.side==='give'?TR.give:TR.get,b.dataset.id);trSave();trRenderAll()});
 root.querySelectorAll('.trDelR').forEach(b=>b.onclick=()=>{let s=b.dataset.side;rm(TR[s],b.dataset.id);rm(TR.send[s],b.dataset.id);trSave();trRenderAll()});
 root.querySelectorAll('.trSend').forEach(b=>b.onclick=()=>{let a=TR.send[b.dataset.side];if(a.includes(b.dataset.id))rm(a,b.dataset.id);else a.push(b.dataset.id);trSave();trRenderAll()});
 root.querySelectorAll('.trClrR').forEach(b=>b.onclick=()=>{TR[b.dataset.side]=[];TR.send[b.dataset.side]=[];trSave();trRenderAll()});
 root.querySelectorAll('.trRi').forEach(i=>i.onchange=()=>{TR.rec[i.dataset.side][+i.dataset.i]=Math.max(0,Math.min(17,Math.round(+i.value||0)));trSave();trRenderAll()});
 root.querySelectorAll('.trk').forEach(b=>b.onclick=()=>{TRV[b.dataset.k]=b.dataset.v;TRopen.flineups=true;trRenderAll()});
 let eS=root.querySelector('#trSug');if(eS)eS.onclick=()=>{TRSUG=true;trRenderAll()};
 let eC=root.querySelector('#trCopy');if(eC)eC.onclick=()=>{let t=TRSUM,done=()=>{eC.textContent='Copied!';setTimeout(()=>eC.textContent='Copy summary to share',1500)};try{navigator.clipboard.writeText(t).then(done,()=>{window.prompt('Copy this:',t)})}catch(x){window.prompt('Copy this:',t)}};
 root.querySelectorAll('.trD').forEach(d=>d.ontoggle=()=>{TRopen[d.dataset.k]=d.open});
 root.querySelectorAll('.trUse').forEach(b=>b.onclick=()=>{let d0=trDeals(TR.my,TR.their),d=(d0.length?d0:trDeals(TR.my,TR.their,true))[+b.dataset.i];if(!d)return;TR.send.my=d.A.map(x=>x.p.id);TR.send.their=d.B.map(x=>x.p.id);trSave();trRenderAll();let v=document.querySelector('#trBody .trVerdict');if(v)v.scrollIntoView({behavior:'smooth',block:'center'})});
 root.querySelectorAll('.trIdea').forEach(b=>{b.onclick=()=>{TR.give=b.dataset.a.split(',');TR.get=[b.dataset.t];trSave();TR.mode='quick';openTrade('quick')}});
 let e=root.querySelector('#trEx');if(e)e.onclick=()=>{let f=n=>{let q=allPlayers().find(p=>p.name===n);return q&&q.id},ids=['Patrick Mahomes','Lamar Jackson','Cam Skattebo'].map(f);if(ids.every(Boolean)){TR.give=[ids[0]];TR.get=[ids[1],ids[2]]}else{let c=trCtx().vs,top=(pos,n)=>c.filter(v=>v.p.pos===pos).sort((a,b)=>b.raw-a.raw).slice(n,n+1).map(v=>v.p.id)[0];TR.give=[top('QB',2)];TR.get=[top('QB',4),top('RB',9)]}trSave();trRenderAll()};
 e=root.querySelector('#trClr');if(e)e.onclick=()=>{TR.give=[];TR.get=[];trSave();trRenderAll()};
 let samp=()=>{let c=trCtx().vs,pools={QB:5,RB:15,WR:15,TE:6},A=[],B=[];TRPOS.forEach(pos=>{c.filter(v=>v.p.pos===pos&&!v.pl.gone).sort((a,b)=>b.raw-a.raw).slice(0,pools[pos]).forEach((v,i)=>((i+(pos==='RB'||pos==='TE'?1:0))%2?B:A).push(v.p.id))});TR.my=A;TR.their=B;TR.send={my:[],their:[]};trSave();trRenderAll()};
 e=root.querySelector('#trSamp');if(e)e.onclick=()=>{let c=trCtx().vs,pick=(pos,rs)=>{let a=c.filter(v=>v.p.pos===pos&&!v.pl.gone).sort((x,y)=>y.raw-x.raw);return rs.map(r=>a[r]&&a[r].p.id).filter(Boolean)};TR.my=[...pick('QB',[5,16]),...pick('RB',[3,11,19,27,38]),...pick('WR',[2,9,17,25,33,44]),...pick('TE',[4,13])];TR.send={my:[],their:[]};trSave();trRenderAll()};
 e=root.querySelector('#trSamp2');if(e)e.onclick=samp;e=root.querySelector('#trClrAll');if(e)e.onclick=()=>{TR.my=[];TR.their=[];TR.send={my:[],their:[]};TRSUG=false;trSave();trRenderAll()}}
function trRenderAll(){TRC=null;TRTV={};TRCVmap=null;let set=$('trSet'),body=$('trBody');
 if(set&&body){set.innerHTML=trSettings();body.innerHTML=TR.mode==='full'?trFull():trQuick();document.querySelectorAll('#trModes .tsb').forEach(b=>b.classList.toggle('on',b.dataset.v===TR.mode));trBind(set);trBind(body)}
 let ms=$('plMineSet'),mb=$('plMineBody');if(ms&&mb&&!$('plMine').hidden){ms.innerHTML=trSettings();mb.innerHTML=trMine();trBind(ms);trBind(mb)}}
var renderTrade=trRenderAll;
document.querySelectorAll('#trModes .tsb').forEach(b=>b.onclick=()=>{TR.mode=b.dataset.v;trRenderAll()});
document.querySelectorAll('#plModes .tsb').forEach(b=>b.onclick=()=>{TR.tool=b.dataset.v;document.querySelectorAll('#plModes .tsb').forEach(x=>x.classList.toggle('on',x===b));$('plFind').hidden=TR.tool!=='find';$('plMine').hidden=TR.tool!=='mine';trRenderAll()});
function openTrade(mode,give){if(mode)TR.mode=mode;if(give){TR.get=TR.get.filter(i=>i!==give);trAdd(TR.give,give);trSave()}showView('trade');trRenderAll()}
document.addEventListener('click',e=>{if(e.target.id==='profTrade'&&curPlayer)openTrade('quick',curPlayer.id)});
document.querySelectorAll('.navb[data-v=trade]').forEach(b=>b.addEventListener('click',()=>trRenderAll()));

</script></body></html>'''

html = html.replace("__PAYLOAD__", payload).replace("__META__", meta).replace("__REFS__", refs_json)
os.makedirs("public", exist_ok=True)
with open("public/index.html","w",encoding="utf-8") as f:f.write(html)
print("7. FANTASY MANIA v3 built successfully.")
print(f"   {len(players)} players | Through Week {max_week} | Week {next_week} outlook")
print("   Output: public/index.html")
