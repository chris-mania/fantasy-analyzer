"""
Fantasy Mania projection engine.

Projects PPR points for a week from: recent + last-season usage, team scoring environment (Vegas spread/total),
opponent strength, the injury report (teammates out -> bigger slice of the pool, QB out), and the player's own status.
Everything is a linear (ridge) model trained on 2022 -> now, so every number can be explained.

Used two ways:
  project(...)            -> called by build.py for the upcoming week
  python proj_engine.py backtest 2025 3 18   -> honest week-by-week test (train only on the past)
"""
import sys, re
import numpy as np, pandas as pd

POS = ["RB", "WR", "TE"]
ALPHA = 0.40          # recency weight of the running average
STARTER_DECAY = 0.3   # how fast older games fade when deciding who the starting QB is
K_LY = 2.0            # how many games of last-season prior a player starts with
RIDGE = 30.0
COMP = {"receptions": 1.0, "receiving_yards": 0.1, "rushing_yards": 0.1, "tds": 6.0}
STAT0 = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "tds", "ppr", "offense_pct", "tgt_share"]

R4 = ["m3_opp", "l2_opp", "m3_snap", "l2_snap", "shift_opp", "shift_snap", "shift_x_vac_c", "shift_x_vac_t"]
R3 = ["r3_targets", "r3_carries", "r3_ppr", "r3_offense_pct", "r3_receptions", "r3_rushing_yards", "r3_receiving_yards"]
BASE_FEATS = ["b_targets", "b_carries", "b_receptions", "b_receiving_yards", "b_rushing_yards", "b_tds", "b_ppr",
              "b_offense_pct", "b_tgt_share", "n_prior", "ly_n", "pos_rb", "pos_wr", "pos_te", "tm_tgt", "tm_car",
              "spread", "total_line", "implied", "opp_implied", "is_home", "own_avail", "qb_out", "vac_t", "vac_c",
              "vac_t_WR", "vac_t_TE", "vac_t_RB", "vac_c_RB", "ypt", "cr", "ypc", "opp_def", "exp_tgt_team", "rest"] + R3 + R4
INTER = ["spr_rb", "spr_wr", "spr_te", "imp_rb", "imp_wr", "imp_te", "imp_x_tgt", "imp_x_car", "spr_x_car", "spr_x_tgt"]
BOOST = ["boost_t", "boost_c", "bt_x_tgt", "bt_x_share", "bc_x_car", "bc_x_ppr", "bt_x_ppr"]
FEATS = BASE_FEATS + INTER + BOOST
GROUPS = {   # for "why" explanations
    "usage": ["m3_opp", "l2_opp", "m3_snap", "l2_snap", "shift_opp", "shift_snap", "r3_targets", "r3_carries", "r3_ppr", "r3_offense_pct", "r3_receptions", "r3_rushing_yards", "r3_receiving_yards", "b_targets", "b_carries", "b_receptions", "b_receiving_yards", "b_rushing_yards", "b_tds", "b_ppr", "b_offense_pct",
              "b_tgt_share", "n_prior", "ly_n", "pos_rb", "pos_wr", "pos_te", "ypt", "cr", "ypc", "exp_tgt_team"],
    "script": ["tm_tgt", "tm_car", "spread", "total_line", "implied", "opp_implied", "is_home", "rest"] + INTER,
    "matchup": ["opp_def"],
    "injuries": ["shift_x_vac_c", "shift_x_vac_t", "own_avail", "qb_out", "vac_t", "vac_c", "vac_t_WR", "vac_t_TE", "vac_t_RB", "vac_c_RB"] + BOOST,
}
# neutral values used to measure how much each group moves a projection
NEUTRAL = {"spread": 0.0, "total_line": 45.0, "implied": 22.5, "opp_implied": 22.5, "is_home": 0.5, "rest": 7.0, "opp_def": 0.0,
           "own_avail": 1.0, "qb_out": 0.0, "vac_t": 0.0, "vac_c": 0.0, "vac_t_WR": 0.0, "vac_t_TE": 0.0, "vac_t_RB": 0.0, "vac_c_RB": 0.0,
           "boost_t": 0.0, "boost_c": 0.0, "bt_x_tgt": 0.0, "bt_x_share": 0.0, "bc_x_car": 0.0, "bc_x_ppr": 0.0, "bt_x_ppr": 0.0}


def avail_weight(status, practice):
    """Chance a player plays, measured on 2022-2025 injury reports."""
    st = str(status).strip().lower() if pd.notna(status) else ""
    pr = str(practice).strip().lower() if pd.notna(practice) else ""
    if st == "out":
        return 0.0
    if st == "doubtful":
        return 0.02
    if st == "questionable":
        if "did not" in pr:
            return 0.45
        if "limited" in pr:
            return 0.63
        return 0.70
    if "did not" in pr:
        return 0.70
    if "limited" in pr:
        return 0.91
    return 1.0


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------------ data
class Loader:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def read(self, folder, name):
        last = None
        for ext in ("", ".gz"):
            try:
                return pd.read_csv(f"{self.base}/{folder}/{name}{ext}", low_memory=False)
            except Exception as e:
                last = e
        raise last


def load_all(L, seasons):
    pl = L.read("players", "players.csv")[["gsis_id", "pfr_id"]].dropna()
    pfr2g = dict(zip(pl.pfr_id, pl.gsis_id))
    frames, inj_frames = [], []
    for y in seasons:
        try:
            s = L.read("stats_player", f"stats_player_week_{y}.csv")
        except Exception:
            continue
        s = s[s["season_type"].astype(str).eq("REG")].copy()
        try:
            sn = L.read("snap_counts", f"snap_counts_{y}.csv")
            sn = sn[sn["game_type"].astype(str).eq("REG")].copy()
            sn["player_id"] = sn["pfr_player_id"].map(pfr2g)
            sn = sn.dropna(subset=["player_id"]).groupby(["player_id", "week"], as_index=False)["offense_pct"].max()
            s = s.merge(sn, on=["player_id", "week"], how="left")
        except Exception:
            s["offense_pct"] = np.nan
        frames.append(s)
        try:
            i = L.read("injuries", f"injuries_{y}.csv")
            if "game_type" in i.columns:
                i = i[i["game_type"].astype(str).eq("REG")]
            inj_frames.append(i)
        except Exception:
            pass
    S = pd.concat(frames, ignore_index=True)
    S["week"] = _num(S["week"]).astype(int)
    S["season"] = _num(S["season"]).astype(int)
    for c in ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds", "fantasy_points_ppr",
              "offense_pct", "target_share", "attempts"]:
        if c not in S.columns:
            S[c] = 0.0
        S[c] = _num(S[c])
    S["tds"] = S["receiving_tds"].fillna(0) + S["rushing_tds"].fillna(0)
    S["ppr"] = S["fantasy_points_ppr"]
    S["tgt_share"] = S["target_share"]
    inj = pd.concat(inj_frames, ignore_index=True) if inj_frames else pd.DataFrame()
    if len(inj):
        inj["season"] = _num(inj["season"]).astype(int)
        inj["week"] = _num(inj["week"]).astype(int)
        inj["avail"] = [avail_weight(a, b) for a, b in zip(inj["report_status"], inj["practice_status"])]
        inj = inj.drop_duplicates(["gsis_id", "season", "week"], keep="last")[["gsis_id", "season", "week", "avail"]]
    else:
        inj = pd.DataFrame(columns=["gsis_id", "season", "week", "avail"])
    g = L.read("schedules", "games.csv")
    g = g[g["game_type"].astype(str).eq("REG")].copy()
    g["season"] = _num(g["season"]).astype(int)
    g["week"] = _num(g["week"]).astype(int)
    h = g.rename(columns={"home_team": "team", "away_team": "opponent_team"}).copy()
    h["is_home"] = 1
    h["spread"] = _num(h["spread_line"])
    h["rest"] = _num(h["home_rest"]) if "home_rest" in h else np.nan
    a = g.rename(columns={"away_team": "team", "home_team": "opponent_team"}).copy()
    a["is_home"] = 0
    a["spread"] = -_num(a["spread_line"])
    a["rest"] = _num(a["away_rest"]) if "away_rest" in a else np.nan
    ctx = pd.concat([h, a], ignore_index=True)
    ctx["total_line"] = _num(ctx["total_line"])
    ctx["implied"] = ctx["total_line"] / 2 + ctx["spread"] / 2
    ctx["opp_implied"] = ctx["total_line"] / 2 - ctx["spread"] / 2
    ctx = ctx[["season", "week", "team", "opponent_team", "is_home", "spread", "total_line", "implied", "opp_implied", "rest"]]
    return S, inj, ctx


# ------------------------------------------------------------------ running states
CUTSHORT = True      # skip games where a regular starter left early injured
CUT_SNAP = 0.40
CUT_PRIOR = 0.65

def player_states(S, last_week=None, stats=None, positions=None):
    stats = stats or STAT0
    positions = positions or POS
    out = []
    Sp = S[S.position.isin(positions)].sort_values(["player_id", "season", "week"])
    last = {}
    for (pid, y), d in Sp.groupby(["player_id", "season"]):
        d = d.set_index("week")
        d = d[~d.index.duplicated(keep="last")]
        pos = d.position.iloc[0]
        m = {c: np.nan for c in stats}
        sums = {c: 0.0 for c in stats}
        cnt = {c: 0 for c in stats}
        hist = {c: [] for c in stats}
        n = 0
        lm = last.get(pid)
        for w in range(1, 19):
            row = {"player_id": pid, "season": y, "week": w, "position": pos, "n_prior": n}
            for c in stats:
                row["ew_" + c] = m[c]
                row["sm_" + c] = sums[c] / cnt[c] if cnt[c] else np.nan
                row["ly_" + c] = lm[0][c] if lm else np.nan
                row["l3_" + c] = float(np.mean(hist[c][-3:])) if hist[c] else np.nan
                if c == "targets" and "carries" in hist:
                    op = [a + b for a, b in zip(hist["targets"], hist["carries"])]
                    row["m3_opp"] = float(np.median(op[-3:])) if op else np.nan
                    row["l2_opp"] = float(np.mean(op[-2:])) if op else np.nan
                    row["sm_opp"] = float(np.mean(op)) if op else np.nan
                if c == "offense_pct":
                    row["m3_snap"] = float(np.median(hist[c][-3:])) if hist[c] else np.nan
                    row["l2_snap"] = float(np.mean(hist[c][-2:])) if hist[c] else np.nan
            row["ly_n"] = lm[1] if lm else 0
            out.append(row)
            if w in d.index:
                x = d.loc[w]
                if CUTSHORT and n >= 1 and "offense_pct" in stats and hist["offense_pct"]:
                    sp = x["offense_pct"]
                    if (not pd.isna(sp)) and sp < CUT_SNAP and float(np.mean(hist["offense_pct"])) >= CUT_PRIOR:
                        continue
                for c in stats:
                    v = x[c]
                    if pd.isna(v):
                        if c in ("offense_pct", "tgt_share"):
                            continue
                        v = 0.0
                    m[c] = v if np.isnan(m[c]) else ALPHA * v + (1 - ALPHA) * m[c]
                    sums[c] += v
                    cnt[c] += 1
                    hist[c].append(v)
                n += 1
        if n:
            last[pid] = ({c: (sums[c] / cnt[c] if cnt[c] else np.nan) for c in stats}, n)
    return pd.DataFrame(out)


def team_states(S):
    t = S.groupby(["team", "season", "week"], as_index=False).agg(t_tgt=("targets", "sum"), t_car=("carries", "sum"))
    out, last = [], {}
    for (tm, y), d in t.groupby(["team", "season"]):
        d = d.set_index("week")
        m = {"t_tgt": np.nan, "t_car": np.nan}
        sums = {"t_tgt": 0.0, "t_car": 0.0}
        n = 0
        for w in range(1, 19):
            out.append({"team": tm, "season": y, "week": w, "tm_tgt": m["t_tgt"], "tm_car": m["t_car"]})
            if w in d.index:
                for c in ("t_tgt", "t_car"):
                    v = d.loc[w, c]
                    m[c] = v if np.isnan(m[c]) else ALPHA * v + (1 - ALPHA) * m[c]
                    sums[c] += v
                n += 1
        if n:
            last[tm] = (sums["t_tgt"] / n, sums["t_car"] / n)
    df = pd.DataFrame(out)
    # last season's team volume fills the first weeks of a season
    for y in sorted(df.season.unique()):
        prev = df[(df.season == y - 1)].groupby("team").agg(p_t=("tm_tgt", "last"), p_c=("tm_car", "last"))
        if len(prev):
            idx = df.season == y
            df.loc[idx, "tm_tgt"] = df.loc[idx, "tm_tgt"].fillna(df.loc[idx, "team"].map(prev.p_t))
            df.loc[idx, "tm_car"] = df.loc[idx, "tm_car"].fillna(df.loc[idx, "team"].map(prev.p_c))
    return df


# ------------------------------------------------------------------ feature table
def build_features(S, inj, ctx, rows):
    """rows: DataFrame[player_id, position, season, week, team, opponent_team] (+ y_* for training rows)."""
    ps = player_states(S, None)
    ts = team_states(S)
    T = rows.copy()
    T = T.merge(ps.drop(columns=["position"]), on=["player_id", "season", "week"], how="left")
    T = T.merge(ts, on=["team", "season", "week"], how="left")
    T = T.merge(ctx, on=["team", "season", "week", "opponent_team"], how="left")
    ij = inj.rename(columns={"gsis_id": "player_id", "avail": "own_avail"})
    T = T.merge(ij, on=["player_id", "season", "week"], how="left")
    T["own_avail"] = T["own_avail"].fillna(1.0)
    # starting QB per team-week (most attempts so far), and his report status
    q = S[S.position == "QB"].groupby(["team", "season", "player_id", "week"], as_index=False)["attempts"].sum()
    qb_rows = []
    for (tm, y), d in q.groupby(["team", "season"]):
        att = {}
        by_w = {w: dd for w, dd in d.groupby("week")}
        for w in range(1, 19):
            qb_rows.append({"team": tm, "season": y, "week": w, "starter_qb": max(att, key=att.get) if att else None})
            if w in by_w:
                att = {k: v * STARTER_DECAY for k, v in att.items()}      # the most recent game counts most (picks the right starter ~90% of the time vs ~81% for season totals)
                for r in by_w[w].itertuples():
                    att[r.player_id] = att.get(r.player_id, 0) + (r.attempts if pd.notna(r.attempts) else 0)
    qbs = pd.DataFrame(qb_rows).merge(inj.rename(columns={"gsis_id": "starter_qb", "avail": "qb_avail"}), on=["starter_qb", "season", "week"], how="left")
    qbs["qb_avail"] = qbs["qb_avail"].fillna(1.0)
    T = T.merge(qbs[["team", "season", "week", "qb_avail"]], on=["team", "season", "week"], how="left")
    T["qb_avail"] = T["qb_avail"].fillna(1.0)
    T["qb_out"] = 1 - T["qb_avail"]
    # teammates' expected usage (running state, blended with last season) and their availability
    st = ps.copy()
    n = st.n_prior.clip(0, 6) / 6.0
    for c in ("targets", "carries"):
        st["x_" + c] = (n * st["ew_" + c].fillna(0) + (1 - n) * st["ly_" + c].fillna(st["ew_" + c].fillna(0))).where(st["ew_" + c].notna() | st["ly_" + c].notna(), 0)
    st = st.merge(inj.rename(columns={"gsis_id": "player_id"}), on=["player_id", "season", "week"], how="left")
    st["avail"] = st["avail"].fillna(1.0)
    tmap = S[["player_id", "season", "week", "team"]].drop_duplicates(["player_id", "season", "week"])
    tmp = st.merge(tmap, on=["player_id", "season", "week"], how="left")
    # a player's team for weeks he did not play: take the nearest known team in that season; else the newest team seen
    tmp["team"] = tmp.groupby(["player_id", "season"])["team"].transform(lambda s: s.ffill().bfill())
    cur_team = rows.drop_duplicates("player_id").set_index("player_id")["team"]
    fut = (tmp.team.isna())
    tmp.loc[fut, "team"] = tmp.loc[fut, "player_id"].map(cur_team)
    # players listed for a forecast week keep the team given by the caller (trades)
    over = rows[rows.get("y_ppr").isna()] if "y_ppr" in rows else rows.iloc[0:0]
    if len(over):
        o = over.set_index(["player_id", "season", "week"])["team"]
        key = list(zip(tmp.player_id, tmp.season, tmp.week))
        mapped = pd.Series([o.get(k, np.nan) for k in key], index=tmp.index)
        tmp["team"] = mapped.fillna(tmp["team"])
    tmp["vac_t"] = (1 - tmp.avail) * tmp.x_targets
    tmp["vac_c"] = (1 - tmp.avail) * tmp.x_carries
    for p in POS:
        tmp["vac_t_" + p] = np.where(tmp.position == p, tmp.vac_t, 0.0)
        tmp["vac_c_" + p] = np.where(tmp.position == p, tmp.vac_c, 0.0)
    tmp["xa_t"] = tmp.avail * tmp.x_targets
    tmp["rb_x_c"] = np.where(tmp.position == "RB", tmp.x_carries, 0.0)
    tmp["rb_xa_c"] = np.where(tmp.position == "RB", tmp.avail * tmp.x_carries, 0.0)
    vcols = [c for c in tmp.columns if c.startswith("vac_")] + ["x_targets", "xa_t", "rb_x_c", "rb_xa_c"]
    agg = tmp.groupby(["team", "season", "week"], as_index=False)[vcols].sum().rename(columns={"x_targets": "P_t", "xa_t": "PA_t", "rb_x_c": "P_c", "rb_xa_c": "PA_c"})
    T = T.merge(agg, on=["team", "season", "week"], how="left")
    own = tmp[["player_id", "season", "week", "vac_t", "vac_c", "x_targets", "x_carries", "avail"]].rename(
        columns={"vac_t": "own_vac_t", "vac_c": "own_vac_c", "x_targets": "own_xt", "x_carries": "own_xc", "avail": "own_av2"})
    T = T.merge(own, on=["player_id", "season", "week"], how="left")
    for c in ("vac_t", "vac_c"):
        T[c] = T[c] - T["own_" + c].fillna(0)
    for p in POS:
        mine = (T.position == p)
        T["vac_t_" + p] = T["vac_t_" + p] - np.where(mine, T.own_vac_t.fillna(0), 0.0)
        T["vac_c_" + p] = T["vac_c_" + p] - np.where(mine, T.own_vac_c.fillna(0), 0.0)
    own_av = T.own_av2.fillna(1.0)
    ox_t, ox_c = T.own_xt.fillna(0), T.own_xc.fillna(0)
    T["boost_t"] = ((ox_t + (T.P_t - ox_t)) / (ox_t + (T.PA_t - ox_t * own_av)).clip(lower=1.0) - 1).clip(0, 1.5)
    isrb = (T.position == "RB")
    T["boost_c"] = np.where(isrb, ((ox_c + (T.P_c - ox_c)) / (ox_c + (T.PA_c - ox_c * own_av)).clip(lower=1.0) - 1).clip(0, 2.0), 0.0)
    return T


def prep(T):
    T = T.copy()
    for p in POS:
        T["pos_" + p.lower()] = (T.position == p) * 1
    n = T.n_prior
    for c in STAT0:
        ew, ly, sm = T["ew_" + c], T["ly_" + c], T["sm_" + c]
        est = (n * sm.fillna(0) * 0.5 + n * ew.fillna(0) * 0.5 + K_LY * ly.fillna(ew)) / (n + K_LY)
        T["b_" + c] = est.where(~(n.eq(0) & ly.isna()), np.nan)
    med_ppr = T.groupby("position")["y_ppr"].transform("median") if T["y_ppr"].notna().any() else 7.0
    T["b_ppr"] = T["b_ppr"].fillna(med_ppr * 0.5 if not np.isscalar(med_ppr) else 3.5)
    for c in STAT0:
        if c != "ppr":
            T["b_" + c] = T["b_" + c].fillna(0)
    for c in ("targets", "carries", "ppr", "offense_pct", "receptions", "rushing_yards", "receiving_yards"):
        T["r3_" + c] = T["l3_" + c].fillna(T["b_" + c]) if "l3_" + c in T else T["b_" + c]
    if "m3_opp" in T:
        so = (T.sm_targets.fillna(0) + T.sm_carries.fillna(0)).where(T.n_prior > 0)
        ly_o = (T.ly_targets.fillna(0) + T.ly_carries.fillna(0))
        base_o = so.fillna(ly_o)
        for c in ("m3_opp", "l2_opp"):
            T[c] = T[c].fillna(base_o)
        T["shift_opp"] = (T.l2_opp - base_o).where(T.n_prior >= 2, 0.0).fillna(0.0).clip(-12, 12)
        base_s = T.sm_offense_pct.where(T.n_prior > 0).fillna(T.ly_offense_pct)
        for c in ("m3_snap", "l2_snap"):
            T[c] = T[c].fillna(base_s)
        T["m3_snap"] = T.m3_snap.fillna(50.0); T["l2_snap"] = T.l2_snap.fillna(50.0)
        T["shift_snap"] = (T.l2_snap - base_s.fillna(50.0)).where(T.n_prior >= 2, 0.0).fillna(0.0).clip(-30, 30)
        T["shift_x_vac_c"] = T.shift_opp * T.vac_c.fillna(0.0)
        T["shift_x_vac_t"] = T.shift_opp * T.vac_t.fillna(0.0)
    T["ypt"] = (T.b_receiving_yards + 8) / (T.b_targets + 1.2)
    T["cr"] = (T.b_receptions + 2) / (T.b_targets + 3)
    T["ypc"] = (T.b_rushing_yards + 20) / (T.b_carries + 5)
    # opponent strength: how much a defense has allowed above what players usually score, earlier weeks only, shrunk
    T = T.reset_index(drop=True)
    res = (T.y_ppr - T.b_ppr)
    T["opp_def"] = 0.0
    for (y, pos, o), d in T.groupby(["season", "position", "opponent_team"]):
        wk, r = d.week.values, res.loc[d.index].values
        ok = ~np.isnan(r)
        T.loc[d.index, "opp_def"] = [np.nansum(np.where(ok & (wk < w), r, 0.0)) / ((ok & (wk < w)).sum() + 25) for w in wk]
    T["exp_tgt_team"] = T.b_tgt_share / 100 * T.tm_tgt
    for p in ("rb", "wr", "te"):
        T["spr_" + p] = T.spread * T["pos_" + p]
        T["imp_" + p] = (T.implied - 22.5) * T["pos_" + p]
    T["imp_x_tgt"] = (T.implied - 22.5) * T.b_targets
    T["imp_x_car"] = (T.implied - 22.5) * T.b_carries
    T["spr_x_car"] = T.spread * T.b_carries
    T["spr_x_tgt"] = T.spread * T.b_targets
    T["bt_x_tgt"] = T.boost_t * T.b_targets
    T["bt_x_share"] = T.boost_t * T.b_tgt_share / 100 * T.tm_tgt
    T["bc_x_car"] = T.boost_c * T.b_carries
    T["bc_x_ppr"] = T.boost_c * T.b_ppr
    T["bt_x_ppr"] = T.boost_t * T.b_ppr
    return T


# ------------------------------------------------------------------ model
class Ridge:
    def fit(self, X, y, alpha=RIDGE):
        self.mu, self.sd = X.mean(0), X.std(0).replace(0, 1)
        Z = ((X - self.mu) / self.sd).values
        self.ym = float(np.mean(y))
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (np.asarray(y) - self.ym))
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd).values @ self.coef + self.ym

    def contrib(self, X, cols):
        """each feature's push on the prediction relative to the 'neutral' feature values"""
        Z = ((X - self.mu) / self.sd).values * self.coef
        return pd.DataFrame(Z, index=X.index, columns=cols)


def fit_models(Ttrain):
    tr = Ttrain[(Ttrain.week >= 2) & Ttrain.y_ppr.notna()]
    med = tr[FEATS].median()
    X = tr[FEATS].fillna(med)
    models = {c: Ridge().fit(X, tr["y_" + c]) for c in list(COMP) + ["targets", "carries"]}
    models["ppr"] = Ridge().fit(X, tr["y_ppr"])
    pred = sum(w * models[c].predict(X) for c, w in COMP.items())
    resid = tr["y_ppr"].values - pred
    # spread of outcomes by projection level -> floor / ceiling
    bins = np.quantile(pred, [0, .2, .4, .6, .8, 1.0])
    qs = []
    for i in range(5):
        m = (pred >= bins[i]) & (pred <= bins[i + 1])
        qs.append((float(pred[m].mean()), float(np.quantile(resid[m], .10)), float(np.quantile(resid[m], .90))))
    return {"models": models, "med": med, "bands": qs}


def predict(fit, Tnew):
    X = Tnew[FEATS].fillna(fit["med"])
    out = pd.DataFrame(index=Tnew.index)
    for c in COMP:
        out["p_" + c] = fit["models"][c].predict(X)
    out["p_tds"] = out["p_tds"].clip(lower=0)
    out["p_receptions"] = out["p_receptions"].clip(lower=0)
    out["p_targets"] = fit["models"]["targets"].predict(X).clip(min=0)
    out["p_carries"] = fit["models"]["carries"].predict(X).clip(min=0)
    out["proj"] = sum(w * out["p_" + c] for c, w in COMP.items()).clip(lower=0.3)
    bx = [b[0] for b in fit["bands"]]
    lo = np.interp(out["proj"], bx, [b[1] for b in fit["bands"]])
    hi = np.interp(out["proj"], bx, [b[2] for b in fit["bands"]])
    out["floor"] = (out["proj"] + lo).clip(lower=0)
    out["ceil"] = out["proj"] + hi
    # group contributions (points) relative to neutral inputs
    Xn = X.copy()
    for k, v in NEUTRAL.items():
        if k in Xn.columns:
            Xn[k] = v
    # keep derived interaction features consistent with the neutral base values
    for p in ("rb", "wr", "te"):
        Xn["spr_" + p] = Xn.spread * Xn["pos_" + p]
        Xn["imp_" + p] = (Xn.implied - 22.5) * Xn["pos_" + p]
    Xn["imp_x_tgt"] = (Xn.implied - 22.5) * Xn.b_targets
    Xn["imp_x_car"] = (Xn.implied - 22.5) * Xn.b_carries
    Xn["spr_x_car"] = Xn.spread * Xn.b_carries
    Xn["spr_x_tgt"] = Xn.spread * Xn.b_targets
    Xn["bt_x_tgt"] = Xn.boost_t * Xn.b_targets
    Xn["bt_x_share"] = Xn.boost_t * Xn.b_tgt_share / 100 * Xn.tm_tgt
    Xn["bc_x_car"] = Xn.boost_c * Xn.b_carries
    Xn["bc_x_ppr"] = Xn.boost_c * Xn.b_ppr
    Xn["bt_x_ppr"] = Xn.boost_t * Xn.b_ppr
    if "shift_opp" in Xn.columns:
        Xn["shift_x_vac_c"] = Xn.shift_opp * Xn.vac_c
        Xn["shift_x_vac_t"] = Xn.shift_opp * Xn.vac_t
    for gname, cols in GROUPS.items():
        if gname == "usage":
            continue
        Xg = X.copy()
        Xg[[c for c in cols if c in Xg.columns]] = Xn[[c for c in cols if c in Xg.columns]]
        out["g_" + gname] = out["proj"] - sum(w * fit["models"][c].predict(Xg) for c, w in COMP.items())
    return out


def project(base, season, next_week, roster, opp_map=None, log=print):
    """
    roster: DataFrame[player_id, position, team, opponent_team] for players with a game in next_week.
    Returns DataFrame indexed by player_id with proj / floor / ceil / components / group effects.
    """
    L = Loader(base)
    seasons = [y for y in range(max(2022, season - 4), season + 1)]
    S, inj, ctx = load_all(L, seasons)
    S = S[(S.season < season) | (S.week < next_week)]            # never look at the week being projected
    played = S[S.position.isin(POS)]
    train_rows = played[["player_id", "position", "season", "week", "team", "opponent_team", "targets", "carries", "receptions",
                         "receiving_yards", "rushing_yards", "tds", "ppr"]].rename(columns={c: "y_" + c for c in
                         ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "tds", "ppr"]})
    fr = roster.copy()
    fr["season"] = season
    fr["week"] = next_week
    for c in ["y_targets", "y_carries", "y_receptions", "y_receiving_yards", "y_rushing_yards", "y_tds", "y_ppr"]:
        fr[c] = np.nan
    rows = pd.concat([train_rows, fr[train_rows.columns]], ignore_index=True)
    T = prep(build_features(S, inj, ctx, rows))
    # no betting lines yet? use a neutral game rather than dropping the player
    for c, v in (("spread", 0.0), ("total_line", 45.0), ("implied", 22.5), ("opp_implied", 22.5), ("is_home", 0.5), ("rest", 7.0)):
        T[c] = T[c].fillna(v)
    for c in ("spr_rb", "spr_wr", "spr_te", "imp_rb", "imp_wr", "imp_te", "imp_x_tgt", "imp_x_car", "spr_x_car", "spr_x_tgt"):
        T[c] = T[c].fillna(0.0)
    is_new = T.y_ppr.isna() & (T.season == season) & (T.week == next_week)
    fit = fit_models(T[~is_new])
    out = predict(fit, T[is_new])
    out["player_id"] = T.loc[is_new, "player_id"].values
    out["n_prior"] = T.loc[is_new, "n_prior"].values
    # what the model would have said before each game already played this season (for the game logs)
    hist = pd.DataFrame(columns=["player_id", "week", "proj"])
    try:
        hs = []
        Tp = T[~is_new]
        for w in range(1, next_week):
            tr = Tp[(Tp.season < season) | ((Tp.season == season) & (Tp.week < w))]
            te = Tp[(Tp.season == season) & (Tp.week == w)]
            if len(te) == 0 or len(tr) < 500:
                continue
            o = predict(fit_models(tr), te)
            hs.append(pd.DataFrame({"player_id": te.player_id.values, "week": w, "proj": o.proj.values}))
        if hs:
            hist = pd.concat(hs, ignore_index=True)
    except Exception as e:
        log("   Past projections unavailable:", e)
    qb = pd.DataFrame()
    if opp_map:
        try:
            nm = S.drop_duplicates("player_id").set_index("player_id")["player_display_name"].to_dict() if "player_display_name" in S.columns else {}
            qb = qb_project(S, inj, ctx, season, next_week, opp_map, nm)
        except Exception as e:
            log("   QB projections unavailable:", e)
    return out.drop_duplicates("player_id").set_index("player_id"), fit, qb, hist


# ------------------------------------------------------------------ honest backtest
def backtest(base, season, w0, w1):
    L = Loader(base)
    seasons = list(range(max(2022, season - 4), season + 1))
    S, inj, ctx = load_all(L, seasons)
    played = S[S.position.isin(POS)]
    cols = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "tds", "ppr"]
    base_rows = played[["player_id", "position", "season", "week", "team", "opponent_team"] + cols].rename(columns={c: "y_" + c for c in cols})
    T = prep(build_features(S, inj, ctx, base_rows))
    for c, v in (("spread", 0.0), ("total_line", 45.0), ("implied", 22.5), ("opp_implied", 22.5), ("is_home", 0.5), ("rest", 7.0)):
        T[c] = T[c].fillna(v)
    for c in ("spr_rb", "spr_wr", "spr_te", "imp_rb", "imp_wr", "imp_te", "imp_x_tgt", "imp_x_car", "spr_x_car", "spr_x_tgt"):
        T[c] = T[c].fillna(0.0)
    rep = []
    for w in range(w0, w1 + 1):
        tr = T[(T.season < season) | ((T.season == season) & (T.week < w))]
        te = T[(T.season == season) & (T.week == w)]
        fit = fit_models(tr)
        o = predict(fit, te)
        rep.append(pd.DataFrame({"week": w, "y": te.y_ppr.values, "proj": o.proj.values, "floor": o.floor.values, "ceil": o.ceil.values,
                                 "b_ppr": te.b_ppr.values, "sm": te.sm_ppr.fillna(te.ly_ppr).fillna(te.b_ppr).values}))
    R = pd.concat(rep)
    for name, p in (("season-average baseline", R.sm), ("blended average baseline", R.b_ppr), ("Mania projection", R.proj)):
        e = R.y - p
        print(f"{name:26s} MAE {np.mean(abs(e)):.3f}  RMSE {np.sqrt(np.mean(e**2)):.3f}  R2 {1-np.sum(e**2)/np.sum((R.y-R.y.mean())**2):.3f}  n={len(R)}")
    print("share of games inside floor-ceiling:", round(float(((R.y >= R.floor) & (R.y <= R.ceil)).mean()), 3), "(target 0.80)")
    return R


# ================================================================== QUARTERBACKS
QSTAT = ["attempts", "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds", "ppr", "sacks_suffered"]
QCOMP = {"passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0, "rushing_yards": 0.1, "rushing_tds": 6.0}
QK_LY = 2.0
QRIDGE = 300.0
QFEATS = ["b_attempts", "b_passing_yards", "b_passing_tds", "b_passing_interceptions", "b_rushing_yards", "b_rushing_tds", "b_ppr",
          "b_sacks_suffered", "n_prior", "ly_n", "spread", "total_line", "implied", "opp_implied", "is_home", "rest", "own_avail",
          "rec_vac", "rec_vac_share", "opp_def", "spr_x_att", "imp_x_ppr", "imp_x_pyd", "vac_x_ppr"]
QNEUTRAL = {"spread": 0.0, "total_line": 45.0, "implied": 22.5, "opp_implied": 22.5, "is_home": 0.5, "rest": 7.0, "opp_def": 0.0,
            "own_avail": 1.0, "rec_vac": 0.0, "rec_vac_share": 0.0}
QGROUPS = {"script": ["spread", "total_line", "implied", "opp_implied", "is_home", "rest", "spr_x_att", "imp_x_ppr", "imp_x_pyd"],
           "matchup": ["opp_def"], "injuries": ["own_avail", "rec_vac", "rec_vac_share", "vac_x_ppr"]}


def _qb_numeric(S):
    S = S.copy()
    for c in QSTAT + ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds", "sacks_suffered"]:
        if c not in S.columns:
            S[c] = 0.0
        S[c] = _num(S[c])
    return S


def team_vac_table(S, inj):
    """per team-week: how many expected targets the RB/WR/TE group is missing according to the injury report"""
    ps = player_states(S, None)
    n = ps.n_prior.clip(0, 6) / 6.0
    ps["x_t"] = (n * ps["ew_targets"].fillna(0) + (1 - n) * ps["ly_targets"].fillna(ps["ew_targets"].fillna(0))).where(ps["ew_targets"].notna() | ps["ly_targets"].notna(), 0)
    ps = ps.merge(inj.rename(columns={"gsis_id": "player_id"}), on=["player_id", "season", "week"], how="left")
    ps["avail"] = ps["avail"].fillna(1.0)
    tmap = S[S.position.isin(POS)][["player_id", "season", "week", "team"]].drop_duplicates(["player_id", "season", "week"])
    ps = ps.merge(tmap, on=["player_id", "season", "week"], how="left").sort_values(["player_id", "season", "week"])
    ps["team"] = ps.groupby("player_id")["team"].transform(lambda s: s.ffill().bfill())
    ps["vac"] = (1 - ps.avail) * ps.x_t
    out = ps.groupby(["team", "season", "week"], as_index=False).agg(rec_vac=("vac", "sum"), pool=("x_t", "sum"))
    out["rec_vac_share"] = out.rec_vac / out.pool.clip(lower=1.0)
    return out[["team", "season", "week", "rec_vac", "rec_vac_share"]]


def qb_features(S, inj, ctx, rows, vac):
    Sq = _qb_numeric(S[S.position == "QB"])
    ps = player_states(Sq, None, stats=QSTAT, positions=["QB"])
    T = rows.merge(ps.drop(columns=["position"]), on=["player_id", "season", "week"], how="left")
    T = T.merge(ctx, on=["team", "season", "week", "opponent_team"], how="left")
    T = T.merge(inj.rename(columns={"gsis_id": "player_id", "avail": "own_avail"}), on=["player_id", "season", "week"], how="left")
    T["own_avail"] = T["own_avail"].fillna(1.0)
    T = T.merge(vac, on=["team", "season", "week"], how="left")
    T["rec_vac"] = T["rec_vac"].fillna(0.0)
    T["rec_vac_share"] = T["rec_vac_share"].fillna(0.0)
    n = T.n_prior
    for c in QSTAT:
        ew, ly, sm = T["ew_" + c], T["ly_" + c], T["sm_" + c]
        est = (n * sm.fillna(0) * 0.5 + n * ew.fillna(0) * 0.5 + QK_LY * ly.fillna(ew)) / (n + QK_LY)
        T["b_" + c] = est.where(~(n.eq(0) & ly.isna()), np.nan)
    return T


def qb_prep(T):
    T = T.copy()
    T["b_ppr"] = T["b_ppr"].fillna(13.0)
    for c in QSTAT:
        if c != "ppr":
            T["b_" + c] = T["b_" + c].fillna(0)
    for c, v in (("spread", 0.0), ("total_line", 45.0), ("implied", 22.5), ("opp_implied", 22.5), ("is_home", 0.5), ("rest", 7.0)):
        T[c] = T[c].fillna(v)
    T = T.reset_index(drop=True)
    res = T.y_ppr - T.b_ppr
    T["opp_def"] = 0.0
    for (y, o), d in T.groupby(["season", "opponent_team"]):
        wk, r = d.week.values, res.loc[d.index].values
        ok = ~np.isnan(r)
        T.loc[d.index, "opp_def"] = [np.nansum(np.where(ok & (wk < w), r, 0.0)) / ((ok & (wk < w)).sum() + 12) for w in wk]
    T["spr_x_att"] = T.spread * T.b_attempts
    T["imp_x_ppr"] = (T.implied - 22.5) * T.b_ppr
    T["imp_x_pyd"] = (T.implied - 22.5) * T.b_passing_yards
    T["vac_x_ppr"] = T.rec_vac_share * T.b_ppr
    return T


def qb_fit(Ttrain):
    tr = Ttrain[Ttrain.y_ppr.notna() & (Ttrain.week >= 2) & (Ttrain.y_attempts >= 10)]
    med = tr[QFEATS].median()
    X = tr[QFEATS].fillna(med)
    models = {c: Ridge().fit(X, tr["y_" + c], QRIDGE) for c in QCOMP}
    pred = sum(w * models[c].predict(X) for c, w in QCOMP.items())
    resid = tr["y_ppr"].values - pred
    bins = np.quantile(pred, [0, .2, .4, .6, .8, 1.0])
    qs = []
    for i in range(5):
        m = (pred >= bins[i]) & (pred <= bins[i + 1])
        qs.append((float(pred[m].mean()), float(np.quantile(resid[m], .07)), float(np.quantile(resid[m], .93))))
    return {"models": models, "med": med, "bands": qs}


def qb_predict(fit, Tn):
    X = Tn[QFEATS].fillna(fit["med"])
    out = pd.DataFrame(index=Tn.index)
    for c in QCOMP:
        out["p_" + c] = fit["models"][c].predict(X)
    out["p_passing_tds"] = out["p_passing_tds"].clip(lower=0)
    out["p_passing_interceptions"] = out["p_passing_interceptions"].clip(lower=0)
    out["p_rushing_tds"] = out["p_rushing_tds"].clip(lower=0)
    out["proj"] = sum(w * out["p_" + c] for c, w in QCOMP.items()).clip(lower=2.0)
    bx = [b[0] for b in fit["bands"]]
    out["floor"] = (out["proj"] + np.interp(out["proj"], bx, [b[1] for b in fit["bands"]])).clip(lower=0)
    out["ceil"] = out["proj"] + np.interp(out["proj"], bx, [b[2] for b in fit["bands"]])
    Xn = X.copy()
    for k, v in QNEUTRAL.items():
        Xn[k] = v
    Xn["spr_x_att"] = Xn.spread * Xn.b_attempts
    Xn["imp_x_ppr"] = (Xn.implied - 22.5) * Xn.b_ppr
    Xn["imp_x_pyd"] = (Xn.implied - 22.5) * Xn.b_passing_yards
    Xn["vac_x_ppr"] = Xn.rec_vac_share * Xn.b_ppr
    for gname, cols in QGROUPS.items():
        Xg = X.copy()
        Xg[cols] = Xn[cols]
        out["g_" + gname] = out["proj"] - sum(w * fit["models"][c].predict(Xg) for c, w in QCOMP.items())
    return out


def _qb_rows_played(S):
    q = _qb_numeric(S[S.position == "QB"])
    cols = ["attempts", "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds", "ppr"]
    r = q[["player_id", "position", "season", "week", "team", "opponent_team"] + cols].rename(columns={c: "y_" + c for c in cols})
    return r


def qb_project(S, inj, ctx, season, next_week, opp_map, names):
    """Project the starting QB of every team that plays in next_week. Returns DataFrame indexed by player_id."""
    Sq = _qb_numeric(S[S.position == "QB"])
    cw = Sq[(Sq.season == season) & (Sq.week < next_week)].copy()
    cw["attempts"] = cw.attempts * (STARTER_DECAY ** (next_week - 1 - cw.week).clip(lower=0))
    cur = cw.groupby(["team", "player_id"]).attempts.sum().reset_index()
    prv = Sq[(Sq.season == season - 1)].groupby(["team", "player_id"]).attempts.sum().reset_index()
    rows, meta = [], {}
    for team, opp in opp_map.items():
        c = cur[cur.team == team].sort_values("attempts", ascending=False)
        p = prv[prv.team == team].sort_values("attempts", ascending=False)
        cands = list(c.player_id) + [x for x in p.player_id if x not in set(c.player_id)]
        if not cands:
            continue
        starter = cands[0]
        a = inj[(inj.gsis_id == starter) & (inj.season == season) & (inj.week == next_week)]
        av = float(a.avail.iloc[0]) if len(a) else 1.0
        use, sub = starter, False
        if av <= 0.1 and len(cands) > 1:
            use, sub = cands[1], True
        rows.append({"player_id": use, "position": "QB", "season": season, "week": next_week, "team": team, "opponent_team": opp})
        meta[use] = {"sub": sub, "starter": starter}
    if not rows:
        return pd.DataFrame()
    fr = pd.DataFrame(rows)
    for c in ["y_attempts", "y_passing_yards", "y_passing_tds", "y_passing_interceptions", "y_rushing_yards", "y_rushing_tds", "y_ppr"]:
        fr[c] = np.nan
    train = _qb_rows_played(S[(S.season < season) | (S.week < next_week)])
    vac = team_vac_table(S, inj)
    T = qb_prep(qb_features(S, inj, ctx, pd.concat([train, fr[train.columns]], ignore_index=True), vac))
    is_new = T.y_ppr.isna() & (T.season == season) & (T.week == next_week)
    fit = qb_fit(T[~is_new])
    tn = T[is_new].copy()
    # a backup filling in for an injured starter: use the starter's passing volume (a little lower) instead of the backup's own thin history
    for i, r in tn.iterrows():
        m = meta.get(r.player_id)
        if m and m["sub"]:
            st = T[(T.player_id == m["starter"]) & (T.season == season) & (T.week == next_week)]
            if len(st):
                for c in ("b_attempts", "b_passing_yards", "b_passing_tds", "b_passing_interceptions", "b_ppr", "b_sacks_suffered"):
                    tn.at[i, c] = 0.85 * float(st.iloc[0][c])
                tn.at[i, "imp_x_ppr"] = (tn.at[i, "implied"] - 22.5) * tn.at[i, "b_ppr"]
                tn.at[i, "imp_x_pyd"] = (tn.at[i, "implied"] - 22.5) * tn.at[i, "b_passing_yards"]
                tn.at[i, "spr_x_att"] = tn.at[i, "spread"] * tn.at[i, "b_attempts"]
                tn.at[i, "vac_x_ppr"] = tn.at[i, "rec_vac_share"] * tn.at[i, "b_ppr"]
    o = qb_predict(fit, tn)
    o["player_id"] = tn.player_id.values
    o["team"] = tn.team.values
    o["opp"] = tn.opponent_team.values
    o["sub"] = [meta[p]["sub"] for p in o["player_id"]]
    o["name"] = [names.get(p, p) for p in o["player_id"]]
    return o.set_index("player_id")


def qb_backtest(base, season, w0, w1):
    L = Loader(base)
    S, inj, ctx = load_all(L, list(range(max(2022, season - 4), season + 1)))
    vac = team_vac_table(S, inj)
    T = qb_prep(qb_features(S, inj, ctx, _qb_rows_played(S), vac))
    T = T[T.y_attempts >= 10]
    rep = []
    for w in range(w0, w1 + 1):
        tr = T[(T.season < season) | ((T.season == season) & (T.week < w))]
        te = T[(T.season == season) & (T.week == w)]
        o = qb_predict(qb_fit(tr), te)
        rep.append(pd.DataFrame({"y": te.y_ppr.values, "proj": o.proj.values, "floor": o.floor.values, "ceil": o.ceil.values, "b_ppr": te.b_ppr.values,
                                 "sm": te.sm_ppr.fillna(te.ly_ppr).fillna(te.b_ppr).values}))
    R = pd.concat(rep)
    for name, p in (("QB season-average", R.sm), ("QB blended average", R.b_ppr), ("QB Mania projection", R.proj)):
        e = R.y - p
        print(f"{name:22s} MAE {np.mean(abs(e)):.3f}  RMSE {np.sqrt(np.mean(e**2)):.3f}  R2 {1-np.sum(e**2)/np.sum((R.y-R.y.mean())**2):.3f}  n={len(R)}")
    print("inside floor-ceiling:", round(float(((R.y >= R.floor) & (R.y <= R.ceil)).mean()), 3))
    return R


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        base = sys.argv[5] if len(sys.argv) > 5 else "https://github.com/nflverse/nflverse-data/releases/download"
        backtest(base, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    if len(sys.argv) > 1 and sys.argv[1] == "qbtest":
        base = "https://github.com/nflverse/nflverse-data/releases/download"
        qb_backtest(base, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
