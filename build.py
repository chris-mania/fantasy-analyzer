import json, math, os, re
import numpy as np
import pandas as pd

SEASON=2026
BASE='https://github.com/nflverse/nflverse-data/releases/download'
POSITIONS=['WR','RB','TE']
QUALIFY_SNAPS=30
BASELINE=50.0

print('1. Loading nflverse player stats, snaps, and schedule...')
stats=pd.read_csv(f'{BASE}/stats_player/stats_player_week_{SEASON}.csv',low_memory=False)
snaps=pd.read_csv(f'{BASE}/snap_counts/snap_counts_{SEASON}.csv',low_memory=False)
sched=pd.read_csv(f'{BASE}/schedules/games.csv',low_memory=False)
if 'season_type' in stats: stats=stats[stats.season_type.eq('REG')]
if 'game_type' in snaps: snaps=snaps[snaps.game_type.eq('REG')]
sched=sched[(sched.season==SEASON) & (sched.game_type.eq('REG') if 'game_type' in sched else True)].copy()

name_col='player_display_name' if 'player_display_name' in stats else 'player_name'
team_col='team' if 'team' in stats else 'recent_team'
needed=['targets','receptions','receiving_yards','receiving_tds','receiving_air_yards','carries','rushing_yards','rushing_tds','fantasy_points_ppr']
for c in needed:
    if c not in stats: stats[c]=0
    stats[c]=pd.to_numeric(stats[c],errors='coerce').fillna(0)

# Team-week denominators: shares only use games in which the player actually appeared.
tw=(stats.groupby([team_col,'week'])[['targets','carries','receiving_air_yards']].sum().reset_index()
    .rename(columns={team_col:'team','targets':'team_targets','carries':'team_carries','receiving_air_yards':'team_air'}))
keep=['player_id',name_col,team_col,'position','week']+needed
s=stats[stats.position.isin(POSITIONS)][keep].copy().rename(columns={name_col:'player_name',team_col:'team'})
s=s.merge(tw,on=['team','week'],how='left')

# Snap merge. Missing snaps remain NaN, never fake 0%.
def norm_name(x):
    x=str(x).lower(); x=re.sub(r'\b(jr|sr|ii|iii|iv)\.?\b','',x); return re.sub(r'[^a-z0-9]','',x)
s['norm_name']=s.player_name.map(norm_name)
sn=snaps.copy()
sn['norm_name']=sn.player.map(norm_name)
for c in ['offense_snaps','offense_pct']:
    sn[c]=pd.to_numeric(sn[c],errors='coerce')
sn=sn[['norm_name','team','week','offense_snaps','offense_pct']].drop_duplicates(['norm_name','team','week'])
m=s.merge(sn,on=['norm_name','team','week'],how='left')
# impossible joins are missing, not zero
bad=((m.targets+m.carries)>0)&(m.offense_pct.fillna(0)<=0)
m.loc[bad,['offense_snaps','offense_pct']]=np.nan

print('2. Loading red-zone / goal-line play-by-play...')
rz_ok=True
try:
    want=['season_type','week','posteam','yardline_100','pass_attempt','rush_attempt','receiver_player_id','rusher_player_id']
    urls=[f'{BASE}/pbp/play_by_play_{SEASON}.csv.gz',f'{BASE}/pbp/play_by_play_{SEASON}.csv']
    pbp=None
    for u in urls:
        try:
            pbp=pd.read_csv(u,usecols=lambda c:c in want,low_memory=False); break
        except Exception: pass
    if pbp is None: raise RuntimeError('PBP file unavailable')
    if 'season_type' in pbp: pbp=pbp[pbp.season_type.eq('REG')]
    rzpbp=pbp[pbp.yardline_100.le(20)].copy()
    rec=rzpbp[(rzpbp.pass_attempt.eq(1)) & rzpbp.receiver_player_id.notna()][['receiver_player_id','week']].copy()
    rec.columns=['player_id','week']; rec['rz_targets']=1
    rus=rzpbp[(rzpbp.rush_attempt.eq(1)) & rzpbp.rusher_player_id.notna()][['rusher_player_id','week','yardline_100']].copy()
    rus.columns=['player_id','week','yardline_100']; rus['rz_carries']=1; rus['gl_carries']=rus.yardline_100.le(5).astype(int)
    rz=pd.concat([rec.assign(rz_carries=0,gl_carries=0),rus.assign(rz_targets=0)],ignore_index=True)
    rz=rz.groupby(['player_id','week'])[['rz_targets','rz_carries','gl_carries']].sum().reset_index()
    m=m.merge(rz,on=['player_id','week'],how='left')
except Exception as e:
    print('WARNING: PBP unavailable; red-zone fields disabled:',e); rz_ok=False
for c in ['rz_targets','rz_carries','gl_carries']:
    if c not in m: m[c]=0
    m[c]=m[c].fillna(0)

# Per-game shares. This fixes the Nico-type denominator problem.
def safe_div(a,b): return np.where(pd.to_numeric(b,errors='coerce').fillna(0)>0,a/b*100,0)
m['target_share']=safe_div(m.targets,m.team_targets)
m['air_share']=safe_div(m.receiving_air_yards,m.team_air)
m['touches']=m.targets+m.carries
m['team_opps']=m.team_targets+m.team_carries
m['touch_share']=safe_div(m.touches,m.team_opps)
m['total_yards']=m.receiving_yards+m.rushing_yards
m['tds']=m.receiving_tds+m.rushing_tds

# Participation intelligence: typical role + downweight extreme abbreviated games.
print('3. Detecting low-participation games and calculating player profiles...')
m['snap_pct']=m.offense_pct*100
m['typical_snap']=m.groupby('player_id').snap_pct.transform('median')
m['participation_ratio']=m.snap_pct/m.typical_snap.replace(0,np.nan)
# Conservative: established role >=50%; game <=50% of normal; at least 2 usable snap games.
snap_n=m.groupby('player_id').snap_pct.transform('count')
m['low_participation']=(snap_n>=2)&(m.typical_snap>=50)&(m.participation_ratio<=.50)&m.snap_pct.notna()
m['mania_game_weight']=1.0
m.loc[m.low_participation,'mania_game_weight']=m.loc[m.low_participation,'participation_ratio'].clip(.10,.50)

# Opportunity points are descriptive, not a separate public score.
m['opp_pts']=1.55*m.targets+.65*m.carries+1.35*m.rz_targets+.45*m.rz_carries+1.75*m.gl_carries

def wavg(df,col):
    z=df[[col,'mania_game_weight']].dropna()
    return np.average(z[col],weights=z.mania_game_weight) if len(z) else np.nan

rows=[]
for keys,d in m.groupby(['player_id','player_name','team','position'],dropna=False):
    pid,name,team,pos=keys; games=len(d); w=d.mania_game_weight.sum()
    r={'player_id':pid,'player_name':name,'team':team,'position':pos,'games':games,
       'targets_pg':wavg(d,'targets'),'receptions_pg':wavg(d,'receptions'),'carries_pg':wavg(d,'carries'),
       'touches_pg':wavg(d,'touches'),'rec_yards_pg':wavg(d,'receiving_yards'),'rush_yards_pg':wavg(d,'rushing_yards'),
       'total_yards_pg':wavg(d,'total_yards'),'td_pg':wavg(d,'tds'),'ppr_ppg':wavg(d,'fantasy_points_ppr'),
       'target_share':wavg(d,'target_share'),'touch_share':wavg(d,'touch_share'),'air_share':wavg(d,'air_share'),
       'snap_pct':wavg(d,'snap_pct'),'typical_snap':d.snap_pct.median(skipna=True),'rz_targets_pg':wavg(d,'rz_targets'),
       'rz_carries_pg':wavg(d,'rz_carries'),'gl_carries_pg':wavg(d,'gl_carries'),'opp_pg':wavg(d,'opp_pts'),
       'actual_ppr_pg':d.fantasy_points_ppr.mean(),'low_games':int(d.low_participation.sum()),'weight_games':w}
    rows.append(r)
g=pd.DataFrame(rows)

# Position-relative percentiles.
def percentile_by_pos(df,col):
    out=pd.Series(np.nan,index=df.index,dtype=float)
    for pos,grp in df.groupby('position'):
        vals=grp[col]
        ref=vals[vals.notna()]
        if len(ref): out.loc[grp.index]=vals.rank(pct=True,method='average')*100
    return out
metrics=['targets_pg','receptions_pg','carries_pg','touches_pg','rec_yards_pg','rush_yards_pg','total_yards_pg','td_pg','ppr_ppg','target_share','touch_share','air_share','snap_pct','typical_snap','rz_targets_pg','rz_carries_pg','gl_carries_pg','opp_pg']
for c in metrics: g['p_'+c]=percentile_by_pos(g,c).fillna(50)

# Position-specific Mania Rating. Production matters most; volume/role prevents TD luck from dominating.
WEIGHTS={
'WR':{'ppr_ppg':.24,'rec_yards_pg':.12,'receptions_pg':.08,'targets_pg':.13,'target_share':.10,'air_share':.07,'typical_snap':.07,'rz_targets_pg':.07,'td_pg':.06,'opp_pg':.06},
'TE':{'ppr_ppg':.25,'rec_yards_pg':.11,'receptions_pg':.09,'targets_pg':.14,'target_share':.12,'air_share':.05,'typical_snap':.08,'rz_targets_pg':.08,'td_pg':.05,'opp_pg':.03},
'RB':{'ppr_ppg':.23,'rush_yards_pg':.10,'rec_yards_pg':.06,'carries_pg':.13,'targets_pg':.09,'touches_pg':.10,'touch_share':.10,'typical_snap':.06,'rz_carries_pg':.05,'gl_carries_pg':.04,'td_pg':.04}
}
def raw_rating(r):
    w=WEIGHTS[r.position]; return sum(w[k]*r['p_'+k] for k in w)/sum(w.values())
g['raw_mania']=g.apply(raw_rating,axis=1)
# Confidence shrinkage: DNPs don't become zeroes; tiny samples are simply less certain.
# 1 GP=.48, 2=.70, 3=.82, 4=.90, 5=.95, 6+=.98
conf_map={1:.48,2:.70,3:.82,4:.90,5:.95}
g['confidence']=g.games.map(conf_map).fillna(.98)
g['mania_rating']=g.confidence*g.raw_mania+(1-g.confidence)*BASELINE

# Rank Mania Rating across all skill players and within position.
g['overall_rank']=g.mania_rating.rank(ascending=False,method='min').astype(int)
g['pos_rank']=g.groupby('position').mania_rating.rank(ascending=False,method='min').astype(int)
g['team_pos_rank']=g.groupby(['team','position']).mania_rating.rank(ascending=False,method='min').astype(int)

# Trend: weighted opportunity over latest up-to-3 appearances vs prior appearances.
def trend_for(pid,team):
    d=m[(m.player_id==pid)&(m.team==team)].sort_values('week')
    if len(d)<4:return 0.0
    recent=d.tail(3).opp_pts.mean(); prior=d.iloc[:-3].opp_pts.mean()
    return float(recent-prior) if pd.notna(prior) else 0.0
g['trend']=g.apply(lambda r:trend_for(r.player_id,r.team),axis=1)

# Upcoming schedule.
max_week=int(m.week.max())
future=sched[sched.week>max_week].sort_values('week')
current_week=int(future.week.min()) if len(future) else max_week
next_games=sched[sched.week.eq(current_week)].copy()
opp={}
for _,x in next_games.iterrows():
    opp[str(x.home_team)]=(str(x.away_team),'vs')
    opp[str(x.away_team)]=(str(x.home_team),'@')

# Defense matchup from 2026 historical player games: how opponent has allowed production to each position.
# We infer player's opponent from schedule by team/week.
week_opp={}
for _,x in sched[sched.week<=max_week].iterrows():
    week_opp[(str(x.home_team),int(x.week))]=str(x.away_team)
    week_opp[(str(x.away_team),int(x.week))]=str(x.home_team)
m['opponent']=[week_opp.get((str(t),int(w))) for t,w in zip(m.team,m.week)]
def_pp=(m.groupby(['opponent','position']).agg(allowed_ppr=('fantasy_points_ppr','mean'),allowed_opp=('opp_pts','mean'),allowed_rz=('rz_targets','mean')).reset_index())
# percentile matchup by position (higher = easier)
for c in ['allowed_ppr','allowed_opp','allowed_rz']:
    def_pp['p_'+c]=def_pp.groupby('position')[c].rank(pct=True)*100
def matchup(team,pos):
    o=opp.get(str(team))
    if not o:return (None,50.0)
    opponent=o[0]; q=def_pp[(def_pp.opponent==opponent)&(def_pp.position==pos)]
    if q.empty:return (o,50.0)
    rr=q.iloc[0]; score=.55*rr.p_allowed_ppr+.30*rr.p_allowed_opp+.15*rr.p_allowed_rz
    return (o,float(score))

g['opponent']='';g['site']='';g['matchup_score']=50.0
for i,r in g.iterrows():
    o,ms=matchup(r.team,r.position)
    if o:g.at[i,'opponent']=o[0];g.at[i,'site']=o[1]
    g.at[i,'matchup_score']=ms
# Early season: player quality dominates. Start Score = 75% Mania, 15% matchup, 10% trend percentile.
g['trend_pct']=g.groupby('position').trend.rank(pct=True)*100
g['start_score']=.75*g.mania_rating+.15*g.matchup_score+.10*g.trend_pct
# Slightly reduce weekly certainty for tiny player samples without destroying talent rating.
g['start_score']=g.start_score*(.90+.10*g.confidence)
g['week_rank']=g.start_score.rank(ascending=False,method='min').astype(int)
g['week_pos_rank']=g.groupby('position').start_score.rank(ascending=False,method='min').astype(int)

# Expected PPR is a transparent blend of player's weighted PPR and opponent allowance.
league_pos=m.groupby('position').fantasy_points_ppr.mean().to_dict()
def exp_ppr(r):
    o=opp.get(str(r.team)); defense=None
    if o:
        q=def_pp[(def_pp.opponent==o[0])&(def_pp.position==r.position)]
        if len(q): defense=float(q.iloc[0].allowed_ppr)
    if defense is None:defense=float(league_pos.get(r.position,10))
    return .72*r.ppr_ppg+.28*defense
g['expected_ppr']=g.apply(exp_ppr,axis=1)
g['luck']=g.actual_ppr_pg-g.expected_ppr
g['flag']=np.where((g.games>=3)&(g.luck<=-3),'BUY LOW',np.where((g.games>=3)&(g.luck>=3),'SELL HIGH',''))

print('4. Packaging players, game logs, teammate rooms, rankings...')
players=[]
for _,r in g.sort_values('mania_rating',ascending=False).iterrows():
    d=m[(m.player_id==r.player_id)&(m.team==r.team)].sort_values('week')
    logs=[]
    for x in d.itertuples():
        logs.append({'w':int(x.week),'t':int(x.targets),'rec':int(x.receptions),'ry':round(float(x.receiving_yards),1),'c':int(x.carries),'ruy':round(float(x.rushing_yards),1),'td':int(x.tds),'ppr':round(float(x.fantasy_points_ppr),1),'sp':None if pd.isna(x.snap_pct) else round(float(x.snap_pct),1),'ts':round(float(x.target_share),1),'touchs':round(float(x.touch_share),1),'rt':int(x.rz_targets),'rc':int(x.rz_carries),'gc':int(x.gl_carries),'low':bool(x.low_participation),'weight':round(float(x.mania_game_weight),2)})
    vals={k:(None if pd.isna(r[k]) else round(float(r[k]),1)) for k in metrics+['actual_ppr_pg','typical_snap','trend','expected_ppr','matchup_score']}
    players.append({'id':str(r.player_id),'name':r.player_name,'team':r.team,'pos':r.position,'games':int(r.games),'mania':round(float(r.mania_rating),1),'raw':round(float(r.raw_mania),1),'conf':round(float(r.confidence)*100),'overall_rank':int(r.overall_rank),'pos_rank':int(r.pos_rank),'team_rank':int(r.team_pos_rank),'start':round(float(r.start_score),1),'week_rank':int(r.week_rank),'week_pos_rank':int(r.week_pos_rank),'opp':r.opponent,'site':r.site,'flag':r.flag,'low_games':int(r.low_games),'m':vals,'logs':logs})

payload=json.dumps(players,allow_nan=False)
meta=json.dumps({'season':SEASON,'through':max_week,'week':current_week,'rz':rz_ok})

html=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mania Fantasy Analyzer</title>
<style>
:root{--bg:#0b0f14;--card:#131922;--card2:#0e141c;--bd:#293241;--tx:#cbd5e1;--hi:#f8fafc;--mu:#8491a3;--ac:#60a5fa;--gr:#4ade80;--or:#fb923c;--red:#fb7185;--pur:#c084fc}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px}.wrap{max-width:1180px;margin:auto}header{text-align:center;margin-bottom:20px}h1{margin:0;color:var(--hi);font-size:2.25rem}.sub,.mu{color:var(--mu)}.tabs,.filters{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:16px 0}.btn,select,input{background:var(--card);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:10px 14px}.btn{cursor:pointer;font-weight:700}.btn.on{background:var(--ac);color:#07111e;border-color:var(--ac)}.search{position:relative;margin:18px 0}.search input{width:100%;font-size:1rem}.dd{position:absolute;left:0;right:0;top:46px;background:var(--card);border:1px solid var(--bd);z-index:20;display:none;max-height:280px;overflow:auto;border-radius:8px}.dd div{padding:11px 14px;border-bottom:1px solid var(--bd);cursor:pointer}.dd div:hover{background:#1d2633}.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:22px;margin:16px 0}.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}.rating{font-size:4rem;line-height:1;font-weight:900;color:var(--hi)}.start{font-size:2.2rem;font-weight:850;color:var(--ac)}.pill{display:inline-block;border:1px solid var(--bd);border-radius:999px;padding:5px 9px;margin:3px;font-size:.8rem}.buy{color:var(--gr);border-color:var(--gr)}.sell{color:var(--red);border-color:var(--red)}.warn{padding:10px 12px;border:1px solid var(--or);color:var(--or);background:#fb923c12;border-radius:8px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px}.stat{background:var(--card2);border:1px solid var(--bd);border-radius:10px;padding:13px}.stat b{display:block;color:var(--hi);font-size:1.35rem;margin-top:4px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:10px 9px;border-bottom:1px solid var(--bd);text-align:center}th{font-size:.72rem;color:var(--mu);text-transform:uppercase;cursor:pointer}td.l,th.l{text-align:left}.hi{color:var(--gr);font-weight:800}.low{color:var(--or)}.rankrow:hover{background:#1b2430;cursor:pointer}.current{background:#60a5fa12}.checks input{width:auto}.sectionTitle{color:var(--hi);margin:22px 0 8px}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:720px){body{padding:12px}.split{grid-template-columns:1fr}.rating{font-size:3rem}}
</style></head><body><div class="wrap"><header><h1>Mania Fantasy Analyzer</h1><div class="sub" id="subtitle"></div></header>
<div class="tabs"><button class="btn on" id="bp" onclick="mode('profile')">Player Profile</button><button class="btn" id="br" onclick="mode('rank')">Rankings</button><button class="btn" id="bc" onclick="mode('compare')">Head-to-Head</button></div>
<section id="profile"><div class="search"><input placeholder="Search a player..." oninput="searchPlayer(this.value,'p')"><div class="dd" id="ddp"></div></div><div id="profileOut"></div></section>
<section id="rank" style="display:none"><div class="filters"><button class="btn on" id="ro" onclick="rankMode('overall')">Overall</button><button class="btn" id="rw" onclick="rankMode('week')">Week</button><select id="posFilter" onchange="renderRanks()"><option>ALL</option><option>WR</option><option>RB</option><option>TE</option></select><select id="sortFilter" onchange="renderRanks()"><option value="mania">Mania Rating</option><option value="start">Start Score</option><option value="ppr_ppg">PPR/G</option><option value="targets_pg">Targets/G</option><option value="receptions_pg">Receptions/G</option><option value="carries_pg">Carries/G</option><option value="touches_pg">Touches/G</option><option value="rec_yards_pg">Rec Yards/G</option><option value="rush_yards_pg">Rush Yards/G</option><option value="total_yards_pg">Total Yards/G</option><option value="target_share">Target Share</option><option value="touch_share">Touch Share</option><option value="typical_snap">Typical Snap</option><option value="rz_targets_pg">RZ Targets/G</option><option value="gl_carries_pg">Goal-Line/G</option></select></div><div id="rankOut"></div></section>
<section id="compare" style="display:none"><div class="split"><div class="search"><input placeholder="Player A..." oninput="searchPlayer(this.value,'a')"><div class="dd" id="dda"></div></div><div class="search"><input placeholder="Player B..." oninput="searchPlayer(this.value,'b')"><div class="dd" id="ddb"></div></div></div><div id="compareOut"></div></section>
</div><script>const DB=__PAYLOAD__;const META=__META__;let A=null,B=null,RANKMODE='overall';
subtitle.innerHTML=`${META.season} NFL Season &bull; Through Week ${META.through} &bull; Week ${META.week} Start/Sit`;
const fmt=(v,s='')=>v===null||v===undefined?'—':v+s;const byId=id=>DB.find(p=>p.id===id);
function mode(x){['profile','rank','compare'].forEach(k=>document.getElementById(k).style.display=k===x?'block':'none');bp.classList.toggle('on',x==='profile');br.classList.toggle('on',x==='rank');bc.classList.toggle('on',x==='compare');if(x==='rank')renderRanks()}
function searchPlayer(v,t){let d=document.getElementById('dd'+t),q=v.toLowerCase().trim();if(!q){d.style.display='none';return}let a=DB.filter(p=>p.name.toLowerCase().includes(q)).slice(0,9);d.innerHTML='';d.style.display=a.length?'block':'none';a.forEach(p=>{let e=document.createElement('div');e.textContent=`${p.name} (${p.team} • ${p.pos}) — Mania ${p.mania}`;e.onclick=()=>{d.style.display='none';if(t==='p')showPlayer(p);else{if(t==='a')A=p;else B=p;showCompare()}};d.appendChild(e)})}
function teammateRoom(p){let mates=DB.filter(x=>x.team===p.team&&x.pos===p.pos).sort((a,b)=>b.mania-a.mania);let label=p.pos==='RB'?`${p.team} BACKFIELD`:p.pos==='WR'?`${p.team} WR ROOM`:`${p.team} TE ROOM`;return `<h3 class="sectionTitle">${label}</h3><div class="scroll"><table><thead><tr><th class="l">Player</th><th>Mania</th><th>Tgt/G</th><th>Car/G</th><th>Share</th><th>Typical Snap</th><th>RZ Tgt</th><th>GL Car</th></tr></thead><tbody>${mates.map(x=>`<tr class="rankrow ${x.id===p.id?'current':''}" onclick="showPlayer(byId('${x.id}'))"><td class="l"><b>${x.name}</b></td><td>${x.mania}</td><td>${fmt(x.m.targets_pg)}</td><td>${fmt(x.m.carries_pg)}</td><td>${fmt(x.pos==='RB'?x.m.touch_share:x.m.target_share,'%')}</td><td>${fmt(x.m.typical_snap,'%')}</td><td>${fmt(x.m.rz_targets_pg)}</td><td>${fmt(x.m.gl_carries_pg)}</td></tr>`).join('')}</tbody></table></div>`}
function calcCustom(p,chosen){let logs=p.logs.filter(x=>chosen.has(x.w));if(!logs.length)return null;let avg=k=>logs.reduce((s,x)=>s+(Number(x[k])||0),0)/logs.length;return {ppr:avg('ppr'),t:avg('t'),rec:avg('rec'),ry:avg('ry'),c:avg('c'),ruy:avg('ruy'),sp:(()=>{let z=logs.filter(x=>x.sp!==null);return z.length?z.reduce((s,x)=>s+x.sp,0)/z.length:null})()}}
function showPlayer(p){mode('profile');let chosen=new Set(p.logs.map(x=>x.w));function draw(){let custom=calcCustom(p,chosen);let logs=p.logs.map(x=>`<tr><td><input type="checkbox" ${chosen.has(x.w)?'checked':''} onchange="toggle(${x.w},this.checked)"></td><td>W${x.w}</td><td>${x.t}</td><td>${x.rec}</td><td>${x.ry}</td><td>${x.c}</td><td>${x.ruy}</td><td>${x.td}</td><td>${fmt(x.sp,'%')}</td><td>${x.rt}</td><td>${x.rc}</td><td>${x.gc}</td><td class="hi">${x.ppr}</td><td class="${x.low?'low':''}">${x.low?'⚠ Low participation':''}</td></tr>`).join('');let sample=p.games<3?`<div class="warn">Limited sample: ${p.games} game${p.games===1?'':'s'}. Missed games are not counted as zeroes; Mania Rating is confidence-adjusted until the sample grows.</div>`:'';let low=p.low_games?`<div class="warn">${p.low_games} unusually low-participation game${p.low_games>1?'s':''} detected. It remains in actual stats but is downweighted when estimating normal player quality.</div>`:'';let customBox=chosen.size!==p.logs.length&&custom?`<div class="warn"><b>Custom view (${chosen.size}/${p.logs.length} games):</b> ${custom.ppr.toFixed(1)} PPR/G • ${custom.t.toFixed(1)} Tgt/G • ${custom.c.toFixed(1)} Car/G • ${custom.sp===null?'—':custom.sp.toFixed(1)+'%'} Snap. Official Mania Rating above is unchanged.</div>`:'';profileOut.innerHTML=`<div class="card"><div class="hero"><div><h2 style="margin:0;color:var(--hi)">${p.name}</h2><div class="mu">${p.team} • ${p.pos} • ${p.games} games • Team ${p.pos}${p.team_rank}</div><div style="margin-top:12px"><span class="pill">Overall #${p.overall_rank}</span><span class="pill">${p.pos} #${p.pos_rank}</span>${p.flag?`<span class="pill ${p.flag==='BUY LOW'?'buy':'sell'}">${p.flag}</span>`:''}</div></div><div><div class="rating">${p.mania}</div><b>MANIA RATING</b><div class="mu">Raw profile ${p.raw} • Confidence ${p.conf}%</div></div><div><div class="start">${p.start}</div><b>WEEK ${META.week} START SCORE</b><div class="mu">#${p.week_rank} overall • ${p.pos} #${p.week_pos_rank}<br>${p.site||''} ${p.opp||'TBD'} • Matchup ${fmt(p.m.matchup_score)}</div></div></div>${sample}${low}<h3 class="sectionTitle">Production & Usage</h3><div class="grid"><div class="stat">PPR / Game<b>${fmt(p.m.actual_ppr_pg)}</b></div><div class="stat">Targets / Game<b>${fmt(p.m.targets_pg)}</b></div><div class="stat">Receptions / Game<b>${fmt(p.m.receptions_pg)}</b></div><div class="stat">Carries / Game<b>${fmt(p.m.carries_pg)}</b></div><div class="stat">Rec Yards / Game<b>${fmt(p.m.rec_yards_pg)}</b></div><div class="stat">Rush Yards / Game<b>${fmt(p.m.rush_yards_pg)}</b></div><div class="stat">Target Share<b>${fmt(p.m.target_share,'%')}</b></div><div class="stat">Touch Share<b>${fmt(p.m.touch_share,'%')}</b></div><div class="stat">Air Yard Share<b>${fmt(p.m.air_share,'%')}</b></div><div class="stat">Typical Snap %<b>${fmt(p.m.typical_snap,'%')}</b></div><div class="stat">RZ Targets / G<b>${fmt(p.m.rz_targets_pg)}</b></div><div class="stat">Goal-Line Carries / G<b>${fmt(p.m.gl_carries_pg)}</b></div><div class="stat">Expected PPR / G<b>${fmt(p.m.expected_ppr)}</b></div><div class="stat">Trend (Opp Pts)<b>${p.m.trend>0?'+':''}${fmt(p.m.trend)}</b></div></div>${teammateRoom(p)}<h3 class="sectionTitle">Game Logs — toggle games for your own view</h3>${customBox}<div class="scroll checks"><table><thead><tr><th>Use</th><th>Game</th><th>Tgt</th><th>Rec</th><th>Rec Yd</th><th>Car</th><th>Rush Yd</th><th>TD</th><th>Snap%</th><th>RZ Tgt</th><th>RZ Car</th><th>GL Car</th><th>PPR</th><th>Note</th></tr></thead><tbody>${logs}</tbody></table></div><p class="mu">Mania Rating estimates overall fantasy quality from position-adjusted production, volume, team share, role, scoring opportunities and controlled efficiency. Missed games are excluded rather than entered as zeroes; extreme low-participation games are downweighted, not erased.</p></div>`;window.toggle=(w,on)=>{on?chosen.add(w):chosen.delete(w);draw()}}draw()}
function rankMode(x){RANKMODE=x;ro.classList.toggle('on',x==='overall');rw.classList.toggle('on',x==='week');sortFilter.value=x==='overall'?'mania':'start';renderRanks()}
function renderRanks(){let pos=posFilter.value,sort=sortFilter.value,a=DB.filter(p=>pos==='ALL'||p.pos===pos);a.sort((x,y)=>{let xv=sort==='mania'?x.mania:sort==='start'?x.start:(x.m[sort]??-999),yv=sort==='mania'?y.mania:sort==='start'?y.start:(y.m[sort]??-999);return yv-xv});let title=RANKMODE==='overall'?'Overall Rankings — Mania Rating':`Week ${META.week} Rankings — Start Score`;rankOut.innerHTML=`<div class="card"><h2 style="color:var(--hi)">${title}</h2><div class="scroll"><table><thead><tr><th>#</th><th class="l">Player</th><th>Pos</th><th>Team</th><th>Opp</th><th>Mania</th><th>Start</th><th>PPR/G</th><th>Tgt/G</th><th>Rec/G</th><th>Car/G</th><th>Yds/G</th><th>Share</th><th>Typical Snap</th><th>RZ Tgt</th><th>GL Car</th></tr></thead><tbody>${a.map((p,i)=>`<tr class="rankrow" onclick="showPlayer(byId('${p.id}'))"><td>${RANKMODE==='overall'?p.overall_rank:p.week_rank}</td><td class="l"><b>${p.name}</b></td><td>${p.pos}</td><td>${p.team}</td><td>${p.site} ${p.opp}</td><td class="hi">${p.mania}</td><td>${p.start}</td><td>${fmt(p.m.actual_ppr_pg)}</td><td>${fmt(p.m.targets_pg)}</td><td>${fmt(p.m.receptions_pg)}</td><td>${fmt(p.m.carries_pg)}</td><td>${fmt(p.m.total_yards_pg)}</td><td>${fmt(p.pos==='RB'?p.m.touch_share:p.m.target_share,'%')}</td><td>${fmt(p.m.typical_snap,'%')}</td><td>${fmt(p.m.rz_targets_pg)}</td><td>${fmt(p.m.gl_carries_pg)}</td></tr>`).join('')}</tbody></table></div></div>`}
function showCompare(){if(!A||!B)return;let row=(l,a,b,s='')=>`<tr><td class="${a>b?'hi':''}">${fmt(a,s)}</td><td>${l}</td><td class="${b>a?'hi':''}">${fmt(b,s)}</td></tr>`;compareOut.innerHTML=`<div class="card"><div class="split"><div style="text-align:center"><h2>${A.name}</h2><div class="rating">${A.mania}</div><b>MANIA RATING</b></div><div style="text-align:center"><h2>${B.name}</h2><div class="rating">${B.mania}</div><b>MANIA RATING</b></div></div><h3 class="sectionTitle" style="text-align:center">Who's better overall? — Mania Rating</h3><table><tbody>${row('Mania Rating',A.mania,B.mania)}${row('PPR / G',A.m.actual_ppr_pg,B.m.actual_ppr_pg)}${row('Targets / G',A.m.targets_pg,B.m.targets_pg)}${row('Carries / G',A.m.carries_pg,B.m.carries_pg)}${row('Total Yards / G',A.m.total_yards_pg,B.m.total_yards_pg)}${row('Typical Snap',A.m.typical_snap,B.m.typical_snap,'%')}</tbody></table><h3 class="sectionTitle" style="text-align:center">Who should I start Week ${META.week}?</h3><table><tbody>${row('Start Score',A.start,B.start)}${row('Matchup Score',A.m.matchup_score,B.m.matchup_score)}${row('Expected PPR',A.m.expected_ppr,B.m.expected_ppr)}${row('Recent Opportunity Trend',A.m.trend,B.m.trend)}</tbody></table><div class="split"><p class="mu" style="text-align:center">${A.name}: ${A.site} ${A.opp||'TBD'} • Week rank #${A.week_rank}</p><p class="mu" style="text-align:center">${B.name}: ${B.site} ${B.opp||'TBD'} • Week rank #${B.week_rank}</p></div></div>`}
window.onload=()=>{if(DB.length)showPlayer(DB[0])};</script></body></html>'''
html=html.replace('__PAYLOAD__',payload).replace('__META__',meta)
os.makedirs('public',exist_ok=True)
with open('public/index.html','w',encoding='utf-8') as f:f.write(html)
print(f'5. DONE — {len(players)} players, through Week {max_week}, Week {current_week} matchup view.')
