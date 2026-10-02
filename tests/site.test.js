// Runs the page script inside build.py under Node, with fixture players and a stub DOM,
// so the in-browser logic can be checked without a build or a network call.
// Usage: node tests/site.test.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const script = fs.readFileSync(path.join(__dirname, '..', 'build.py'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

// Every DOM lookup, property and call resolves to this same do-nothing object.
const dom = new Proxy(function () {}, { get: (_, k) => (k === Symbol.toPrimitive ? () => '' : dom), apply: () => dom, set: () => true });

function loadSite(players, week, refs = { RB: {}, WR: {}, TE: {} }, meta = {}) {
  const src = script
    .replace('__PAYLOAD__', () => JSON.stringify(players))
    .replace('__META__', () => JSON.stringify({ season: 2026, week, next_week: week + 1, ...meta }))
    .replace('__REFS__', () => JSON.stringify(refs));
  return new Function('document', 'window', src + ';return {movers,market,marketWhy,defenseVs,schedule,scheduleWeeks,schedHTML}')(dom, dom);
}

const LOG = { tgt: 0, rec: 0, ry: 0, rtd: 0, air: 0, car: 0, ruy: 0, rutd: 0, ppr: 0, snap: 0, tshare: 0, ashare: 0, touch: 0, rzt: 0, ez: 0, rzc: 0, gl: 0, partial: false, weight: 1 };
let nextId = 0;
// logs: one object per game, in week order starting at Week 1 unless a log sets its own `w`.
function player(name, pos, logs, extra = {}) {
  return {
    id: 'p' + nextId++, name, pos, team: 'AAA', opp: 'BBB', games: logs.length, mania: 70, start: 70, start_pos_rank: 1,
    matchup_adj: 0, trend_pct: 0, similar: [], room: [], m: {}, buckets: {},
    logs: logs.map((l, i) => ({ ...LOG, w: i + 1, ...l })), ...extra,
  };
}
const names = rows => rows.map(x => x.p.name);

// ---- Risers & Fallers ----
{
  const site = loadSite([
    player('Riser', 'WR', [{ snap: 20 }, { snap: 22 }, { snap: 80 }]),
    player('Faller', 'WR', [{ snap: 90 }, { snap: 88 }, { snap: 40, partial: true, weight: 0.45 }]),
    player('Steady', 'WR', [{ snap: 60 }, { snap: 60 }, { snap: 62 }]),
    player('Debut', 'RB', [{ w: 3, snap: 70 }]),
    player('Sat Out Week 3', 'RB', [{ snap: 10 }, { snap: 90 }]),
  ], 3);

  assert.deepStrictEqual(names(site.movers('snap', 1, 'ALL')), ['Riser'], 'steady, one-game and absent players are left out');
  const [riser] = site.movers('snap', 1, 'ALL');
  assert.strictEqual(riser.before, 21);
  assert.strictEqual(riser.now, 80);
  assert.strictEqual(riser.d, 59);

  assert.deepStrictEqual(names(site.movers('snap', -1, 'ALL')), ['Faller'], 'fallers are the mirror image');
  assert.strictEqual(site.movers('snap', -1, 'ALL')[0].last.partial, true, 'the shortened game is still flagged');

  // A game he left early barely counts toward "before": (80*1 + 20*0.2) / 1.2 = 70, not the plain mean of 50.
  // So his return to 78% is a +8 move, not a +28 one.
  const back = loadSite([player('Hurt Early Once', 'RB', [{ snap: 80 }, { snap: 20, partial: true, weight: 0.2 }, { snap: 78 }])], 3).movers('snap', 1, 'ALL')[0];
  assert.ok(Math.abs(back.before - 70) < 1e-9 && Math.abs(back.d - 8) < 1e-9, 'earlier games are weighted by game weight');
}
{
  // Week 2: the 5-point cutoff is inclusive, and the position filter narrows the list.
  const site = loadSite([
    player('Exactly Five', 'TE', [{ snap: 50 }, { snap: 55 }]),
    player('Just Under', 'TE', [{ snap: 50 }, { snap: 54.9 }]),
    player('Big Riser', 'WR', [{ snap: 30 }, { snap: 70 }]),
  ], 2);
  assert.deepStrictEqual(names(site.movers('snap', 1, 'ALL')), ['Big Riser', 'Exactly Five'], 'sorted by size of the move');
  assert.deepStrictEqual(names(site.movers('snap', 1, 'TE')), ['Exactly Five']);
  assert.deepStrictEqual(site.movers('snap', 1, 'RB'), []);
}
{
  // "share" means touch share for running backs and target share for everyone else.
  const site = loadSite([
    player('Back', 'RB', [{ touch: 10, tshare: 50 }, { touch: 30, tshare: 50 }]),
    player('Receiver', 'WR', [{ touch: 50, tshare: 10 }, { touch: 50, tshare: 25 }]),
  ], 2);
  assert.deepStrictEqual(site.movers('share', 1, 'ALL').map(x => [x.p.name, x.d]), [['Back', 20], ['Receiver', 15]]);
}
{
  // Week 1: nobody has an earlier game to compare with.
  assert.deepStrictEqual(loadSite([player('Opening Day', 'WR', [{ snap: 95 }])], 1).movers('snap', 1, 'ALL'), []);
  assert.deepStrictEqual(loadSite([], 1).movers('snap', -1, 'ALL'), []);
}

// ---- Buy Low / Sell High ----
// With these reference lists a bucket score of N sits at the Nth percentile, so the fixtures read as percentiles.
const TENS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const POS_REFS = { b_production: TENS, b_opportunity: TENS, b_role: TENS, b_high_value: TENS, b_efficiency: TENS };
const REFS = { RB: POS_REFS, WR: POS_REFS, TE: POS_REFS };
const TWO_GAMES = [{ ppr: 10 }, { ppr: 10 }];
// usage: one number used for all three usage buckets, or [volume, role, red zone].
function rated(name, pos, usage, production, { efficiency = 50, logs = TWO_GAMES } = {}) {
  const [Opportunity, Role, HighValue] = Array.isArray(usage) ? usage : [usage, usage, usage];
  return player(name, pos, logs, { buckets: { Production: production, Opportunity, Role, 'High Value': HighValue, Efficiency: efficiency } });
}
{
  const site = loadSite([
    rated('Buy Me', 'WR', 90, 50),
    rated('Buy Me More', 'RB', 90, 30),
    rated('Small Gap', 'WR', 80, 70),
    rated('Nobody Wants Him', 'WR', 40, 10),
    rated('Sell Me', 'TE', 50, 90),
    rated('Nothing To Sell', 'TE', 10, 40),
    rated('Balanced Star', 'WR', 95, 95),
    rated('One Game', 'WR', 90, 20, { logs: [{ ppr: 10 }] }),
  ], 2, REFS);

  const all = site.market('ALL');
  assert.deepStrictEqual(names(all.buy), ['Buy Me More', 'Buy Me'], 'biggest gap first; small gaps, low usage and one-game players are left out');
  assert.deepStrictEqual(names(all.sell), ['Sell Me'], 'low producers are not sell-highs');
  assert.deepStrictEqual(names(site.market('WR').buy), ['Buy Me']);
  assert.deepStrictEqual(site.market('RB').sell, []);
}
{
  // Usage uses the Mania bucket weights, which differ by position:
  // RB (.30*100 + .20*50 + .125*0) / .625 = 64, WR (.30*100 + .225*50 + .10*0) / .625 = 66.
  const { buy } = loadSite([rated('Back', 'RB', [100, 50, 0], 10), rated('Receiver', 'WR', [100, 50, 0], 10)], 2, REFS).market('ALL');
  const usage = Object.fromEntries(buy.map(x => [x.p.name, x.usage]));
  assert.ok(Math.abs(usage.Back - 64) < 1e-9 && Math.abs(usage.Receiver - 66) < 1e-9, JSON.stringify(usage));
}
{
  // The one-line reason picks the most concrete fact available.
  const site = loadSite([
    rated('Snakebit', 'WR', 90, 50, { logs: [{ ppr: 8, rzt: 2 }, { ppr: 8, rzc: 1 }] }),
    rated('Cold', 'WR', 90, 50, { efficiency: 20 }),
    rated('Plain Buy', 'WR', 90, 50, { logs: [{ ppr: 14, rtd: 1 }, { ppr: 6 }] }),
    rated('TD Luck', 'RB', 30, 90, { logs: [{ ppr: 12, rutd: 1 }, { ppr: 12, rtd: 1 }] }),
    rated('Hot Hand', 'RB', 30, 90, { efficiency: 90 }),
    rated('Plain Sell', 'RB', 30, 90),
  ], 2, REFS);
  const { buy, sell } = site.market('ALL');
  const why = (rows, isBuy) => Object.fromEntries(rows.map(x => [x.p.name, site.marketWhy(x, isBuy)]));
  assert.deepStrictEqual(why(buy, true), {
    Snakebit: 'No touchdowns yet on 3 red-zone looks.',
    Cold: 'He is getting the work, but his efficiency is only 20th percentile. Volume tends to outlast a cold stretch.',
    'Plain Buy': 'His role is bigger than his box scores so far.',
  });
  assert.deepStrictEqual(why(sell, false), {
    'TD Luck': '50% of his points have come from touchdowns.',
    'Hot Hand': '90th percentile efficiency on 30th percentile usage. That is hard to keep up.',
    'Plain Sell': 'He is scoring more than his role usually supports.',
  });
}
assert.deepStrictEqual(loadSite([], 1, REFS).market('ALL'), { buy: [], sell: [] });
{
  // Injury flags come from the game logs: he left the latest week early, or his team played it without him.
  const site = loadSite([
    rated('Healthy', 'WR', 90, 50, { logs: [{ ppr: 10 }, { ppr: 10 }, { ppr: 10 }] }),
    rated('Left Early', 'WR', 90, 45, { logs: [{ ppr: 10 }, { ppr: 10 }, { ppr: 2, partial: true, weight: 0.3 }] }),
    rated('Sat Out', 'WR', 90, 40),
    Object.assign(rated('On Bye', 'WR', 90, 35), { team: 'BYE' }),
  ], 3, REFS);
  const { buy } = site.market('ALL');
  assert.deepStrictEqual(Object.fromEntries(buy.map(x => [x.p.name, x.hurt])), { 'On Bye': '', 'Sat Out': 'Missed Wk 3', 'Left Early': 'Left Wk 3 early', Healthy: '' });
  assert.strictEqual(site.marketWhy(buy.find(x => x.p.name === 'Sat Out'), true), 'Check his status before you make a move. His role is bigger than his box scores so far.');
  assert.strictEqual(site.marketWhy(buy.find(x => x.p.name === 'Healthy'), true), 'His role is bigger than his box scores so far.');
}

// ---- Defense vs Position ----
const NO_REFS = { RB: {}, WR: {}, TE: {} };
// usual: his season PPR per game, which "vs usual" compares each game against.
const scorer = (name, pos, team, usual, logs) => player(name, pos, logs, { team, m: { ppr: usual } });
{
  // AAA played DDD in Week 1 and EEE in Week 2; BBB played them the other way round.
  const opps = { AAA: { 1: 'DDD', 2: 'EEE' }, BBB: { 1: 'EEE', 2: 'DDD' }, DDD: { 1: 'AAA', 2: 'BBB' }, EEE: { 1: 'BBB', 2: 'AAA' } };
  const site = loadSite([
    scorer('Back One', 'RB', 'AAA', 10, [{ ppr: 20 }, { ppr: 5 }]),
    scorer('Back Two', 'RB', 'AAA', 10, [{ ppr: 10 }, { ppr: 5 }]),
    scorer('Back Three', 'RB', 'BBB', 20, [{ ppr: 10 }, { ppr: 30 }]),
    scorer('Deep Bench', 'RB', 'BBB', 1, [{ ppr: 0 }, { ppr: 6 }]),
    scorer('Receiver', 'WR', 'AAA', 10, [{ ppr: 40 }, { ppr: 40 }]),
  ], 2, NO_REFS, { opps });

  const [easy, tough] = site.defenseVs('RB');
  // DDD: backs who usually score 10 + 10 + 20 scored 20 + 10 + 30, which is +50%. The 1-PPR back is ignored for
  // "vs usual" but his 6 points still count toward points allowed: 66 over 2 games.
  assert.deepStrictEqual([easy.team, easy.games, easy.pg, Math.round(easy.vs)], ['DDD', 2, 33, 50]);
  // EEE: usual 10 + 10 + 20, actual 5 + 5 + 10, which is -50%.
  assert.deepStrictEqual([tough.team, tough.games, tough.pg, Math.round(tough.vs)], ['EEE', 2, 10, -50]);
  assert.deepStrictEqual(site.defenseVs('WR').map(x => x.team), ['DDD', 'EEE'], 'positions are kept apart');
  assert.deepStrictEqual(site.defenseVs('TE'), []);
}
{
  // A game he left early counts less: (30*1 + 0*0.25) / (10*1 + 10*0.25) = +140%, where a plain average would say +50%.
  const site = loadSite([
    scorer('Finished', 'RB', 'AAA', 10, [{ ppr: 30 }]),
    scorer('Left Early', 'RB', 'AAA', 10, [{ ppr: 0, partial: true, weight: 0.25 }]),
  ], 1, NO_REFS, { opps: { AAA: { 1: 'DDD' }, DDD: { 1: 'AAA' } } });
  assert.strictEqual(Math.round(site.defenseVs('RB')[0].vs), 140);
  // No schedule in the page: nothing to show, and nothing breaks.
  assert.deepStrictEqual(loadSite([scorer('Finished', 'RB', 'AAA', 10, [{ ppr: 30 }])], 1).defenseVs('RB'), []);
}

// ---- Schedule Strength ----
{
  // Through Week 1: DDD has been easy on backs (+100%), EEE tough (-50%), FFF has not played yet.
  // AAA still has DDD, a bye, FFF, EEE and DDD again; BBB gets EEE every week.
  const opps = {
    AAA: { 1: 'DDD', 2: 'DDD', 4: 'FFF', 5: 'EEE', 6: 'DDD', 16: 'EEE' },
    BBB: { 1: 'EEE', 2: 'EEE', 3: 'EEE', 4: 'EEE', 5: 'EEE', 6: 'EEE', 16: 'DDD' },
  };
  const site = loadSite([
    scorer('Home Back', 'RB', 'AAA', 10, [{ ppr: 20 }]),
    scorer('Away Back', 'RB', 'BBB', 10, [{ ppr: 5 }]),
  ], 1, NO_REFS, { opps });

  assert.deepStrictEqual(site.scheduleWeeks('next4'), [2, 3, 4, 5]);
  assert.deepStrictEqual(site.scheduleWeeks('all'), [2, 3, 4, 5, 6, 16]);
  assert.deepStrictEqual(site.scheduleWeeks('playoffs'), [16]);

  const [first, second] = site.schedule('RB', 'next4');
  assert.strictEqual(first.team, 'AAA', 'easiest schedule first');
  assert.deepStrictEqual(first.games.map(g => [g.w, g.opp, g.vs === null ? null : Math.round(g.vs)]), [[2, 'DDD', 100], [3, '', null], [4, 'FFF', 0], [5, 'EEE', -50]]);
  assert.ok(Math.abs(first.avg - 50 / 3) < 1e-9, 'the bye is left out of the average and the unseen defense counts as neutral');
  assert.deepStrictEqual([second.team, Math.round(second.avg)], ['BBB', -50]);
  assert.deepStrictEqual(site.schedule('RB', 'playoffs').map(x => [x.team, Math.round(x.avg)]), [['BBB', 100], ['AAA', -50]]);

  // The profile panel lists every game left, byes included, and is empty when there is no schedule.
  const panel = site.schedHTML({ pos: 'RB', team: 'AAA' });
  assert.ok(panel.includes('Wk 2 DDD') && panel.includes('Wk 3 bye') && panel.includes('Wk 16 EEE'), panel);
  assert.strictEqual(loadSite([scorer('Home Back', 'RB', 'AAA', 10, [{ ppr: 20 }])], 1).schedHTML({ pos: 'RB', team: 'AAA' }), '');
  assert.deepStrictEqual(loadSite([], 1).schedule('RB', 'all'), []);
}

console.log('site tests passed');
