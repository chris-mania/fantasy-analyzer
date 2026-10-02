// Runs the page script inside build.py under Node, with fixture players and a stub DOM,
// so the in-browser logic can be checked without a build or a network call.
// Usage: node tests/site.test.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const script = fs.readFileSync(path.join(__dirname, '..', 'build.py'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

// Every DOM lookup, property and call resolves to this same do-nothing object.
const dom = new Proxy(function () {}, { get: (_, k) => (k === Symbol.toPrimitive ? () => '' : dom), apply: () => dom, set: () => true });

function loadSite(players, week, refs = { RB: {}, WR: {}, TE: {} }) {
  const src = script
    .replace('__PAYLOAD__', () => JSON.stringify(players))
    .replace('__META__', () => JSON.stringify({ season: 2026, week, next_week: week + 1 }))
    .replace('__REFS__', () => JSON.stringify(refs));
  return new Function('document', 'window', src + ';return {movers}')(dom, dom);
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

console.log('site tests passed');
