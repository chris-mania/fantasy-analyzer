// Future weeks should use the imported ESPN projection; the current week stays Mania's own. (V2 rule.)
// Run from the repo root: node tests/espn_future_projections.test.js
const fs = require('fs'), assert = require('assert');
const src = fs.readFileSync('build.py', 'utf8').replace(/\r/g, '');
const lift = name => { const m = src.match(new RegExp('^function ' + name + '\\(.*', 'm')); assert(m, name + ' not found'); return m[0]; };
eval([lift('trEspnW'), lift('trWeekPpg')].join('\n'));

global.META = { next_week: 11 };
// id 'x' has an ESPN projection for wk 12, none for wk 13 (null)
global.TR = { espnPj: { x: { 12: 20, 13: null } } };
const v = { gw: [11, 12, 13], arr: [10, 8, 9], p: { id: 'x', proj: { pts: 12 }, out: false } };

assert.strictEqual(trWeekPpg(v, 11), 12, 'current week keeps Mania projection, not ESPN');
assert.strictEqual(trWeekPpg(v, 12), 20, 'future week uses the ESPN projection');
assert.strictEqual(trWeekPpg(v, 13), 9, 'future week with no ESPN value falls back to Mania');

// a week the player does not have a game for is still null (bye handling untouched)
assert.strictEqual(trWeekPpg(v, 14), null, 'absent week stays null');

// current week with a played actual is unaffected by ESPN
const vp = { gw: [11, 12], arr: [10, 8], p: { id: 'x', played: { ppr: 5, proj: 7 }, proj: { pts: 12 }, out: false } };
assert.strictEqual(trWeekPpg(vp, 11), 7, 'current-week actual/proj path unchanged');

// no league imported -> every week is Mania's
global.TR = { espnPj: {} };
assert.strictEqual(trWeekPpg(v, 12), 8, 'without an import, future weeks use Mania');
assert.strictEqual(trWeekPpg(v, 11), 12);

console.log('espn future projections: ok');
