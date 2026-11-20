// Checks the phone-only ESPN import (issue #7). Run from the repo root: node tests/espn_phone_import.test.js
// The app's script has no module boundary, so this lifts the three functions out of the source by name.
const fs = require('fs'), assert = require('assert');
const files = ['build.py'].concat(fs.existsSync('app/js') ? fs.readdirSync('app/js').map(f => 'app/js/' + f) : []);
const src = files.map(f => fs.readFileSync(f, 'utf8')).join('\n').replace(/\r/g, '');
const lift = name => { const m = src.match(new RegExp('^function ' + name + '\\(.*(?:\\n .*)*', 'm')); assert(m, name + ' not found'); return m[0]; };

const applied = [];
const E = { textContent: '' }, link = { href: '' };
const modal = { querySelector: s => (s === '.espnPhGo' ? link : null) };
const trEspnSeason = () => 2026;
const trEspnApply = (m, d) => applied.push(d);
eval(['trEspnPhoneUrl', 'trEspnPhoneFix', 'trEspnPhonePaste', 'trEspnBookmarklet'].map(lift).join('\n'));

// the link: from a pasted league address, from a bare ID, and nothing for junk
const light = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/123456?view=mSettings&view=mTeam&view=mBoxscore';
assert.strictEqual(trEspnPhoneUrl('https://fantasy.espn.com/football/league?leagueId=123456'), light);
assert.strictEqual(trEspnPhoneUrl(' 123456 '), light);
assert.match(trEspnPhoneUrl('https://fantasy.espn.com/football/team?leagueId=123456&seasonId=2025&teamId=3'), /seasons\/2025\//);
assert.match(trEspnPhoneUrl('123456', true), /view=mRoster$/);
assert.strictEqual(trEspnPhoneUrl('my league'), '');
assert.strictEqual(trEspnPhoneUrl(''), '');

// rosters move from this week's matchups onto their teams; a bye week has no away side
const roster = n => ({ entries: [{ playerId: n }] });
const feed = () => ({ id: 123456, seasonId: 2026, teams: [{ id: 1 }, { id: 2 }, { id: 3 }], schedule: [
  { home: { teamId: 1, totalPoints: 90 }, away: { teamId: 2, totalPoints: 80 } },
  { home: { teamId: 1, rosterForCurrentScoringPeriod: roster(11) }, away: { teamId: 2, rosterForCurrentScoringPeriod: roster(22) } },
  { home: { teamId: 3, rosterForCurrentScoringPeriod: roster(33) } }] });
assert.deepStrictEqual(trEspnPhoneFix(feed()).teams.map(t => t.roster.entries[0].playerId), [11, 22, 33]);

// a good paste is handed on to the normal ESPN reader
trEspnPhonePaste(modal, JSON.stringify(feed()), E);
assert.strictEqual(applied.length, 1);
assert.strictEqual(E.textContent, '');

// a team with no game this week: nothing imported, and the link switches to the full feed
const short = feed(); short.schedule.pop();
trEspnPhonePaste(modal, JSON.stringify(short), E);
assert.strictEqual(applied.length, 1);
assert.match(E.textContent, /left out 1 team /);
assert.match(link.href, /leagues\/123456\?.*view=mRoster$/);

// the full feed already has rosters on the teams and no schedule
trEspnPhonePaste(modal, JSON.stringify({ teams: [{ id: 1, roster: roster(1) }] }), E);
assert.strictEqual(applied.length, 2);

// a cut-off copy, a logged-out browser, and some other JSON each get their own message
trEspnPhonePaste(modal, JSON.stringify(feed()).slice(0, 200), E);
assert.match(E.textContent, /cut off/);
trEspnPhonePaste(modal, '{"messages":["You are not authorized to view this League."]}', E);
assert.match(E.textContent, /not logged in/);
trEspnPhonePaste(modal, '{"hello":1}', E);
assert.match(E.textContent, /not ESPN league data/);
assert.strictEqual(applied.length, 2);

// the bookmarklet (desktop private import, Route B): a javascript: URL that fetches this league from ESPN
// with the user's own session and posts it back to our origin. No password or cookie reaches our server.
global.location = { origin: 'https://fantasymania.org', pathname: '/', href: 'https://fantasymania.org/' };
const bm = decodeURIComponent(trEspnBookmarklet().replace(/^javascript:/, ''));
assert.match(trEspnBookmarklet(), /^javascript:/);
assert.ok(bm.includes('leagueId=') && bm.includes('lm-api-reads.fantasy.espn.com'), 'reads the ESPN league feed');
assert.ok(bm.includes('credentials:"include"') || bm.includes("credentials:'include'"), 'uses the ESPN session');
assert.ok(bm.includes('https://fantasymania.org'), 'posts back only to our origin');
assert.ok(!/password/i.test(bm), 'never touches a password');

console.log('espn phone import: ok');
