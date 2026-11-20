# Ideas backlog (not building yet)

Found while exploring what the ESPN league import already pulls in vs. what it leaves on the table.
None of this is built. Revisit when there's bandwidth/usage to justify it.

## Trade block + deadline
ESPN lets GMs mark specific players "on the block." Surface that directly in Trade Finder/Ideas so it
shows real players a real GM has said they're open to trading, instead of only guessing. Also show the
league's actual trade deadline (pulled from the import) as a countdown banner.

## GM / League Activity tab
Per-team season totals already in the import: trades made, waiver adds/drops, moves to IR, waiver
priority rank. Good for a "league activity" tab -- who's active on waivers, who never trades, etc.

Caveat: this only gives season *totals* per team, not a per-transaction log. "Best waiver pickup" or
"who made the best trade" (with real dates, specific players, before/after value) needs ESPN's separate
transaction-history endpoint (not pulled today) -- doable, just a different/extra fetch to add first.

## Real playoff odds (the big one)
Two pieces confirmed present in the data, neither used yet:
- The league's actual remaining schedule -- every team's real opponent for every week left in the
  season, not a guess.
- ESPN's own per-player weekly point projections (statSourceId=0 in the roster feed) for every
  rostered player on every team in the league, not just mine. The import currently only grabs the
  *actual* result (statSourceId=1) per week, which is why future weeks come back empty -- the fix is to
  also grab statSourceId=0 for weeks that haven't happened yet.

Combine those two and you get a real simulation: live % chance to make the playoffs, % chance at a
first-round bye, projected final seed -- built on real rosters + real schedule + real per-player
projections, not just points-so-far. This is the highest-value thing in the list.

## Smaller stuff also sitting in the data
- Real team logos (ESPN gives a logo image URL per team) -- could replace generic initials in
  Trade/Compare screens.
- Keeper flags, if the league uses keepers -- matters for trade value (a protected keeper is worth more).
- Home/away/division record splits, current streak -- standings trivia.
- League has 3 prior seasons on ESPN (2023-2025) -- an all-time/franchise-history page is possible but
  needs a separate pull per past season, not just the current snapshot.
- Commissioner point adjustments, if any were made.

## Explicitly deferred (not worth doing yet / too much work for now)
- Exact league scoring rules (statId-level point values + overrides) to replace the generic PPR
  assumption. Biggest potential accuracy upgrade of anything here, but touches every value calculation
  in the app, so it needs real usage before it's worth the risk.
