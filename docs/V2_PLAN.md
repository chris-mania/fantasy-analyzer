# Fantasy Mania: app version 2 plan (from owner, Nov 17)

## 1. League import: three ways in per platform, each with different output
### ESPN
1. **Public league**: enter league info (league ID). No login.
2. **Run it on your own computer**: no sign-in, nothing stored on the site (extension / bookmarklet / local step) for people who don't want their info on the site.
3. **Private league**: user supplies their ESPN login info the way other sites do (cookies/email), pulled only for that import.
- Each way gives a *different amount of data*. Task: map what each way returns, and optimize each.
- ESPN data we can now use: actual record AND current standing/place, plus ESPN's **future-week projections**.
- Projection rule: current week = Fantasy Mania's own projections; future weeks = ESPN's projections (when the import gives access).
- League rules differ by platform (like Sleeper), so scoring/roster settings must come from the import.

### Sleeper (works well today)
- Import by username. If the league isn't public / can't be read, show a clear note and offer another route.
- Use Sleeper's free API for top waivers / waiver add (all platforms use it for this).

### Yahoo
- Last priority, after ESPN. Official OAuth exists; needs approved developer app + worker for token exchange. Check Yahoo terms first.

## 2. Trade finder / trade suggestions (future fix)
- Bug: trades the user builds that make BOTH teams better do not show up in the suggested trades. Find why the suggester misses them and fix.

## 3. Waivers
- Top waivers / waiver adds driven by Sleeper's free API for everything.

## Open
- Verify ESPN/Yahoo current access rules before building.
- Table: what each import path returns vs. what's missing.
