# Fantasy Mania notes for Claude

- Always `git pull` (or fetch and rebase) right before editing and before pushing. Never push a copy of
  build.py that was made before the latest commit on main: that has wiped out other changes before.
- `site_extras.py` runs after build.py in `.github/workflows/update.yml`. It adds the move from
  fantasymania.pages.dev to fantasymania.org (carries saved data, skips home-screen apps), the canonical
  and site-name tags, robots.txt and sitemap.xml. Keep that workflow step and don't add these to build.py.
- `seo_pages.py` runs after that (about once an hour) and makes /players/<name>/ and /rankings/ pages for
  Google by opening the built app in a hidden browser and saving each real profile and rankings view. It
  makes no API calls. Keep that workflow step too. Profile or rankings design changes in build.py flow into
  these pages automatically.
- seo_pages.py also makes /players/<name>/game-log/ (the profile cut down to its game log), /teams/ and
  /teams/<team>/ (depth chart incl. kicker, snap share, target share, game logs against), and
  /insights/schedule-strength/ and /insights/defense-vs-position/. Rankings pages use the Projections view
  with "Week N" titles. Reloading any of these pages opens the normal app (HANDOFF script).
- Start/sit pages (/start/ and /start/<a>-vs-<b>/): seo_pages.py calls start_sit.py, which writes
  start-data.json, sitemap-start.xml, _routes.json and copies start_worker.js to public/_worker.js. The
  worker makes each matchup page on request and only runs for /start addresses. Pair rules and nicknames
  are at the top of start_sit.py.
- Shared trade links (/trade/<a>-and-<b>-for-<c>/, optional ?n=&sf=1&hz=&st=): the same worker (start_worker.js,
  routes in start_sit.py) opens the app's Quick Trade with those players; the app's "Copy trade link" button
  (fmTradeLink in build.py) makes them. Not in any sitemap.
- site_extras.py also adds the homepage "Who should I start?" button and "Explore Fantasy Mania" links, the
  share preview (og.png + og/twitter tags), llms.txt and the Bing IndexNow key file (key in site_extras_key.py).
  indexnow.py runs after deploy once a day and sends the sitemap pages to Bing.
- Page size: build.py packs the player data before writing the page (`_pack` near `payload =`): each team's
  position group ("room") is stored once, and game logs / similar players / weekly room rows store field
  names once. A small function in the payload rebuilds the exact same DB before the app runs. Keep that
  step when editing the payload; new per-player fields work as usual.
- The live site is https://fantasymania.org.
