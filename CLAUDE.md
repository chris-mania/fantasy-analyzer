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
- Start/sit pages (/start/ and /start/<a>-vs-<b>/): seo_pages.py calls start_sit.py, which writes
  start-data.json, sitemap-start.xml, _routes.json and copies start_worker.js to public/_worker.js. The
  worker makes each matchup page on request and only runs for /start addresses. Pair rules and nicknames
  are at the top of start_sit.py.
- site_extras.py also adds the homepage "Who should I start?" button and "Explore Fantasy Mania" links, the
  share preview (og.png + og/twitter tags), llms.txt and the Bing IndexNow key file (key in site_extras_key.py).
  indexnow.py runs after deploy once a day and sends the sitemap pages to Bing.
- The live site is https://fantasymania.org.
