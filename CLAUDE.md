# Fantasy Mania notes for Claude

- Always `git pull` (or fetch and rebase) right before editing and before pushing. Never push a copy of
  build.py that was made before the latest commit on main: that has wiped out other changes before.
- `site_extras.py` runs after build.py in `.github/workflows/update.yml`. It adds the move from
  fantasymania.pages.dev to fantasymania.org (carries saved data, skips home-screen apps), the canonical
  and site-name tags, robots.txt and sitemap.xml. Keep that workflow step and don't add these to build.py.
- The live site is https://fantasymania.org.
