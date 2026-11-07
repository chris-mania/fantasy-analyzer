"""Tell Bing (and other IndexNow search engines) which Fantasy Mania pages exist, once a day.

Bing powers the web results in ChatGPT search, Copilot and DuckDuckGo. Google does not use IndexNow;
Google finds pages through the sitemaps in Search Console instead.

Reads the page lists from public/sitemap.xml and public/sitemap-start.xml (what was just published)
and sends them in one request. The key file (public/<key>.txt) is written by site_extras.py.
Never fails the workflow: problems are printed and ignored.
"""
import json, re, sys, urllib.request

SITE = "https://fantasymania.org"
HOST = "fantasymania.org"


def main():
    from site_extras_key import KEY  # small module so this script doesn't rerun site_extras
    urls = []
    for f in ("public/sitemap.xml", "public/sitemap-start.xml"):
        try:
            urls += re.findall(r"<loc>([^<]+)</loc>", open(f, encoding="utf-8").read())
        except OSError:
            pass
    urls = [u for u in dict.fromkeys(urls) if u.startswith(SITE)][:10000]
    if not urls:
        print("indexnow: no pages to send"); return
    body = json.dumps({"host": HOST, "key": KEY, "keyLocation": "%s/%s.txt" % (SITE, KEY), "urlList": urls}).encode()
    req = urllib.request.Request("https://api.indexnow.org/indexnow", data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print("indexnow: sent %d pages, status %s" % (len(urls), r.status))
    except urllib.error.HTTPError as e:
        print("indexnow: status %s (%s) for %d pages" % (e.code, e.read()[:200], len(urls)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("indexnow: skipped", repr(e)[:200])
    sys.exit(0)
