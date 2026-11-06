"""Runs after build.py, every build. Adds what the site needs no matter what
build.py looks like: the pages.dev -> fantasymania.org move (carries saved data,
skips home-screen apps), Google tags, robots.txt and sitemap.xml.
Kept separate from build.py so feature edits there can't remove it."""
import os

MOVE = r'''<script id="fmMove">(function(){var H=location.hostname;
if(H==='fantasymania.pages.dev'){
 try{if((window.matchMedia&&matchMedia('(display-mode: standalone)').matches)||navigator.standalone===true)return}catch(e){return}
 var d={},n=0,skip={faAlive:1,faRs:1,fmRs:1,faLast:1,fmNav2:1,fmMoved:1};
 try{for(var i=0;i<localStorage.length;i++){var k=localStorage.key(i);if(!skip[k]){d[k]=localStorage.getItem(k);n++}}}catch(e){return}
 var tail=location.hash;
 if(n){tail='#fmmove='+encodeURIComponent(JSON.stringify({d:d,h:location.hash}));if(tail.length>500000)return}
 location.replace('https://fantasymania.org'+location.pathname+location.search+tail);return}
if(H==='fantasymania.org'&&location.hash.indexOf('#fmmove=')===0){
 var raw=location.hash.slice(8),p=null;
 try{p=JSON.parse(decodeURIComponent(raw))}catch(e){}
 try{history.replaceState(null,'',location.pathname+location.search+(p&&p.h||''))}catch(e){}
 if(!p||!p.d)return;
 try{if(localStorage.getItem('fmMoved'))return;
  for(var k in p.d){if(localStorage.getItem(k)===null)localStorage.setItem(k,p.d[k])}
  localStorage.setItem('fmMoved','1')}catch(e){}}
})();</script>'''

GOOGLE = ('<link rel="canonical" href="https://fantasymania.org/">'
          '<script type="application/ld+json">{"@context":"https://schema.org","@type":"WebSite",'
          '"name":"Fantasy Mania","alternateName":["FantasyMania","Fantasy Mania Fantasy Football"],'
          '"url":"https://fantasymania.org/"}</script>')

def patch(path, extra_head=""):
    if not os.path.exists(path):
        print("site_extras: missing", path); return
    s = open(path, encoding="utf-8").read()
    if 'id="fmMove"' not in s:
        i = s.find("<head>")
        assert i >= 0, path + " has no <head>"
        s = s[:i+6] + MOVE + s[i+6:]
    if extra_head and 'rel="canonical"' not in s:
        i = s.find("</head>")
        assert i >= 0, path + " has no </head>"
        s = s[:i] + extra_head + s[i:]
    open(path, "w", encoding="utf-8").write(s)
    print("site_extras: patched", path)

patch("public/index.html", GOOGLE)
patch("public/fantasy.html")
open("public/robots.txt", "w").write("User-agent: *\nAllow: /\n\nSitemap: https://fantasymania.org/sitemap.xml\n")
open("public/sitemap.xml", "w").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <url><loc>https://fantasymania.org/</loc></url>\n</urlset>\n')
print("site_extras: robots.txt and sitemap.xml written")
