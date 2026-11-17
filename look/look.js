(function(){
  try{
    var l=document.createElement('link');l.rel='stylesheet';l.href='https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,800&display=swap';document.head.appendChild(l);
      }catch(e){}

  // ---- move intro paragraphs to an "About this page" note at the bottom of each page (content kept) ----
  function leaf(sec,re){var out=[];sec.querySelectorAll('div,p').forEach(function(d){if(d.closest('.lkAbout'))return;var own=[].slice.call(d.childNodes).filter(function(n){return n.nodeType===3}).map(function(n){return n.nodeValue}).join('').trim();if(re.test(own)&&d.querySelector('button,input,section')===null&&d.children.length<=3)out.push(d)});return out}
  function lift(){
    try{document.querySelectorAll('section.view').forEach(function(sec){
      var picks=[];
      var h1=sec.querySelector('h1');
      if(h1){var n=h1.nextElementSibling; if(n&&n.tagName==='P'&&n.className.indexOf('sub')>-1)picks.push(n)}
      sec.querySelectorAll('p.sub.subC').forEach(function(n){if(picks.indexOf(n)<0)picks.push(n)});
      leaf(sec,/^(Enter your full roster|Search a player to open the full|Pick the one that matches what you have)/).forEach(function(n){if(picks.indexOf(n)<0)picks.push(n)});
      var hh=sec.querySelector('.homeHero .hhL > p.sub'); if(hh&&picks.indexOf(hh)<0)picks.push(hh);
      if(!picks.length)return;
      var box=sec.querySelector(':scope > .lkAbout');
      if(!box){box=document.createElement('div');box.className='lkAbout';box.innerHTML='<span class="lkL">About this page</span>';sec.appendChild(box)}
      picks.forEach(function(n){if(n.closest('.lkAbout'))return;var k=(n.textContent||'').trim().slice(0,80);var dup=[].slice.call(box.children).some(function(c){return c.classList.contains('lkL')?false:(c.textContent||'').trim().slice(0,80)===k});if(dup){n.style.display='none';n.setAttribute('data-lk-dup','1')}else box.appendChild(n)});
    })}catch(e){}
  }
  // ---- square off every rounded box that is not a circle (rings, faces and dots keep their shape) ----
  function sq(root){
    try{var els=(root.querySelectorAll?root.querySelectorAll('*'):[]);var list=[root].concat(Array.prototype.slice.call(els));
      list.forEach(function(e){if(!e.getBoundingClientRect||e.nodeType!==1||e.tagName==='svg'||e.closest&&e.closest('svg'))return;
        var s=getComputedStyle(e),r=parseFloat(s.borderTopLeftRadius)||0;if(r<4)return;var b=e.getBoundingClientRect();if(!b.width||!b.height)return;
        var m=Math.min(b.width,b.height);if(Math.abs(b.width-b.height)<=3&&r>=m*.35)return; /* circle */
        if(e.tagName==='IMG'||e.closest('.portrait,.cmpPortrait,.hp,.crW,.initials,.thumb,.pctNum,.ring,.homeIm,.a2Tile,.newPill,.faNeon'))return;e.classList.add('lk-sq')})}catch(e){}
  }
  var pend=false,added=[];
  new MutationObserver(function(ms){ms.forEach(function(m){m.addedNodes.forEach(function(n){if(n.nodeType===1)added.push(n)})});
    if(pend)return;pend=true;setTimeout(function(){pend=false;lift();var a=added;added=[];a.forEach(function(n){if(document.body.contains(n))sq(n)})},350)}).observe(document.body,{childList:true,subtree:true});
  setTimeout(function(){lift();sq(document.body)},1200);
  var sv=window.showView; if(typeof sv==='function'){window.showView=function(){var r=sv.apply(this,arguments);setTimeout(function(){lift();sq(document.body)},600);return r}}
  var ct;document.addEventListener('click',function(){clearTimeout(ct);ct=setTimeout(function(){lift();sq(document.body)},700)},true);

  // ---- team color for the profile band + blue "IN THE WORLD" ----
  function tint(){try{var pb=document.getElementById('profileBody');if(pb&&typeof curPlayer!=='undefined'&&curPlayer&&typeof TCOL!=='undefined'){var c=TCOL[curPlayer.team]||'#275DF3';if(pb.style.getPropertyValue('--lkTc')!==c)pb.style.setProperty('--lkTc',c);var pr=document.getElementById('profile');if(pr&&pr.style.getPropertyValue('--lkTc')!==c)pr.style.setProperty('--lkTc',c)}
    var h=document.querySelector('#home .hhL h1');if(h&&!h.querySelector('.lkBl')&&/in the world/i.test(h.textContent)){h.innerHTML=h.textContent.replace(/(in the world)/i,'<span class="lkBl">$1</span>')}}catch(e){}}
  setInterval(function(){if(!document.hidden)tint()},800);

  function words(){try{var pb=document.getElementById('profileBody');if(!pb||typeof curPlayer==='undefined'||!curPlayer)return;
    var host=pb.querySelector('.dkTop')||pb.querySelector('.hero2');if(!host)return;
    var w=host.querySelector(':scope>.lkWm');if(!w){w=document.createElement('div');w.className='lkWm';host.insertBefore(w,host.firstChild)}
    if(w.textContent!==curPlayer.team)w.textContent=curPlayer.team;
    var m=pb.querySelector('.hid .meta');var ag=pb.querySelector('.hid .ageChip');
    if(m&&!m.dataset.lk){var t=m.textContent.replace(/\s*\u2022\s*/g,' \u00b7 ').trim();if(ag){t=ag.textContent.trim()+' \u00b7 '+t;ag.style.display='none'}m.textContent=t;m.dataset.lk='1'}
    var ov=pb.querySelector('.dkOv');if(ov&&!ov.dataset.lk){var sm=ov.querySelector('small');if(sm)sm.textContent='MANIA RATING';var hs=pb.querySelector('.hero2 .hs.left .scoresub');var t2=hs?hs.textContent.replace(/\s*\u2022\s*/g,' \u00b7 ').trim():'';if(t2){var em=document.createElement('em');em.textContent=t2;ov.appendChild(em)}ov.dataset.lk='1'}
    pb.querySelectorAll('.scoresub,.dkT em').forEach(function(e){if(e.textContent.indexOf('\u2022')>-1&&!e.querySelector('*'))e.textContent=e.textContent.replace(/\s*\u2022\s*/g,' \u00b7 ')})}catch(e){}}
  setInterval(function(){if(!document.hidden)words()},800);

  function chips(){try{var pb=document.getElementById('profileBody');if(!pb)return;
    pb.querySelectorAll('.rankline').forEach(function(rl){var dc=rl.querySelector('i.dcLink');if(!dc)return;var t=rl.querySelectorAll('i');
      var c=null;t.forEach(function(x){if(/\bon\b/i.test(x.textContent)&&x!==dc)c=x});
      if(c&&!c.classList.contains('dcLink')){c.classList.add('dcLink','lkDc');c.dataset.dcteam=dc.dataset.dcteam;c.dataset.dcid=dc.dataset.dcid;c.title='Tap to see the depth chart'}});
    var ol=pb.querySelector('.hol .outlook');var pj=pb.querySelector('.hpjW');
    if(pj){var x=pj.querySelector('.lkOl');if(ol){if(!x){x=document.createElement('span');pj.querySelector('.heroPj')?pj.querySelector('.heroPj').appendChild(x):pj.appendChild(x)}
      x.className='lkOl '+(/g90/.test(ol.className)?'g':/g70/.test(ol.className)?'y':/g0/.test(ol.className)?'r':'');if(x.textContent!==ol.textContent)x.textContent=ol.textContent}else if(x)x.remove()}}catch(e){}}
  setInterval(function(){if(!document.hidden)chips()},800);

  function kdTabs(){try{var rs=document.getElementById('rankings'),kb=document.getElementById('kdBoard');if(!rs||!kb||!rs.classList.contains('kdOn'))return;
    if(!kb.dataset.lkv)kb.dataset.lkv='ov';
    if(!kb.querySelector('.lkKdTabs')){var d=document.createElement('div');d.className='lkKdTabs';
      [['ov','Overview'],['st','Stats']].forEach(function(a){var b=document.createElement('button');b.type='button';b.textContent=a[1];b.dataset.v=a[0];b.onclick=function(){kb.dataset.lkv=a[0];d.querySelectorAll('button').forEach(function(x){x.classList.toggle('on',x.dataset.v===a[0])})};if(a[0]===kb.dataset.lkv)b.className='on';d.appendChild(b)});
      kb.insertBefore(d,kb.firstChild)}}catch(e){}}
  setInterval(function(){if(!document.hidden)kdTabs()},800);

  try{var tp=new URLSearchParams(location.search).get('top');if(tp)setInterval(function(){var e=document.getElementById('profile');if(e&&e.getAttribute('data-lktop')!==tp)e.setAttribute('data-lktop',tp)},300)}catch(e){}


})();
