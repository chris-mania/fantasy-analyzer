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



  /* ===== profile: badges on the meta line + one info line (projection / rank tag / games counted) ===== */
  function liFit(el,prop,start,min,step,tight){
    var s=start;el.style.setProperty(prop,s+'px');
    var n=0;while(tight()&&s>min&&n++<40){s-=step;el.style.setProperty(prop,s+'px')}
  }
  function liInfo(){
    var pr=document.getElementById('profile');if(!pr)return;
    var body=document.getElementById('profileBody');if(!body)return;
    var hero=body.querySelector('.hero2');if(!hero)return;
    var meta=hero.querySelector('.hid .meta'),bd=hero.querySelector('.hol .bdgs');
    if(meta&&bd&&meta.getAttribute('data-lk')==='1'&&!bd.closest('.lkBdW')){var w=document.createElement('span');w.className='lkBdW';w.appendChild(bd);meta.appendChild(w)}
    if(meta){
      bd=meta.querySelector('.bdgs');
      var sg=meta.textContent+'|'+(bd?bd.children.length:0)+'|'+meta.clientWidth;
      if(meta.getAttribute('data-fit')!==sg){
        meta.style.setProperty('--bs','40px');meta.style.setProperty('--mf','21px');
        var k=0;while(meta.scrollWidth>meta.clientWidth+1&&k++<40){
          var bs=parseFloat(meta.style.getPropertyValue('--bs')),mf=parseFloat(meta.style.getPropertyValue('--mf'));
          if(mf>16)meta.style.setProperty('--mf',(mf-.5)+'px');
          else if(bs>16)meta.style.setProperty('--bs',(bs-2)+'px');
          else if(mf>10)meta.style.setProperty('--mf',(mf-.5)+'px');else break;
        }
        meta.setAttribute('data-fit',meta.textContent+'|'+(bd?bd.children.length:0)+'|'+meta.clientWidth);
      }
    }
    var h1=hero.querySelector('.hid h1'),rl=hero.querySelector('.hid .rankline');
    var s2=(h1?h1.textContent:'')+(hero.querySelector('.hid').clientWidth)+'|'+(rl?rl.textContent:'')+'|'+hero.clientWidth;
    if(hero.getAttribute('data-fit2')!==s2){
      if(h1){h1.style.whiteSpace='nowrap';h1.style.setProperty('font-size','48px','important');var hs=48,n1=0;var avH=function(){var hb=hero.getBoundingClientRect(),eb=h1.getBoundingClientRect();return hb.right-16-eb.left};while(h1.scrollWidth>h1.clientWidth+1&&hs>18&&n1++<40){hs-=1;h1.style.setProperty('font-size',hs+'px','important')}}
      if(rl){rl.classList.remove('lkTight');rl.style.flexWrap='nowrap';rl.style.removeProperty('justify-content');rl.style.setProperty('font-size','19px','important');var rf=19,n2=0;var avR=function(e){var hb=hero.getBoundingClientRect(),eb=e.getBoundingClientRect();return hb.right-16-eb.left};while(rl.scrollWidth>rl.clientWidth+1&&rf>9&&n2++<40){rf-=.25;rl.style.setProperty('font-size',rf+'px','important')}if(rl.scrollWidth>rl.clientWidth+1){rl.classList.add('lkTight');rf=Math.min(rf+2,13);n2=0;rl.style.setProperty('font-size',rf+'px','important');while(rl.scrollWidth>rl.clientWidth+1&&rf>7&&n2++<40){rf-=.25;rl.style.setProperty('font-size',rf+'px','important')}}else{rl.classList.remove('lkTight')}}
      hero.setAttribute('data-fit2',s2);
    }
    var pj=hero.querySelector('.heroPj'),ol=hero.querySelector('.hol .outlook'),pm=body.querySelector('.pgMin');
    var pt=pj?pj.textContent.replace(/\s+/g,' ').trim():'';
    var m=pt.match(/projection\s*([\d.]+)\s*(PPR|Half|Std)?/i);
    var proj=m?m[1]:'',fmt=m&&m[2]?m[2]:'PPR';
    var wk=(pt.match(/Week\s*(\d+)/i)||[])[1]||'';
    var tag=ol?ol.textContent.replace(/\s*this week/i,'').trim():'';
    var gm=pm&&pm.querySelector('b')?pm.querySelector('b').textContent.trim():'';
    var im=body.querySelector('.injMini');var warn=!!im;
    var sig=[proj,fmt,wk,tag,gm,warn,hero.clientWidth].join('|');
    var el=document.getElementById('lkInfo');
    if(el&&el.getAttribute('data-sig')===sig&&el.previousElementSibling===hero)return;
    if(el)el.remove();
    el=document.createElement('div');el.id='lkInfo';el.className='lkInfo n'+((proj?1:0)+(tag?1:0)+(gm?1:0));el.setAttribute('data-sig',sig);
    el.innerHTML=(proj?'<button type="button" class="liP"><small>Week '+wk+' projection</small><span><b>'+proj+'</b> <em>'+fmt+'</em> <i>\u203A</i></span></button>':'')
      +(tag?'<div class="liR"><small>This week</small><span><b>'+tag+'</b></span></div>':'')
      +(gm?'<button type="button" class="liG'+(warn?' warn':'')+'"><small>Games analyzed</small><span><b>'+gm+'</b> <u>Edit</u></span></button>':'');
    var p=el.querySelector('.liP');if(p)p.onclick=function(){if(pj)pj.click()};
    var g=el.querySelector('.liG');if(g)g.onclick=function(){var q=body.querySelector('.injMini');if(q)q.classList.add('lkShow');if(pm)pm.click()};
    hero.parentNode.insertBefore(el,hero.nextSibling);
    liFit(el,'--if',15,9.5,.5,function(){var o=false;[].forEach.call(el.children,function(c){if(c.scrollWidth>c.clientWidth+1)o=true});return o||el.scrollWidth>el.clientWidth+1});
  }

  /* tap anywhere outside a badge popup closes it */
  document.addEventListener('click',function(e){
    var t=e.target;if(t.closest&&(t.closest('.bdgBtn')||t.closest('.bdgD')))return;
    [].forEach.call(document.querySelectorAll('.bdgD:not([hidden])'),function(d){d.hidden=true;d.innerHTML=''});
    [].forEach.call(document.querySelectorAll('.bdgBtn.on'),function(b){b.classList.remove('on')});
  },true);
  setInterval(function(){if(!document.hidden)try{liInfo()}catch(e){}},500);

  /* ===== homepage: match the new mock (subtitle on the Who-should-I-start card, Explore heading) ===== */
  function homeMock(){try{
    var ss=document.getElementById('hmSS');
    if(ss){var h=ss.querySelector('.hsH');if(h&&!ss.querySelector('.lkHsSub')){var sub=document.createElement('div');sub.className='lkHsSub';sub.textContent='Compare two players for this week.';h.parentNode.insertBefore(sub,h.nextSibling)}}
    var g=document.getElementById('hmGrid');
    if(g&&!document.getElementById('lkExploreHd')){var hd=document.createElement('div');hd.id='lkExploreHd';hd.className='lkExploreHd';hd.textContent='Explore FantasyMania';g.parentNode.insertBefore(hd,g)}
    var h1=document.querySelector('#home .hhL>h1');
    if(h1&&!document.getElementById('lkTagline')){var tl=document.createElement('div');tl.id='lkTagline';tl.textContent='Player ratings. Weekly projections. Smarter decisions.';h1.parentNode.insertBefore(tl,h1.nextSibling)}
    if(h1&&!h1.dataset.lkWrap){
      var tn=h1.firstChild;
      if(tn&&tn.nodeType===3){
        var t=tn.textContent,ix=t.toLowerCase().indexOf('accurate');
        if(ix>-1){
          ix+='accurate'.length;
          var before=t.slice(0,ix),after=t.slice(ix);
          tn.textContent=before;
          var br1=document.createElement('br');h1.insertBefore(br1,tn.nextSibling);
          h1.insertBefore(document.createTextNode(after),br1.nextSibling)
        }
      }
      var span=h1.querySelector('.lkBl');
      if(span){var br2=document.createElement('br');h1.insertBefore(br2,span)}
      h1.dataset.lkWrap='1'
    }
  }catch(e){}}
  setInterval(function(){if(!document.hidden)homeMock()},800);
})();
