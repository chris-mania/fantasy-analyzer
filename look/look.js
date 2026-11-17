(function(){
  try{
    var l=document.createElement('link');l.rel='stylesheet';l.href='https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,800&display=swap';document.head.appendChild(l);
    var t=document.createElement('div');t.id='lkTick';document.body.insertBefore(t,document.body.firstChild);
    function sc(){try{var seen={},out=[];(DB||[]).forEach(function(p){if(p.played&&p.played.score&&!seen[p.played.score]){seen[p.played.score]=1;out.push(p.played.score.replace(',',''))}});
      t.innerHTML='<span><b>WEEK '+(META.next_week||'')+'</b>'+(out.length?'Final: '+out.slice(0,14).join(' · '):'Scores, ratings and who to start')+'</span>'}catch(e){}}
    var n=0,iv=setInterval(function(){if(typeof DB!=='undefined'&&typeof META!=='undefined'){sc();clearInterval(iv)}else if(++n>60)clearInterval(iv)},300);
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
        if(e.tagName==='IMG'||e.closest('.portrait,.hp,.crW,.initials,.thumb,.pctNum,.ring'))return;e.classList.add('lk-sq')})}catch(e){}
  }
  var pend=false,added=[];
  new MutationObserver(function(ms){ms.forEach(function(m){m.addedNodes.forEach(function(n){if(n.nodeType===1)added.push(n)})});
    if(pend)return;pend=true;setTimeout(function(){pend=false;lift();var a=added;added=[];a.forEach(function(n){if(document.body.contains(n))sq(n)})},350)}).observe(document.body,{childList:true,subtree:true});
  setTimeout(function(){lift();sq(document.body)},1200);
  var sv=window.showView; if(typeof sv==='function'){window.showView=function(){var r=sv.apply(this,arguments);setTimeout(function(){lift();sq(document.body)},600);return r}}
  var ct;document.addEventListener('click',function(){clearTimeout(ct);ct=setTimeout(function(){lift();sq(document.body)},700)},true);
})();
