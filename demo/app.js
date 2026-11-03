(function(){
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const tip=$('tip');
function showTip(text,x,y){tip.textContent=text;tip.hidden=false;const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(Math.max(8,x+12),innerWidth-r.width-8)+'px';tip.style.top=Math.max(8,y-r.height-10)+'px'}
function hideTip(){tip.hidden=true}
function el(tag,attrs,parent,text){const n=document.createElementNS(NS,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(text!=null)n.textContent=text;if(parent)parent.appendChild(n);return n}
function bindTip(node,label){node.addEventListener('pointermove',e=>showTip(label,e.clientX,e.clientY));node.addEventListener('pointerleave',hideTip)}
function focusable(node,label){node.setAttribute('tabindex','0');node.setAttribute('role','img');node.setAttribute('aria-label',label);node.classList.add('hit');bindTip(node,label);
  node.addEventListener('focus',()=>{const r=node.getBoundingClientRect();showTip(label,r.left+r.width/3,r.top)});node.addEventListener('blur',hideTip)}
function html(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!=null)n.textContent=text;return n}

/* one row per arm: a dot per seed, a mean tick, and a mean +- std band */
function strip(host,cfg){
  const {rows,key,dom,ticks,fmt,refs=[],title}=cfg;
  const W=520,rowH=46,L=112,R=14,T=refs.length?20:6,H=T+rows.length*rowH+24,x=v=>L+(W-L-R)*(v-dom[0])/(dom[1]-dom[0]);
  const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'group','aria-label':title},host);
  const g=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  ticks.forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:H-22},g);el('text',{x:x(v),y:H-7,'text-anchor':'middle'},a,fmt(v))});
  refs.forEach(r=>{el('line',{x1:x(r.v),x2:x(r.v),y1:T-6,y2:H-22,stroke:'var(--ink)','stroke-dasharray':'3 3'},s);el('text',{x:x(r.v),y:T-9,class:'rs','text-anchor':r.anchor||'middle'},s,r.label)});
  rows.forEach((row,i)=>{const y=T+i*rowH+rowH/2-4,sum=row.arm.summary[key],color=`var(${row.color||'--gnn'})`;
    el('text',{x:0,y:y-3,class:'rl'},s,row.label);if(row.sub)el('text',{x:0,y:y+12,class:'rs'},s,row.sub);
    el('rect',{x:x(sum.mean-sum.std),y:y-11,width:Math.max(1,x(sum.mean+sum.std)-x(sum.mean-sum.std)),height:22,fill:color,opacity:.14,rx:2},s);
    const summary=`${row.label}: mean ${sum.mean.toFixed(4)}, std ${sum.std.toFixed(4)}, ${row.arm.n} seeds`;
    focusable(el('rect',{x:0,y:y-rowH/2+2,width:W,height:rowH-2},s),summary);
    el('line',{x1:x(sum.mean),x2:x(sum.mean),y1:y-13,y2:y+13,stroke:color,'stroke-width':2.5},s);
    row.arm.rows.forEach((r,j)=>{const c=el('circle',{cx:x(r[key]),cy:y+(j%2?4:-4),r:4,fill:color,stroke:'var(--bg)','stroke-width':1.5},s);bindTip(c,`${row.label}, seed ${r.seed}: ${r[key].toFixed(4)}. ${r.run}`)})});
  const d=html('details','dt');d.append(html('summary','', 'Seed values as a table'));
  const t=html('table');const head=html('tr');['Arm','Seed',key.replace('illicit_','').replace('fraud_','').toUpperCase()].forEach(h=>head.append(html('th','',h)));const th=html('thead');th.append(head);t.append(th);const tb=html('tbody');
  rows.forEach(row=>row.arm.rows.forEach(r=>{const tr=html('tr');tr.append(html('td','',row.label),html('td','',r.seed),html('td','',r[key].toFixed(4)));tb.append(tr)}));t.append(tb);d.append(t);host.append(d)}

function bootChart(host,boot){
  const W=520,rowH=24,L=62,R=14,T=4,H=T+rowH*8+24,dom=[0,.04],x=v=>L+(W-L-R)*(v-dom[0])/(dom[1]-dom[0]);
  const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'group','aria-label':'Paired bootstrap 95 percent intervals of the ROC-AUC difference by seed; none includes zero'},host);
  const g=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  [0,.01,.02,.03,.04].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:H-22},g);el('text',{x:x(v),y:H-7,'text-anchor':'middle'},a,v===0?'0':'+'+v.toFixed(2))});
  boot.forEach((b,i)=>{const y=T+i*rowH+rowH/2,[m,lo,hi]=b.roc;
    el('text',{x:0,y:y+4,class:'rs'},s,'seed '+b.seed);
    el('line',{x1:x(lo),x2:x(hi),y1:y,y2:y,stroke:'var(--gnn)','stroke-width':2.5,'stroke-linecap':'round'},s);
    el('circle',{cx:x(m),cy:y,r:4.5,fill:'var(--gnn)',stroke:'var(--bg)','stroke-width':1.5},s);
    focusable(el('rect',{x:0,y:y-rowH/2,width:W,height:rowH},s),`Seed ${b.seed}: ROC-AUC difference +${m.toFixed(4)}, 95% CI +${lo.toFixed(4)} to +${hi.toFixed(4)}. AUPRC difference +${b.auprc[0].toFixed(4)}, 95% CI +${b.auprc[1].toFixed(4)} to +${b.auprc[2].toFixed(4)}`)})}

fetch('data/results.json').then(r=>r.json()).then(d=>{
  const rf=d.ell1g0.rf,g1=d.ell1g1,g3=d.ell1g3,dg=d.dgf1g1;
  strip($('c-g1'),{title:'Illicit F1 by seed for random forest, GraphSAGE and GCN',key:'illicit_f1',dom:[.5,.85],ticks:[.5,.6,.7,.8],fmt:v=>v.toFixed(1),rows:[
    {label:'Random forest',sub:'tabular floor',arm:rf,color:'--floor'},{label:'GraphSAGE',sub:'frozen config',arm:g1.sage},{label:'GCN',sub:'comparator',arm:g1.gcn,color:'--gnn'}]});
  strip($('c-g3'),{title:'Illicit F1 by seed for the real graph and four ablations',key:'illicit_f1',dom:[.4,.85],ticks:[.4,.5,.6,.7,.8],fmt:v=>v.toFixed(1),
    refs:[{v:rf.summary.illicit_f1.mean,label:'forest 0.806',anchor:'end'}],rows:[
    {label:'real',sub:'true graph',arm:g3.real},{label:'no_edges',sub:'gated',arm:g3.no_edges,color:'--floor'},{label:'random',sub:'gated',arm:g3.random,color:'--floor'},{label:'config',sub:'reported',arm:g3.config,color:'--floor'},{label:'scrambled',sub:'gated, informed',arm:g3.scrambled,color:'--floor'}]});
  strip($('c-auc'),{title:'ROC-AUC by seed for DGF-1 and the two floors',key:'fraud_auc',dom:[.69,.77],ticks:[.70,.72,.74,.76],fmt:v=>v.toFixed(2),rows:[
    {label:'DGF-1',sub:'GraphSAGE, gated',arm:dg.gnn},{label:'Parity floor',sub:'XGBoost, gated',arm:dg.parity,color:'--floor'},{label:'Raw-17 floor',sub:'reported',arm:dg.raw17,color:'--floor'}]});
  strip($('c-pr'),{title:'AUPRC by seed for DGF-1 and the two floors, with the chance level',key:'fraud_auprc',dom:[.01,.045],ticks:[.01,.02,.03,.04],fmt:v=>v.toFixed(2),refs:[{v:.014809,label:'chance 0.0148'}],rows:[
    {label:'DGF-1',sub:'GraphSAGE, gated',arm:dg.gnn},{label:'Parity floor',sub:'XGBoost, gated',arm:dg.parity,color:'--floor'},{label:'Raw-17 floor',sub:'reported',arm:dg.raw17,color:'--floor'}]});
  bootChart($('c-boot'),d.bootstrap);
  /* local-94 */
  const body=document.querySelector('#t-local tbody');
  [['GNN on 94 local features','gnn_local94'],['GNN on all 165 features','gnn_all165'],['Random forest on all 165','rf_all165']].forEach(([label,k])=>{const a=d.local94[k],tr=html('tr');const th=html('th','',label);th.scope='row';tr.append(th);
    ['illicit_f1','illicit_precision','illicit_recall','illicit_auc'].forEach(f=>tr.append(html('td','n',a[f].toFixed(4))));body.append(tr)});
  /* decisions */
  $('reg-count').textContent=d.registry.rows;
  const list=$('adrs'),items=d.adrs.map(a=>{const li=html('li');li.append(html('span','id',a.id));const link=html('a','',a.title);link.href='https://github.com/coderback/cambium/blob/f6c415d/decisions/'+a.file;li.append(link);
    li.append(html('span','m',[a.state,a.date,a.erratum?'has erratum':null].filter(Boolean).join(' · ')));list.append(li);return {li,text:(a.id+' '+a.title+' '+a.state).toLowerCase()}});
  $('adr-q').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();let n=0;items.forEach(i=>{const on=!q||i.text.includes(q);i.li.hidden=!on;if(on)n++});$('adr-empty').hidden=n>0});
});
})();
