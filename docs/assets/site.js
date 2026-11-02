/* Supplied observations, reported stage values, and explicitly recomputed searches. */
'use strict';
const research=window.RESEARCH,el=id=>document.getElementById(id);
const colors=['#7b8cde','#e96a73','#eaa05d','#aa7bcb','#59bf78','#af805e'];
const symbols=['circle','square','triangle-up','diamond','triangle-down','pentagon'];
const stages={'k = 1':'#70b7eb','k = 2':'#ac7be1','Sparsification':'#76bea5','Rationalization':'#e7b56d'};
const stageNames={'k = 1':'1 group','k = 2':'2 groups','Sparsification':'Sparsified','Rationalization':'Rationalized'};
const regimeName=r=>r==='global'?'Global':'Regime '+(Number.isInteger(Number(r))?String.fromCharCode(65+Number(r)):r);
let selected='mhd',selectedScope='global';
const layout={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#fff',font:{family:'Arial, sans-serif',size:14,color:'#29454a'},margin:{l:60,r:20,t:18,b:100},legend:{orientation:'h',y:-.25,yanchor:'top',x:.5,xanchor:'center',font:{size:13}},xaxis:{gridcolor:'#edf2ed',zeroline:false,tickfont:{size:14},title:{font:{size:17},standoff:14}},yaxis:{gridcolor:'#edf2ed',zeroline:false,tickfont:{size:14},title:{font:{size:17},standoff:12}}};
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',scale:2}};
const display=()=>{const c=research.cases[selected];return c.scopes?{...c,...c.scopes[selectedScope]}:c;};
function drawScatter(id,c,raw=false){
  const x=raw?c.raw_x:c.x,y=raw?c.raw_y:c.y,groups=[...new Set(c.group)];
  const traces=groups.map((name,j)=>{const indices=c.group.flatMap((g,i)=>g===name?[i]:[]);return {x:indices.map(i=>x[i]),y:indices.map(i=>y[i]),customdata:indices.map(i=>c.rows[i]),type:'scatter',mode:'markers',name,marker:{size:10,opacity:.86,color:name.includes('Rough turbulent')||name.includes('Outer')?'#d55e00':name.includes('Intermediate')?'#009e73':name.includes('Near-wall')||name.includes('Viscous')?'#0072b2':colors[j%colors.length],symbol:name.includes('Rough turbulent')||name.includes('Outer')?'triangle-up':name.includes('Intermediate')?'diamond':name.includes('Viscous')?'square':symbols[j%symbols.length],line:{width:.6,color:'white'}},hovertemplate:'x: %{x:.5g}<br>y: %{y:.5g}<br>CSV row: %{customdata}<extra>%{fullData.name}</extra>'};});
  const prefix=raw?'raw':'learned';
  Plotly.react(id,traces,{...structuredClone(layout),xaxis:{...layout.xaxis,title:{...layout.xaxis.title,text:raw?c.raw_label:'Π₁'},type:el(prefix+'-log-x').checked?'log':'linear'},yaxis:{...layout.yaxis,title:{...layout.yaxis.title,text:raw?c.output:c.output_formula?'Rescaled response':c.output},type:el(prefix+'-log-y').checked?'log':'linear'}},config);
}
function evidence(){
  const rows=research.reference.stages.filter(r=>r.case===selected),multiple=new Set(rows.map(r=>r.regime)).size>1,traces=[];
  for(const [stage,color] of Object.entries(stages)){
    const values=rows.filter(r=>r.stage===stage);if(!values.length)continue;
    for(const [index,metric] of ['score','mi'].entries())traces.push({type:'bar',name:stageNames[stage],legendgroup:stage,showlegend:multiple&&index===0,x:values.map(r=>multiple?regimeName(r.regime):stageNames[stage]),y:values.map(r=>r[metric]),xaxis:index?'x2':'x',yaxis:index?'y2':'y',width:multiple?.16:.46,error_y:{type:'data',array:values.map(r=>r[metric+'_sd']),visible:true,thickness:1,width:3},marker:{color}});
  }
  Plotly.react('evidence-chart',traces,{...structuredClone(layout),height:360,barmode:'group',bargap:.25,margin:{l:52,r:18,t:44,b:multiple?95:64},xaxis:{...layout.xaxis,domain:[0,.42]},xaxis2:{...layout.xaxis,domain:[.58,1],anchor:'y2'},yaxis:{...layout.yaxis,title:{text:'ε'}},yaxis2:{...layout.yaxis,anchor:'x2',title:{text:'MI / nats'}},annotations:[{text:'ε = exp[−max(MI, 0)]',x:.21,y:1.13,xref:'paper',yref:'paper',showarrow:false},{text:'Mutual information',x:.79,y:1.13,xref:'paper',yref:'paper',showarrow:false}],legend:{...layout.legend,y:-.25}},config);
  const tbody=el('evidence-table');tbody.replaceChildren();
  rows.forEach(r=>{const row=document.createElement('tr');[regimeName(r.regime),r.stage,`${r.mi.toFixed(3)} ± ${r.mi_sd.toFixed(3)}`,`${r.score.toFixed(3)} ± ${r.score_sd.toFixed(3)}`,r.r2===null?'—':r.r2.toFixed(3)].forEach(v=>{const td=document.createElement('td');td.textContent=v;row.append(td);});tbody.append(row);});
  el('evidence-source').textContent='Reported stage values and uncertainty. R² evaluates the selected representation.';
  el('limitation').hidden=selected!=='pipe';el('limitation').textContent='Regime B · Viscous: the rationalized Reynolds representation fails predictive validation (R² = −3.079).';
}
function landscapePlot(){
  const c=display(),p=c.landscape,z=p.z.map(row=>row.map(value=>-value)),lo=Math.min(...z.flat()),hi=Math.max(...z.flat());
  const colorbar={title:{text:'MI'},thickness:10,len:.76,tickfont:{size:12}};
  const traces2=[{type:'contour',x:p.x,y:p.y,z,colorscale:'Viridis',autocontour:false,contours:{start:lo,end:hi,size:Math.max((hi-lo)/45,1e-8),coloring:'fill',showlines:false},colorbar}];
  const traces3=[{type:'surface',x:p.x,y:p.y,z,colorscale:'Viridis',colorbar}],run=c.trajectory;
  if(run?.trace.length){
    const t=run.initial?[run.initial,...run.trace]:run.trace,x=t.map(v=>v.mean_parameters[0]),y=t.map(v=>v.mean_parameters[1]),mi=t.map(v=>v.mean_mi);
    traces2.push({type:'scatter',x,y,mode:'lines',line:{color:'#E63946',width:3}},{type:'scatter',x:[x[0],x.at(-1)],y:[y[0],y.at(-1)],mode:'markers',marker:{size:[8,10],symbol:['circle','diamond'],color:['white','#f4c95d'],line:{color:'#19344f',width:1.2}}});
    traces3.push({type:'scatter3d',x,y,z:mi,mode:'lines',line:{color:'#E63946',width:5}},{type:'scatter3d',x:[x[0],x.at(-1)],y:[y[0],y.at(-1)],z:[mi[0],mi.at(-1)],mode:'markers',marker:{size:[3.5,4.5],symbol:['circle','diamond'],color:['white','#f4c95d'],line:{color:'#19344f',width:1}}});
    Plotly.react('convergence-chart',[{type:'scatter',x:run.trace.map(v=>v.generation),y:run.trace.map(v=>-v.best_objective),mode:'lines',line:{color:'#507e74',width:2}}],{...structuredClone(layout),showlegend:false,margin:{l:60,r:15,t:18,b:65},xaxis:{...layout.xaxis,title:{text:'CMA-ES generation'}},yaxis:{...layout.yaxis,title:{text:'Best MI / nats'}}},config);
  }
  const base={...structuredClone(layout),showlegend:false,margin:{l:55,r:48,t:18,b:65},xaxis:{...layout.xaxis,type:'linear',constrain:'domain',title:{text:'c₁',standoff:14}},yaxis:{...layout.yaxis,type:'linear',scaleanchor:'x',scaleratio:1,constrain:'domain',title:{text:'c₂',standoff:12}}};
  Plotly.react('landscape-contour-chart',traces2,structuredClone(base),config);
  Plotly.react('landscape-chart',traces3,{...structuredClone(base),margin:{l:3,r:48,t:18,b:65},scene:{aspectmode:'manual',aspectratio:{x:1,y:1,z:.82},xaxis:{title:{text:'c₁',font:{size:17}},nticks:4},yaxis:{title:{text:'c₂',font:{size:17}},nticks:4},zaxis:{title:{text:'MI',font:{size:17}},nticks:4},camera:{eye:{x:1.55,y:1.55,z:1.18}}}},config);
  el('landscape-title').textContent=research.cases[selected].title+(selectedScope==='global'?'':' · Regime '+selectedScope+' · '+c.name);
  el('landscape-note').textContent=`${p.rows} samples · ${p.method} (${p.param}) · Fixed two-coefficient slice. Paths are new searches`+(selectedScope==='global'?'.':' within this detected physical regime.');
}
function drawCase(){
  const c=display(),original=research.cases[selected];el('case-meta').textContent=`${original.n_raw.toLocaleString()} observations`+(selectedScope==='global'?'':` · ${c.n_scope} in Regime ${selectedScope}`);
  const formula=el('case-formula');formula.replaceChildren();for(const path of c.formula_assets||[]){const image=document.createElement('img');image.src=path;image.alt='Reported dimensionless representation';formula.append(image);}
  el('case-note').textContent=c.exponent_note||'';
  const defaults=c.axis_defaults||[[true,true],[true,true]];
  for(const [i,prefix] of ['raw','learned'].entries()){el(prefix+'-log-x').checked=defaults[i][0];el(prefix+'-log-y').checked=defaults[i][1];}
  drawScatter('raw-chart',c,true);drawScatter('collapse-chart',c);evidence();landscapePlot();
}
function choose(key){
  selected=key;const c=research.cases[key];selectedScope=c.scopes?'A':'global';
  for(const b of el('case-list').children){const active=b.dataset.key===key;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;}
  el('case-title').textContent=c.title;el('case-type').textContent=c.category;el('case-question').textContent=c.system;
  el('case-schematic').src='assets/schematics/'+key+'.png';el('case-schematic').alt=c.short+' physical schematic';el('download-data').href='data/'+c.file;
  const picker=el('learning-scope');picker.replaceChildren();picker.parentElement.hidden=!c.scopes;
  for(const [scope,record] of Object.entries(c.scopes||{})){const option=document.createElement('option');option.value=scope;option.textContent='Regime '+scope+' · '+record.name;picker.append(option);}
  document.querySelectorAll('.atlas-card').forEach(card=>card.classList.toggle('selected',card.dataset.case===key));drawCase();
}
function initializeAtlas(){
  const atlas=el('case-atlas');
  Object.entries(research.cases).forEach(([key,c])=>{
    const card=document.createElement('article');card.className='atlas-card';card.dataset.case=key;card.dataset.family=c.category;
    const image=document.createElement('img');image.src='assets/schematics/'+key+'.png';image.alt=c.short+' physical schematic';image.loading='lazy';card.append(image);
    const body=document.createElement('div');body.className='atlas-body';const meta=document.createElement('div');meta.className='atlas-meta';const tag=document.createElement('span');tag.className='tag '+c.category.toLowerCase().replaceAll(' ','-');tag.textContent=c.category;const n=document.createElement('span');n.textContent=c.n_raw.toLocaleString()+' observations';meta.append(tag,n);body.append(meta);
    const title=document.createElement('h3');title.textContent=c.short;body.append(title);const result=document.createElement('p');result.textContent=c.result;body.append(result);
    const button=document.createElement('button');button.className='atlas-open';button.textContent='Explore results ↗';button.setAttribute('aria-label','Explore '+c.short);button.addEventListener('click',()=>{choose(key);el('benchmarks').scrollIntoView({behavior:'smooth',block:'start'});});body.append(button);card.append(body);atlas.append(card);
  });
  document.querySelectorAll('.atlas-filters button').forEach(button=>button.addEventListener('click',()=>{const family=button.dataset.family;document.querySelectorAll('.atlas-filters button').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});atlas.querySelectorAll('.atlas-card').forEach(card=>card.hidden=family!=='All'&&card.dataset.family!==family);}));
}
Object.entries(research.cases).forEach(([key,c])=>{const b=document.createElement('button');b.type='button';b.role='tab';b.dataset.key=key;b.textContent=c.title;const s=document.createElement('small');s.textContent=c.category;b.append(s);b.addEventListener('click',()=>choose(key));el('case-list').append(b);});
for(const prefix of ['raw','learned'])for(const axis of ['x','y'])el(prefix+'-log-'+axis).addEventListener('change',()=>drawScatter(prefix==='raw'?'raw-chart':'collapse-chart',display(),prefix==='raw'));
el('learning-scope').addEventListener('change',e=>{selectedScope=e.target.value;drawCase();});
el('copy-command').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(el('quickstart').textContent);el('copy-command').textContent='Copied';setTimeout(()=>el('copy-command').textContent='Copy',1800);}catch{el('copy-command').textContent='Select text below';}});
el('total-n').textContent=Object.values(research.cases).reduce((s,c)=>s+c.n_raw,0).toLocaleString();
initializeAtlas();choose('mhd');
