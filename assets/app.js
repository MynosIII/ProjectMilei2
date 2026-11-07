'use strict';
(async()=>{
try {
 const response=await fetch('./data/charts.json',{cache:'no-cache'});
 if(!response.ok)throw new Error('HTTP '+response.status);
 const figures=await response.json();
 document.getElementById('loading-data').hidden=true;
const drawn=new Set();function draw(section){const id=section.id;if(drawn.has(id))return;const f=figures[id];if(!f)return;drawn.add(id);Plotly.newPlot('plot-'+id,f.data,f.layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}}).catch(e=>{document.getElementById('plot-'+id).textContent='No se pudo dibujar este gráfico: '+e.message;});}const obs=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)draw(e.target)}),{rootMargin:'500px'});document.querySelectorAll('section').forEach(s=>obs.observe(s));document.querySelectorAll('[data-cat]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-cat]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('section').forEach(s=>{s.hidden=b.dataset.cat!=='Todas'&&s.dataset.section!==b.dataset.cat;});window.dispatchEvent(new Event('resize'));}));window.addEventListener('hashchange',()=>{const s=document.querySelector(location.hash);if(s&&s.tagName==='SECTION'){s.hidden=false;draw(s);}});
}catch(error){
 const element=document.getElementById('loading-data');
 element.hidden=false;element.textContent='No se pudo cargar la base de gráficos. Recargá la página. Detalle: '+error.message;
}
})();
