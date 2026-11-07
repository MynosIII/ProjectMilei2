from pathlib import Path
import re,json,shutil,sys
P=Path(sys.argv[1])
R=Path(sys.argv[2])
for sub in ['assets','data','scripts']:(R/sub).mkdir(parents=True,exist_ok=True)
source=(P/'PJM_analisis_avanzado.html').read_text(encoding='utf8')
summary=json.loads((P/'resumen_resultados.json').read_text(encoding='utf8'))
source=source.replace('977 filas',str(summary['n'])+' filas').replace('las 977 respuestas','las '+str(summary['n'])+' respuestas')
scripts=re.findall(r'<script>(.*?)</script>',source,re.S)
assert len(scripts)==2
js=scripts[1]
prefix='const figures='
end=js.index(';const drawn=')
data=json.loads(js[len(prefix):end])
assert len(data)==32
(R/'data'/'charts.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
(R/'assets'/'plotly.min.js').write_text(scripts[0],encoding='utf8')
app="""'use strict';
(async()=>{
try {
 const response=await fetch('./data/charts.json',{cache:'no-cache'});
 if(!response.ok)throw new Error('HTTP '+response.status);
 const figures=await response.json();
 document.getElementById('loading-data').hidden=true;
"""+js[end+1:]+"""
}catch(error){
 const element=document.getElementById('loading-data');
 element.hidden=false;element.textContent='No se pudo cargar la base de gráficos. Recargá la página. Detalle: '+error.message;
}
})();
"""
(R/'assets'/'app.js').write_text(app,encoding='utf8')
source=source.replace('<script>'+scripts[0]+'</script>','<script src="./assets/plotly.min.js" defer></script>')
source=source.replace('<script>'+scripts[1]+'</script>','<script src="./assets/app.js" defer></script>')
source=source.replace('<title>PJM · Explorador de opinión</title>','<title>ProjectMilei2 · Explorador de opinión</title><meta name="description" content="Prueba de concepto de análisis de opinión: 32 visualizaciones interactivas de posiciones políticas, identidades declaradas y diferencias sociales.">')
source=source.replace('<div class="filters">','<p id="loading-data" role="status">Cargando la base de gráficos…</p><p><a href="./data/manifest.json">Base y archivos descargables</a> · <a href="./data/charts.json" download>Datos de los 32 gráficos (JSON)</a></p><div class="filters">')
source=source.replace('section{background:', 'section{overflow-x:auto;background:')
source=source.replace('.plot{width:100%}', '.plot{width:100%;min-width:720px}a{color:#55d5ba}')
(R/'index.html').write_text(source,encoding='utf8')
(R/'.nojekyll').write_text('',encoding='utf8')
files=['asociaciones_cramer.csv','asociaciones_eta2.csv','dispersion_identidades.csv','diversidad_preguntas.csv','identificacion_peronista_agregada.csv','preguntas_identidad_cramer.csv','clave_auditada.csv','resumen_resultados.json']
for f in files:shutil.copyfile(P/f,R/'data'/f)
manifest={'project':'ProjectMilei2','purpose':'Prueba de concepto para presentación','responses':summary['n'],'charts':32,'unit':'Una respuesta a la encuesta','public_database':'charts.json','database_format':'Diccionario de figuras Plotly con datos calculados; no es una copia de filas sociodemográficas individuales','tables':files,'scoring':'La clave por número de pregunta y opción se conserva en clave_auditada.csv','source':'PJM 2 (Responses), versión utilizada en el análisis','limitations':['Análisis exploratorio de esta base; sin ponderación poblacional','El enlace numérico de la clave usa el orden actual del formulario','Los referentes se muestran por centros de seguidores, sin respuestas simuladas'],'privacy':'No incluye fechas, contactos ni una tabla individual que vincule demografía y política. Las figuras espaciales contienen coordenadas sin identificadores de personas.'}
(R/'data'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'directory':str(R),'charts':len(data),'files':[str(f.relative_to(R)) for f in R.rglob('*') if f.is_file()],'html_bytes':(R/'index.html').stat().st_size}))
