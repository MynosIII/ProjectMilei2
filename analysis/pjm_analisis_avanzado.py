"""PJM: análisis exploratorio agregado; ejecutar con numpy pandas scipy scikit-learn plotly.
Entradas: datos_ampliados.json, datos_pjm_exacto.json.
Salida: PJM_analisis_avanzado.html, tablas agregadas y resumen_resultados.json.
"""
from pathlib import Path
import json,html,re,itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,spearmanr
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,leaves_list
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from pjm_grafico_corregido import norm,family,COLORS

P=Path(__file__).parent
RNG=np.random.default_rng(20261106)
raw=json.loads((P/'datos_pjm_exacto.json').read_text(encoding='utf8'))
df=pd.DataFrame(json.loads((P/'datos_ampliados.json').read_text(encoding='utf8')))
mapping=raw['mapping']; dims=['Economía','Social','Político']
Q=np.array([[mapping[q]['form_options'].index(a) if a in mapping[q]['form_options'] else -1 for q,a in enumerate(ans)] for ans in df.answers])
vectors=np.array([[mapping[q]['answers'].get(a,[np.nan]*3) for q,a in enumerate(ans)] for ans in df.answers])
maximum=np.array([sum(max(v[d] for v in c['answers'].values()) for c in mapping) for d in range(3)])
valid=np.isfinite(vectors).all(axis=(1,2))
assert valid.all(), 'Hay respuestas que no corresponden al formulario; revisar antes de graficar.'
df[dims]=vectors.sum(axis=1)/maximum*100
df['Identidad']=df.identity.map(family)
df['Edad']=df.edad.fillna('Sin dato').str.strip()
df['Género']=df.genero.fillna('Sin dato').str.strip().replace({'Helicóptero Apache':'Otra respuesta'})
df['Ingresos']=df.ingresos.fillna('Sin dato').str.strip()
df['Educación']=df.educacion.fillna('Sin dato').str.strip()
df['Residencia']=df.residencia.fillna('Sin dato').str.strip()
def channel(x):
    t=norm(x)
    if 'reddit' in t:return 'Reddit'
    if t=='x (twitter)':return 'X / Twitter'
    if t=='television':return 'Televisión'
    if t=='radio':return 'Radio'
    if t=='instagram':return 'Instagram'
    if any(w in t for w in ['youtube','twitch','streaming']):return 'Video / streaming'
    if any(w in t for w in ['online','internet','fact-check']):return 'Web / otras fuentes'
    return 'Otras fuentes'
df['Fuente']=df.plataforma.map(channel)
def region(x):
    t=norm(x)
    if t in ['caba','pba (gba)']:return 'AMBA'
    if t=='pba (interior)':return 'PBA interior'
    if t in ['jujuy','tucuman','salta','santiago del estero','catamarca','la rioja']:return 'NOA'
    if t in ['misiones','corrientes','chaco','formosa']:return 'NEA'
    if t in ['mendoza','san juan','san luis']:return 'Cuyo'
    if t in ['chubut','neuquen','rio negro','santa cruz','tierra del fuego','la pampa']:return 'Patagonia / La Pampa'
    if t in ['cordoba','santa fe','entre rios']:return 'Centro / Litoral'
    return 'Sin dato / otra'
df['Región']=df.Residencia.map(region)
df['Problema']=df.problema.fillna('Sin dato').str.strip()
ages=['-15','15-18','18-25','25-30','30-40','40-50','50-60','60-70','+70']
incomes=['-200.000','200.000-500.000','500.000-1.000.000','1.000.000-2.000.000','2.000.000-5.000.000','+5.000.000','Estoy desempleado']
demographics=['Edad','Género','Ingresos','Educación','Región','Fuente']
ratings=pd.DataFrame(df.ratings.tolist(),columns=['Milei','Villarruel','C. Kirchner','Kicillof','Jorge Macri','M. Macri','Moreno','Bregman','Grabois']).apply(pd.to_numeric,errors='coerce')
invalid_ratings=int(((ratings<0)|(ratings>10)).sum().sum())
ratings=ratings.where((ratings>=0)&(ratings<=10))
charts=[]; aggregates=[]
def add(fig,title,section,explanation,height=620):
    idx=len(charts)+1
    fig.update_layout(template='plotly_dark',paper_bgcolor='#111a28',plot_bgcolor='#111a28',font=dict(family='Arial',size=13,color='#e9eef6'),height=height,margin=dict(l=90,r=45,t=55,b=100),title=None,legend_title_text='',colorway=['#46d3b3','#6aa9ff','#efb65e','#cf8aef','#ef8397'],hoverlabel=dict(font_size=13))
    charts.append({'id':f'g{idx}','title':title,'section':section,'explanation':explanation,'figure':json.loads(fig.to_json())})
def ordercat(series,name):
    available=series.value_counts()
    base=ages if name=='Edad' else incomes if name=='Ingresos' else available.index.tolist()
    return [c for c in base if available.get(c,0)>=10]
def heat(z,x,y,limit=None,colors='Blues',fmt='.1f',counts=None):
    z=np.asarray(z,float)
    tr=go.Heatmap(z=z,x=list(x),y=list(y),colorscale=colors,text=np.where(np.isfinite(z),np.vectorize(lambda a:format(a,fmt))(z),''),texttemplate='%{text}',hoverongaps=False)
    if limit:tr.update(zmin=limit[0],zmax=limit[1])
    if counts is not None:tr.update(customdata=np.asarray(counts),hovertemplate='%{y} / %{x}<br>Valor: %{z:.2f}<br>n=%{customdata}<extra></extra>')
    f=go.Figure(tr);f.update_layout(margin=dict(l=220,b=160));return f
def smallcats(col):
    s=df[col].fillna('Sin dato');counts=s.value_counts();return s.where(s.map(counts)>=10,'Otras / n<10')

# 1. Espacio tridimensional con nubes descriptivas opcionales.
fig=go.Figure()
for group,g in df.groupby('Identidad'):
    fig.add_trace(go.Scatter3d(x=g[dims[0]],y=g[dims[1]],z=g[dims[2]],mode='markers',name=f'{group} (n={len(g)})',legendgroup=group,marker=dict(size=3.5,color=COLORS[group],opacity=.64),hovertemplate=html.escape(group)+'<br>Economía %{x:.1f}<br>Social %{y:.1f}<br>Político %{z:.1f}<extra></extra>'))
    if len(g)>=10:
        pts=g[dims].to_numpy();val,vec=np.linalg.eigh(np.cov(pts.T))
        if val.min()>1e-7:
            u,v=np.mgrid[0:2*np.pi:24j,0:np.pi:14j];sphere=np.array([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)])
            ell=(vec@np.diag(np.sqrt(val))@sphere.reshape(3,-1)).reshape(sphere.shape)+pts.mean(axis=0)[:,None,None]
            fig.add_trace(go.Surface(x=ell[0],y=ell[1],z=ell[2],opacity=.12,colorscale=[[0,COLORS[group]],[1,COLORS[group]]],showscale=False,showlegend=False,legendgroup=group,hoverinfo='skip',visible=False))
fig.update_layout(scene={a:dict(title=d,range=[0,100]) for a,d in zip(['xaxis','yaxis','zaxis'],dims)},updatemenus=[dict(type='buttons',direction='left',buttons=[dict(label='Solo puntos',method='update',args=[{'visible':[t.type!='surface' for t in fig.data]}]),dict(label='Puntos + nubes',method='update',args=[{'visible':[True]*len(fig.data)}])])])
add(fig,'Mapa 3D de posiciones y nubes por identidad','Espacio político','Cada punto es una respuesta. Los elipsoides describen una desviación estándar; no son fronteras de grupos. Las tres dimensiones conservan la clave literal del archivo.',800)

# 2. Distancia entre centros, reordenada por similitud.
centers=df.groupby('Identidad')[dims].mean();sizes=df.Identidad.value_counts();centers=centers.loc[sizes[sizes>=10].index]
dist=squareform(pdist(centers));idx=leaves_list(linkage(pdist(centers),method='average'));labels=centers.index[idx]
add(heat(dist[np.ix_(idx,idx)],labels,labels,fmt='.0f'),'Distancias entre centros de identidades','Espacio político','Distancia euclídea entre promedios de los tres ejes de 0 a 100. El orden reúne grupos cercanos. Una media cercana puede ocultar distribuciones diferentes.',760)

# 3. Dispersión por eje, no confundir cantidad de personas con volumen.
sd=df.groupby('Identidad')[dims].std().loc[centers.index]
add(heat(sd.values,dims,[f'{g} · n={sizes[g]}' for g in sd.index]),'Amplitud de cada identidad en los tres ejes','Espacio político','Desviación estándar: valores altos indican mayor dispersión en ese eje. Permite distinguir grupos amplios en economía de grupos amplios en otras dimensiones.',650)
spread=[]
for g,sub in df.groupby('Identidad'):
    if len(sub)>=10:
        a=sub[dims].to_numpy();r=np.sqrt(((a-a.mean(axis=0))**2).sum(axis=1));spread.append({'Identidad':g,'n':len(a),'Dispersión':float(np.sqrt((r*r).mean())),'P90':float(np.quantile(r,.9))})
sp=pd.DataFrame(spread).sort_values('Dispersión')
fig=px.scatter(sp,x='Dispersión',y='Identidad',size='n',color='Identidad',color_discrete_map=COLORS,hover_data=['n','P90'],size_max=38)
add(fig,'Qué grupos abarcan un espacio más amplio','Espacio político','Distancia cuadrática media al centro propio. El tamaño de la burbuja muestra n; la posición horizontal muestra dispersión, no tamaño poblacional.')
aggregates.append(('dispersion_identidades.csv',sp))

# 5. PCA one-hot: no impone orden a las tres opciones.
O=OneHotEncoder(sparse_output=False,handle_unknown='error').fit_transform(Q)
pca=PCA(n_components=2).fit(O);pc=pca.transform(O)
pdf=pd.DataFrame({'Componente 1':pc[:,0],'Componente 2':pc[:,1],'Identidad':df.Identidad})
fig=px.scatter(pdf,x='Componente 1',y='Componente 2',color='Identidad',color_discrete_map=COLORS,opacity=.48)
fig.update_traces(marker_size=5)
fig.update_xaxes(title=f'Componente 1 · {pca.explained_variance_ratio_[0]:.1%} de variación')
fig.update_yaxes(title=f'Componente 2 · {pca.explained_variance_ratio_[1]:.1%} de variación')
add(fig,'Una segunda geometría: PCA de las 90 opciones','Espacio político','Se codifica cada opción como presencia/ausencia y se proyectan las respuestas, sin usar los puntajes originales. Los colores provienen de identidades declaradas. Los ejes no tienen un significado ideológico preasignado.',720)

# 6. Entropía de cada pregunta.
ent=[]
for q,c in enumerate(mapping):
    probs=np.bincount(Q[:,q],minlength=3)/len(df);H=-sum(p*np.log2(p) for p in probs if p>0)/np.log2(3)
    ent.append({'Pregunta':c['question'],'Diversidad':H*100,'Opción mayoritaria (%)':probs.max()*100,'n':len(df)})
en=pd.DataFrame(ent).sort_values('Diversidad')
fig=px.bar(en,x='Diversidad',y='Pregunta',orientation='h',hover_data=['Opción mayoritaria (%)','n'])
fig.update_xaxes(range=[0,100],title='Entropía normalizada · 0 = unanimidad; 100 = reparto igual')
add(fig,'Preguntas con más consenso y más diversidad','Espacio político','La entropía mide el reparto entre opciones, no cuán extremas son las posiciones. Un reparto equilibrado puede indicar diversidad sin demostrar polarización.',1100)
aggregates.append(('diversidad_preguntas.csv',en))

# 7–12. Composición de identidades por variable, porcentajes dentro de cada categoría.
for col in demographics:
    cats=ordercat(df[col],col);ct=pd.crosstab(df[col],df.Identidad).reindex(cats).fillna(0);percent=ct.div(ct.sum(axis=1),axis=0)*100
    fig=go.Figure()
    for g in percent.columns:fig.add_trace(go.Bar(y=[f'{x} · n={ct.loc[x].sum():.0f}' for x in cats],x=percent[g],name=g,orientation='h',marker_color=COLORS[g],customdata=ct[g],hovertemplate='%{y}<br>%{fullData.name}: %{x:.1f}%<br>Respuestas: %{customdata}<extra></extra>'))
    fig.update_layout(barmode='stack',legend=dict(orientation='h',y=-.3),margin=dict(l=250,b=200));fig.update_xaxes(range=[0,100],title='% dentro de cada categoría')
    add(fig,f'Identidades declaradas según {col.lower()}','Cruces sociales','Cada barra suma 100%. La base es el total de respuestas de esa categoría; se muestran categorías con n≥10. Asociación descriptiva dentro de esta muestra.',max(630,len(cats)*40+300))

# 13. Medias sociodemográficas con intervalos por remuestreo.
for col in ['Edad','Género','Ingresos','Educación']:
    cats=ordercat(df[col],col);fig=make_subplots(rows=1,cols=3,subplot_titles=dims,shared_yaxes=True,horizontal_spacing=.07)
    for j,d in enumerate(dims):
        m=[];low=[];high=[];ns=[]
        for cat in cats:
            a=df.loc[df[col].eq(cat),d].to_numpy();boot=RNG.choice(a,(800,len(a)),replace=True).mean(axis=1);m.append(a.mean());low.append(np.quantile(boot,.025));high.append(np.quantile(boot,.975));ns.append(len(a))
        fig.add_trace(go.Scatter(x=m,y=[f'{x} (n={n})' for x,n in zip(cats,ns)],mode='markers',marker=dict(size=8,color=['#6aa9ff','#efb65e','#cf8aef'][j]),error_x=dict(type='data',array=np.array(high)-m,arrayminus=np.array(m)-low),showlegend=False,hovertemplate='%{y}<br>Media %{x:.1f}<extra></extra>'),row=1,col=j+1)
        fig.update_xaxes(range=[0,100],row=1,col=j+1)
    fig.update_layout(margin=dict(l=255,b=80))
    add(fig,f'Promedios y variación del promedio por {col.lower()}','Cruces sociales','Puntos: medias de los ejes 0–100. Barras: percentiles 2,5 y 97,5 de 800 remuestreos dentro del grupo; describen estabilidad interna bajo remuestreo, no margen de error representativo de Argentina.',max(590,len(cats)*40+180))

# 17. Tamaños de asociación η² para categoría vs puntajes.
eta=[]
for col in demographics+['Identidad','Residencia']:
    for d in dims:
        grand=df[d].mean();total=((df[d]-grand)**2).sum();between=sum(len(g)*(g[d].mean()-grand)**2 for _,g in df.groupby(col))
        eta.append({'Variable':col,'Dimensión':d,'Eta2':between/total,'n':len(df)})
et=pd.DataFrame(eta);mat=et.pivot(index='Variable',columns='Dimensión',values='Eta2').reindex(columns=dims)
add(heat(mat.values,dims,mat.index,limit=(0,1),fmt='.2f'),'Qué variables separan más los puntajes','Asociaciones','η²: proporción de variación del puntaje asociada a diferencias entre promedios de categorías. No es causal ni una medida de predicción individual; categorías numerosas pueden elevarla.')
aggregates.append(('asociaciones_eta2.csv',et))

# 18. Cramér V con corrección por sesgo y p BH global de pares categóricos.
variables=demographics+['Identidad','Problema'];catdf=pd.DataFrame({c:smallcats(c) for c in variables});V=np.eye(len(variables));pairs=[]
for i,j in itertools.combinations(range(len(variables)),2):
    ct=pd.crosstab(catdf.iloc[:,i],catdf.iloc[:,j]);chi,p,_,expected=chi2_contingency(ct,correction=False);n=ct.values.sum();r,k=ct.shape
    phi=max(0,chi/n-(k-1)*(r-1)/(n-1));rc=r-(r-1)**2/(n-1);kc=k-(k-1)**2/(n-1);v=np.sqrt(phi/min(rc-1,kc-1)) if min(rc,kc)>1 else 0
    V[i,j]=V[j,i]=v;pairs.append({'Variable A':variables[i],'Variable B':variables[j],'V corregida':v,'p':p,'n':int(n),'% esperados <5':float((expected<5).mean()*100)})
pa=pd.DataFrame(pairs);order=np.argsort(pa.p.values);adj=np.minimum.accumulate((pa.p.values[order]*len(pa)/np.arange(1,len(pa)+1))[::-1])[::-1];pa['q_BH']=0.;pa.loc[order,'q_BH']=np.minimum(adj,1)
add(heat(V,variables,variables,limit=(0,1),fmt='.2f'),'Red de asociaciones entre variables categóricas','Asociaciones','V de Cramér corregida por sesgo, de 0 a 1. Las categorías con menos de 10 respuestas se agrupan como otras. La tabla descargable incluye p, ajuste Benjamini–Hochberg y celdas esperadas escasas; no se usan estrellas de significación.',700)
aggregates.append(('asociaciones_cramer.csv',pa.sort_values('V corregida',ascending=False)))

# 19. Residuales: fuente de información e identidad.
ct=pd.crosstab(df.Fuente,df.Identidad);obs=ct.to_numpy();exp=np.outer(obs.sum(axis=1),obs.sum(axis=0))/obs.sum();res=(obs-exp)/np.sqrt(exp)
res[exp<5]=np.nan
add(heat(res,ct.columns,ct.index,limit=(-5,5),colors='RdBu_r',fmt='.1f',counts=obs),'Fuentes e identidades: sobrerrepresentación y ausencia relativa','Asociaciones','Residuo de Pearson: positivo significa más respuestas que bajo independencia; negativo, menos. Se ocultan celdas con frecuencia esperada menor que 5. Es un diagnóstico exploratorio, no una prueba individual por celda.',700)

# 20. Preferencia explícita peronista/kirchnerista, porcentaje y Wilson.
per=df.Identidad.isin(['Peronismo','Kirchnerismo']);forest=[]
for col in demographics:
    for cat in ordercat(df[col],col):
        mask=df[col].eq(cat);n=int(mask.sum());k=int(per[mask].sum());ph=k/n;z=1.96;den=1+z*z/n;cen=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        forest.append({'Variable':col,'Categoría':cat,'n':n,'k':k,'Porcentaje':ph*100,'Inferior':(cen-half)*100,'Superior':(cen+half)*100})
fo=pd.DataFrame(forest);fig=go.Figure()
for c in demographics:
    g=fo[fo.Variable.eq(c)];fig.add_trace(go.Scatter(x=g.Porcentaje,y=[c+' / '+x for x in g.Categoría],mode='markers',name=c,error_x=dict(type='data',array=g.Superior-g.Porcentaje,arrayminus=g.Porcentaje-g.Inferior),customdata=g[['n','k']],hovertemplate='%{y}<br>%{x:.1f}% · n=%{customdata[0]}, casos=%{customdata[1]}<extra></extra>'))
fig.add_vline(x=per.mean()*100,line_dash='dash',line_color='#d6dbe4');fig.update_xaxes(range=[0,100],title='% que se declara peronista o kirchnerista');fig.update_layout(margin=dict(l=350))
add(fig,'Dónde aparece más la identificación peronista o kirchnerista','Asociaciones','Solo se cuentan identidades explícitamente declaradas. Línea punteada: promedio de la muestra. Intervalos Wilson 95% bajo supuesto binomial, sin incorporar sesgo de selección ni ponderación. No es un predictor de personas.',max(1000,len(fo)*28))
aggregates.append(('identificacion_peronista_agregada.csv',fo))

# 21. Percepción de políticos: medias por identidad y divergencia entre nombres.
ratingmean=ratings.groupby(df.Identidad).mean();ratingcount=ratings.groupby(df.Identidad).count();ratingmean=ratingmean.where(ratingcount>=10)
add(heat(ratingmean.values,ratings.columns,ratingmean.index,limit=(0,10),fmt='.1f',counts=ratingcount.values),'Evaluación de dirigentes por identidad declarada','Dirigentes y medios','Promedio de las respuestas numéricas a cada dirigente (0–10). Se muestran celdas con al menos 10 respuestas válidas. No se presupone qué significa cada extremo más allá de la escala del cuestionario.',720)
add(heat(ratings.corr(method='spearman').values,ratings.columns,ratings.columns,limit=(-1,1),colors='RdBu_r',fmt='.2f'),'Dirigentes que reciben evaluaciones parecidas u opuestas','Dirigentes y medios','Correlación de Spearman entre evaluaciones individuales válidas. Una relación negativa indica ordenamientos opuestos de las puntuaciones; no identifica una causa ni una transferencia de votos.',680)

# 23. Medios multirrespuesta: uso declarado y lift de identidades, bases >=20.
known=['TN','C5N','A24','CANAL 9','CRONICA','LaDerechaDiario','Clarin','LaIzquierdaDiario','Infobae','La Nación','LN+','ElDestape']
M=pd.DataFrame({m:df.medios.fillna('').map(lambda x,m=m:norm(m) in [norm(s) for s in str(x).split(',')]) for m in known})
media=[]
for m in known:
    n=int(M[m].sum())
    if n>=20:
        for g in centers.index:
            k=int((M[m]&df.Identidad.eq(g)).sum());base=df.Identidad.eq(g).mean();lift=(k/n)/base
            media.append({'Medio':m,'Identidad':g,'n medio':n,'casos':k,'Lift':lift if k>=5 else np.nan})
md=pd.DataFrame(media);mt=md.pivot(index='Medio',columns='Identidad',values='Lift');nums=md.pivot(index='Medio',columns='Identidad',values='casos').reindex(index=mt.index,columns=mt.columns)
add(heat(mt.values,mt.columns,[f'{m} (n={int(M[m].sum())})' for m in mt.index],limit=(0,3),colors='Blues',fmt='.1f',counts=nums),'Medios e identidades: afinidad relativa declarada','Dirigentes y medios','Lift = proporción de una identidad entre quienes consumen un medio / proporción de esa identidad en toda la muestra. 1 indica igual presencia; 2, el doble. Consumos se superponen: no sumar audiencias. Se omiten celdas con menos de 5 casos y medios con n<20.',750)

# 24. Problemas por identidad: porcentajes dentro de grupo; Other explicito.
top=df.Problema.value_counts().head(10).index;pbl=df.Problema.where(df.Problema.isin(top),'Otros problemas');ct=pd.crosstab(df.Identidad,pbl).loc[centers.index];pc=ct.div(ct.sum(axis=1),axis=0)*100
add(heat(pc.values,pc.columns,[f'{g} (n={int(ct.loc[g].sum())})' for g in pc.index],limit=(0,60),fmt='.0f',counts=ct.values),'Qué problema prioriza cada identidad','Dirigentes y medios','Porcentaje dentro de cada identidad. Se conservan las diez respuestas de problema más frecuentes y se agrupa el resto. Las categorías son declaraciones; no se interpretan como causas de la identidad.',760)

# 25. Preguntas que distinguen identidades: diferencias en opción vs todo el conjunto.
questionV=[]
for q,c in enumerate(mapping):
    ct=pd.crosstab(df.Identidad,Q[:,q]);chi,_,_,_=chi2_contingency(ct,correction=False);n=ct.values.sum();rr,kk=ct.shape;phi=max(0,chi/n-(kk-1)*(rr-1)/(n-1));den=min(rr-(rr-1)**2/(n-1)-1,kk-(kk-1)**2/(n-1)-1);v=np.sqrt(phi/den) if den>0 else 0
    questionV.append({'Pregunta':c['question'],'V corregida':v,'n':int(n)})
qv=pd.DataFrame(questionV).sort_values('V corregida')
fig=px.bar(qv,x='V corregida',y='Pregunta',orientation='h');fig.update_xaxes(range=[0,1])
add(fig,'Qué preguntas distinguen más las identidades declaradas','Asociaciones','V de Cramér corregida entre opción elegida e identidad declarada. No emplea los puntajes originales y no impone un orden a las tres opciones. No mide calidad de una pregunta ni causalidad.',1100)
aggregates.append(('preguntas_identidad_cramer.csv',qv))

# 26. Pregunta vs demografía: no hace falta que cada pregunta entre bien en los ejes.
assoc=np.zeros((30,len(demographics)))
for q in range(30):
    for j,col in enumerate(demographics):
        ct=pd.crosstab(smallcats(col),Q[:,q]);chi,_,_,_=chi2_contingency(ct,correction=False);n=ct.values.sum();r,k=ct.shape;phi=max(0,chi/n-(k-1)*(r-1)/(n-1));den=min(r-(r-1)**2/(n-1)-1,k-(k-1)**2/(n-1)-1);assoc[q,j]=np.sqrt(phi/den) if den>0 else 0
add(heat(assoc,demographics,[c['question'] for c in mapping],limit=(0,1),fmt='.2f'),'Cada pregunta frente a las variables sociales','Asociaciones','V de Cramér corregida: asociación entre la opción de cada pregunta y cada variable. Sirve para encontrar diferencias específicas que podrían desaparecer al sumar puntajes. Exploración sin afirmaciones de significación por celda.',1150)

# 27. Correlaciones ordinales y evaluaciones vs puntajes (exploratorio).
numeric=df[dims].copy();numeric['Edad (bandas)']=df.Edad.map({a:i for i,a in enumerate(ages)})
numeric['Ingresos (bandas)']=df.Ingresos.map({a:i for i,a in enumerate(incomes[:-1])})
numeric=pd.concat([numeric,ratings],axis=1);rho=numeric.corr(method='spearman');nn=numeric.notna().astype(int).T@numeric.notna().astype(int)
add(heat(rho.values,rho.columns,rho.index,limit=(-1,1),colors='RdBu_r',fmt='.2f',counts=nn),'Correlaciones ordinales: ejes, edad, ingresos y evaluaciones','Asociaciones','Spearman por pares completos. Edad e ingresos usan orden de bandas; desempleo queda fuera del orden de ingresos. La educación no se fuerza a un ranking porque mezcla nivel, cursado y abandono. El tooltip informa n de cada par.',850)

# 28. Distribución económica por grupo: boxplots sin exhibir fichas individuales.
fig=go.Figure()
for g in centers.index:
    sub=df[df.Identidad.eq(g)];fig.add_trace(go.Violin(x=sub.Economía,name=f'{g} · n={len(sub)}',orientation='h',side='positive',points=False,box_visible=True,meanline_visible=True,line_color=COLORS[g],spanmode='hard'))
fig.update_xaxes(range=[0,100]);fig.update_layout(showlegend=False,margin=dict(l=260))
add(fig,'Distribuciones económicas: superposición y heterogeneidad','Espacio político','Ancho de la forma: densidad estimada, normalizada por grupo; caja: mediana y cuartiles. Permite ver si los promedios ocultan varios conjuntos de respuestas. El ancho no representa cantidad de participantes.',780)

# 29. Diversidad de evaluaciones dentro de cada persona, luego agregado por identidad.
ev=ratings.std(axis=1).where(ratings.notna().sum(axis=1)>=7);evdf=pd.DataFrame({'Identidad':df.Identidad,'Dispersión de evaluaciones':ev}).dropna()
fig=px.box(evdf,x='Dispersión de evaluaciones',y='Identidad',color='Identidad',color_discrete_map=COLORS,points=False)
fig.update_layout(showlegend=False,margin=dict(l=220))
add(fig,'Cuánto se diferencian las evaluaciones de dirigentes','Dirigentes y medios','Para cada respuesta se calcula la desviación estándar de sus evaluaciones (al menos siete válidas); luego se muestra la distribución por identidad. Un valor bajo puede ser aprobación o rechazo parejos: no equivale a indiferencia.',720)

# 30. Geometría de referentes por seguidores: conservar sin atribuir postura al político.
ref=df.reference.map(norm).replace({'milei':'javier milei','moreno':'guillermo moreno','cfk':'cristina kirchner','cristina':'cristina kirchner','cristina fernandez de kirchner':'cristina kirchner','alberdi':'juan bautista alberdi'})
top=ref[ref.ne('')].value_counts().head(12);f=go.Figure()
for k,(name,n) in enumerate(top.items()):
    pts=df.loc[ref.eq(name),dims].to_numpy();c=pts.mean(axis=0);color=px.colors.qualitative.Light24[k];lines=np.array([a for p in pts for a in [p,c,[None]*3]],dtype=object)
    f.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',name=f'{name.title()} (n={n})',legendgroup=name,marker=dict(size=3,opacity=.6,color=color)))
    f.add_trace(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines',showlegend=False,legendgroup=name,opacity=.15,line=dict(color=color,width=1),hoverinfo='skip'))
    f.add_trace(go.Scatter3d(x=[c[0]],y=[c[1]],z=[c[2]],mode='markers+text',text=[name.title()],marker=dict(size=7,symbol='diamond',color=color),showlegend=False,legendgroup=name,hovertemplate='Centro de seguidores<extra></extra>'))
f.update_layout(scene={a:dict(title=d,range=[0,100]) for a,d in zip(['xaxis','yaxis','zaxis'],dims)})
add(f,'Referentes: cercanía entre quienes los eligen','Dirigentes y medios','Se muestran los doce referentes con más menciones. Cada diamante es el promedio de quienes lo eligieron y las líneas conectan sus respuestas. No representa la posición propia del dirigente.',800)

# 31. Sensibilidad a patrones idénticos sin borrar encuestados.
unique=~pd.Series([tuple(a) for a in Q]).duplicated();sens=[]
for name,mask in [('Todas las respuestas',np.ones(len(df),bool)),('Un ejemplar por patrón de 30 opciones',unique.to_numpy())]:
    for d in dims:sens.append({'Base':name,'Dimensión':d,'Media':float(df.loc[mask,d].mean()),'n':int(mask.sum())})
se=pd.DataFrame(sens);f=px.bar(se,x='Dimensión',y='Media',color='Base',barmode='group',hover_data=['n']);f.update_yaxes(range=[0,100])
add(f,'Sensibilidad: respuestas completas frente a patrones distintos','Calidad y lectura','Comparación de medias con todas las filas y con un ejemplar por patrón de 30 opciones. No es una limpieza ni demuestra duplicación de personas: respuestas idénticas pueden pertenecer a personas diferentes. La versión reducida cambia deliberadamente los pesos.',500)
missing=df[['edad','genero','nacionalidad','residencia','ingresos','educacion','plataforma','medios','identity','reference']].isna().sum()
missing['identity']=(df.identity=='').sum();missing['reference']=(df.reference=='').sum()
f=go.Figure(go.Bar(x=missing.values/len(df)*100,y=missing.index,orientation='h',customdata=missing.values,hovertemplate='%{y}<br>%{x:.1f}% · %{customdata} respuestas<extra></extra>'));f.update_xaxes(range=[0,100],title='% sin dato')
add(f,'Cobertura de las variables utilizadas','Calidad y lectura','Faltantes sobre las 977 respuestas. Algunas preguntas son condicionales: residencia argentina y nacionalidad no deben interpretarse como fallas automáticas de captura.',570)

for name,table in aggregates:table.to_csv(P/name,index=False,encoding='utf-8-sig')
top_assoc=pa[pa['Variable A'].ne('Identidad') & pa['Variable B'].eq('Identidad')].sort_values('V corregida',ascending=False)
summary={'n':len(df),'charts':len(charts),'unique_answer_patterns':int(unique.sum()),'invalid_ratings':invalid_ratings,'pca_variance':pca.explained_variance_ratio_.tolist(),'top_categorical_associations':pa.sort_values('V corregida',ascending=False).head(5).to_dict('records'),'largest_spread':sp.sort_values('Dispersión',ascending=False).head(3).to_dict('records'),'most_discriminating_questions':qv.tail(3).to_dict('records'),'peronista_kirchnerista_pct':float(per.mean()*100)}
(P/'resumen_resultados.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')

sections=list(dict.fromkeys(c['section'] for c in charts))
nav=''.join('<a href="#'+c['id']+'">'+str(i+1).zfill(2)+' · '+html.escape(c['title'])+'</a>' for i,c in enumerate(charts))
filters='<button class="active" data-cat="Todas">Todas</button>'+''.join('<button data-cat="'+html.escape(s)+'">'+html.escape(s)+'</button>' for s in sections)
cards=''.join('<section id="'+c['id']+'" data-section="'+html.escape(c['section'])+'"><div class="eyebrow">'+str(i+1).zfill(2)+' / '+html.escape(c['section'])+'</div><h2>'+html.escape(c['title'])+'</h2><p>'+html.escape(c['explanation'])+'</p><div class="plot" id="plot-'+c['id']+'" style="min-height:'+str(c['figure']['layout']['height'])+'px"></div></section>' for i,c in enumerate(charts))
payload=json.dumps({c['id']:c['figure'] for c in charts},ensure_ascii=False).replace('</','<\\/')
notes='''<details><summary>Leer la base, la clave y los alcances</summary><p>Fuente: PJM 2 (Responses), 977 filas. Unidad: una respuesta, no una persona verificada. No se aplicaron ponderaciones ni un diseño de muestra probabilística. Los cruces describen este conjunto y no estiman automáticamente a la población argentina. Se conserva cada fila, incluso si repite las treinta opciones de otra.</p><p>Clave: se verificó el orden actual de las 30 preguntas y sus tres opciones en el formulario. Se aplica la tabla Copy of Sheet10 por número de pregunta y opción. No se reordena por contenido. La tabla numérica carece de títulos: la correspondencia histórica no puede demostrarse si el formulario fue reorganizado. En particular, Modos de producción (10) aporta al segundo puntaje y Moral (20) al primero. Se preservan aportes cruzados y decimales. Los puntajes se normalizan por sus máximos teóricos (20; 21,5; 20). Los nombres Economía, Social y Político reflejan los nombres del archivo, no una validación de constructos.</p><p>Identidades: se normalizan declaraciones mediante reglas explícitas y editables, no se infieren ideologías individuales desde demografía. Las etiquetas mixtas pueden quedar en una familia: consultar family() en Python. Las figuras combinan peronismo y kirchnerismo solo donde lo indican expresamente. Medios: coincidencia exacta de nombres separados por comas. Son selecciones múltiples, no audiencias excluyentes. Los ingresos son bandas monetarias nominales declaradas, sin ajuste por fecha ni conversión a poder adquisitivo.</p><p>Estadística: asociaciones exploratorias; no causalidad. V de Cramér incluye corrección por sesgo. La tabla de pares incluye p asintótica y q Benjamini–Hochberg: si hay celdas esperadas escasas, esas p requieren cautela. Los gráficos priorizan tamaños de asociación, no selección por p. Los intervalos bootstrap y Wilson suponen estabilidad dentro de esta base y no incorporan sesgo de selección. Los puntajes de dirigentes se validan en 0–10; valores fuera de rango se tratan como faltantes. Los gráficos con grupos pequeños indican sus umbrales.</p><p>Se incluyen los centros de seguidores de referentes y las evaluaciones efectivamente declaradas. No se han simulado respuestas de dirigentes. Las visualizaciones sociodemográficas son agregadas, sin fichas individuales.</p></details>'''
style='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#0b111c;color:#e9eef6;font:16px Arial,sans-serif}aside{position:fixed;width:260px;height:100vh;overflow:auto;top:0;left:0;background:#101827;padding:24px 18px;border-right:1px solid #273247}aside strong{font-size:24px}aside a{display:block;color:#aebdd2;text-decoration:none;font-size:12px;line-height:1.5;padding:9px 0;border-bottom:1px solid #202b3c}aside a:hover{color:#55d5ba}main{margin-left:260px;max-width:1750px;padding:38px 40px}h1{font-size:44px;letter-spacing:-1px;max-width:980px;margin:16px 0}h2{font-size:26px;line-height:1.25;margin:12px 0}.eyebrow{color:#55d5ba;font-size:12px;letter-spacing:2px;text-transform:uppercase}p{color:#b7c6db;line-height:1.6;max-width:1050px}.kpis{display:flex;gap:18px;flex-wrap:wrap;margin:28px 0}.kpi{padding:20px 26px;background:#111a28;border-left:3px solid #55d5ba;min-width:180px}.kpi b{display:block;font-size:32px;margin-bottom:6px}.kpi span{font-size:13px;color:#b7c6db}section{background:#111a28;margin:28px 0;padding:25px;border-top:1px solid #27384e;scroll-margin-top:25px}details{background:#111a28;padding:22px;margin:22px 0}summary{cursor:pointer;color:#e9eef6}.plot{width:100%}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}button{border:1px solid #34465e;background:#111a28;color:#d8e3f0;border-radius:4px;padding:11px 15px;cursor:pointer}button.active{border-color:#55d5ba;color:#55d5ba}footer{padding:35px 0;color:#91a6c1;font-size:13px}@media(max-width:1000px){aside{display:none}main{margin:0;padding:20px}h1{font-size:34px}section{padding:14px}.kpi{min-width:130px}}'''
js='''const figures=PAYLOAD;const drawn=new Set();function draw(section){const id=section.id;if(drawn.has(id))return;const f=figures[id];if(!f)return;drawn.add(id);Plotly.newPlot('plot-'+id,f.data,f.layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}}).catch(e=>{document.getElementById('plot-'+id).textContent='No se pudo dibujar este gráfico: '+e.message;});}const obs=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)draw(e.target)}),{rootMargin:'500px'});document.querySelectorAll('section').forEach(s=>obs.observe(s));document.querySelectorAll('[data-cat]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-cat]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('section').forEach(s=>{s.hidden=b.dataset.cat!=='Todas'&&s.dataset.section!==b.dataset.cat;});window.dispatchEvent(new Event('resize'));}));window.addEventListener('hashchange',()=>{const s=document.querySelector(location.hash);if(s&&s.tagName==='SECTION'){s.hidden=false;draw(s);}});'''.replace('PAYLOAD',payload)
page='<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PJM · Explorador de opinión</title><style>'+style+'</style></head><body><aside><strong>PJM / Explorador</strong><p>Opiniones, identidades<br>y diferencias sociales</p>'+nav+'</aside><main><div class="eyebrow">Prueba de concepto · análisis exploratorio</div><h1>Posiciones políticas y diferencias sociales</h1><p>Una exploración interactiva de las relaciones entre respuestas al test, identidades declaradas, características sociales, medios y evaluaciones de dirigentes. Filtrá por tema, girá los mapas y consultá las bases en cada gráfico.</p><div class="kpis"><div class="kpi"><b>'+str(len(df))+'</b><span>respuestas analizadas</span></div><div class="kpi"><b>'+str(len(charts))+'</b><span>visualizaciones</span></div><div class="kpi"><b>30 × 3</b><span>preguntas y opciones auditables</span></div><div class="kpi"><b>'+str(int(unique.sum()))+'</b><span>patrones distintos de 30 respuestas</span></div></div>'+notes+'<div class="filters">'+filters+'</div>'+cards+'<footer>Python + Plotly · visualización autónoma sin conexión · análisis descriptivo y agregado.</footer></main><script>'+get_plotlyjs()+'</script><script>'+js+'</script></body></html>'
(P/'PJM_analisis_avanzado.html').write_text(page,encoding='utf8')
assert len(df)==len(Q) and len(charts)>=25
assert np.isfinite(df[dims]).all().all()
print(json.dumps(summary,ensure_ascii=True))
