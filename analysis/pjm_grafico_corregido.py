"""PJM: prueba de concepto reproducible. Python 3 + numpy, pandas, plotly.
Ejecutar: python pjm_grafico.py (datos_pjm_exacto.json junto al script).
Opciones verificadas en el formulario; tabla aplicada por pregunta y opción sin reordenar.
"""
from pathlib import Path
import json, html, unicodedata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parent
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(s or '').lower()) if not unicodedata.combining(c)).strip()
def family(s):
    t=norm(s)
    if not t: return 'Sin declaración'
    if 'kirchner' in t: return 'Kirchnerismo'
    if any(w in t for w in ['peronis','justicial','nacional y popular']): return 'Peronismo'
    if 'socialdem' in t or 'social dem' in t: return 'Socialdemocracia'
    if any(w in t for w in ['mao','comunis','socialista','izquierda','marxis'] ) and 'liberal' not in t: return 'Izquierda / socialismo'
    if 'radical' in t: return 'Radicalismo'
    if any(w in t for w in ['libertar','minarqui','anarcocapi']): return 'Libertarismo'
    if 'liberal' in t: return 'Liberalismo'
    if 'capitalis' in t: return 'Capitalismo'
    if any(w in t for w in ['derecha','conserva']): return 'Derecha / centroderecha'
    if 'centro' in t: return 'Centro'
    if any(w in t for w in ['independ','personas no partidos','buscando']): return 'Independiente / sin definición'
    return 'Otras declaraciones'

COLORS={'Kirchnerismo':'#45b5f5','Peronismo':'#2874d0','Socialdemocracia':'#fa7c91','Izquierda / socialismo':'#e34545','Radicalismo':'#b22c3d','Libertarismo':'#955cf5','Liberalismo':'#e5b844','Capitalismo':'#ed8b37','Derecha / centroderecha':'#dfc843','Centro':'#38c9b0','Independiente / sin definición':'#9babbf','Otras declaraciones':'#d798d4','Sin declaración':'#627184'}

def build():
    raw=json.loads((ROOT/'datos_pjm_exacto.json').read_text(encoding='utf-8'))
    df=pd.DataFrame(raw['records'])
    conf=raw['mapping']
    scores=np.full((len(df),30,3),np.nan)
    for q,c in enumerate(conf):
        for i,answers in enumerate(df['answers']):
            if answers[q] in c['answers']:
                scores[i,q]=c['answers'][answers[q]]
    # Solamente cuestionarios con las treinta preguntas reconocidas.
    complete=np.isfinite(scores).all(axis=(1,2))
    maximum=np.array([sum(max(v[d] for v in c['answers'].values()) for c in conf) for d in range(3)])
    coords=scores.sum(axis=1)/maximum*100
    df[['Economía','Social','Político']]=coords
    df['Grupo']=df['identity'].map(family)
    plot=df.loc[complete].copy()
    plot['Detalle']=plot['identity'].map(lambda x:html.escape(str(x or 'Sin declaración')))
    fig=go.Figure()
    for group,g in plot.groupby('Grupo',sort=True):
        fig.add_trace(go.Scatter3d(x=g['Economía'],y=g['Social'],z=g['Político'],name=f'{group} · {len(g)}',legendgroup=group,mode='markers',marker=dict(size=4,color=COLORS[group],opacity=.73),customdata=g[['id','Detalle']].values,hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>Economía: %{x:.1f}<br>Social: %{y:.1f}<br>Político: %{z:.1f}<extra></extra>'))
        # Superficie descriptiva de una desviación estándar: no es intervalo de confianza.
        if len(g)>=8:
            pts=g[['Economía','Social','Político']].to_numpy()
            cov=np.cov(pts.T)
            vals,vec=np.linalg.eigh(cov)
            if np.min(vals)>1e-8:
                u,v=np.mgrid[0:2*np.pi:24j,0:np.pi:14j]
                sphere=np.array([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)])
                ell=(vec@np.diag(np.sqrt(vals))@sphere.reshape(3,-1)).reshape(sphere.shape)+pts.mean(axis=0)[:,None,None]
                fig.add_trace(go.Surface(x=ell[0],y=ell[1],z=ell[2],colorscale=[[0,COLORS[group]],[1,COLORS[group]]],opacity=.13,showscale=False,hoverinfo='skip',visible=False,name=group,legendgroup=group))
    fig.update_layout(template='plotly_dark',height=780,paper_bgcolor='#101824',margin=dict(l=0,r=0,b=0,t=65),title='01 / Espacio político · respuestas e identidades declaradas',scene=dict(xaxis=dict(title='Economía · Score 1 (0–100)',range=[0,100]),yaxis=dict(title='Social · Score 2 (0–100)',range=[0,100]),zaxis=dict(title='Político · Score 3 (0–100)',range=[0,100]),aspectmode='cube'),legend=dict(font=dict(size=11)),updatemenus=[dict(type='buttons',direction='left',x=0,y=1.08,buttons=[dict(label='Puntos',method='update',args=[{'visible':[t.type!='surface' for t in fig.data]}]),dict(label='Puntos + nubes',method='update',args=[{'visible':[True]*len(fig.data)}])])])
    charts=[fig]
    proj=px.scatter_matrix(plot,dimensions=['Economía','Social','Político'],color='Grupo',color_discrete_map=COLORS,hover_name='id',title='02 / Tres perspectivas del mismo espacio',height=800)
    proj.update_traces(diagonal_visible=False,marker=dict(size=4,opacity=.55))
    charts.append(proj)
    means=plot.groupby('Grupo')[['Economía','Social','Político']].mean()
    heat=go.Figure(go.Heatmap(z=means.values,x=means.columns,y=means.index,colorscale='RdBu_r',zmin=0,zmax=100,text=np.round(means.values,1),texttemplate='%{text}',colorbar=dict(title='0–100')))
    heat.update_layout(title='03 / Promedios por identidad declarada',height=620,margin=dict(l=210))
    charts.append(heat)
    counts=plot['Grupo'].value_counts().rename_axis('Grupo').reset_index(name='Respuestas')
    charts.append(px.bar(counts,x='Respuestas',y='Grupo',orientation='h',color='Grupo',color_discrete_map=COLORS,title='04 / Composición de las respuestas graficadas',height=600))
    corr=plot[['Economía','Social','Político']].corr()
    ch=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,zmin=-1,zmax=1,colorscale='RdBu_r',text=np.round(corr.values,2),texttemplate='%{text}',colorbar=dict(title='Pearson r')))
    ch.update_layout(title='05 / Correlaciones entre dimensiones',height=500)
    charts.append(ch)
    # Centros de seguidores: observados; no son posiciones imputadas a los referentes.
    plot['Referente']=plot['reference'].map(norm)
    aliases={'milei':'javier milei','javier gerardo milei':'javier milei','moreno':'guillermo moreno','cristina':'cristina kirchner','cfk':'cristina kirchner','cristina fernandez de kirchner':'cristina kirchner','alberdi':'juan bautista alberdi'}
    plot['Referente']=plot['Referente'].replace(aliases)
    ref=go.Figure()
    top=plot.loc[plot.Referente.ne(''),'Referente'].value_counts().head(12)
    palette=px.colors.qualitative.Light24
    for k,(name,n) in enumerate(top.items()):
        g=plot[plot.Referente.eq(name)]
        pts=g[['Economía','Social','Político']].to_numpy(); center=pts.mean(axis=0)
        ref.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',name=f'{name.title()} · {n}',legendgroup=name,marker=dict(size=3,color=palette[k],opacity=.6),hovertemplate='Respuesta que eligió este referente<extra>%{fullData.name}</extra>'))
        line=np.array([[*p] for p in pts for p in (p,center,[None]*3)],dtype=object)
        ref.add_trace(go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode='lines',line=dict(color=palette[k],width=1),opacity=.22,showlegend=False,legendgroup=name,hoverinfo='skip'))
        ref.add_trace(go.Scatter3d(x=[center[0]],y=[center[1]],z=[center[2]],mode='markers+text',text=[name.title()],textposition='top center',marker=dict(size=8,symbol='diamond',color=palette[k]),legendgroup=name,showlegend=False,hovertemplate='Centro de quienes eligieron este referente<extra></extra>'))
    ref.update_layout(title='06 / Referentes: centros de sus seguidores y vínculos',height=800,scene=fig.layout.scene)
    charts.append(ref)
    panels=[]
    for i,ch in enumerate(charts):
        ch.update_layout(template='plotly_dark',paper_bgcolor='#101824',plot_bgcolor='#101824',font=dict(family='Arial',color='#e7edf5'))
        panels.append(ch.to_html(full_html=False,include_plotlyjs=False,div_id=f'chart{i}',config=dict(responsive=True,displaylogo=False)))
    notes='''<details><summary>Clave exacta aplicada y auditoría</summary><p>Las preguntas y las tres opciones se verificaron en el editor del formulario PJM 2. Se aplica la fila de Copy of Sheet10 correspondiente al número de pregunta (1–30) y a la posición de la opción (1–3), sin reorganizar preguntas por contenido ni invertir opciones por interpretación.</p><p>Se conservan literalmente las asignaciones de la tabla: la pregunta 10, Modos de producción, aporta al Score 2; la 20, Moral, al Score 1. Libertad de Expresión y Libertad de Culto conservan los valores de su clave aunque su orientación resulte llamativa. La tabla no tiene títulos de preguntas: el enlace pregunta/fila usa su orden en el formulario, sin afirmar que ese orden coincida con una intención histórica distinta.</p><p>La suma original se exporta en coordenadas.csv y se normaliza a 0–100 por máximo teórico únicamente para visualizar. clave_auditada.csv contiene las 90 combinaciones pregunta/opción con texto y sus tres puntajes. Derecho y Justicia utiliza primero la columna del final; la columna anterior duplicada solo se usa si falta aquella.</p><p>Los colores agrupan identidades declaradas mediante reglas explícitas. Las nubes son elipsoides descriptivos de una desviación estándar, no límites de grupos. Los referentes siguen representados por centros de sus seguidores; no se han simulado sus respuestas.</p></details>'''
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>PJM · Mapa político 3D</title><style>body{margin:0;background:#0b111b;color:#e7edf5;font:16px Arial}main{max-width:1440px;margin:auto;padding:36px}h1{font-size:42px;margin:12px 0}p{line-height:1.6;max-width:1000px}.tag{color:#44d4b4;letter-spacing:3px;font-size:12px}section,details{background:#101824;padding:22px;margin:24px 0;border:1px solid #253449;border-radius:12px}summary{cursor:pointer}a{color:#44d4b4}</style><main><div class="tag">PJM / PRUEBA DE CONCEPTO</div><h1>Un mapa de posiciones políticas</h1><p>'+str(len(df))+' respuestas leídas · '+str(len(plot))+' cuestionarios completos graficados · '+str((~complete).sum())+' incompletos excluidos. Clave por número de pregunta y opción, verificada con el formulario.</p><p>Arrastrá para girar, usá la rueda para acercarte y hacé clic en la leyenda para filtrar grupos. Doble clic en una categoría para aislarla. En el primer gráfico podés activar las nubes.</p>'+notes+'<script>'+get_plotlyjs()+'</script>'+''.join('<section>'+p+'</section>' for p in panels)+'</main></html>'
    (ROOT/'PJM_mapa_3D_corregido.html').write_text(page,encoding='utf-8')
    plot[['id','Grupo','Economía','Social','Político']].assign(Score1_original=scores.sum(axis=1)[complete,0],Score2_original=scores.sum(axis=1)[complete,1],Score3_original=scores.sum(axis=1)[complete,2]).to_csv(ROOT/'coordenadas.csv',index=False,encoding='utf-8-sig')
    assert len(plot)>0 and np.isfinite(plot[['Economía','Social','Político']]).all().all()
    assert ((plot[['Economía','Social','Político']]>=0)&(plot[['Economía','Social','Político']]<=100)).all().all()
    print(json.dumps({'read':len(df),'plotted':len(plot),'excluded':int((~complete).sum()),'charts':len(charts),'maxima':maximum.tolist(),'duplicate_answer_patterns':int(df['answers'].map(tuple).duplicated().sum())}))

if __name__=='__main__': build()
