"""Fetch the configured public Sheet into a temporary directory, calculate, publish aggregates.
Never commits source rows. Unknown options abort the update instead of guessing scores.
"""
from pathlib import Path
import csv,io,json,hashlib,shutil,subprocess,sys,tempfile,urllib.request,urllib.parse,time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'sheets_source.json').read_text(encoding='utf8'))
class GoogleExportRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlparse(newurl)
        assert parsed.scheme=='https' and (parsed.hostname=='docs.google.com' or parsed.hostname.endswith('.googleusercontent.com')), 'Unexpected export redirect blocked.'
        return super().redirect_request(req,fp,code,msg,headers,newurl)
OPENER=urllib.request.build_opener(GoogleExportRedirect())
def download(gid):
    url=f"https://docs.google.com/spreadsheets/d/{CONFIG['spreadsheet_id']}/export?format=csv&gid={gid}"
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'ProjectMilei2-public-data-refresh/1.0'})
            with OPENER.open(req,timeout=60) as response:
                data=response.read()
            if data.lstrip().startswith(b'<'):raise ValueError('Expected CSV; Google returned a page. No permissions were changed.')
            return data
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)

response_csv=download(CONFIG['responses_gid']);scoring_csv=download(CONFIG['scoring_gid'])
form_bytes=(ROOT/'analysis'/'form_mapping.json').read_bytes()
fingerprint=hashlib.sha256(response_csv+scoring_csv+form_bytes).hexdigest()
manifest_file=ROOT/'data'/'manifest.json'
if manifest_file.exists():
    previous=json.loads(manifest_file.read_text(encoding='utf8'))
    if previous.get('source_fingerprint')==fingerprint and (ROOT/'data'/'charts.json').exists():
        print('Sheet unchanged. Existing public chart database retained.');sys.exit(0)
rows=list(csv.reader(io.StringIO(response_csv.decode('utf-8-sig'))))
header=rows[0]
assert len(header)>=55 and header[11]=='Autoridad Política' and header[53].startswith('¿Como se define'), 'Sheet columns changed; review the schema before publishing.'
keyrows=list(csv.reader(io.StringIO(scoring_csv.decode('utf-8-sig'))))
lookup={}
for line in keyrows[1:]:
    if not line or not line[0]:continue
    question=int(line[3]);option=int(line[0]);values=[float(x.replace(',','.')) for x in line[4:7]]
    assert len(values)==3 and (question,option) not in lookup
    lookup[question,option]=values
assert set(lookup)=={(q,o) for q in range(1,31) for o in range(1,4)}, 'Expected 90 scoring combinations.'
mapping=json.loads(form_bytes)
for q,c in enumerate(mapping,1):
    assert c['key_question']==q and len(c['form_options'])==3
    c['answers']={a:lookup[q,o] for o,a in enumerate(c['form_options'],1)}
records=[];excluded=0
for line in rows[1:]:
    if not any(line):continue
    r=line+['']*max(0,55-len(line))
    answers=[r[q+11] or None for q in range(29)]+[r[52] or r[40] or None]
    if any(a is None for a in answers):excluded+=1;continue
    for q,a in enumerate(answers):
        assert a in mapping[q]['answers'], f'Unknown option in question {q+1}; manual form mapping review required.'
    records.append(dict(id=f'R{len(records)+1:04}',edad=r[2] or None,genero=r[3] or None,nacionalidad=r[4] or None,residencia=r[5] or None,pais=r[6] or None,ingresos=r[7] or None,educacion=r[8] or None,plataforma=r[9] or None,medios=r[10] or None,ratings=r[41:50],problema=r[50] or None,answers=answers,identity=r[53],reference=r[54]))
assert len(records)>=30, 'Too few complete responses to refresh the advanced analysis safely.'
with tempfile.TemporaryDirectory(prefix='pjm-private-analysis-') as temp:
    work=Path(temp)
    for name in ['pjm_analisis_avanzado.py','pjm_grafico_corregido.py','offline_run.py']:
        shutil.copyfile(ROOT/'analysis'/name,work/name)
    (work/'datos_pjm_exacto.json').write_text(json.dumps({'mapping':mapping}),encoding='utf8')
    (work/'datos_ampliados.json').write_text(json.dumps(records),encoding='utf8')
    subprocess.run([sys.executable,'-I','-S',str(work/'offline_run.py')],check=True,cwd=work)
    with (work/'clave_auditada.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Pregunta','Título','Opción','Texto','Score 1','Score 2','Score 3'])
        for c in mapping:
            for o,a in enumerate(c['form_options'],1):writer.writerow([c['key_question'],c['question'],o,a,*c['answers'][a]])
    subprocess.run([sys.executable,'-I','-S',str(ROOT/'scripts'/'package_public.py'),str(work),str(ROOT)],check=True)
    manifest=json.loads(manifest_file.read_text(encoding='utf8'))
    timestamp=datetime.now(timezone.utc).isoformat()
    manifest.update(source_fingerprint=fingerprint,updated_at=timestamp,source_rows=len(records)+excluded,excluded_incomplete=excluded,refresh='GitHub Actions every hour, or manual workflow dispatch')
    manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    index=ROOT/'index.html';page=index.read_text(encoding='utf8')
    status=f'<p class="data-update">Datos actualizados: <time datetime="{timestamp}">{timestamp[:16].replace("T"," ")} UTC</time> · {len(records)} respuestas completas · {excluded} incompletas excluidas · actualización automática cada hora.</p>'
    page=page.replace('<div class="filters">',status+'<div class="filters">')
    index.write_text(page,encoding='utf8')
print(json.dumps({'complete_responses':len(records),'excluded_incomplete':excluded,'published':'32 chart figures and aggregate tables; source rows discarded'}))
