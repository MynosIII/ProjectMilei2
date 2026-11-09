"""Radar: tono, emoción y agresividad de cada publicación.

Dos métodos, con el mismo formato de salida:
  - "modelo": pysentimiento (modelos RoBERTuito/BERTweet entrenados con tuits en español y portugués).
    Se usa en GitHub Actions cuando la librería está instalada. Es gratis y corre en CPU.
  - "lexico": diccionarios cortos y explicables. Respaldo si el modelo no está disponible.

Los resultados se guardan por id en radar/.store/nlp.jsonl.gz (fuera del repositorio, como los textos)
para no reclasificar todo cada día. Al repositorio solo llegan porcentajes agregados.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
from pathlib import Path

MAX_NEW = int(os.environ.get("RADAR_NLP_MAX", "6000"))
# Tiempo máximo de modelos por corrida; lo que no llega se clasifica con diccionarios y se reintenta al día siguiente
BUDGET = float(os.environ.get("RADAR_NLP_SECONDS", "900"))
DEADLINE = [0.0]

POS = set("""
bien buen bueno buena buenos buenas excelente genial gran grande mejor mejores logro logros exito exitos
apoyo apoyar felicito felicitaciones gracias agradezco orgullo orgulloso orgullosa esperanza crecimiento
crece crecen recuperacion avance avances acierto correcto justo justa honesto honesta valiente
victoria gana gano ganamos triunfo festejo celebra celebramos alegria feliz felices hermoso hermosa
solucion soluciones baja bajo estable estabilidad confianza libertad paz unidad trabajo empleo
bom boa otimo otima melhor parabens obrigado obrigada vitoria venceu conquista apoio esperanca
crescimento sucesso orgulho feliz lindo linda forte acerto justica honesto
""".split())

NEG = set("""
mal malo mala malos malas peor peores pesimo pesima desastre desastroso fracaso fracasos crisis caos
corrupcion corrupto corrupta corruptos chorro chorros ladron ladrones robo roban robaron estafa fraude
mentira mentiras miente mentiroso mentirosa vergüenza verguenza vergonzoso escandalo escandaloso
ajuste hambre pobreza desempleo despidos despido inflacion deuda recesion cae caida derrumbe
miedo amenaza amenazas violencia violento represion odio ataque ataca atacan culpa culpable
traidor traicion traidores inutil inutiles incapaz ineptos inepto basura asco horrible terrible
injusto injusta abuso abusos preso presa prision condena condenado condenada denuncia denuncias
problema problemas peligro peligroso grave rechazo rechaza repudio repudiamos indignante
pior ruim pessimo pessima fracasso corrupcao corrupto ladrao ladroes roubo mentira mentiroso vergonha
escandalo crise golpe golpista golpistas fome pobreza desemprego ameaca violencia odio culpa traidor
inutil incompetente lixo nojo horrivel terrivel injusto abuso preso prisao condenado denuncia perigo
""".split())

EMO = {
    "ira": set("""bronca furia furioso furiosa indignado indignada indignante enojo enojado harto harta
              basta repudio odio rabia raiva odeio revolta revoltado absurdo inaceptable vergonha""".split()),
    "miedo": set("""miedo temor temo preocupa preocupante preocupacion peligro amenaza alarma panico
                   incertidumbre medo receio perigo ameaca risco""".split()),
    "tristeza": set("""triste tristeza dolor duele lamentable lamento luto pena desolador angustia
                      llorar lloro tristeza saudade lamentavel dor""".split()),
    "alegria": set("""alegria feliz felices genial celebra celebramos festejo emocion orgullo hermoso
                     gracias felicitaciones viva vamos alegria feliz parabens lindo""".split()),
    "asco": set("""asco asqueroso asquerosa nauseabundo repugnante basura lacra nojo nojento lixo""".split()),
    "sorpresa": set("""increible sorpresa sorprende insolito inesperado wow impresionante
                      inacreditavel surpresa chocante""".split()),
}

INSULT = set("""
idiota idiotas imbecil imbeciles estupido estupida pelotudo pelotuda boludo boluda forro forra
mogolico tarado tarada negro de mierda mierda carajo puto puta hdp hijo de puta gorila gorilas
kuka kukas kuca cucas zurdo zurdos zurda facho fachos nazi nazis rata ratas lacra lacras parasito
parasitos mandril mandriles libertonto libertontos vende patria cipayo cipayos chorra burro burra
babaca idiota imbecil otario otaria vagabundo vagabunda canalha bandido bandidos lixo petralha
petralhas bolsominion bolsominions minion minions esquerdopata comuna comunas genocida""".split())

WORD = re.compile(r"[a-z]+")


def lexicon(clean: str) -> dict:
    w = WORD.findall(clean)
    ws = set(w)
    p, n = len(ws & POS), len(ws & NEG)
    tono = "POS" if p > n else ("NEG" if n > p else "NEU")
    emo = max(EMO, key=lambda k: len(ws & EMO[k]))
    if not ws & EMO[emo]:
        emo = "otra"
    joined = " ".join(w)
    insult = bool(ws & INSULT) or any(f" {x} " in f" {joined} " for x in INSULT if " " in x)
    return {"tono": tono, "emocion": emo, "agresivo": insult, "metodo": "lexico"}


# ------------------------------------------------------------------ modelo (pysentimiento)

EMO_MAP = {"joy": "alegria", "anger": "ira", "fear": "miedo", "sadness": "tristeza", "disgust": "asco",
           "surprise": "sorpresa", "others": "otra"}


def _analyzers():
    try:
        from pysentimiento import create_analyzer  # noqa: WPS433
    except Exception:
        return None
    out = {}
    for lang in ("es", "pt"):
        for task in ("sentiment", "emotion", "hate_speech"):
            try:
                out[(lang, task)] = create_analyzer(task=task, lang=lang)
            except Exception as e:  # no todas las tareas existen en portugués
                print(f"nlp: sin modelo {task}/{lang}: {str(e)[:80]}", flush=True)
    return out or None


def _out(r):
    return r if isinstance(r, list) else [r]


def model_labels(rows: list[dict], analyzers: dict, batch: int = 64) -> list[dict]:
    """rows: [{"text", "clean", "lang"}] -> etiquetas. Cada lote recibe las tres tareas juntas, así un texto queda
    entero con modelo o entero con diccionario (y se reintenta en la próxima corrida si se acabó el tiempo)."""
    out = [lexicon(r["clean"]) for r in rows]
    DEADLINE[0] = time.time() + BUDGET
    for lang in ("es", "pt"):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        if not idx or (lang, "sentiment") not in analyzers:
            continue
        done = 0
        for k in range(0, len(idx), batch):
            if time.time() > DEADLINE[0]:
                break
            ids = idx[k:k + batch]
            texts = [rows[i]["text"][:500] for i in ids]
            for i, p in zip(ids, _out(analyzers[(lang, "sentiment")].predict(texts))):
                out[i]["tono"] = p.output
                out[i]["metodo"] = "modelo"
            if (lang, "emotion") in analyzers:
                for i, p in zip(ids, _out(analyzers[(lang, "emotion")].predict(texts))):
                    out[i]["emocion"] = EMO_MAP.get(p.output, "otra")
            if (lang, "hate_speech") in analyzers:
                for i, p in zip(ids, _out(analyzers[(lang, "hate_speech")].predict(texts))):
                    labels = p.output if isinstance(p.output, list) else [p.output]
                    out[i]["agresivo"] = bool({"aggressive", "hateful"} & set(labels))
            done += len(ids)
        print(f"nlp: {done} de {len(idx)} textos en {lang} clasificados con modelo", flush=True)
    return out


PT_MARK = set("nao voce uma isso com sao muito ele ela pra esse essa voces governo tambem entao ainda agora nos".split())
ES_MARK = set("una pero muy este esta porque los las del gobierno tambien entonces todavia ahora nosotros hay".split())


def guess_lang(clean: str, lang) -> str:
    if isinstance(lang, str) and lang[:2] in ("es", "pt"):
        return lang[:2]
    ws = set(WORD.findall(clean))
    return "pt" if len(ws & PT_MARK) > len(ws & ES_MARK) else "es"


def label(df, cache_path: Path) -> "pd.DataFrame":  # noqa: F821
    """Agrega columnas tono / emocion / agresivo / metodo_nlp al DataFrame del radar."""
    cache = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[r.pop("id")] = r
    analyzers = _analyzers() if os.environ.get("RADAR_NLP", "auto") != "lexico" else None
    want = "modelo" if analyzers else "lexico"
    langs = [guess_lang(c, l) for c, l in zip(df["clean"], df["lang"])]
    todo = [i for i, pid in enumerate(df["id"]) if pid not in cache or (want == "modelo" and cache[pid]["metodo"] != "modelo")]
    if want == "modelo":  # lo más reciente primero
        order = df["ts"].rank(method="first", ascending=False).to_numpy() if "ts" in df else None
        todo = sorted(todo, key=lambda i: order[i]) if order is not None else todo
        todo = todo[:MAX_NEW]
    rows = [{"text": df["text"].iat[i] or "", "clean": df["clean"].iat[i], "lang": langs[i]} for i in todo]
    labels = model_labels(rows, analyzers) if analyzers else [lexicon(r["clean"]) for r in rows]
    for i, lab in zip(todo, labels):
        cache[df["id"].iat[i]] = lab
    keep = set(df["id"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf8") as f:
        for pid, r in cache.items():
            if pid in keep:
                f.write(json.dumps({"id": pid, **r}, ensure_ascii=False) + "\n")
    got = [cache.get(pid) or lexicon(c) for pid, c in zip(df["id"], df["clean"])]
    print(f"nlp: {len(todo)} textos nuevos ({want}); total {len(got)}", flush=True)
    return df.assign(tono=[g["tono"] for g in got], emocion=[g["emocion"] for g in got],
                     agresivo=[bool(g["agresivo"]) for g in got], metodo_nlp=[g["metodo"] for g in got], lang2=langs)
