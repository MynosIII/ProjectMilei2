"""Genera datos SINTÉTICOS para probar el pipeline sin red. No son publicaciones reales.

Uso: python radar/tests/make_fixture.py /ruta/posts.jsonl.gz
"""
import gzip
import json
import random
import sys
from datetime import datetime, timedelta, timezone

random.seed(4)
now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
themes = {
    "economia": ["la inflación de {m} bajó pero los salarios siguen sin recuperarse", "el dólar blue vuelve a subir y {a} culpa al gobierno",
                 "jubilados marchan otra vez al congreso por el ajuste en las jubilaciones", "tarifas de luz y gas aumentan en diciembre"],
    "seguridad": ["{a} anuncia nuevo protocolo antipiquetes y operativo en la autopista", "detenidos tras la marcha en plaza de mayo y denuncia por represión"],
    "corrupcion": ["filtración de audios complica a {a} en la causa por coimas", "nueva denuncia por corrupción en la agencia de discapacidad"],
    "elecciones": ["encuesta muestra a {a} arriba en provincia de buenos aires rumbo a las elecciones",
                   "fiscalización y boleta única: lo que hay que saber antes de votar"],
    "brasil": ["lula critica a {a} por los aranceles y defiende el mercosur", "bolsonaro y la causa por el intento de golpe en brasilia"],
}
actors = ["Milei", "Kicillof", "Cristina", "Bullrich", "Villarruel", "Macri", "Lula", "Bolsonaro"]
plats = ["Bluesky", "Mastodon", "Reddit"]
rows = []
for i in range(3200):
    th = random.choices(list(themes), weights=[5, 2, 2, 3, 1])[0]
    ts = now - timedelta(hours=random.randint(0, 24 * 21))
    if th == "corrupcion" and random.random() < 0.6:
        ts = now - timedelta(days=6, hours=random.randint(0, 20))  # pico
    txt = random.choice(themes[th]).format(a=random.choice(actors), m=random.choice(["octubre", "septiembre"]))
    txt += " " + random.choice(["", "#Argentina", "#Milei", "#Elecciones2027", "#Jubilados", "#Inflación"]) + " " + random.choice(
        ["una vergüenza", "a ver qué pasa", "lo veníamos diciendo", "increíble", "", "fuente en el link"]) + f" ({i % 97})"
    p = random.choice(plats)
    rows.append({"id": f"{p}:{i}", "platform": p, "kind": "post", "author": f"user{random.randint(1, 900)}", "outlet": "", "text": txt,
                 "created_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "url": "", "engagement": random.randint(0, 80),
                 "hashtags": [w[1:].lower() for w in txt.split() if w.startswith("#")],
                 "domains": [random.choice(["infobae.com", "clarin.com", "lanacion.com.ar", "pagina12.com.ar", "youtube.com", "c5n.com"])]
                 if random.random() < 0.3 else [], "query": "fixture"})
# Grupo coordinado: 14 cuentas con el mismo mensaje en minutos
base = now - timedelta(days=3, hours=5)
msgs = ["URGENTE difundan: el fraude en la provincia ya está confirmado, los audios lo prueban todo #FraudeBA #Elecciones2027",
        "Los medios ocultan el fraude en la provincia, compartan antes de que lo borren #FraudeBA",
        "Fiscales denuncian urnas abiertas en La Matanza, nadie habla de esto #FraudeBA #Elecciones2027"]
for m, msg in enumerate(msgs):
    for j in range(14):
        rows.append({"id": f"Bluesky:c{m}-{j}", "platform": "Bluesky", "kind": "post", "author": f"bot{j}", "outlet": "", "text": msg,
                     "created_at": (base + timedelta(hours=9 * m, minutes=2 * j)).strftime("%Y-%m-%dT%H:%M:%SZ"), "url": "",
                     "engagement": 3, "hashtags": [w[1:].lower() for w in msg.split() if w.startswith("#")], "domains": [], "query": "fixture"})
for k in range(400):
    th = random.choice(list(themes))
    ts = now - timedelta(hours=random.randint(0, 24 * 21))
    outlet = random.choice(["Infobae", "Clarín", "La Nación", "Página/12", "Ámbito", "Folha de S.Paulo"])
    rows.append({"id": f"Google News:{k}", "platform": "Google News", "kind": "news", "author": outlet, "outlet": outlet,
                 "text": random.choice(themes[th]).format(a=random.choice(actors), m="octubre") + f" ({k})",
                 "created_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "url": "", "engagement": 0, "hashtags": [], "domains": [], "query": "fixture"})
with gzip.open(sys.argv[1], "wt", encoding="utf8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "filas sintéticas")
