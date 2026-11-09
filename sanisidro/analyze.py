"""Agrega el almacén de San Isidro en sanisidro/data/sanisidro.json (lo único que se publica).

Clasifica cada texto por tema de reclamo y por localidad con diccionarios auditables (config.json),
arma los polígonos de las localidades desde OpenStreetMap y cuenta equipamiento urbano por localidad.
No publica autores ni identificadores de usuarios; los fragmentos se recortan y se enmascaran @menciones,
correos y teléfonos.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
STORE = HERE / ".store" / "items.jsonl"
GEO = HERE / ".store" / "osm.json"
STATUS = HERE / ".store" / "status.json"
OUT = HERE / "data" / "sanisidro.json"
LOCS = list(CONFIG["localidades"])


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def rx(terms):
    terms = sorted({fold(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + ")", re.I)


TEMAS = {k: rx(v) for k, v in CONFIG["temas"].items()}
RECLAMO = rx(CONFIG["marcadores_reclamo"])
POS, NEG = rx(CONFIG["positivo"]), rx(CONFIG["negativo"])
POLS = {k: rx(v) for k, v in CONFIG["politicos"].items()}
LOC_RX = []
for name, d in CONFIG["localidades"].items():
    if name != "San Isidro":
        LOC_RX.append((name, rx(d["alias"] + [name])))
for barrio, loc in CONFIG["barrios"].items():
    LOC_RX.append((loc, rx([barrio])))
LOC_RX.append(("San Isidro", rx(CONFIG["localidades"]["San Isidro"]["alias"])))
MASK = [(re.compile(r"@[\w.\-]{2,}"), "@…"), (re.compile(r"\S+@\S+\.\w+"), "[correo]"),
        (re.compile(r"(?:\+?54)?[\s\-]?(?:11)?[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[tel]")]


def localidades_en(text_f: str) -> list[str]:
    found = []
    for name, r in LOC_RX:
        if name not in found and r.search(text_f):
            found.append(name)
    return found


def classify(item):
    t = fold(item.get("text", ""))
    temas = [k for k, r in TEMAS.items() if r.search(t)]
    marker = bool(RECLAMO.search(t))
    pos, neg = len(POS.findall(t)), len(NEG.findall(t))
    return {"temas": temas, "reclamo": bool(temas) and (marker or neg > pos), "pos": pos, "neg": neg,
            "locs": localidades_en(t), "pols": [k for k, r in POLS.items() if r.search(t)]}


def excerpt(text: str, n=240) -> str:
    for r, rep in MASK:
        text = r.sub(rep, text)
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- geometría
def rings_from_relation(rel):
    """Une los 'ways' externos de una relación en anillos cerrados [[lon,lat],…]."""
    segs = [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in rel.get("members", [])
            if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry")]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[::-1][:-1] + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if len(ring) > 3:
            rings.append(ring)
    return rings


def area_km2(ring):
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    xy = [(math.radians(p[0]) * 6371 * math.cos(lat0), math.radians(p[1]) * 6371) for p in ring]
    return abs(sum(xy[i][0] * xy[i - 1][1] - xy[i - 1][0] * xy[i][1] for i in range(len(xy)))) / 2


def inside(lon, lat, ring):
    c, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi:
            c = not c
        j = i
    return c


def simplify(ring, step=0.00012):
    out = [ring[0]]
    for p in ring[1:]:
        if abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) >= step:
            out.append(p)
    out.append(ring[-1])
    return [[round(x, 5), round(y, 5)] for x, y in out]


def match_loc(name: str):
    f = fold(name).strip()
    for loc in LOCS:
        lf = fold(loc)
        if f == lf or (loc == "Boulogne Sur Mer" and f.startswith("boulogne")):
            return loc
    return None


def build_geo():
    if not GEO.exists():
        return {}, None, {}, None
    g = json.loads(GEO.read_text(encoding="utf8"))
    polys, partido, pops = {}, None, {}
    for el in g.get("boundaries", []):
        tags = el.get("tags", {})
        lvl = tags.get("admin_level")
        rings = rings_from_relation(el)
        if not rings:
            continue
        if lvl in ("5", "6") and "san isidro" in fold(tags.get("name", "")):
            partido = rings
            continue
        loc = match_loc(tags.get("name", ""))
        if loc and (loc not in polys or len(sum(rings, [])) > len(sum(polys[loc], []))):
            polys[loc] = rings
            if tags.get("population", "").isdigit():
                pops[loc] = {"valor": int(tags["population"]), "fecha": tags.get("population:date", ""), "fuente": "OpenStreetMap (etiqueta population)"}
    return polys, partido, g.get("pois", {}), pops


def which_loc(lon, lat, polys):
    for loc, rings in polys.items():
        if any(inside(lon, lat, r) for r in rings):
            return loc
    return None


# ---------------------------------------------------------------- principal
def main():
    items = []
    if STORE.exists():
        for line in STORE.read_text(encoding="utf8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=CONFIG["window_days"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = [i for i in items if i.get("created_at", "") >= since]

    # Localidad del video (para heredar en comentarios que no la nombran)
    video_loc = {}
    for i in items:
        if i["kind"] == "video":
            video_loc[i["video_id"]] = localidades_en(fold(i.get("text", "")))

    rows = []
    for i in items:
        c = classify(i)
        locs, inferida = c["locs"], False
        if not locs and i["kind"] == "comment":
            vl = video_loc.get(i.get("video_id"), [])
            if len(vl) == 1:
                locs, inferida = vl, True
        rows.append(i | c | {"loc": locs[0] if len(locs) == 1 else ("Varias" if locs else "Sin localidad"), "loc_inferida": inferida})

    voces = [r for r in rows if r["kind"] == "comment"]
    news = [r for r in rows if r["kind"] == "news"]
    reclamos = [r for r in rows if r["reclamo"] and r["kind"] != "video"]

    temas = list(CONFIG["temas"])
    loc_keys = LOCS + ["Varias", "Sin localidad"]
    tema_tbl = []
    for t in temas:
        rv = [r for r in reclamos if t in r["temas"]]
        tema_tbl.append({"tema": t, "reclamos": len(rv),
                         "vecinos": sum(1 for r in rv if r["kind"] == "comment"),
                         "medios_reclamo": sum(1 for r in rv if r["kind"] == "news"),
                         "agenda_medios": sum(1 for r in news if t in r["temas"]),
                         "apoyo": sum(int(r.get("engagement") or 0) for r in rv)})
    tema_tbl.sort(key=lambda x: -x["reclamos"])

    matriz = {loc: {t: 0 for t in temas} for loc in loc_keys}
    for r in reclamos:
        for t in r["temas"]:
            matriz[r["loc"]][t] += 1

    # Serie semanal por tema
    def week(s):
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")
    serie = defaultdict(Counter)
    for r in reclamos:
        for t in r["temas"]:
            serie[week(r["created_at"])][t] += 1
    semanas = sorted(serie)

    # Dirigentes mencionados y tono
    pol = {}
    for k in CONFIG["politicos"]:
        ms = [r for r in rows if k in r["pols"] and r["kind"] != "video"]
        if not ms:
            continue
        por_loc = Counter(r["loc"] for r in ms)
        pol[k] = {"menciones": len(ms), "positivas": sum(1 for r in ms if r["pos"] > r["neg"]),
                  "negativas": sum(1 for r in ms if r["neg"] > r["pos"]),
                  "en_reclamos": sum(1 for r in ms if r["reclamo"]),
                  "temas": Counter(t for r in ms if r["reclamo"] for t in r["temas"]).most_common(3),
                  "por_localidad": dict(por_loc)}

    # Geografía
    polys, partido, pois, pops = build_geo()
    poi_loc = {loc: Counter() for loc in LOCS}
    poi_pts = {}
    for k, pts in (pois or {}).items():
        keep = []
        for lat, lon, name in pts:
            loc = which_loc(lon, lat, polys) if polys else None
            if partido and not any(inside(lon, lat, r) for r in partido):
                continue
            if loc:
                poi_loc[loc][k] += 1
            keep.append([round(lon, 5), round(lat, 5)])
        poi_pts[k] = keep
    locs_out = []
    for loc in LOCS:
        rv = [r for r in reclamos if r["loc"] == loc]
        rings = polys.get(loc, [])
        locs_out.append({
            "nombre": loc, "cp": CONFIG["localidades"][loc]["cp"], "centro": CONFIG["localidades"][loc]["centro"],
            "poligono": [simplify(rg) for rg in rings], "area_km2": round(sum(area_km2(rg) for rg in rings), 2) if rings else None,
            "poblacion": pops.get(loc) if pops else None,
            "reclamos": len(rv), "inferidos": sum(1 for r in rv if r["loc_inferida"]),
            "menciones": sum(1 for r in rows if r["loc"] == loc and r["kind"] != "video"),
            "tema_top": Counter(t for r in rv for t in r["temas"]).most_common(3),
            "equipamiento": dict(poi_loc.get(loc, {})),
        })

    # Fragmentos de ejemplo (sin autor), balanceados por tema y priorizando apoyo
    ejemplos, seen_txt = [], set()
    for t in temas:
        pool = sorted([r for r in reclamos if t in r["temas"]], key=lambda r: -(r.get("engagement") or 0))
        for r in pool[:14]:
            key = fold(r["text"])[:80]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            ejemplos.append({"tema": t, "temas": r["temas"], "localidad": r["loc"], "inferida": r["loc_inferida"],
                             "fuente": "Comentario en YouTube" if r["kind"] == "comment" else (r.get("outlet") or "Medio"),
                             "fecha": r["created_at"][:10], "texto": excerpt(r["text"]),
                             "url": f"https://www.youtube.com/watch?v={r['video_id']}" if r["kind"] == "comment" else r.get("url", ""),
                             "contexto": r.get("video_title", ""), "apoyo": int(r.get("engagement") or 0)})

    videos = Counter(r["video_id"] for r in reclamos if r["kind"] == "comment")
    vinfo = {i["video_id"]: i for i in items if i["kind"] == "video"}
    top_videos = [{"titulo": vinfo[v]["title"], "canal": vinfo[v]["channel"], "fecha": vinfo[v]["created_at"][:10],
                   "url": vinfo[v]["url"], "reclamos": n,
                   "comentarios": sum(1 for r in voces if r["video_id"] == v)}
                  for v, n in videos.most_common(12) if v in vinfo]
    titulares = sorted(news, key=lambda r: r["created_at"], reverse=True)
    titulares = [{"titulo": r["title"], "medio": r.get("outlet", ""), "fecha": r["created_at"][:10], "url": r["url"],
                  "temas": r["temas"], "localidad": r["loc"], "reclamo": r["reclamo"]} for r in titulares[:80]]

    status = json.loads(STATUS.read_text(encoding="utf8")) if STATUS.exists() else {}
    out = {
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "municipio": CONFIG["municipio"], "window_days": CONFIG["window_days"],
        "poblacion_partido": {"valor": CONFIG["poblacion_censo_2022"], "fuente": CONFIG["poblacion_fuente"]},
        "status": status,
        "totales": {"textos": len([r for r in rows if r["kind"] != "video"]), "comentarios": len(voces), "videos": len(vinfo),
                    "noticias": len(news), "reclamos": len(reclamos),
                    "reclamos_con_localidad": sum(1 for r in reclamos if r["loc"] in LOCS),
                    "medios": len({r.get("outlet") for r in news if r.get("outlet")})},
        "temas": tema_tbl,
        "matriz": {"localidades": loc_keys, "temas": temas, "valores": [[matriz[l][t] for t in temas] for l in loc_keys]},
        "serie": {"semanas": semanas, "temas": {t: [serie[w][t] for w in semanas] for t in temas}},
        "localidades": locs_out,
        "partido_poligono": [simplify(rg) for rg in (partido or [])],
        "equipamiento_puntos": poi_pts,
        "politicos": pol,
        "ejemplos": ejemplos,
        "videos": top_videos,
        "titulares": titulares,
        "metodo": {
            "temas": CONFIG["temas"], "marcadores_reclamo": CONFIG["marcadores_reclamo"],
            "regla": "Un texto es reclamo si menciona al menos un tema y contiene un marcador de queja o más términos negativos que positivos.",
            "localidad": "Se busca el nombre de la localidad o de un barrio/lugar conocido; si un comentario no nombra ninguna, hereda la del video cuando el video nombra una sola (se marca como inferida).",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    print(json.dumps(out["totales"], ensure_ascii=False))


if __name__ == "__main__":
    main()
