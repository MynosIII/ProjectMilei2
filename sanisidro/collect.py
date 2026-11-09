"""Reclamos vecinales en San Isidro: recolecta fuentes públicas y gratuitas.

Fuentes (cada una falla por separado sin frenar a las demás):
  - YouTube Data API v3: videos sobre San Isidro y sus comentarios (YOUTUBE_API_KEY, gratis).
  - Google News RSS: notas de medios sobre San Isidro y sus localidades.
  - OpenStreetMap (Overpass): límites de las localidades y equipamiento urbano.
Facebook e Instagram no tienen API abierta para buscar publicaciones y scrapearlas viola sus términos.

Uso: python sanisidro/collect.py            (recolecta y luego corre analyze.py)
El almacén con textos (.store/) no se sube al repositorio; en Actions vive en la caché del workflow.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
STORE = HERE / ".store" / "items.jsonl"
GEO = HERE / ".store" / "osm.json"
UA = "SanIsidroReclamos/0.1 (+https://github.com/MynosIII/ProjectMilei2; investigación agregada)"
TAG_RE = re.compile(r"<[^>]+>")


def get(url, data=None, headers=None, retries=2, timeout=60):
    h = {"User-Agent": UA}
    h.update(headers or {})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(20 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(5)
                continue
            raise


def get_json(url, **kw):
    return json.loads(get(url, **kw).decode("utf8"))


def clean(t):
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", t or ""))).strip()


def iso(d):
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- YouTube
def youtube():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("falta YOUTUBE_API_KEY")
    cfg = CONFIG["youtube"]
    api = "https://www.googleapis.com/youtube/v3/"
    since = iso(datetime.now(timezone.utc) - timedelta(days=CONFIG["window_days"]))
    ctx, excl = cfg["contexto_ar"], cfg["excluir"]
    out, seen = [], set()
    for q in cfg["queries"]:
        params = {"part": "snippet", "type": "video", "q": q, "regionCode": cfg["region"], "relevanceLanguage": cfg["language"],
                  "publishedAfter": since, "maxResults": cfg["videos_per_query"], "order": "relevance", "key": key}
        try:
            items = get_json(api + "search?" + urllib.parse.urlencode(params)).get("items", [])
        except urllib.error.HTTPError as e:
            log(youtube_query=q, error=e.code)
            if e.code == 403:  # cuota agotada
                break
            continue
        for v in items:
            vid = v["id"]["videoId"]
            if vid in seen:
                continue
            seen.add(vid)
            sn = v["snippet"]
            vtext = clean(sn["title"] + ". " + sn.get("description", "")).lower()
            # "San Isidro" existe en Perú, Costa Rica, etc.: exigir contexto argentino y descartar homónimos
            if not any(c in vtext for c in ctx if c != "san isidro") and "san isidro" not in vtext:
                continue
            if any(re.search(rf"\b{re.escape(x)}\b", vtext) for x in excl):
                continue
            video = {"id": f"YouTube:{vid}", "platform": "YouTube", "kind": "video", "video_id": vid,
                     "text": clean(sn["title"] + ". " + sn.get("description", ""))[:600], "title": clean(sn["title"]),
                     "channel": sn["channelTitle"], "created_at": sn["publishedAt"][:19] + "Z",
                     "url": f"https://www.youtube.com/watch?v={vid}", "query": q, "engagement": 0}
            out.append(video)
            try:
                d = get_json(api + "commentThreads?" + urllib.parse.urlencode(
                    {"part": "snippet", "videoId": vid, "order": "relevance", "maxResults": cfg["comments_per_video"],
                     "textFormat": "plainText", "key": key}))
            except urllib.error.HTTPError:  # comentarios desactivados
                continue
            for t in d.get("items", []):
                c = t["snippet"]["topLevelComment"]["snippet"]
                out.append({"id": f"YouTube:{t['id']}", "platform": "YouTube", "kind": "comment", "video_id": vid,
                            "video_title": video["title"], "channel": video["channel"],
                            "text": clean(c.get("textOriginal") or c.get("textDisplay", ""))[:1500],
                            "created_at": c["publishedAt"][:19] + "Z",
                            "url": f"https://www.youtube.com/watch?v={vid}&lc={t['id']}", "query": q,
                            "engagement": c.get("likeCount", 0) + t["snippet"].get("totalReplyCount", 0)})
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------- Google News
def google_news():
    atom_out = []
    for q in CONFIG["google_news"]["queries"]:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": q + " when:1y", "hl": "es-419", "gl": "AR", "ceid": "AR:es-419"})
        try:
            root = ET.fromstring(get(url))
        except Exception as e:
            log(google_news=q, error=str(e)[:120])
            continue
        for it in root.iter("item"):
            title = clean(it.findtext("title"))
            src = it.find("source")
            outlet = src.text if src is not None and src.text else ""
            if outlet and title.endswith(" - " + outlet):
                title = title[: -len(outlet) - 3]
            try:
                created = iso(parsedate_to_datetime(it.findtext("pubDate")))
            except (TypeError, ValueError):
                continue
            link = it.findtext("link") or ""
            atom_out.append({"id": "Noticias:" + (it.findtext("guid") or link), "platform": "Medios", "kind": "news",
                             "text": title, "title": title, "outlet": outlet, "created_at": created, "url": link,
                             "query": q, "engagement": 0})
        time.sleep(1)
    return atom_out


# ---------------------------------------------------------------- OpenStreetMap
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
BBOX = "-34.545,-58.605,-34.435,-58.475"  # partido de San Isidro con margen; se recorta con el polígono del partido
POI = {
    "salud": ("amenity", {"hospital", "clinic", "doctors"}),
    "educacion": ("amenity", {"school", "kindergarten", "college", "university"}),
    "seguridad": ("amenity", {"police", "fire_station"}),
    "verdes": ("leisure", {"park", "playground", "garden"}),
    "paradas": ("highway", {"bus_stop"}),
    "estaciones": ("railway", {"station", "halt"}),
    "camaras": ("man_made", {"surveillance"}),
}
QUERY = f"""[out:json][timeout:120][bbox:{BBOX}];
relation["boundary"="administrative"]["admin_level"~"^(5|6|7|8|9|10)$"];out geom tags;
(nwr["amenity"~"^(hospital|clinic|doctors|school|kindergarten|college|university|police|fire_station)$"];
 nwr["leisure"~"^(park|playground|garden)$"];node["highway"="bus_stop"];nwr["railway"~"^(station|halt)$"];
 node["man_made"="surveillance"];);out center tags;"""


def osm():
    data, last = None, None
    for ep in OVERPASS:
        try:
            data = json.loads(get(ep, data=urllib.parse.urlencode({"data": QUERY}).encode(), retries=0, timeout=150).decode("utf8"))
            break
        except Exception as e:
            last = e
            log(overpass=ep, error=str(e)[:120])
    if data is None:
        raise last
    bounds, pois = [], {k: [] for k in POI}
    for e in data.get("elements", []):
        tags = e.get("tags", {})
        if e["type"] == "relation" and tags.get("boundary") == "administrative" and "members" in e:
            bounds.append(e)
            continue
        lat, lon = e.get("lat") or (e.get("center") or {}).get("lat"), e.get("lon") or (e.get("center") or {}).get("lon")
        if lat is None:
            continue
        for k, (key, vals) in POI.items():
            if tags.get(key) in vals:
                pois[k].append([lat, lon, tags.get("name", "")])
    GEO.write_text(json.dumps({"boundaries": bounds, "pois": pois, "fetched_at": iso(datetime.now(timezone.utc))},
                              ensure_ascii=False), encoding="utf8")
    return len(bounds), {k: len(v) for k, v in pois.items()}


def save(known):
    cutoff = iso(datetime.now(timezone.utc) - timedelta(days=CONFIG["window_days"] + 30))
    keep = [r for r in known.values() if r.get("created_at", "") >= cutoff]
    STORE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n", encoding="utf8")


def main():
    STORE.parent.mkdir(parents=True, exist_ok=True)
    known = {}
    if STORE.exists():
        for line in STORE.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
                known[r["id"]] = r
            except (ValueError, KeyError):
                pass
    status = {}
    for name, fn in (("youtube", youtube), ("google_news", google_news)):
        try:
            rows = fn()
            new = sum(1 for r in rows if r["id"] not in known)
            for r in rows:
                known[r["id"]] = r | {"collected_at": iso(datetime.now(timezone.utc))}
            status[name] = {"ok": True, "items": len(rows), "new": new}
        except Exception as e:
            status[name] = {"ok": False, "error": str(e)[:200]}
        log(source=name, **status[name])
    save(known)  # guardar antes de OSM, que es la fuente más lenta
    status_path = HERE / ".store" / "status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf8")
    try:
        nb, npoi = osm()
        status["openstreetmap"] = {"ok": nb > 0, "boundaries": nb, "pois": npoi}
    except Exception as e:
        status["openstreetmap"] = {"ok": GEO.exists(), "error": str(e)[:200]}
    log(source="openstreetmap", **status["openstreetmap"])
    status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
    sys.exit(0)
