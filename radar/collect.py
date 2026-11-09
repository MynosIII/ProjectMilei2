"""Radar: recolecta publicaciones públicas y las agrega al almacén local.

Fuentes abiertas, sin claves obligatorias: Mastodon (etiquetas públicas), Bluesky
(búsqueda; usa BSKY_HANDLE y BSKY_APP_PASSWORD si existen), Reddit (listados públicos),
Google News RSS y GDELT. Cada fuente falla por separado sin detener las demás.

Uso: python radar/collect.py [--store radar/.store/posts.jsonl]
El almacén contiene texto de publicaciones públicas y NO se publica: queda fuera de git
(ver .gitignore) y en GitHub Actions se conserva en la caché del workflow.
"""
from __future__ import annotations

import argparse
import gzip
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
UA = "RadarConversacion/0.1 (+https://github.com/MynosIII/ProjectMilei2; investigación agregada)"
TAG_RE = re.compile(r"<[^>]+>")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w{2,60})", re.UNICODE)
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def get(url: str, headers: dict | None = None, data: bytes | None = None, retries: int = 2) -> bytes:
    h = {"User-Agent": UA, "Accept": "application/json, application/xml;q=0.9, */*;q=0.5"}
    h.update(headers or {})
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=h, data=data)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(15 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(3)
                continue
            raise
    raise RuntimeError("unreachable")


def get_json(url: str, **kw):
    return json.loads(get(url, **kw).decode("utf8"))


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", text or ""))).strip()


def domain(u: str) -> str:
    try:
        d = urllib.parse.urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except ValueError:
        return ""


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(platform, pid, author, text, created, url="", engagement=0, links=None, query="", kind="post", outlet="", lang=None):
    text = clean(text)
    links = list(dict.fromkeys((links or []) + URL_RE.findall(text)))
    return {
        "id": f"{platform}:{pid}",
        "platform": platform,
        "kind": kind,  # post | news
        "author": author or "",
        "outlet": outlet,
        "text": text,
        "created_at": created,
        "url": url,
        "engagement": int(engagement or 0),
        "hashtags": sorted({h.lower() for h in HASHTAG_RE.findall(text)}),
        "domains": sorted({d for d in (domain(l) for l in links) if d}),
        "query": query,
        "lang": (lang or "")[:2].lower() or None,
    }


# ---------------------------------------------------------------- fuentes

def mastodon(cfg):
    out = []
    for inst in cfg["instances"]:
        for tag in cfg["tags"]:
            max_id = None
            for _ in range(cfg.get("pages_per_tag", 3)):
                q = {"limit": 40}
                if max_id:
                    q["max_id"] = max_id
                rows = get_json(f"https://{inst}/api/v1/timelines/tag/{urllib.parse.quote(tag)}?{urllib.parse.urlencode(q)}")
                if not rows:
                    break
                for s in rows:
                    s = s.get("reblog") or s
                    acct = s["account"]["acct"]
                    if "@" not in acct:
                        acct = f"{acct}@{inst}"
                    out.append(record(
                        "Mastodon", s.get("uri") or s["id"], acct, s.get("content", ""), s["created_at"][:19] + "Z",
                        s.get("url") or "", (s.get("favourites_count", 0) + s.get("reblogs_count", 0) + s.get("replies_count", 0)),
                        [c.get("url") for c in [s.get("card") or {}] if c.get("url")], f"#{tag}", lang=s.get("language")))
                max_id = rows[-1]["id"]
                time.sleep(1)
    return out


def bluesky(cfg):
    base, headers = "https://public.api.bsky.app", {}
    handle, pw = os.environ.get("BSKY_HANDLE"), os.environ.get("BSKY_APP_PASSWORD")
    if handle and pw:
        s = get_json("https://bsky.social/xrpc/com.atproto.server.createSession",
                     headers={"Content-Type": "application/json"},
                     data=json.dumps({"identifier": handle, "password": pw}).encode())
        base, headers = "https://bsky.social", {"Authorization": f"Bearer {s['accessJwt']}"}
    out = []
    for qtext in cfg["queries"]:
        cursor = None
        for _ in range(cfg.get("pages_per_query", 3)):
            q = {"q": qtext, "limit": 100, "sort": "latest"}
            if cursor:
                q["cursor"] = cursor
            d = get_json(f"{base}/xrpc/app.bsky.feed.searchPosts?{urllib.parse.urlencode(q)}", headers=headers)
            for p in d.get("posts", []):
                rec = p.get("record", {})
                links = []
                emb = (p.get("embed") or {}).get("external") or {}
                if emb.get("uri"):
                    links.append(emb["uri"])
                for f in rec.get("facets", []) or []:
                    for feat in f.get("features", []):
                        if feat.get("uri"):
                            links.append(feat["uri"])
                rkey = p["uri"].rsplit("/", 1)[-1]
                out.append(record(
                    "Bluesky", p["uri"], p["author"]["handle"], rec.get("text", ""), (rec.get("createdAt") or p.get("indexedAt"))[:19] + "Z",
                    f"https://bsky.app/profile/{p['author']['handle']}/post/{rkey}",
                    p.get("likeCount", 0) + p.get("repostCount", 0) + p.get("replyCount", 0) + p.get("quoteCount", 0), links, qtext,
                    lang=(rec.get("langs") or [None])[0]))
            cursor = d.get("cursor")
            if not cursor:
                break
            time.sleep(1)
    return out


def reddit(cfg):
    out = []
    for sub in cfg["subreddits"]:
        after = None
        for _ in range(cfg.get("pages", 2)):
            q = {"limit": 100, "raw_json": 1}
            if after:
                q["after"] = after
            d = get_json(f"https://www.reddit.com/r/{sub}/new.json?{urllib.parse.urlencode(q)}")["data"]
            for c in d["children"]:
                p = c["data"]
                out.append(record(
                    "Reddit", p["id"], p.get("author", ""), f"{p.get('title', '')}. {p.get('selftext', '')[:2000]}",
                    iso(datetime.fromtimestamp(p["created_utc"], timezone.utc)), "https://www.reddit.com" + p.get("permalink", ""),
                    p.get("score", 0) + p.get("num_comments", 0), [] if p.get("is_self") else [p.get("url", "")], f"r/{sub}"))
            after = d.get("after")
            if not after:
                break
            time.sleep(2)
    return out


NEWS_LOCALE = {"AR": {"hl": "es-419", "gl": "AR", "ceid": "AR:es-419"}, "BR": {"hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}}


def actor_feeds(cfg):
    """Una búsqueda de Google News por cada político configurado (nombre completo entre comillas)."""
    if not cfg.get("per_actor"):
        return []
    meta = CONFIG.get("actor_meta", {})
    return [dict(q=f'"{name}"', **NEWS_LOCALE.get(meta.get(name, {}).get("pais", "AR"), NEWS_LOCALE["AR"]))
            for name in CONFIG.get("actors", {})]


def google_news(cfg):
    out, failed = [], 0
    feeds = cfg["feeds"] + actor_feeds(cfg) + [dict(f, kind="factcheck") for f in cfg.get("factcheck_feeds", [])]
    for f in feeds:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": f["q"] + " when:" + f.get("when", "7d"), "hl": f["hl"], "gl": f["gl"], "ceid": f["ceid"]})
        try:
            root = ET.fromstring(get(url))
        except Exception:  # una búsqueda caída no frena las demás
            failed += 1
            continue
        for it in root.iter("item"):
            title = it.findtext("title", "")
            src = it.find("source")
            outlet = src.text if src is not None else ""
            src_url = src.get("url", "") if src is not None else ""
            if outlet and title.endswith(" - " + outlet):
                title = title[: -len(outlet) - 3]
            try:
                created = iso(parsedate_to_datetime(it.findtext("pubDate", "")))
            except (TypeError, ValueError):
                continue
            out.append(record("Google News", it.findtext("guid", "") or it.findtext("link", ""), outlet, title, created,
                              it.findtext("link", ""), 0, [src_url] if src_url else [], f["q"], kind=f.get("kind", "news"), outlet=outlet, lang=f["hl"]))
        time.sleep(1)
    if failed == len(feeds):
        raise RuntimeError("todas las búsquedas de Google News fallaron")
    return out


def gdelt(cfg):
    out = []
    for qtext in cfg["queries"]:
        q = {"query": qtext, "mode": "artlist", "format": "json", "maxrecords": cfg.get("max_records", 250),
             "timespan": cfg.get("timespan", "7d"), "sort": "datedesc"}
        raw = get("https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(q))
        try:
            d = json.loads(raw.decode("utf8", "replace"))
        except json.JSONDecodeError:
            d = {}
        for a in d.get("articles", []):
            try:
                created = iso(datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
            except (KeyError, ValueError):
                continue
            out.append(record("GDELT", a["url"], a.get("domain", ""), a.get("title", ""), created, a["url"], 0,
                              [a["url"]], qtext, kind="news", outlet=a.get("domain", ""),
                              lang={"Spanish": "es", "Portuguese": "pt"}.get(a.get("language", ""), "xx")))
        time.sleep(20)  # GDELT limita fuerte: ~1 consulta cada 5-10 s
    return out


def rss(cfg):
    """RSS/Atom de medios. En feeds generales se guardan solo notas políticas (actores o palabras clave)."""
    terms = [t.lower() for t in cfg.get("keywords", [])]
    for alias in CONFIG["actors"].values():
        terms += [a.lower().lstrip("@") for a in alias if len(a) > 4]
    atom = "{http://www.w3.org/2005/Atom}"
    out = []
    for f in cfg["feeds"]:
        try:
            root = ET.fromstring(get(f["url"], headers={"Accept": "application/rss+xml, application/xml, */*"}))
        except Exception as e:  # un medio caído no frena a los demás
            print(json.dumps({"rss_feed": f["name"], "error": str(e)[:120]}, ensure_ascii=False), flush=True)
            continue
        items = list(root.iter("item")) or list(root.iter(atom + "entry"))
        for it in items:
            title = it.findtext("title") or it.findtext(atom + "title") or ""
            desc = it.findtext("description") or it.findtext(atom + "summary") or ""
            link = it.findtext("link") or ""
            if not link:
                el = it.find(atom + "link")
                link = el.get("href", "") if el is not None else ""
            date = it.findtext("pubDate") or it.findtext(atom + "updated") or it.findtext(atom + "published") or ""
            try:
                created = iso(parsedate_to_datetime(date)) if "," in date else iso(datetime.fromisoformat(date.replace("Z", "+00:00")))
            except (TypeError, ValueError):
                continue
            text = f"{clean(title)}. {clean(desc)[:400]}"
            if not f.get("politics_only") and not any(t in text.lower() for t in terms):
                continue
            out.append(record("RSS medios", link or title, f["name"], text, created, link, 0, [link] if link else [],
                              f["name"], kind="news", outlet=f["name"], lang=f.get("lang")))
        time.sleep(0.5)
    return out


def youtube(cfg):
    """Comentarios de videos políticos recientes (YouTube Data API v3; necesita YOUTUBE_API_KEY, gratis)."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("falta YOUTUBE_API_KEY")
    api = "https://www.googleapis.com/youtube/v3/"
    since = iso(datetime.now(timezone.utc) - timedelta(days=cfg.get("days", 14)))
    out, seen = [], set()
    # Búsquedas por tema + canales de todo el espectro (cada uno rotulado con su línea editorial)
    jobs = [({"q": q, "regionCode": cfg.get("region", "AR"), "relevanceLanguage": cfg.get("language", "es"), "order": "viewCount"}, "")
            for q in cfg["queries"]]
    # En canales se filtra por palabras políticas (algunos son de entretenimiento)
    jobs += [({"channelId": ch["channel_id"], "order": "date", "q": cfg.get("channel_query", ""),
               "maxResults": cfg.get("videos_per_channel", 4)}, ch.get("linea", "")) for ch in cfg.get("channels", [])]
    for extra, linea in jobs:
        params = {"part": "snippet", "type": "video", "publishedAfter": since, "maxResults": cfg.get("videos_per_query", 8), "key": key, **extra}
        q = "canal" if "channelId" in extra else extra["q"]
        for v in get_json(api + "search?" + urllib.parse.urlencode(params)).get("items", []):
            vid = v["id"]["videoId"]
            if vid in seen:
                continue
            seen.add(vid)
            channel = v["snippet"]["channelTitle"]
            try:
                d = get_json(api + "commentThreads?" + urllib.parse.urlencode(
                    {"part": "snippet", "videoId": vid, "order": "relevance", "maxResults": cfg.get("comments_per_video", 100),
                     "textFormat": "plainText", "key": key}))
            except urllib.error.HTTPError:  # comentarios desactivados
                continue
            for t in d.get("items", []):
                c = t["snippet"]["topLevelComment"]["snippet"]
                out.append(record("YouTube", t["id"], c.get("authorChannelId", {}).get("value") or c.get("authorDisplayName", ""),
                                  c.get("textOriginal") or c.get("textDisplay", ""), c["publishedAt"][:19] + "Z",
                                  f"https://www.youtube.com/watch?v={vid}&lc={t['id']}",
                                  c.get("likeCount", 0) + t["snippet"].get("totalReplyCount", 0), [], f"{q} · {channel}",
                                  lang=None) | {"channel": channel, "linea": linea})
    return out


def google_trends(cfg, path):
    """Interés de búsqueda relativo (0-100) en Google para los términos configurados. Guarda una serie, no publicaciones."""
    import http.cookiejar
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) RadarConversacion/0.1"), ("Accept-Language", "es-419,es")]
    opener.open(f"https://trends.google.com/trends/explore?geo={cfg['geo']}", timeout=30).read()

    def call(url):
        for attempt in range(3):
            try:
                raw = opener.open(url, timeout=30).read().decode("utf8")
                return json.loads(raw[raw.find("{"):])
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(20 * (attempt + 1))
                    continue
                raise

    req = {"comparisonItem": [{"keyword": t, "geo": cfg["geo"], "time": cfg["timeframe"]} for t in cfg["terms"]], "category": 0, "property": ""}
    ex = call("https://trends.google.com/trends/api/explore?" + urllib.parse.urlencode({"hl": "es-419", "tz": 180, "req": json.dumps(req)}))
    w = next(x for x in ex["widgets"] if x["id"] == "TIMESERIES")
    data = call("https://trends.google.com/trends/api/widgetdata/multiline?" + urllib.parse.urlencode(
        {"hl": "es-419", "tz": 180, "req": json.dumps(w["request"]), "token": w["token"]}))
    rows = [{"time": int(p["time"]), "values": p["value"]} for p in data["default"]["timelineData"]]
    path.write_text(json.dumps({"geo": cfg["geo"], "terms": cfg["terms"], "rows": rows}, ensure_ascii=False), encoding="utf8")
    return rows


COLLECTORS = {"mastodon": mastodon, "bluesky": bluesky, "reddit": reddit, "google_news": google_news, "rss": rss, "youtube": youtube, "gdelt": gdelt}


# ---------------------------------------------------------------- almacén

def load(path: Path) -> dict:
    rows = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def save(path: Path, rows: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf8") as f:
        for r in sorted(rows.values(), key=lambda r: r["created_at"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(HERE / ".store" / "posts.jsonl.gz"))
    ap.add_argument("--only", nargs="*", choices=list(COLLECTORS) + ["trends"])
    args = ap.parse_args()
    store = Path(args.store)
    store.parent.mkdir(parents=True, exist_ok=True)
    rows = load(store)
    before = len(rows)
    cutoff = iso(datetime.now(timezone.utc) - timedelta(days=CONFIG["window_days"]))
    status = []
    for name, fn in COLLECTORS.items():
        if args.only and name not in args.only:
            continue
        if name not in CONFIG["sources"] or not CONFIG["sources"][name].get("enabled", True):
            continue
        t0 = time.time()
        try:
            got = fn(CONFIG["sources"][name])
            new = sum(1 for r in got if r["id"] not in rows)
            for r in got:
                if r["text"]:
                    rows[r["id"]] = r
            status.append({"source": name, "ok": True, "fetched": len(got), "new": new, "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # una fuente caída no detiene las demás
            status.append({"source": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}", "seconds": round(time.time() - t0, 1)})
        print(json.dumps(status[-1], ensure_ascii=False), flush=True)
    if not args.only or "trends" in (args.only or []):
        t0 = time.time()
        try:
            got = google_trends(CONFIG["trends"], store.parent / "trends.json")
            status.append({"source": "google_trends", "ok": True, "fetched": len(got), "new": len(got), "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            status.append({"source": "google_trends", "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}", "seconds": round(time.time() - t0, 1)})
        print(json.dumps(status[-1], ensure_ascii=False), flush=True)
    rows = {k: v for k, v in rows.items() if v["created_at"] >= cutoff}
    save(store, rows)
    log = store.parent / "collect_status.json"
    log.write_text(json.dumps({"run_at": iso(datetime.now(timezone.utc)), "rows_before": before, "rows_after": len(rows), "sources": status},
                              ensure_ascii=False, indent=1), encoding="utf8")
    print(f"Almacén: {before} → {len(rows)} publicaciones")
    if not any(s["ok"] for s in status):
        sys.exit("Ninguna fuente respondió.")


if __name__ == "__main__":
    main()
