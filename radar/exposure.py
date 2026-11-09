"""Radar: ficha de exposición digital de instituciones públicas (solo fuentes pasivas).

Para cada dominio institucional configurado consulta, sin escanear nada:
  - crt.sh: certificados públicos -> cantidad de subdominios visibles.
  - DNS sobre HTTPS (dns.google): SPF y DMARC -> protección contra suplantación de correo.
  - Shodan InternetDB: puertos y vulnerabilidades ya indexados públicamente para las IP del dominio.
  - Have I Been Pwned (catálogo público de brechas por dominio, sin datos personales).

Se publican solo agregados por dominio (conteos y semáforo). No se publican qué hosts ni qué CVE,
para no convertir la página en un mapa de ataque.

Uso: python radar/exposure.py [--out radar/data/exposure.json]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from collect import CONFIG, HERE, get, get_json, iso

RISKY_PORTS = {21, 23, 445, 1433, 3306, 3389, 5432, 5900, 6379, 9200, 27017}


def subdomains(domain: str) -> set[str]:
    rows = get_json(f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json", retries=3)
    names = set()
    for r in rows:
        for n in str(r.get("name_value", "")).split("\n"):
            n = n.strip().lower().lstrip("*.")
            if n.endswith(domain):
                names.add(n)
    return names


def dns(name: str, rtype: str) -> list[str]:
    d = get_json("https://dns.google/resolve?" + urllib.parse.urlencode({"name": name, "type": rtype}))
    return [a.get("data", "").strip('"') for a in d.get("Answer", []) if a.get("data")]


def mail_protection(domain: str) -> dict:
    spf = [t for t in dns(domain, "TXT") if t.lower().startswith("v=spf1")]
    dmarc = [t for t in dns(f"_dmarc.{domain}", "TXT") if t.lower().startswith("v=dmarc1")]
    policy = ""
    if dmarc:
        for part in dmarc[0].split(";"):
            k, _, v = part.strip().partition("=")
            if k.lower() == "p":
                policy = v.strip().lower()
    return {"spf": bool(spf), "spf_strict": bool(spf) and "-all" in spf[0], "dmarc": bool(dmarc), "dmarc_policy": policy or "ninguna"}


def internetdb(ips: list[str]) -> dict:
    ports, vuln_hosts, risky = set(), 0, set()
    for ip in ips:
        try:
            d = get_json(f"https://internetdb.shodan.io/{ip}")
        except Exception:  # 404 = IP sin datos indexados
            continue
        ports.update(d.get("ports", []))
        risky.update(p for p in d.get("ports", []) if p in RISKY_PORTS)
        vuln_hosts += 1 if d.get("vulns") else 0
        time.sleep(1)
    return {"ips": len(ips), "open_ports": len(ports), "risky_services": len(risky), "hosts_with_known_cves": vuln_hosts}


def breaches(domain: str) -> int:
    try:
        rows = json.loads(get("https://haveibeenpwned.com/api/v3/breaches?" + urllib.parse.urlencode({"Domain": domain})).decode())
        return len(rows)
    except Exception:
        return -1


def light(r: dict) -> str:
    """Semáforo simple y explicable."""
    score = 0
    score += 0 if r["mail"]["dmarc_policy"] in ("reject", "quarantine") else (1 if r["mail"]["dmarc"] else 2)
    score += 0 if r["mail"]["spf"] else 1
    score += 2 if r["services"]["hosts_with_known_cves"] else 0
    score += 1 if r["services"]["risky_services"] else 0
    return "verde" if score <= 1 else ("amarillo" if score <= 3 else "rojo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "exposure.json"))
    args = ap.parse_args()
    cfg = CONFIG.get("exposure", {})
    out = []
    for item in cfg.get("domains", []):
        dom = item["domain"]
        r = {"name": item["name"], "domain": dom, "tipo": item.get("tipo", "")}
        try:
            subs = subdomains(dom)
        except Exception as e:
            subs = set()
            r["error_crtsh"] = str(e)[:100]
        r["subdomains"] = len(subs)
        try:
            r["mail"] = mail_protection(dom)
        except Exception as e:
            r["mail"] = {"spf": False, "spf_strict": False, "dmarc": False, "dmarc_policy": "sin dato"}
            r["error_dns"] = str(e)[:100]
        # IP del dominio principal y de una muestra acotada de subdominios (solo resolución DNS pública)
        ips = set()
        for name in [dom, f"www.{dom}"] + sorted(subs)[: cfg.get("max_subdomains", 15)]:
            try:
                ips.update(a for a in dns(name, "A") if a.count(".") == 3)
            except Exception:
                pass
        r["services"] = internetdb(sorted(ips)[: cfg.get("max_ips", 25)])
        r["breaches_catalog"] = breaches(dom)
        r["semaforo"] = light(r)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps({"updated_at": iso(datetime.now(timezone.utc)), "domains": out}, ensure_ascii=False, indent=1),
                              encoding="utf8")


if __name__ == "__main__":
    main()
