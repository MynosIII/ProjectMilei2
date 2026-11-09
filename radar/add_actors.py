"""Agrega políticos a radar/config.json desde el formulario "Run workflow" de GitHub Actions.

Formato: nombres separados por punto y coma. Opcionalmente país y espacio entre paréntesis.
  python radar/add_actors.py "Sergio Massa; Tarcísio de Freitas (BR, Republicanos); Juan Pérez (AR)"

Alias que se buscan en los textos: el nombre completo y el apellido (si tiene 5 letras o más).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

PATH = Path(__file__).resolve().parent / "config.json"


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def parse(raw: str):
    for item in re.split(r"[;\n]", raw):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^(.*?)\s*(?:\((.*)\))?$", item)
        name, extra = m.group(1).strip(), [x.strip() for x in (m.group(2) or "").split(",") if x.strip()]
        pais = next((x.upper() for x in extra if x.upper() in ("AR", "BR")), "AR")
        espacio = next((x for x in extra if x.upper() not in ("AR", "BR")), "")
        yield name, pais, espacio


def main(raw: str):
    cfg = json.loads(PATH.read_text(encoding="utf8"))
    added = []
    for name, pais, espacio in parse(raw):
        if name in cfg["actors"]:
            continue
        full = norm(name)
        last = full.split()[-1]
        cfg["actors"][name] = [full] + ([last] if len(last) >= 5 and last != full else [])
        cfg.setdefault("actor_meta", {})[name] = {"pais": pais, "espacio": espacio}
        added.append(name)
    PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("Agregados:", ", ".join(added) or "ninguno (ya estaban)")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
