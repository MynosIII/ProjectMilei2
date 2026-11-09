"""Radar: análisis de la conversación pública y figuras para la web.

Entrada: radar/.store/posts.jsonl.gz (de collect.py). Salida: radar/data/radar.json,
con figuras Plotly y tablas agregadas. Las cuentas de redes se publican con un
seudónimo (hash con sal); los medios aparecen con su nombre.

Módulos:
  1. Volumen por plataforma y detección de picos (z-score robusto sobre la serie diaria).
  2. Narrativas: TF-IDF + NMF; etiqueta = términos principales; evolución diaria.
  3. Actores: menciones, narrativa dominante y exposición a términos de riesgo reputacional.
  4. Coordinación: textos casi idénticos publicados por cuentas distintas en una ventana corta.
  5. Hashtags que viajan juntos y dominios más enlazados.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
TZ = CONFIG.get("timezone", "America/Argentina/Buenos_Aires")
SALT = os.environ.get("RADAR_SALT", "radar-demo")
# Paleta categórica validada para fondo oscuro (orden fijo; no se recicla más allá de 8 series)
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#5fb3c9"]
TONE_COLORS = {"NEG": "#e66767", "NEU": "#5d6470", "POS": "#3987e5"}
BG, PANEL, GRID, TXT, MUTED = "#0b111c", "#111a28", "#223047", "#e9eef6", "#b7c6db"

STOP = set("""
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque bajo bien cada como con contra cual cuales cuando
de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estado estan estar este esto
estos fue fueron ha hace hacer han hasta hay la las le les lo los mas me mi mis mucho muy nada ni no nos nosotros o otra otras
otro otros para pero poco por porque que quien quienes se sea segun ser si sido sin sobre solo son su sus tambien tan tanto te
tiene tienen todo todos tras tu tus un una uno unos usted va van y ya yo q xq pq vez ver dijo dice decir hoy ayer anos ano dia
dias asi cosa cosas gente parte vamos voy creo puede pueden hacia mientras luego tengo tenes sos mismo misma ahi alla
o os um uma uns umas do da dos das no na nos nas em com por para pelo pela mais mas mais ao aos ele ela eles elas seu sua seus
suas isso esse essa este esta nao sim ja foi sao ser ter tem como quando onde muito tambem ate sobre entre depois
https http www com amp rt via video nota leer mas segun tras
html shtml php noticias noticia news breakingnews latestnews featurednews noticiasdestacadas ultimasnoticias ultimahora tardigram acional biobiochile
""".split())


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def pseudo(platform: str, author: str) -> str:
    return "cuenta_" + hashlib.sha256(f"{SALT}|{platform}|{author}".encode()).hexdigest()[:6]


def layout(title: str, height=480, **kw):
    lay = dict(title=dict(text=title, font=dict(size=17)), height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
               font=dict(color=TXT, family="Arial"), margin=dict(l=60, r=30, t=70, b=60),
               legend=dict(orientation="h", y=-0.18), hoverlabel=dict(bgcolor=BG))
    lay.update(kw)
    return lay


def axes(fig):
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def fig_json(fig) -> dict:
    return json.loads(fig.to_json())


ES_PT = set("de la que el en los las del se por un una para con no es su al lo como mas pero sus le ya o fue este ha si porque esta son entre cuando muy sin sobre tambien hay donde quien desde todo nos durante uno les ni contra otros ese eso ante ellos esto antes algunos unos yo otro otras otra el tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros os em um uma nao com dos das na no ao ele mais isso".split())
EN_ETC = set("the and of to is in for on that with this are was it be as by at from have has not but they he she you we or an will his their its after der die und das ist nicht mit ein eine les des et est pour pas sur".split())


def is_es_pt(text: str, lang) -> bool:
    if lang in ("es", "pt"):
        return True
    if lang:  # idioma declarado distinto
        return False
    words = re.findall(r"[a-z]+", norm(text))
    a = sum(w in ES_PT for w in words)
    b = sum(w in EN_ETC for w in words)
    return a >= 2 and a > b


def read_jsonl(path: Path) -> list:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load(path: Path, extra=()) -> pd.DataFrame:
    rows = read_jsonl(path) if Path(path).exists() else []
    for e in extra:  # otros JSONL con el mismo esquema (por ejemplo, una recolección manual)
        for f in (sorted(Path(e).glob("*.jsonl*")) if Path(e).is_dir() else [Path(e)]):
            if not f.name.startswith("trends"):  # las series de Google Trends no son publicaciones
                rows += read_jsonl(f)
    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="last")
    df = df[~df["platform"].isin(CONFIG.get("exclude_platforms", []))]
    for col, default in [("hashtags", None), ("domains", None), ("outlet", ""), ("engagement", 0), ("kind", "post")]:
        if col not in df:
            df[col] = default
    df["hashtags"] = df["hashtags"].map(lambda x: x if isinstance(x, list) else [])
    df["domains"] = df["domains"].map(lambda x: x if isinstance(x, list) else [])
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0)
    if "lang" not in df:
        df["lang"] = None
    # El análisis es sobre conversación en español y portugués: Mastodon trae mucho contenido global por etiqueta
    df = df[[is_es_pt(t, l if isinstance(l, str) else None) for t, l in zip(df["text"], df["lang"])]].copy()
    df["ts"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(TZ)
    df["day"] = df["ts"].dt.tz_localize(None).dt.normalize()
    df["clean"] = (df["text"].str.replace(r"https?://\S+", " ", regex=True)
                   .str.replace(r"@[\w.\-]+", " ", regex=True).map(norm))
    # Mismo título en varias fuentes de noticias: una sola vez por medio
    df = df.drop_duplicates(subset=["platform", "author", "clean"]).reset_index(drop=True)
    df["who"] = np.where(df["kind"] == "news", df["outlet"].fillna("").replace("", "Medio sin nombre"),
                         [pseudo(p, a) for p, a in zip(df["platform"], df["author"])])
    return df


# ------------------------------------------------------------- 1. volumen y picos

def volume(df):
    daily = df.groupby(["day", "platform"]).size().unstack(fill_value=0).sort_index()
    total = daily.sum(axis=1)
    med = total.rolling(7, min_periods=3).median().shift(1)
    mad = (total - med).abs().rolling(7, min_periods=3).median().shift(1).replace(0, np.nan)
    z = ((total - med) / (1.4826 * mad)).fillna(0)
    spikes = []
    for day in total.index[(z > 3) & (total >= 20)]:
        txt = df.loc[df["day"] == day, "clean"]
        rest = df.loc[df["day"] != day, "clean"]
        terms = distinctive_terms(txt, rest, 5)
        spikes.append({"day": day.strftime("%Y-%m-%d"), "posts": int(total[day]), "z": round(float(z[day]), 1), "terms": terms})
    fig = go.Figure()
    for i, col in enumerate(daily.columns):
        fig.add_bar(x=daily.index, y=daily[col], name=col, marker_color=PALETTE[i % len(PALETTE)])
    for s in spikes:
        fig.add_annotation(x=s["day"], y=s["posts"], text="▲ " + ", ".join(s["terms"][:3]), showarrow=True, arrowhead=0,
                           ay=-40, font=dict(size=11, color="#f2b84b"), arrowcolor="#f2b84b")
    fig.update_layout(**layout("Publicaciones por día y plataforma; ▲ = pico con sus términos distintivos", barmode="stack",
                                yaxis_title="publicaciones"))
    return axes(fig), spikes


def distinctive_terms(target: pd.Series, rest: pd.Series, k=5):
    def counts(s):
        c = Counter()
        for t in s:
            c.update({w for w in re.findall(r"(?<![a-z0-9])[a-z]{4,}(?![a-z0-9])", t) if w not in STOP})
        return c
    a, b = counts(target), counts(rest)
    na, nb = max(len(target), 1), max(len(rest), 1)
    score = {w: (a[w] / na) / ((b[w] + 1) / nb) for w in a if a[w] >= 3}
    return [w for w, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


# ------------------------------------------------------------- 2. narrativas

def narratives(df, k=None):
    docs = df["clean"]
    vec = TfidfVectorizer(stop_words=list(STOP), token_pattern=r"(?u)\b[a-z]{4,}\b", ngram_range=(1, 2),
                          min_df=max(3, len(df) // 2000), max_df=0.4, max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(docs)
    k = k or int(np.clip(len(df) // 250, 4, 8))
    model = NMF(n_components=k, init="nndsvda", random_state=20261109, max_iter=400)
    W = model.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    labels = []
    for comp in model.components_:
        top = [t for t in terms[np.argsort(comp)[::-1]] if " " not in t][:4]
        labels.append(" · ".join(top))
    strength = W.max(axis=1)
    topic = np.where(strength > 0.01, W.argmax(axis=1), -1)
    df = df.assign(narrative=[labels[t] if t >= 0 else "Sin narrativa clara" for t in topic], _w=strength)
    sizes = df.loc[df["narrative"] != "Sin narrativa clara", "narrative"].value_counts()
    order = list(sizes.index)
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(order)}

    # Evolución: participación diaria (%)
    share = (df[df["narrative"].isin(order)].groupby(["day", "narrative"]).size().unstack(fill_value=0))
    share = share.div(share.sum(axis=1), axis=0).mul(100)[order]
    f_share = go.Figure()
    for n in order:
        f_share.add_scatter(x=share.index, y=share[n].round(1), name=n, stackgroup="one", mode="lines",
                            line=dict(width=0.5, color=colors[n]), hovertemplate="%{y:.1f}%<extra>" + n + "</extra>")
    f_share.update_layout(**layout("Peso de cada narrativa en la conversación diaria (%)", height=540, yaxis_title="% del día",
                                   yaxis_range=[0, 100]))

    # Tamaño + mezcla de plataformas
    mix = df[df["narrative"].isin(order)].groupby(["narrative", "platform"]).size().unstack(fill_value=0).loc[order[::-1]]
    f_size = go.Figure()
    for i, p in enumerate(mix.columns):
        f_size.add_bar(y=mix.index, x=mix[p], name=p, orientation="h", marker_color=PALETTE[i % len(PALETTE)])
    f_size.update_layout(**layout("Narrativas detectadas: tamaño y plataforma donde circulan", height=120 + 46 * len(order),
                                  barmode="stack", xaxis_title="publicaciones", margin=dict(l=320, r=30, t=70, b=60)))
    table = [{"narrativa": n, "publicaciones": int(sizes[n]),
              "ejemplo": df.loc[df["narrative"] == n].sort_values("_w", ascending=False)["text"].iloc[0][:180]}
             for n in order]
    return df, axes(f_share), axes(f_size), table


# ------------------------------------------------------------- 3. actores y riesgo

def alias_pattern(aliases):
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(norm(x)) for x in aliases) + r")(?![a-z])")


RISK = alias_pattern(CONFIG["risk_terms"])


def mentions(df) -> pd.DataFrame:
    """Una fila por (publicación, político mencionado). En textos en portugués, los políticos argentinos solo cuentan
    con alias de nombre completo (evita falsos positivos como 'massa' = 'genial' en portugués)."""
    meta = CONFIG.get("actor_meta", {})
    pt = df["lang2"].eq("pt") if "lang2" in df else pd.Series(False, index=df.index)
    rows = []
    for a, alias in CONFIG["actors"].items():
        m = df["clean"].str.contains(alias_pattern(alias))
        if meta.get(a, {}).get("pais") == "AR":
            safe = [x for x in alias if " " in x or norm(x) == "milei"]
            m &= ~pt | (df["clean"].str.contains(alias_pattern(safe)) if safe else False)
        if m.any():
            rows.append(df.loc[m].assign(actor=a))
    return pd.concat(rows) if rows else pd.DataFrame(columns=list(df.columns) + ["actor"])


def actors(df):
    df = df.assign(risk=df["clean"].str.contains(RISK))
    men = mentions(df)
    if men.empty:
        return None, None, None, []
    top = men["actor"].value_counts().index[:8].tolist()
    men = men[men["actor"].isin(top)]

    daily = men.groupby(["day", "actor"]).size().unstack(fill_value=0)[top]
    f_line = go.Figure()
    for i, a in enumerate(top):
        f_line.add_scatter(x=daily.index, y=daily[a], name=a, mode="lines+markers", line=dict(color=PALETTE[i % len(PALETTE)], width=2))
    f_line.update_layout(**layout("Menciones diarias por actor", yaxis_title="publicaciones que lo mencionan"))

    narr = [n for n in men["narrative"].value_counts().index if n != "Sin narrativa clara"][:8]
    heat = (men[men["narrative"].isin(narr)].groupby(["actor", "narrative"]).size().unstack(fill_value=0)
            .reindex(index=top, columns=narr, fill_value=0))
    pct = heat.div(heat.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)
    f_heat = go.Figure(go.Heatmap(z=pct.values.round(1), x=pct.columns, y=pct.index, colorscale=[[0, PANEL], [1, "#55d5ba"]],
                                  text=heat.values, hovertemplate="%{y}<br>%{x}<br>%{z:.1f}% de sus menciones (n=%{text})<extra></extra>",
                                  colorbar=dict(title="%")))
    f_heat.update_layout(**layout("Con qué narrativa aparece cada actor (% de sus menciones)", height=150 + 44 * len(top),
                                  margin=dict(l=170, r=30, t=70, b=200)))
    f_heat.update_xaxes(tickangle=30)

    r = men.groupby("actor").agg(n=("risk", "size"), riesgo=("risk", "mean")).reindex(top)
    r["riesgo"] *= 100
    se = np.sqrt(r["riesgo"] / 100 * (1 - r["riesgo"] / 100) / r["n"]) * 100 * 1.96
    r = r.sort_values("riesgo")
    f_risk = go.Figure(go.Bar(x=r["riesgo"].round(1), y=r.index, orientation="h", marker_color="#ef6f6c",
                              error_x=dict(type="data", array=se.reindex(r.index).round(1), color=MUTED),
                              text=[f"n={n}" for n in r["n"]], textposition="inside", insidetextanchor="start", textfont=dict(color=BG),
                              hovertemplate="%{y}: %{x:.1f}% ± IC95<extra></extra>"))
    f_risk.update_layout(**layout("Exposición reputacional: % de menciones junto a términos de riesgo", height=150 + 44 * len(top),
                                  xaxis_title="% de menciones (barra = IC 95%)", margin=dict(l=170, r=60, t=70, b=60)))
    summary = [{"actor": a, "menciones": int(r.loc[a, "n"]), "riesgo_pct": round(float(r.loc[a, "riesgo"]), 1)} for a in top]
    return axes(f_line), f_heat, axes(f_risk), summary


# ------------------------------------------------------------- 4. coordinación

def coordination(df, window_min=60, threshold=0.9, min_repeats=2):
    posts = df[(df["kind"] == "post") & (df["clean"].str.split().str.len() >= 8)].reset_index(drop=True)
    empty = (None, None, [], {"cuentas": 0, "grupos": 0, "pares": 0})
    if len(posts) < 20:
        return empty
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2, sublinear_tf=True).fit_transform(posts["clean"])
    nn = NearestNeighbors(radius=1 - threshold, metric="cosine", algorithm="brute").fit(X)
    dist, idx = nn.radius_neighbors(X)
    t = posts["ts"].astype("int64").to_numpy() // 60_000_000_000
    G = nx.Graph()
    pairs, co = [], {}
    for i, (ds, js) in enumerate(zip(dist, idx)):
        for d, j in zip(ds, js):
            if j <= i or posts.at[i, "who"] == posts.at[j, "who"] or abs(t[i] - t[j]) > window_min:
                continue
            a, b = sorted((posts.at[i, "who"], posts.at[j, "who"]))
            pairs.append((i, j))
            co.setdefault((a, b), set()).add(posts.at[i, "clean"][:60])
    # Una coincidencia aislada puede ser casualidad (dos personas que copian el mismo titular).
    # Se exige que el par de cuentas coincida en al menos `min_repeats` mensajes distintos.
    for (a, b), msgs in co.items():
        if len(msgs) >= min_repeats:
            G.add_edge(a, b, w=len(msgs))
    pairs = [(i, j) for i, j in pairs if G.has_edge(posts.at[i, "who"], posts.at[j, "who"])]
    comps = [c for c in nx.connected_components(G) if len(c) >= 3]
    if not comps:
        return None, None, [], {"cuentas": 0, "grupos": 0, "pares": len(pairs)}
    H = G.subgraph(set().union(*comps)).copy()
    comp_of = {n: k for k, c in enumerate(sorted(comps, key=len, reverse=True)) for n in c}
    pos = nx.spring_layout(H, seed=7, weight="w", k=0.6 / np.sqrt(max(len(H), 1)))
    plat = posts.groupby("who")["platform"].agg(lambda s: s.mode().iat[0])
    fig = go.Figure()
    ex, ey = [], []
    for a, b in H.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#34465e", width=1), hoverinfo="skip", showlegend=False)
    nodes = list(H.nodes())
    fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", showlegend=False,
                    marker=dict(size=[8 + 2.5 * H.degree(n) for n in nodes], color=[PALETTE[comp_of[n] % len(PALETTE)] for n in nodes],
                                line=dict(color=BG, width=1)),
                    text=[f"{n} · {plat.get(n, '')}<br>grupo {comp_of[n] + 1} · {H.degree(n)} cuentas enlazadas" for n in nodes],
                    hovertemplate="%{text}<extra></extra>")
    fig.update_layout(**layout(f"Posible coordinación: {len(nodes)} cuentas en {len(comps)} grupos publican textos casi idénticos "
                               f"en menos de {window_min} min, al menos {min_repeats} veces", height=620))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Mensajes repetidos más amplificados
    groups = []
    used = set()
    for i, j in pairs:
        if posts.at[i, "who"] not in comp_of:
            continue
        key = posts.at[i, "clean"][:80]
        if key in used:
            continue
        used.add(key)
        sim = posts[posts["clean"].str[:80] == key]
        groups.append({"mensaje": posts.at[i, "text"][:200], "cuentas": int(sim["who"].nunique()) if len(sim) > 1 else 2,
                       "plataforma": posts.at[i, "platform"], "primer_registro": posts.at[i, "ts"].strftime("%d/%m %H:%M"),
                       "grupo": comp_of[posts.at[i, "who"]] + 1})
    groups = sorted(groups, key=lambda g: -g["cuentas"])[:12]
    tbl = go.Figure(go.Table(
        columnwidth=[60, 420, 70, 90, 90],
        header=dict(values=["Grupo", "Mensaje repetido", "Cuentas", "Plataforma", "Primera vez"], fill_color=BG, font=dict(color=TXT),
                    align="left", line_color=GRID),
        cells=dict(values=[[g["grupo"] for g in groups], [g["mensaje"] for g in groups], [g["cuentas"] for g in groups],
                           [g["plataforma"] for g in groups], [g["primer_registro"] for g in groups]],
                   fill_color=PANEL, font=dict(color=MUTED), align="left", height=30, line_color=GRID)))
    tbl.update_layout(**layout("Mensajes casi idénticos de los grupos detectados", height=140 + 52 * max(len(groups), 1)))
    return fig, tbl, groups, {"cuentas": len(nodes), "grupos": len(comps), "pares": len(pairs)}


# ------------------------------------------------------------- 5. hashtags y dominios

def hashtags(df, top=30):
    tags = df["hashtags"].map(lambda h: [x for x in h if norm(x) not in {"argentina"}] if isinstance(h, list) else [])
    cnt = Counter(t for h in tags for t in set(h))
    keep = {t for t, _ in cnt.most_common(top)}
    if len(keep) < 4:
        return None
    G = nx.Graph()
    for h in tags:
        for a, b in combinations(sorted(set(h) & keep), 2):
            G.add_edge(a, b, w=G[a][b]["w"] + 1 if G.has_edge(a, b) else 1)
    if G.number_of_nodes() < 4:
        return None
    pos = nx.spring_layout(G, seed=3, weight="w", k=1.2 / np.sqrt(G.number_of_nodes()))
    comm = {n: k for k, c in enumerate(nx.community.greedy_modularity_communities(G, weight="w")) for n in c}
    fig = go.Figure()
    mw = max([d["w"] for *_, d in G.edges(data=True)] or [1])
    for a, b, d in G.edges(data=True):
        fig.add_scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color="#34465e", width=0.5 + 4 * d["w"] / mw))
    nodes = list(G.nodes())
    fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=["#" + n if cnt[n] >= sorted(cnt[x] for x in nodes)[-12] else "" for n in nodes],
                    textposition="top center", textfont=dict(size=11, color=MUTED), showlegend=False,
                    marker=dict(size=[8 + 30 * np.sqrt(cnt[n] / cnt.most_common(1)[0][1]) for n in nodes],
                                color=[PALETTE[comm[n] % len(PALETTE)] for n in nodes]),
                    hovertext=[f"#{n}: {cnt[n]} publicaciones" for n in nodes], hoverinfo="text")
    fig.update_layout(**layout("Hashtags que viajan juntos (color = comunidad; grosor = publicaciones compartidas)", height=620))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def domains(df, top=15):
    skip = {"bsky.app", "reddit.com", "i.redd.it", "v.redd.it", "news.google.com", "mastodon.social", "mas.to", "x.com", "twitter.com"}
    posts = df[df["kind"] == "post"].explode("domains").dropna(subset=["domains"])
    posts = posts[~posts["domains"].isin(skip)]
    if posts.empty:
        return None
    t = posts.groupby(["domains", "platform"]).size().unstack(fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index[-top:]]
    fig = go.Figure()
    for i, p in enumerate(t.columns):
        fig.add_bar(y=t.index, x=t[p], name=p, orientation="h", marker_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(**layout("Sitios más enlazados desde redes", height=120 + 34 * len(t), barmode="stack",
                               xaxis_title="publicaciones con enlace", margin=dict(l=200, r=30, t=70, b=60)))
    return axes(fig)


def amplifiers(df, top=15):
    posts = df[df["kind"] == "post"]
    if posts.empty:
        return None, {}
    by = posts.groupby("who").agg(n=("id", "size"), plat=("platform", lambda x: x.mode().iat[0]),
                                  narr=("narrative", lambda x: x.mode().iat[0])).sort_values("n", ascending=False)
    k = max(1, int(np.ceil(len(by) * 0.01)))
    conc = {"cuentas": int(len(by)), "top1pct_cuentas": k, "top1pct_share": round(float(by["n"].iloc[:k].sum() / by["n"].sum() * 100), 1),
            "top10_share": round(float(by["n"].iloc[:10].sum() / by["n"].sum() * 100), 1)}
    t = by.iloc[:top][::-1]
    fig = go.Figure(go.Bar(x=t["n"], y=[f"{w} · {p}" for w, p in zip(t.index, t["plat"])], orientation="h", marker_color="#7aa7ff",
                           customdata=t["narr"], hovertemplate="%{y}<br>%{x} publicaciones<br>narrativa principal: %{customdata}<extra></extra>"))
    fig.update_layout(**layout(f"Cuentas más activas: el {conc['top1pct_share']}% del volumen lo generan {k} cuentas (1% del total)",
                               height=120 + 30 * len(t), xaxis_title="publicaciones", margin=dict(l=230, r=30, t=70, b=60)))
    return axes(fig), conc


def trends(path: Path, extra=()):
    """Google Trends. Acepta trends.json (del recolector) o trends_*.jsonl (una fila por punto: group, term, series, date, value)."""
    rows = []
    for e in extra:
        for f in (sorted(Path(e).glob("trends_*.jsonl")) if Path(e).is_dir() else [Path(e)] if "trends" in Path(e).name else []):
            rows += read_jsonl(f)
    if not rows and path.exists():
        t = json.loads(path.read_text(encoding="utf8"))
        for r in t.get("rows", []):
            for term, v in zip(t["terms"], r["values"]):
                rows.append({"series": "time", "group": "actores", "term": term, "geo": t["geo"],
                             "date": datetime.fromtimestamp(r["time"], timezone.utc).strftime("%Y-%m-%d"), "value": v, "timeframe": "1m"})
    if not rows:
        return None, None
    t = pd.DataFrame(rows)
    ts = t[t["series"] == "time"]
    if "timeframe" in ts and ts["timeframe"].nunique() > 1:  # el período más corto, más legible
        ts = ts[ts["timeframe"] == sorted(ts["timeframe"].unique(), key=lambda x: ("12" in x, x))[0]]
    # Un término en 0 durante todo el período suele ser un dato faltante de Trends, no ausencia de búsquedas: se omite
    mx = ts.groupby(["group", "term"])["value"].max()
    ts = ts[~ts.set_index(["group", "term"]).index.isin(mx[mx == 0].index)]
    groups = [g for g in ["actores", "actores2", "economia", "agenda"] if g in set(ts["group"])] or sorted(ts["group"].unique())
    from plotly.subplots import make_subplots
    names = {"actores": "Dirigentes", "actores2": "Dirigentes (2)", "economia": "Economía", "agenda": "Agenda"}
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[names.get(g, g) + " (100 = máximo del grupo)" for g in groups])
    for k, g in enumerate(groups, 1):
        d = ts[ts["group"] == g]
        for i, term in enumerate(d["term"].unique()):
            x = d[d["term"] == term].sort_values("date")
            fig.add_scatter(x=x["date"], y=x["value"], name=term, mode="lines", legendgroup=g, legendgrouptitle_text=names.get(g, g),
                            line=dict(width=2, color=PALETTE[i % len(PALETTE)]), row=k, col=1)
    geo = ts["geo"].iloc[0] if "geo" in ts else "AR"
    fig.update_layout(**layout(f"Qué busca la gente en Google ({geo})", height=260 * len(groups) + 120,
                               legend=dict(orientation="v", x=1.02, y=1), margin=dict(l=60, r=200, t=90, b=50)))
    for a in fig.layout.annotations:
        a.font = dict(size=13, color=MUTED)
    axes(fig)

    reg = t[(t["series"] == "region") & (t["group"] == "actores")] if "region" in set(t["series"]) else pd.DataFrame()
    fig_r = None
    if not reg.empty:
        if "timeframe" in reg and reg["timeframe"].nunique() > 1:
            reg = reg[reg["timeframe"] == sorted(reg["timeframe"].unique(), key=lambda x: ("12" in x, x))[0]]
        piv = reg.pivot_table(index="region", columns="term", values="value", aggfunc="mean").fillna(0)
        order = [x for x in ts.loc[ts["group"] == "actores", "term"].unique() if x in piv.columns]  # mismos colores que la serie
        piv = piv[order + [c for c in piv.columns if c not in order]]
        share = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)
        lead = share.columns[0] if "Milei" not in share.columns else "Milei"
        share = share.sort_values(lead)
        fig_r = go.Figure()
        for i, term in enumerate(share.columns):
            fig_r.add_bar(y=share.index, x=share[term].round(1), name=term, orientation="h", marker_color=PALETTE[i % len(PALETTE)])
        fig_r.update_layout(**layout("Reparto de las búsquedas de dirigentes por provincia (%)", height=120 + 24 * len(share),
                                     barmode="stack", xaxis_title="% de las búsquedas de estos 5 dirigentes", xaxis_range=[0, 100],
                                     margin=dict(l=190, r=30, t=70, b=60)))
        axes(fig_r)
    return fig, fig_r


def editorial_lines(df):
    if "linea" not in df or df["linea"].fillna("").eq("").all():
        return None
    d = df[df["linea"].fillna("") != ""]
    narr = [n for n in d["narrative"].value_counts().index if n != "Sin narrativa clara"][:8]
    t = d[d["narrative"].isin(narr)].groupby(["linea", "narrative"]).size().unstack(fill_value=0).reindex(columns=narr, fill_value=0)
    pct = t.div(t.sum(axis=1), axis=0).mul(100)
    fig = go.Figure(go.Heatmap(z=pct.values.round(1), x=pct.columns, y=pct.index, colorscale=[[0, PANEL], [1, "#f2b84b"]], text=t.values,
                               hovertemplate="%{y}<br>%{x}<br>%{z:.1f}% (n=%{text})<extra></extra>", colorbar=dict(title="%")))
    fig.update_layout(**layout("Comentarios en YouTube según la línea editorial del canal: qué narrativa domina",
                               height=200 + 50 * len(pct), margin=dict(l=200, r=30, t=70, b=200)))
    fig.update_xaxes(tickangle=30)
    return fig


def coverage(df, status):
    c = df.groupby("platform").agg(publicaciones=("id", "size"), cuentas=("who", "nunique"), desde=("ts", "min"), hasta=("ts", "max"))
    rows = [[p, int(r.publicaciones), int(r.cuentas), r.desde.strftime("%d/%m"), r.hasta.strftime("%d/%m")] for p, r in c.iterrows()]
    st = {s["source"]: ("ok" if s["ok"] else "falló: " + s.get("error", "")[:60]) for s in (status or {}).get("sources", [])}
    names = {"mastodon": "Mastodon", "bluesky": "Bluesky", "reddit": "Reddit", "google_news": "Google News", "rss": "RSS medios", "youtube": "YouTube", "google_trends": "Google Trends", "gdelt": "GDELT"}
    for k, v in st.items():
        for r in rows:
            if r[0] == names.get(k):
                r.append(v)
                break
        else:
            rows.append([names.get(k, k), 0, 0, "", "", v])
    for r in rows:
        r += [""] * (6 - len(r))
    fig = go.Figure(go.Table(header=dict(values=["Fuente", "Publicaciones", "Cuentas / medios", "Desde", "Hasta", "Última recolección"],
                                         fill_color=BG, font=dict(color=TXT), align="left", line_color=GRID),
                             cells=dict(values=list(map(list, zip(*rows))), fill_color=PANEL, font=dict(color=MUTED), align="left", height=28,
                                        line_color=GRID)))
    fig.update_layout(**layout("Cobertura por fuente", height=150 + 48 * len(rows)))
    return fig


# ------------------------------------------------------------- principal

# ------------------------------------------------------------- 7. tono, emociones y agresividad

TONE_NAMES = {"NEG": "negativo", "NEU": "neutro", "POS": "positivo"}
EMOTIONS = ["ira", "asco", "miedo", "tristeza", "alegria", "sorpresa"]
EMO_NAMES = {"ira": "Ira", "asco": "Asco", "miedo": "Miedo", "tristeza": "Tristeza", "alegria": "Alegría", "sorpresa": "Sorpresa"}


def source_label(df) -> pd.Series:
    linea = df["linea"].fillna("") if "linea" in df else pd.Series("", index=df.index)
    return pd.Series(np.where(linea != "", "YouTube · " + linea.astype(str), df["platform"]), index=df.index)


def tone_figures(df):
    d = df.assign(fuente=source_label(df))
    order = d["fuente"].value_counts()
    order = order[order >= 30].index.tolist()
    if not order:
        return None, None, None, None
    t = d[d["fuente"].isin(order)].groupby(["fuente", "tono"]).size().unstack(fill_value=0).reindex(columns=["NEG", "NEU", "POS"], fill_value=0)
    pct = t.div(t.sum(axis=1), axis=0).mul(100)
    pct = pct.assign(neto=pct["POS"] - pct["NEG"]).sort_values("neto")
    f_tone = go.Figure()
    for k in ["NEG", "NEU", "POS"]:
        f_tone.add_bar(y=pct.index, x=pct[k].round(1), name=TONE_NAMES[k], orientation="h", marker_color=TONE_COLORS[k],
                       marker_line=dict(color=PANEL, width=2), customdata=t.loc[pct.index, k],
                       hovertemplate="%{y}<br>" + TONE_NAMES[k] + ": %{x:.1f}% (n=%{customdata})<extra></extra>")
    f_tone.update_layout(**layout("Tono por fuente (% de publicaciones)", height=140 + 40 * len(pct), barmode="stack",
                                  xaxis_title="%", xaxis_range=[0, 100], margin=dict(l=230, r=30, t=70, b=60)))

    # Tono neto diario: redes vs medios (promedio móvil de 3 días)
    d["grupo"] = np.where(d["kind"] == "post", "Redes (comentarios)", "Medios (titulares)")
    daily = d.groupby(["day", "grupo"])["tono"].apply(lambda s: (s.eq("POS").mean() - s.eq("NEG").mean()) * 100).unstack()
    daily = daily.rolling(3, min_periods=1).mean()
    f_net = go.Figure()
    for i, g in enumerate(daily.columns):
        f_net.add_scatter(x=daily.index, y=daily[g].round(1), name=g, mode="lines", line=dict(width=2, color=PALETTE[i]))
    f_net.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    f_net.update_layout(**layout("Tono neto diario (% positivo − % negativo, promedio de 3 días)", yaxis_title="puntos"))

    e = d[d["emocion"].isin(EMOTIONS) & d["fuente"].isin(order)]
    f_emo = None
    if len(e) >= 30:
        h = e.groupby(["fuente", "emocion"]).size().unstack(fill_value=0).reindex(columns=EMOTIONS, fill_value=0)
        hp = h.div(h.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)
        f_emo = go.Figure(go.Heatmap(z=hp.values.round(1), x=[EMO_NAMES[c] for c in hp.columns], y=hp.index, text=h.values,
                                     colorscale=[[0, "#1b2230"], [1, "#d95926"]],
                                     hovertemplate="%{y}<br>%{x}: %{z:.1f}% (n=%{text})<extra></extra>", colorbar=dict(title="%")))
        f_emo.update_layout(**layout("Emoción dominante por fuente (% de los textos con emoción)", height=140 + 44 * len(hp),
                                     margin=dict(l=230, r=30, t=70, b=60)))

    a = d[d["fuente"].isin(order)].groupby("fuente")["agresivo"].agg(["mean", "size"])
    a["pct"] = a["mean"] * 100
    a["ci"] = 1.96 * np.sqrt(a["mean"] * (1 - a["mean"]) / a["size"]) * 100
    a = a.sort_values("pct")
    f_agr = go.Figure(go.Bar(x=a["pct"].round(1), y=a.index, orientation="h", marker_color="#d55181",
                             error_x=dict(type="data", array=a["ci"].round(1), color=MUTED), customdata=a["size"],
                             hovertemplate="%{y}: %{x:.1f}% (n=%{customdata})<extra></extra>"))
    f_agr.update_layout(**layout("Agresividad e insultos: % de textos agresivos por fuente (barra = IC 95%)", height=140 + 40 * len(a),
                                 xaxis_title="% de textos", margin=dict(l=230, r=40, t=70, b=60)))
    return f_tone, axes(f_net), f_emo, axes(f_agr)


# ------------------------------------------------------------- 8. perfiles de políticos, agenda y capa OSINT

PROVINCES = {  # nombre: (lat, lon, alias normalizados)
    "Buenos Aires": (-36.6, -60.0, ["provincia de buenos aires", "bonaerense", "conurbano", "la plata", "mar del plata", "bahia blanca", "la matanza"]),
    "CABA": (-34.61, -58.44, ["caba", "ciudad de buenos aires", "porteno", "portenos", "legislatura portena"]),
    "Catamarca": (-28.47, -65.78, ["catamarca", "catamarqueno"]),
    "Chaco": (-26.4, -60.8, ["chaco", "chaqueno", "resistencia chaco"]),
    "Chubut": (-43.8, -68.5, ["chubut", "comodoro rivadavia", "trelew"]),
    "Córdoba": (-31.4, -64.2, ["cordoba", "cordobes", "cordobeses"]),
    "Corrientes": (-28.7, -57.8, ["de corrientes", "en corrientes", "correntino", "correntinos"]),
    "Entre Ríos": (-32.0, -59.2, ["entre rios", "entrerriano", "entrerrianos"]),
    "Formosa": (-24.9, -59.9, ["formosa", "formoseno"]),
    "Jujuy": (-23.3, -65.8, ["jujuy", "jujeno"]),
    "La Pampa": (-37.1, -65.4, ["la pampa", "pampeano"]),
    "La Rioja": (-29.7, -67.2, ["la rioja", "riojano"]),
    "Mendoza": (-34.6, -68.6, ["mendoza", "mendocino", "mendocinos"]),
    "Misiones": (-26.9, -54.6, ["misiones", "misionero", "posadas"]),
    "Neuquén": (-38.6, -70.0, ["neuquen", "neuquino", "vaca muerta"]),
    "Río Negro": (-40.4, -67.2, ["rio negro", "rionegrino", "bariloche"]),
    "Salta": (-24.8, -65.4, ["de salta", "en salta", "salteno", "saltenos"]),
    "San Juan": (-30.9, -68.9, ["san juan", "sanjuanino"]),
    "San Luis": (-33.8, -66.0, ["san luis", "puntano"]),
    "Santa Cruz": (-48.8, -69.6, ["santa cruz", "santacruceno", "rio gallegos"]),
    "Santa Fe": (-30.7, -60.9, ["santa fe", "santafesino", "rosario"]),
    "Santiago del Estero": (-27.8, -63.3, ["santiago del estero", "santiagueno"]),
    "Tierra del Fuego": (-54.3, -67.7, ["tierra del fuego", "ushuaia", "fueguino"]),
    "Tucumán": (-26.9, -65.4, ["tucuman", "tucumano"]),
}
PROV_PATS = {p: alias_pattern(v[2]) for p, v in PROVINCES.items()}
WORD4 = re.compile(r"(?<![a-z0-9])[a-z]{4,}(?![a-z0-9])")


def headline(text: str) -> str:
    """Solo el título (las fuentes RSS traen título + bajada)."""
    return str(text).strip().split("\n")[0][:220]


def ci95(p, n):
    return float(1.96 * np.sqrt(p * (1 - p) / n) * 100) if n else None


def pct(x):
    return round(float(x) * 100, 1)


def agenda_figures(men, df):
    """Agenda de medios vs agenda de redes y red de co-menciones entre políticos."""
    if men.empty:
        return None, None
    news = men[men["kind"] == "news"]["actor"].value_counts()
    soc = men[men["kind"] == "post"]["actor"].value_counts()
    t = pd.DataFrame({"news": news, "social": soc}).fillna(0)
    t = t[(t["news"] + t["social"]) >= 5]
    if t.empty:
        return None, None
    t["news_pct"] = t["news"] / max(t["news"].sum(), 1) * 100
    t["soc_pct"] = t["social"] / max(t["social"].sum(), 1) * 100
    meta = CONFIG.get("actor_meta", {})
    t["pais"] = [meta.get(a, {}).get("pais", "") for a in t.index]
    hi = max(t["news_pct"].max(), t["soc_pct"].max()) * 1.1
    f = go.Figure()
    f.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=MUTED, dash="dot", width=1), hoverinfo="skip", showlegend=False)
    for i, (pais, g) in enumerate(t.groupby("pais")):
        f.add_scatter(x=g["news_pct"].round(1), y=g["soc_pct"].round(1), mode="markers+text", name={"AR": "Argentina", "BR": "Brasil"}.get(pais, pais),
                      text=[a.split()[-1] for a in g.index], textposition="top center", textfont=dict(size=11, color=MUTED),
                      marker=dict(size=np.clip(np.sqrt(g["news"] + g["social"]) * 1.6, 9, 46), color=PALETTE[i], line=dict(color=PANEL, width=2)),
                      customdata=np.c_[g.index, g["news"], g["social"]],
                      hovertemplate="%{customdata[0]}<br>medios: %{x:.1f}% (n=%{customdata[1]})<br>redes: %{y:.1f}% (n=%{customdata[2]})<extra></extra>")
    f.update_layout(**layout("Agenda de medios vs agenda de redes: arriba de la diagonal, las redes hablan más de lo que cubren los medios",
                             height=560, xaxis_title="% de las menciones en medios", yaxis_title="% de las menciones en redes"))

    pairs = Counter()
    for _, g in men.groupby(level=0)["actor"]:
        for a, b in combinations(sorted(set(g)), 2):
            pairs[(a, b)] += 1
    edges = [(a, b, w) for (a, b), w in pairs.items() if w >= 3]
    f_net = None
    if edges:
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        pos = nx.spring_layout(G, seed=7, weight="weight", k=0.9)
        f_net = go.Figure()
        top = max(w for *_, w in edges)
        for a, b, w in edges:
            f_net.add_scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines", hoverinfo="skip", showlegend=False,
                              line=dict(width=0.6 + 5 * w / top, color="#3a4458"))
        nodes = list(G.nodes)
        size = men["actor"].value_counts()
        f_net.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", showlegend=False,
                          text=nodes, textposition="top center", textfont=dict(size=11),
                          marker=dict(size=[float(np.clip(np.sqrt(size.get(n, 1)) * 1.5, 10, 50)) for n in nodes],
                                      color=[PALETTE[0] if meta.get(n, {}).get("pais") == "AR" else PALETTE[1] for n in nodes],
                                      line=dict(color=PANEL, width=2)),
                          customdata=[int(size.get(n, 0)) for n in nodes], hovertemplate="%{text}: %{customdata} menciones<extra></extra>")
        f_net.update_layout(**layout("Quién aparece con quién: red de co-menciones (grosor = textos que nombran a ambos)", height=620,
                                     xaxis=dict(visible=False), yaxis=dict(visible=False)))
    return axes(f), f_net


def pressure(neg_pct, agr_pct, risk_pct, delta):
    """Índice de presión reputacional 0-100: combinación simple y explicable de cuatro señales."""
    parts = {"tono negativo": min(neg_pct / 60, 1) * 35, "agresividad": min(agr_pct / 20, 1) * 20,
             "términos de riesgo": min(risk_pct / 15, 1) * 25, "aceleración": min(max(delta or 0, 0) / 200, 1) * 20}
    return round(sum(parts.values())), {k: round(v, 1) for k, v in parts.items()}


def actor_profiles(df, men, fc, days):
    """Perfil público y agregado de cada político configurado (para el buscador del tablero)."""
    meta = CONFIG.get("actor_meta", {})
    df = df.assign(risk=df["clean"].str.contains(RISK))
    if not men.empty:
        men = men.assign(risk=men["clean"].str.contains(RISK))
    words = [set(w for w in WORD4.findall(c) if w not in STOP) for c in df["clean"]]
    total_words = Counter(w for ws in words for w in ws)
    pos_of = {ix: i for i, ix in enumerate(df.index)}
    last = df["day"].max()
    fc_men = mentions(fc) if not fc.empty else pd.DataFrame(columns=["actor"])
    total_mentions = max(len(men), 1)
    ALIAS_WORDS = set(re.findall(r"[a-z]{4,}", " ".join(norm(x) for k, v in CONFIG["actors"].items() for x in v + [k])))
    by_ix = men.groupby(level=0)["actor"].agg(set).to_dict() if not men.empty else {}
    out = []
    for a, alias in CONFIG["actors"].items():
        m = men[men["actor"] == a] if not men.empty else men
        n = int(len(m))
        info = {"name": a, "pais": meta.get(a, {}).get("pais", ""), "espacio": meta.get(a, {}).get("espacio", ""), "alias": alias, "n": n}
        if n == 0:
            out.append(info)
            continue
        soc, news = m[m["kind"] == "post"], m[m["kind"] == "news"]
        last7 = int((m["day"] > last - pd.Timedelta(days=7)).sum())
        prev7 = int(((m["day"] <= last - pd.Timedelta(days=7)) & (m["day"] > last - pd.Timedelta(days=14))).sum())
        delta = round((last7 - prev7) / prev7 * 100) if prev7 >= 3 else None
        daily_s = soc.groupby("day").size().reindex(days, fill_value=0)
        daily_n = news.groupby("day").size().reindex(days, fill_value=0)
        tone_src = soc if len(soc) >= 10 else m
        tone = {k: pct(tone_src["tono"].eq(k).mean()) for k in ["NEG", "NEU", "POS"]}
        tone_news = {k: pct(news["tono"].eq(k).mean()) for k in ["NEG", "NEU", "POS"]} if len(news) else None
        emo = tone_src["emocion"][tone_src["emocion"].isin(EMOTIONS)]
        emotions = {e: pct(emo.eq(e).mean()) for e in EMOTIONS} if len(emo) >= 5 else None
        agr = pct(tone_src["agresivo"].mean())
        risk = pct(m["risk"].mean())
        idx, parts = pressure(tone["NEG"], agr, risk, delta)
        narr = m.loc[m["narrative"] != "Sin narrativa clara", "narrative"].value_counts()
        own = ALIAS_WORDS
        mine = Counter(w for ix in m.index.unique() for w in words[pos_of[ix]])
        nm, nall = len(m.index.unique()), len(df)
        score = {w: (c / nm) / ((total_words[w] - c + 1) / max(nall - nm, 1)) for w, c in mine.items() if c >= 3 and w not in own}
        co = Counter()
        for ix in m.index.unique():
            co.update(by_ix.get(ix, set()) - {a})
        lines = None
        if "linea" in soc and soc["linea"].fillna("").ne("").any():
            y = soc[soc["linea"].fillna("") != ""]
            lines = {l: {"n": int(len(g)), "neg": pct(g["tono"].eq("NEG").mean()), "pos": pct(g["tono"].eq("POS").mean()),
                         "agresivo": pct(g["agresivo"].mean())} for l, g in y.groupby("linea") if len(g) >= 5}
        prov = {p: int(m["clean"].str.contains(pat).sum()) for p, pat in PROV_PATS.items()}
        heads = news.sort_values("ts", ascending=False).drop_duplicates("clean").head(8)
        f = fc_men[fc_men["actor"] == a].sort_values("ts", ascending=False).head(5) if len(fc_men) else fc_men
        info.update({
            "n_social": int(len(soc)), "n_news": int(len(news)), "share": round(n / total_mentions * 100, 1),
            "last7": last7, "prev7": prev7, "delta": delta,
            "daily_social": daily_s.astype(int).tolist(), "daily_news": daily_n.astype(int).tolist(),
            "tone": tone, "tone_base": "redes" if len(soc) >= 10 else "todas", "tone_news": tone_news, "emotions": emotions,
            "agresivo": agr, "risk": risk, "risk_ci": round(ci95(m["risk"].mean(), n), 1),
            "pressure": idx if n >= 20 else None, "pressure_parts": parts if n >= 20 else None,
            "narratives": [{"n": k, "pct": pct(v / max(narr.sum(), 1))} for k, v in narr.head(5).items()],
            "words": [{"w": w, "s": round(v, 1)} for w, v in sorted(score.items(), key=lambda x: -x[1])[:12]],
            "co": [{"a": b, "n": c} for b, c in co.most_common(6)],
            "platforms": {k: int(v) for k, v in m["platform"].value_counts().items()},
            "lines": lines, "provinces": {k: v for k, v in sorted(prov.items(), key=lambda x: -x[1]) if v},
            "headlines": [{"t": headline(r.text), "o": r.who, "u": r.url, "d": r.ts.strftime("%Y-%m-%d %H:%M"), "tono": r.tono} for r in heads.itertuples()],
            "factchecks": [{"t": headline(r.text), "o": r.outlet or r.author, "u": r.url, "d": r.ts.strftime("%Y-%m-%d")} for r in f.itertuples()],
        })
        out.append(info)
    return out


def province_map(df):
    es = df[df["lang2"] != "pt"] if "lang2" in df else df
    rows = []
    for p, (lat, lon, _) in PROVINCES.items():
        m = es["clean"].str.contains(PROV_PATS[p])
        n = int(m.sum())
        if n:
            sub = es[m]
            rows.append({"p": p, "lat": lat, "lon": lon, "n": n, "news": int((sub["kind"] == "news").sum()),
                         "neg": pct(sub["tono"].eq("NEG").mean()), "net": round((sub["tono"].eq("POS").mean() - sub["tono"].eq("NEG").mean()) * 100, 1),
                         "top": distinctive_terms(sub["clean"], es.loc[~m, "clean"], 3)})
    return rows


def live_layer(df, fc, men, profiles, spikes, coord):
    """Feed de titulares (públicos) y alertas automáticas, al estilo de un monitor OSINT."""
    news = pd.concat([df[df["kind"] == "news"].sort_values("ts", ascending=False).drop_duplicates("clean").head(70),
                      fc.sort_values("ts", ascending=False).drop_duplicates("clean").head(40)]).sort_values("ts", ascending=False)
    tag = {}
    for ix, a in zip(men.index, men["actor"]):
        tag.setdefault(ix, set()).add(a)
    fc_tag = {}
    if len(fc):
        fm = mentions(fc)
        for ix, a in zip(fm.index, fm["actor"]):
            fc_tag.setdefault(ix, set()).add(a)
    feed = []
    for r in news.itertuples():
        is_fc = r.kind == "factcheck"
        feed.append({"d": r.ts.strftime("%Y-%m-%d %H:%M"), "o": (r.outlet or r.author) if is_fc else r.who, "t": headline(r.text), "u": r.url,
                     "k": "verificación" if is_fc else "noticia", "tono": getattr(r, "tono", "NEU") if not is_fc else "",
                     "a": sorted((fc_tag if is_fc else tag).get(r.Index, set()))[:4]})
    alerts = []
    last = df["day"].max()
    for s in spikes:
        if pd.Timestamp(s["day"]) > last - pd.Timedelta(days=7):
            alerts.append({"nivel": "media", "tipo": "Pico de conversación", "fecha": s["day"],
                           "titulo": f"{s['posts']} publicaciones en un día (z={s['z']})", "detalle": "Términos: " + ", ".join(s["terms"][:5])})
    for p in profiles:
        if not p.get("n"):
            continue
        if p.get("delta") is not None and p["last7"] >= 15 and p["delta"] >= 100:
            alerts.append({"nivel": "alta" if p["delta"] >= 200 else "media", "tipo": "Aceleración", "actor": p["name"], "fecha": last.strftime("%Y-%m-%d"),
                           "titulo": f"{p['name']}: menciones +{p['delta']}% en 7 días", "detalle": f"{p['prev7']} → {p['last7']} menciones"})
        if p["n_social"] >= 30 and p["tone"]["NEG"] >= 60:
            alerts.append({"nivel": "media", "tipo": "Tono negativo", "actor": p["name"], "fecha": last.strftime("%Y-%m-%d"),
                           "titulo": f"{p['name']}: {p['tone']['NEG']}% de comentarios negativos", "detalle": f"n={p['n_social']} comentarios"})
        if p["n_social"] >= 30 and p["agresivo"] >= 15:
            alerts.append({"nivel": "media", "tipo": "Agresividad", "actor": p["name"], "fecha": last.strftime("%Y-%m-%d"),
                           "titulo": f"{p['name']}: {p['agresivo']}% de comentarios agresivos", "detalle": "insultos o ataques personales"})
        for f in p.get("factchecks", [])[:2]:
            if pd.Timestamp(f["d"]) > last - pd.Timedelta(days=10):
                alerts.append({"nivel": "alta", "tipo": "Verificación", "actor": p["name"], "fecha": f["d"], "titulo": f"Chequeo publicado: {f['t'][:120]}",
                               "detalle": f["o"], "url": f["u"]})
    if coord.get("grupos"):
        alerts.append({"nivel": "alta", "tipo": "Coordinación", "fecha": last.strftime("%Y-%m-%d"),
                       "titulo": f"{coord['grupos']} grupos de cuentas con mensajes casi idénticos", "detalle": f"{coord['cuentas']} cuentas involucradas"})
    exp = HERE / "data" / "exposure.json"
    if exp.exists():
        for d in json.loads(exp.read_text(encoding="utf8")).get("domains", []):
            if d.get("semaforo") == "rojo":
                alerts.append({"nivel": "baja", "tipo": "Exposición digital", "fecha": "", "titulo": f"{d['name']}: semáforo rojo",
                               "detalle": f"DMARC {d['mail']['dmarc_policy']} · SPF {'sí' if d['mail']['spf'] else 'no'}"})
    rank = {"alta": 0, "media": 1, "baja": 2}
    alerts.sort(key=lambda x: x["fecha"], reverse=True)
    alerts.sort(key=lambda x: rank[x["nivel"]])
    return feed, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(HERE / ".store" / "posts.jsonl.gz"))
    ap.add_argument("--out", default=str(HERE / "data" / "radar.json"))
    ap.add_argument("--extra", nargs="*", default=[], help="Archivos o carpetas JSONL adicionales")
    ap.add_argument("--label", default="", help="Texto de aviso visible (por ejemplo, 'datos de prueba')")
    args = ap.parse_args()
    store = Path(args.store)
    status_path = store.parent / "collect_status.json"
    status = json.loads(status_path.read_text(encoding="utf8")) if status_path.exists() else None
    df = load(store, args.extra)
    # Chequeos de verificadores: se analizan aparte (no son conversación)
    fc = df[df["kind"] == "factcheck"].copy()
    df = df[df["kind"] != "factcheck"].copy()
    import nlp
    df = nlp.label(df, store.parent / "nlp.jsonl.gz")
    fc = fc.assign(lang2=[nlp.guess_lang(c, l) for c, l in zip(fc["clean"], fc["lang"])])
    figs = {}
    figs["r1"], spikes = volume(df)
    df, figs["r2"], figs["r3"], narr_table = narratives(df)
    figs["r4"], figs["r5"], figs["r6"], actor_summary = actors(df)
    figs["r7"], figs["r8"], coord_groups, coord = coordination(df)
    figs["r9"] = hashtags(df)
    figs["r10"] = domains(df)
    figs["r11"] = coverage(df, status)
    figs["r12"], conc = amplifiers(df)
    # Series de Google Trends: del recolector o importadas (agregados públicos, se versionan en radar/data/trends/)
    figs["r13"], figs["r15"] = trends(store.parent / "trends.json", list(args.extra) + [str(HERE / "data" / "trends")])
    figs["r14"] = editorial_lines(df)
    figs["r16"], figs["r17"], figs["r18"], figs["r19"] = tone_figures(df)
    men = mentions(df)
    figs["r20"], figs["r21"] = agenda_figures(men, df)
    days = pd.date_range(max(df["day"].min(), df["day"].max() - pd.Timedelta(days=CONFIG["window_days"] - 1)), df["day"].max(), freq="D")
    profiles = actor_profiles(df, men, fc, days)
    feed, alerts = live_layer(df, fc, men, profiles, spikes, coord)
    nlp_method = df["metodo_nlp"].value_counts().to_dict()
    out = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "summary": {
            "posts": int(len(df)), "social_posts": int((df["kind"] == "post").sum()), "news": int((df["kind"] == "news").sum()),
            "accounts": int(df.loc[df["kind"] == "post", "who"].nunique()), "outlets": int(df.loc[df["kind"] == "news", "who"].nunique()),
            "platforms": sorted(df["platform"].unique().tolist()),
            "from": df["ts"].min().strftime("%Y-%m-%d"), "to": df["ts"].max().strftime("%Y-%m-%d"),
            "narratives": narr_table, "spikes": spikes, "actors": actor_summary, "coordination": coord,
            "coordination_examples": coord_groups[:5], "concentration": conc,
            "tone": {k: pct(df["tono"].eq(k).mean()) for k in ["NEG", "NEU", "POS"]},
            "agresivo": pct(df.loc[df["kind"] == "post", "agresivo"].mean()) if (df["kind"] == "post").any() else None,
            "nlp_method": {k: int(v) for k, v in nlp_method.items()}, "factchecks": int(len(fc)),
        },
        "figures": {k: fig_json(v) for k, v in figs.items() if v is not None},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    # Perfiles por político (buscador del tablero) y capa "en vivo": titulares públicos, alertas y mapa por provincia
    Path(args.out).with_name("actors.json").write_text(json.dumps({
        "updated_at": out["updated_at"], "days": [d.strftime("%Y-%m-%d") for d in days], "actors": profiles},
        ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    Path(args.out).with_name("live.json").write_text(json.dumps({
        "updated_at": out["updated_at"], "feed": feed, "alerts": alerts, "provinces": province_map(df)},
        ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    print(f"{len(df)} publicaciones · {len(out['figures'])} figuras · {len(narr_table)} narrativas · "
          f"{len(spikes)} picos · coordinación: {coord}")


if __name__ == "__main__":
    main()
