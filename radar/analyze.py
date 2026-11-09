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
PALETTE = ["#55d5ba", "#7aa7ff", "#f2b84b", "#ef6f6c", "#b48ef0", "#5fc3e4", "#9bd46a", "#f08bc0", "#c9a27e", "#8f9bb3"]
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
    k = k or int(np.clip(len(df) // 250, 4, 10))
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

def actors(df):
    pats = {a: re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(norm(x)) for x in alias) + r")(?![a-z])")
            for a, alias in CONFIG["actors"].items()}
    risk = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(norm(x)) for x in CONFIG["risk_terms"]) + r")(?![a-z])")
    df = df.assign(risk=df["clean"].str.contains(risk))
    rows = []
    for a, p in pats.items():
        m = df["clean"].str.contains(p)
        if m.sum():
            rows.append(df.loc[m].assign(actor=a))
    if not rows:
        return None, None, None, []
    men = pd.concat(rows)
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
    groups = [g for g in ["actores", "economia", "agenda"] if g in set(ts["group"])] or sorted(ts["group"].unique())
    from plotly.subplots import make_subplots
    names = {"actores": "Dirigentes", "economia": "Economía", "agenda": "Agenda"}
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
        },
        "figures": {k: fig_json(v) for k, v in figs.items() if v is not None},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    print(f"{len(df)} publicaciones · {len(out['figures'])} figuras · {len(narr_table)} narrativas · "
          f"{len(spikes)} picos · coordinación: {coord}")


if __name__ == "__main__":
    main()
