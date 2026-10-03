# -*- coding: utf-8 -*-
"""
FASE 9 — Contraste intertextual debate–plan de gobierno
=======================================================

Compara el discurso de cada candidato en los debates del JNE (por eje temático)
con los 34 planes de gobierno inscritos, y estima si el discurso de cada
candidato se parece más a SU plan que a los planes de los demás.

Etapas (ver GUIA_FASE9.md):
  P2  Extracción y limpieza del texto de los PDF.
  P3  Segmentación en fragmentos de 120 tokens con 20 de solapamiento
      (misma función que la Fase 7) y vectorización con el mismo modelo SBERT.
  9.1 Similitud debate→plan por dos agregaciones (variable independiente):
        - "promedio":           coseno entre vectores promedio (mean pooling).
        - "mejor_coincidencia": para cada fragmento del debate, máxima similitud
                                con algún fragmento del plan; luego promedio.
  9.2 Control de extensión: estandarización z por plan (columna).
  9.3 Correspondencia programática: rango del plan propio, ICP y prueba de
      permutación.
  9.4 Relación con los agrupamientos de la Fase 8 (opcional, si existen).
  9.5 Muestra para validación cualitativa de los fragmentos recuperados.

Uso (desde la carpeta CODIGO, con el entorno venv_embeddings activado):
    python fase9_contraste_planes.py

Requisitos: sentence-transformers, pymupdf, pandas, numpy, scipy.
"""

import argparse
import gc
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIGURACIÓN (valores por defecto; todos se pueden cambiar por argumento)
# ------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TOKENS = 120          # igual que la Fase 7
OVERLAP_TOKENS = 20         # igual que la Fase 7
MIN_TOKENS_FRAGMENTO = 15   # se descartan fragmentos de plan casi vacíos
N_PERMUTACIONES = 10000
SEMILLA = 42
TOP_N_PERFIL_EJE = 10       # fragmentos de plan que definen el perfil del eje (9.4)
TOP_K_MUESTRA = 3           # fragmentos recuperados que se exportan (9.5)
N_MUESTRA = 5               # casos por eje para la revisión manual (9.5)


def parse_args():
    ap = argparse.ArgumentParser(description="Fase 9: contraste debate–plan")
    ap.add_argument("--corpus", default="corpus_total_con_embeddings.csv",
                    help="Corpus de debates por candidato y eje (salida de la Fase 7)")
    ap.add_argument("--planes", default=r"..\PLANES DE GOBIERNO",
                    help="Carpeta con los 34 PDF de planes de gobierno")
    ap.add_argument("--mapeo", default="mapeo_candidato_plan.csv",
                    help="Tabla exponente_id → archivo del plan")
    ap.add_argument("--clusters", default="resultados_clustering",
                    help="Carpeta con validacion_ward_kmeans_*.csv de la Fase 8 (opcional)")
    ap.add_argument("--salida", default="resultados_fase9")
    ap.add_argument("--modelo", default=MODEL_NAME)
    ap.add_argument("--permutaciones", type=int, default=N_PERMUTACIONES)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--batch", type=int, default=64)
    return ap.parse_args()


# ------------------------------------------------------------------
# P2. EXTRACCIÓN Y LIMPIEZA DE LOS PLANES
# ------------------------------------------------------------------
RE_INDICE = re.compile(r"\.{4,}\s*\d+\s*$")          # líneas de índice "Título ........ 12"
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")    # URLs y correos
RE_SOLO_NUM = re.compile(r"^\s*(p[aá]g(ina)?\.?\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.I)


def extraer_texto_pdf(ruta_pdf):
    """Devuelve (texto_limpio, n_paginas). Elimina índices, numeración y
    encabezados/pies repetidos (líneas que aparecen en >= 30% de las páginas)."""
    import pymupdf  # PyMuPDF
    doc = pymupdf.open(str(ruta_pdf))
    paginas = [p.get_text("text") for p in doc]
    n_pag = len(paginas)
    doc.close()

    lineas_por_pagina = [[l.strip() for l in pg.splitlines()] for pg in paginas]
    frecuencia = Counter()
    for lineas in lineas_por_pagina:
        frecuencia.update(set(l for l in lineas if l))
    umbral = max(3, int(0.30 * n_pag))
    repetidas = {l for l, f in frecuencia.items() if f >= umbral and len(l) < 120}

    conservadas = []
    for lineas in lineas_por_pagina:
        for l in lineas:
            if not l or l in repetidas or RE_INDICE.search(l) or RE_SOLO_NUM.match(l):
                continue
            conservadas.append(l)

    texto = "\n".join(conservadas)
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)   # une palabras cortadas con guion
    texto = RE_URL.sub(" ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto, n_pag


# ------------------------------------------------------------------
# P3. SEGMENTACIÓN (misma lógica que embeddings.py de la Fase 7)
# ------------------------------------------------------------------
def tokens_por_palabra(model, palabras):
    """Número de tokens de cada palabra, contado con el tokenizador del modelo.
    Equivale a contar_tokens(palabra) de la Fase 7, pero en lote (más rápido)."""
    if not palabras:
        return []
    enc = model.tokenizer(palabras, add_special_tokens=False, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def dividir_en_chunks_tokens(model, texto, max_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    palabras = re.findall(r"\S+", str(texto).strip())
    if not palabras:
        return [], []
    token_counts = tokens_por_palabra(model, palabras)
    chunks, n_tok = [], []
    start = 0
    while start < len(palabras):
        end, total = start, 0
        while end < len(palabras):
            sig = token_counts[end]
            if total + sig > max_tokens and total > 0:
                break
            total += sig
            end += 1
        if end == start:
            end, total = start + 1, token_counts[start]
        chunks.append(" ".join(palabras[start:end]))
        n_tok.append(total)
        if end >= len(palabras):
            break
        overlap_total, overlap_start = 0, end
        while overlap_start > start and overlap_total < overlap_tokens:
            overlap_start -= 1
            overlap_total += token_counts[overlap_start]
        # Corrección respecto de embeddings.py (Fase 7): si el fragmento tenía una sola
        # "palabra" de más de max_tokens (p. ej., una URL), el solapamiento devolvía el
        # inicio al mismo punto y el bucle no terminaba. Se fuerza el avance; en los
        # debates el resultado es idéntico porque ese caso no ocurre.
        start = max(overlap_start, start + 1)
    return chunks, n_tok


def codificar(model, textos, batch):
    import torch
    with torch.inference_mode():
        emb = model.encode(textos, batch_size=batch, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


def normalizar(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


# ------------------------------------------------------------------
# 9.2 – 9.3 MÉTRICAS DE CORRESPONDENCIA
# ------------------------------------------------------------------
def zscore_por_columna(S):
    mu = S.mean(axis=0, keepdims=True)
    sd = S.std(axis=0, ddof=0, keepdims=True)
    return (S - mu) / np.where(sd == 0, 1, sd)


def rango_e_icp(Z, propio):
    """Z: matriz candidatos×planes (estandarizada). propio[i] = columna del plan propio.
    Rango 1 = el plan propio es el más similar. ICP = z(propio) − media z(ajenos)."""
    n, K = Z.shape
    filas = np.arange(n)
    z_prop = Z[filas, propio]
    rango = 1 + (Z > z_prop[:, None]).sum(axis=1)
    mask = np.ones_like(Z, dtype=bool)
    mask[filas, propio] = False
    icp = z_prop - (Z * mask).sum(axis=1) / (K - 1)
    return rango, icp, z_prop


def prueba_permutacion(Z, propio, B, rng):
    """H0: no hay asociación candidato–plan. Se permuta la asignación de planes
    entre candidatos y se recalcula el rango medio. p = P(rango medio <= observado)."""
    rango_obs, icp_obs, _ = rango_e_icp(Z, propio)
    T_obs = rango_obs.mean()
    I_obs = icp_obs.mean()
    cont_r = cont_i = 0
    for _ in range(B):
        perm = rng.permutation(propio)
        r, i, _ = rango_e_icp(Z, perm)
        cont_r += r.mean() <= T_obs
        cont_i += i.mean() >= I_obs
    return (1 + cont_r) / (1 + B), (1 + cont_i) / (1 + B)


def correlacion_matrices(A, B_, n_perm, rng):
    """Correlación de Spearman entre los triángulos superiores de dos matrices de
    similitud, con significación por permutación conjunta de filas y columnas."""
    from scipy.stats import spearmanr
    iu = np.triu_indices(A.shape[0], 1)
    rho = spearmanr(A[iu], B_[iu]).correlation
    cont = 0
    for _ in range(n_perm):
        p = rng.permutation(A.shape[0])
        cont += spearmanr(A[iu], B_[np.ix_(p, p)][iu]).correlation >= rho
    return rho, (1 + cont) / (1 + n_perm)


def coherencia_clusters(Pm, etiquetas, n_perm, rng):
    """Diferencia entre la similitud media de planes dentro y entre clústeres de
    la Fase 8 (se excluye el ruido, etiqueta -1), con prueba de permutación."""
    et = np.asarray(etiquetas)
    ok = et != -1
    P2, e2 = Pm[np.ix_(ok, ok)], et[ok]
    iu = np.triu_indices(len(e2), 1)

    def diff(lbl):
        mismo = lbl[iu[0]] == lbl[iu[1]]
        if mismo.all() or (~mismo).all():
            return np.nan, np.nan, np.nan
        v = P2[iu]
        return v[mismo].mean() - v[~mismo].mean(), v[mismo].mean(), v[~mismo].mean()

    d_obs, intra, inter = diff(e2)
    cont = sum(diff(rng.permutation(e2))[0] >= d_obs for _ in range(n_perm))
    return d_obs, intra, inter, (1 + cont) / (1 + n_perm)


# ------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------------------------------------------
def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.semilla)
    out = Path(a.salida)
    out.mkdir(exist_ok=True)

    from sentence_transformers import SentenceTransformer
    import sentence_transformers, torch
    model = SentenceTransformer(a.modelo)
    print("Modelo:", a.modelo, "| max_seq_length:", model.max_seq_length,
          "| dispositivo:", model.device)

    # ---------------- Debates ----------------
    corpus = pd.read_csv(a.corpus, encoding="utf-8-sig")
    texto_col = "texto_limpio" if "texto_limpio" in corpus.columns else "texto_completo"
    mapeo = pd.read_csv(a.mapeo, encoding="utf-8-sig")
    planes_ids = mapeo["exponente_id"].tolist()          # orden fijo de columnas
    idx_plan = {e: i for i, e in enumerate(planes_ids)}
    faltan = set(corpus["exponente_id"]) - set(planes_ids)
    if faltan:
        raise ValueError(f"Candidatos sin plan en el mapeo: {faltan}")

    print("Segmentando y vectorizando debates...")
    deb_chunks = {}   # (exponente_id, eje) -> matriz de fragmentos
    for _, r in corpus.iterrows():
        ch, _ = dividir_en_chunks_tokens(model, r[texto_col])
        deb_chunks[(r["exponente_id"], r["eje_tematico"])] = codificar(model, ch, a.batch)

    # ---------------- Planes (P2 + P3) ----------------
    print("Extrayendo, segmentando y vectorizando planes...")
    plan_chunks, plan_textos, info = [], [], []
    for _, r in mapeo.iterrows():
        ruta = Path(a.planes) / r["archivo_plan"]
        texto, n_pag = extraer_texto_pdf(ruta)
        ch, ntok = dividir_en_chunks_tokens(model, texto)
        ch = [c for c, n in zip(ch, ntok) if n >= MIN_TOKENS_FRAGMENTO]
        E = codificar(model, ch, a.batch)
        plan_chunks.append(E)
        plan_textos.append(ch)
        info.append({"exponente_id": r["exponente_id"],
                     "organizacion_politica": r["organizacion_politica"],
                     "archivo_plan": r["archivo_plan"], "paginas": n_pag,
                     "palabras": len(texto.split()), "fragmentos": len(ch)})
        print(f"  {r['organizacion_politica'][:40]:40s} pág={n_pag:4d} fragmentos={len(ch)}")
        del texto
        gc.collect()
    pd.DataFrame(info).to_csv(out / "p2_planes_descriptivo.csv", index=False, encoding="utf-8-sig")
    plan_prom = normalizar(np.stack([normalizar(E.mean(axis=0)) for E in plan_chunks]))

    # ---------------- 9.1 – 9.3 por eje ----------------
    filas_cand, filas_res, muestra, perfiles = [], [], [], {}
    for eje in sorted(corpus["eje_tematico"].unique()):
        cands = corpus.loc[corpus["eje_tematico"] == eje, "exponente_id"].tolist()
        propio = np.array([idx_plan[c] for c in cands])
        K = len(planes_ids)
        S = {"promedio": np.zeros((len(cands), K)),
             "mejor_coincidencia": np.zeros((len(cands), K))}
        mejores = {}  # para la muestra cualitativa (plan propio)
        for i, c in enumerate(cands):
            D = deb_chunks[(c, eje)]
            d_prom = normalizar(D.mean(axis=0))
            S["promedio"][i] = plan_prom @ d_prom
            for k, P in enumerate(plan_chunks):
                M = D @ P.T                         # fragmentos debate × fragmentos plan
                S["mejor_coincidencia"][i, k] = M.max(axis=1).mean()
                if k == propio[i]:
                    mejores[c] = M.max(axis=0)       # afinidad de cada fragmento del plan
        for agg, Sm in S.items():
            Z = zscore_por_columna(Sm)
            rango, icp, zprop = rango_e_icp(Z, propio)
            p_r, p_i = prueba_permutacion(Z, propio, a.permutaciones, rng)
            pd.DataFrame(Sm, index=cands, columns=planes_ids).to_csv(
                out / f"9_1_similitud_{agg}_{slug(eje)}.csv", encoding="utf-8-sig")
            for c, r_, ic, s_, z_ in zip(cands, rango, icp, Sm[np.arange(len(cands)), propio], zprop):
                filas_cand.append({"eje_tematico": eje, "agregacion": agg, "exponente_id": c,
                                   "similitud_plan_propio": s_, "z_plan_propio": z_,
                                   "rango_plan_propio": int(r_), "ICP": ic})
            filas_res.append({"eje_tematico": eje, "agregacion": agg, "n_candidatos": len(cands),
                              "n_planes": K, "rango_medio": rango.mean(),
                              "rango_esperado_azar": (K + 1) / 2,
                              "rango_mediana": float(np.median(rango)),
                              "top1": (rango == 1).mean(), "top5": (rango <= 5).mean(),
                              "ICP_medio": icp.mean(), "p_perm_rango": p_r, "p_perm_ICP": p_i})
        # 9.5 muestra: fragmentos del plan propio más afines al debate (mejor coincidencia)
        for c in cands:
            k = idx_plan[c]
            orden = np.argsort(-mejores[c])[:TOP_K_MUESTRA]
            for j, fi in enumerate(orden, 1):
                muestra.append({"eje_tematico": eje, "exponente_id": c, "orden": j,
                                "similitud": float(mejores[c][fi]),
                                "fragmento_plan": plan_textos[k][fi],
                                "pertinente_al_eje (1/0)": ""})
        # perfil del eje en cada plan (9.4): fragmentos más cercanos al centroide del eje
        centro = normalizar(np.concatenate([deb_chunks[(c, eje)] for c in cands]).mean(axis=0))
        perf = []
        for P in plan_chunks:
            top = np.argsort(-(P @ centro))[:TOP_N_PERFIL_EJE]
            perf.append(normalizar(P[top].mean(axis=0)))
        perfiles[eje] = (cands, np.stack(perf))

    res = pd.DataFrame(filas_res)
    pd.DataFrame(filas_cand).to_csv(out / "9_3_correspondencia_por_candidato.csv", index=False, encoding="utf-8-sig")
    res.to_csv(out / "9_3_resumen_correspondencia.csv", index=False, encoding="utf-8-sig")
    # Muestra reproducible para revisión manual: N_MUESTRA candidatos por eje (semilla fija);
    # se revisa el fragmento más afín (orden 1) de cada caso seleccionado.
    dm = pd.DataFrame(muestra)
    rng_m = np.random.default_rng(a.semilla)
    dm["en_muestra"] = 0
    for eje, g in dm.groupby("eje_tematico"):
        elegidos = rng_m.choice(sorted(g["exponente_id"].unique()), size=N_MUESTRA, replace=False)
        dm.loc[(dm["eje_tematico"] == eje) & dm["exponente_id"].isin(elegidos) & (dm["orden"] == 1), "en_muestra"] = 1
    dm.to_csv(out / "9_5_muestra_validacion_cualitativa.csv", index=False, encoding="utf-8-sig")

    # Prueba global (los 4 ejes juntos, misma permutación candidato–plan)
    glob = []
    for agg in ["promedio", "mejor_coincidencia"]:
        sub = pd.DataFrame(filas_cand).query("agregacion == @agg")
        glob.append({"agregacion": agg, "rango_medio_global": sub["rango_plan_propio"].mean(),
                     "top1_global": (sub["rango_plan_propio"] == 1).mean(),
                     "ICP_medio_global": sub["ICP"].mean()})
    pd.DataFrame(glob).to_csv(out / "9_3_resumen_global.csv", index=False, encoding="utf-8-sig")

    # ---------------- 9.4 Relación con la Fase 8 ----------------
    filas_rel = []
    for eje, (cands, perf) in perfiles.items():
        idx = [idx_plan[c] for c in cands]
        Pm = perf[idx] @ perf[idx].T                                   # similitud entre planes (perfil del eje)
        Dv = np.stack([normalizar(deb_chunks[(c, eje)].mean(axis=0)) for c in cands])
        Dm = Dv @ Dv.T                                                 # similitud entre debates (Fase 7)
        rho, p_rho = correlacion_matrices(Dm, Pm, 999, rng)
        fila = {"eje_tematico": eje, "spearman_debate_vs_plan": rho, "p_perm": p_rho}
        archivo = Path(a.clusters) / f"validacion_ward_kmeans_{slug(eje)}.csv"
        if archivo.exists():
            cl = pd.read_csv(archivo, encoding="utf-8-sig").set_index("exponente_id")["hdbscan_cluster"]
            et = [cl.get(c, -1) for c in cands]
            d, intra, inter, p_d = coherencia_clusters(Pm, et, 999, rng)
            fila.update({"sim_planes_intra_cluster": intra, "sim_planes_inter_cluster": inter,
                         "diferencia": d, "p_perm_clusters": p_d})
        filas_rel.append(fila)
    pd.DataFrame(filas_rel).to_csv(out / "9_4_relacion_debate_plan_clusters.csv", index=False, encoding="utf-8-sig")

    # ---------------- Registro de reproducibilidad ----------------
    meta = {"modelo": a.modelo, "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS,
            "min_tokens_fragmento": MIN_TOKENS_FRAGMENTO, "permutaciones": a.permutaciones,
            "semilla": a.semilla, "top_n_perfil_eje": TOP_N_PERFIL_EJE, "n_muestra_por_eje": N_MUESTRA,
            "python": platform.python_version(), "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__, "dispositivo": str(model.device),
            "segundos": round(time.time() - t0, 1)}
    (out / "registro_ejecucion.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nResumen por eje:\n", res.round(3).to_string(index=False))
    print("\nListo. Resultados en:", out.resolve())


def slug(eje):
    s = eje.lower().replace(" ", "_").replace(",", "")
    for a, b in zip("áéíóúñ", "aeioun"):
        s = s.replace(a, b)
    return s


if __name__ == "__main__":
    main()
