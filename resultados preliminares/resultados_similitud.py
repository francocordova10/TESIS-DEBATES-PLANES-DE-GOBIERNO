import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# ARCHIVOS
# ======================================================

CORPUS_FILE = Path("corpus_total_con_embeddings.csv")
EMBEDDINGS_FILE = Path("embeddings_sbert.npy")

OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================================================
# CARGAR DATOS
# ======================================================

df = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig")
embeddings = np.load(EMBEDDINGS_FILE)

print("Corpus:", df.shape)
print("Embeddings:", embeddings.shape)


# ======================================================
# FUNCIÓN PARA OBTENER PARES MÁS SIMILARES
# ======================================================

def ranking_similitud_por_eje(df, embeddings, eje, top_n=20):
    df_eje = df[df["eje_tematico"] == eje].copy()

    indices = df_eje["embedding_index"].tolist()
    emb_eje = embeddings[indices]

    sim = cosine_similarity(emb_eje)

    filas = []

    for i in range(len(df_eje)):
        for j in range(i + 1, len(df_eje)):
            filas.append({
                "eje_tematico": eje,
                "candidato_1": df_eje.iloc[i]["exponente"],
                "debate_1": df_eje.iloc[i]["debate_id"],
                "candidato_2": df_eje.iloc[j]["exponente"],
                "debate_2": df_eje.iloc[j]["debate_id"],
                "similitud_coseno": sim[i, j]
            })

    ranking = pd.DataFrame(filas)

    ranking = ranking.sort_values(
        "similitud_coseno",
        ascending=False
    ).reset_index(drop=True)

    return ranking.head(top_n), ranking


# ======================================================
# GENERAR RANKINGS POR EJE
# ======================================================

for eje in df["eje_tematico"].unique():
    nombre_eje = (
        eje.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )

    top, completo = ranking_similitud_por_eje(df, embeddings, eje, top_n=30)

    output_completo = OUTPUT_DIR / f"ranking_similitud_{nombre_eje}.csv"
    output_top = OUTPUT_DIR / f"top30_similitud_{nombre_eje}.csv"

    completo.to_csv(output_completo, index=False, encoding="utf-8-sig")
    top.to_csv(output_top, index=False, encoding="utf-8-sig")

    print("\n======================================")
    print(eje)
    print("======================================")
    print(top[[
        "candidato_1",
        "debate_1",
        "candidato_2",
        "debate_2",
        "similitud_coseno"
    ]].to_string(index=False))


print("\nRankings generados en carpeta /resultados")