import pandas as pd
import numpy as np
from pathlib import Path
import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt


CORPUS_FILE = "corpus_total_con_embeddings.csv"
EMBEDDINGS_FILE = "embeddings_sbert.npy"

OUTPUT_DIR = Path("resultados_clustering")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig")
embeddings = np.load(EMBEDDINGS_FILE)

resultados_metricas = []

for eje in df["eje_tematico"].unique():
    df_eje = df[df["eje_tematico"] == eje].copy()
    indices = df_eje["embedding_index"].tolist()
    emb_eje = embeddings[indices]

    print("\n==============================")
    print(eje)
    print("==============================")

    reducer = umap.UMAP(
        n_neighbors=8,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42
    )

    coords = reducer.fit_transform(emb_eje)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(coords)

    df_eje["umap_x"] = coords[:, 0]
    df_eje["umap_y"] = coords[:, 1]
    df_eje["cluster"] = labels

    # Métricas solo si hay al menos 2 clusters reales
    labels_validos = labels[labels != -1]
    coords_validos = coords[labels != -1]

    n_clusters = len(set(labels_validos))

    silhouette = None
    dbi = None

    if n_clusters >= 2:
        silhouette = silhouette_score(coords_validos, labels_validos)
        dbi = davies_bouldin_score(coords_validos, labels_validos)

    resultados_metricas.append({
        "eje_tematico": eje,
        "n_registros": len(df_eje),
        "n_clusters": n_clusters,
        "n_ruido": int((labels == -1).sum()),
        "silhouette": silhouette,
        "davies_bouldin": dbi
    })

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

    output_csv = OUTPUT_DIR / f"clustering_{nombre_eje}.csv"
    df_eje.to_csv(output_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df_eje["umap_x"],
        df_eje["umap_y"],
        c=df_eje["cluster"]
    )

    for _, row in df_eje.iterrows():
        plt.text(
            row["umap_x"],
            row["umap_y"],
            row["exponente"],
            fontsize=6
        )

    plt.title(f"UMAP + HDBSCAN - {eje}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    output_img = OUTPUT_DIR / f"umap_hdbscan_{nombre_eje}.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print("Clusters:")
    print(df_eje["cluster"].value_counts())
    print("Archivo:", output_csv)


df_metricas = pd.DataFrame(resultados_metricas)
df_metricas.to_csv(
    OUTPUT_DIR / "metricas_clustering.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nMétricas guardadas:")
print(df_metricas)