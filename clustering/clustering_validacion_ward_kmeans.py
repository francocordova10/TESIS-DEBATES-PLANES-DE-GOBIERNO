import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "corpus_total_con_embeddings.csv"
EMBEDDINGS_FILE = ROOT / "embeddings_sbert.npy"
OUTPUT_DIR = ROOT / "resultados_clustering"
OUTPUT_DIR.mkdir(exist_ok=True)


def limpiar_nombre_eje(eje: str) -> str:
    return (
        str(eje)
        .lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


def evaluar_kmeans(embeddings: np.ndarray, min_k: int = 2, max_k: int = 6):
    n_samples = len(embeddings)
    max_k = max(min_k, min(max_k, max(2, n_samples - 1)))

    mejores = []

    for k in range(min_k, max_k + 1):
        if k >= n_samples:
            break

        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(embeddings)

        if len(set(labels)) < 2:
            continue

        try:
            sil = silhouette_score(embeddings, labels)
        except ValueError:
            sil = np.nan

        try:
            dbi = davies_bouldin_score(embeddings, labels)
        except ValueError:
            dbi = np.nan

        mejores.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "labels": labels,
        })

    if not mejores:
        return None

    return max(
        mejores,
        key=lambda x: (x["silhouette"], -x["davies_bouldin"])
    )


def evaluar_ward(embeddings: np.ndarray, min_k: int = 2, max_k: int = 8):
    n_samples = len(embeddings)
    max_k = max(min_k, min(max_k, max(2, n_samples - 1)))

    mejores = []

    for k in range(min_k, max_k + 1):
        if k >= n_samples:
            break

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward"
        )
        labels = model.fit_predict(embeddings)

        if len(set(labels)) < 2:
            continue

        try:
            sil = silhouette_score(embeddings, labels)
        except ValueError:
            sil = np.nan

        mejores.append({
            "k": k,
            "silhouette": sil,
            "labels": labels,
        })

    if not mejores:
        return None

    return max(mejores, key=lambda x: x["silhouette"])


def generar_dendrograma(embeddings: np.ndarray, eje: str, output_dir: Path):
    if len(embeddings) < 2:
        return None

    try:
        Z = linkage(embeddings, method="ward")
    except Exception:
        return None

    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=20)
    plt.title(f"Dendrograma Ward - {eje}")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Distancia")
    plt.tight_layout()

    output_path = output_dir / f"dendrograma_ward_{limpiar_nombre_eje(eje)}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def generar_proyeccion_umap(coords: np.ndarray, labels: np.ndarray, eje: str, output_dir: Path, nombres=None):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    unique_labels = sorted(set(labels))
    legend_handles = []

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            color = "lightgray"
            handle = plt.Line2D([], [], marker="o", linestyle="", color="w", markerfacecolor=color, markeredgecolor="0.4", markersize=8, label="Ruido")
        else:
            color = plt.cm.tab10(label % 10)
            handle = plt.Line2D([], [], marker="o", linestyle="", color="w", markerfacecolor=color, markeredgecolor="0.4", markersize=8, label=f"Cluster {label}")

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            s=80,
            alpha=0.85,
            edgecolors="0.3",
            linewidths=0.5,
        )
        legend_handles.append(handle)

    if nombres is not None:
        for i, nombre in enumerate(nombres):
            if pd.isna(nombre) or str(nombre).strip() == "":
                continue
            ax.annotate(
                str(nombre),
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                color="black",
                xytext=(3, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.8),
            )

    ax.set_title(f"Proyección UMAP + HDBSCAN - {eje}", fontsize=13)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.legend(handles=legend_handles, frameon=True, fontsize=9)
    ax.margins(x=0.08, y=0.08)
    plt.tight_layout()

    output_path = output_dir / f"proyeccion_umap_hdbscan_{limpiar_nombre_eje(eje)}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def buscar_mejor_configuracion(embeddings: np.ndarray, eje: str, output_dir: Path):
    resultados_grid = []

    umap_configs = [
        {"n_neighbors": 3, "min_dist": 0.0},
        {"n_neighbors": 5, "min_dist": 0.0},
        {"n_neighbors": 8, "min_dist": 0.0},
        {"n_neighbors": 8, "min_dist": 0.1},
        {"n_neighbors": 12, "min_dist": 0.1},
        {"n_neighbors": 12, "min_dist": 0.25},
    ]

    hdbscan_configs = [
        {"min_cluster_size": 3, "min_samples": 2},
        {"min_cluster_size": 3, "min_samples": 3},
        {"min_cluster_size": 4, "min_samples": 2},
        {"min_cluster_size": 4, "min_samples": 3},
        {"min_cluster_size": 5, "min_samples": 2},
        {"min_cluster_size": 5, "min_samples": 3},
        {"min_cluster_size": 6, "min_samples": 3},
        {"min_cluster_size": 7, "min_samples": 3},
    ]

    for umap_cfg in umap_configs:
        for hdbscan_cfg in hdbscan_configs:
            reducer = umap.UMAP(
                n_neighbors=umap_cfg["n_neighbors"],
                min_dist=umap_cfg["min_dist"],
                n_components=2,
                metric="cosine",
                random_state=42
            )
            coords = reducer.fit_transform(embeddings)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_cfg["min_cluster_size"],
                min_samples=hdbscan_cfg["min_samples"],
                metric="euclidean"
            )
            labels = clusterer.fit_predict(coords)

            valid_mask = labels != -1
            valid_coords = coords[valid_mask]
            valid_labels = labels[valid_mask]

            n_clusters = len(set(valid_labels)) if len(valid_labels) > 0 else 0

            if n_clusters >= 2:
                try:
                    silhouette = silhouette_score(valid_coords, valid_labels)
                except ValueError:
                    silhouette = np.nan

                try:
                    dbi = davies_bouldin_score(valid_coords, valid_labels)
                except ValueError:
                    dbi = np.nan
            else:
                silhouette = np.nan
                dbi = np.nan

            kmeans_result = evaluar_kmeans(
                embeddings,
                min_k=2,
                max_k=min(6, len(embeddings) - 1)
            )

            resultados_grid.append({
                "eje_tematico": eje,
                "n_neighbors": umap_cfg["n_neighbors"],
                "min_dist": umap_cfg["min_dist"],
                "min_cluster_size": hdbscan_cfg["min_cluster_size"],
                "min_samples": hdbscan_cfg["min_samples"],
                "n_clusters": n_clusters,
                "n_ruido": int((labels == -1).sum()),
                "silhouette": silhouette,
                "davies_bouldin": dbi,
                "kmeans_k": kmeans_result["k"] if kmeans_result is not None else None,
                "kmeans_silhouette": kmeans_result["silhouette"] if kmeans_result is not None else np.nan,
                "kmeans_davies_bouldin": kmeans_result["davies_bouldin"] if kmeans_result is not None else np.nan,
            })

    if not resultados_grid:
        return None, None

    grid_df = pd.DataFrame(resultados_grid)
    grid_df.to_csv(
        output_dir / f"grid_busqueda_{limpiar_nombre_eje(eje)}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    valid_rows = grid_df.dropna(subset=["silhouette"]).copy()
    if valid_rows.empty:
        return None, grid_df

    valid_rows = valid_rows[valid_rows["n_clusters"] >= 2]
    if valid_rows.empty:
        return None, grid_df

    candidate_rows = valid_rows[(valid_rows["n_clusters"] <= 7) & (valid_rows["n_ruido"] <= 3)].copy()
    if candidate_rows.empty:
        candidate_rows = valid_rows[(valid_rows["n_clusters"] <= 7) & (valid_rows["n_ruido"] <= 5)].copy()

    if candidate_rows.empty:
        candidate_rows = valid_rows[valid_rows["n_clusters"] <= 7].copy()

    if candidate_rows.empty:
        candidate_rows = valid_rows

    best_row = candidate_rows.sort_values(
        by=["silhouette", "davies_bouldin", "n_ruido", "n_clusters"],
        ascending=[False, True, True, True]
    ).iloc[0]

    return best_row, grid_df


if __name__ == "__main__":
    df = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig")
    embeddings = np.load(EMBEDDINGS_FILE)

    if "embedding_index" not in df.columns:
        raise ValueError("El CSV no tiene la columna embedding_index. Debe generarse con embeddings.py")

    columnas_chunking = {"n_chunks", "tokens_modelo", "se_truncaria"}
    faltantes = columnas_chunking - set(df.columns)
    if faltantes:
        raise ValueError(
            "El archivo no corresponde al flujo con chunking. Falta(n): "
            + ", ".join(sorted(faltantes))
            + ". Reejecuta embeddings.py para regenerar los embeddings con chunking."
        )

    resultados = []

    for eje in sorted(df["eje_tematico"].dropna().unique()):
        df_eje = df[df["eje_tematico"] == eje].copy()
        indices = df_eje["embedding_index"].tolist()
        emb_eje = embeddings[indices]

        print(f"\n==============================")
        print(eje)
        print("==============================")

        # 1) Dendrograma Ward
        dendro_path = generar_dendrograma(emb_eje, eje, OUTPUT_DIR)
        if dendro_path is not None:
            print("Dendrograma Ward guardado:", dendro_path)

        # 2) Estimar k con Ward
        ward_result = evaluar_ward(emb_eje, min_k=2, max_k=min(8, len(emb_eje) - 1))
        k_estimada = ward_result["k"] if ward_result is not None else 2

        print("K estimada por Ward:", k_estimada)

        # 3) Buscar mejor configuración UMAP + HDBSCAN + K-Means
        best_row, _ = buscar_mejor_configuracion(emb_eje, eje, OUTPUT_DIR)

        if best_row is None:
            print("No se encontró una configuración válida para este eje.")
            continue

        print("Mejor configuración UMAP/HDBSCAN:")
        print("- n_neighbors:", int(best_row["n_neighbors"]))
        print("- min_dist:", float(best_row["min_dist"]))
        print("- min_cluster_size:", int(best_row["min_cluster_size"]))
        print("- min_samples:", int(best_row["min_samples"]))
        print("- clusters:", int(best_row["n_clusters"]))
        print("- ruido:", int(best_row["n_ruido"]))
        print("- silhouette:", float(best_row["silhouette"]))
        print("- davies_bouldin:", float(best_row["davies_bouldin"]))
        print("- mejor kmeans:", int(best_row["kmeans_k"]))
        print("- kmeans silhouette:", float(best_row["kmeans_silhouette"]))

        # 4) Repetir con la mejor configuración y guardar resultados
        reducer = umap.UMAP(
            n_neighbors=int(best_row["n_neighbors"]),
            min_dist=float(best_row["min_dist"]),
            n_components=2,
            metric="cosine",
            random_state=42
        )
        coords = reducer.fit_transform(emb_eje)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(best_row["min_cluster_size"]),
            min_samples=int(best_row["min_samples"]),
            metric="euclidean"
        )
        hdbscan_labels = clusterer.fit_predict(coords)

        kmeans_model = KMeans(
            n_clusters=int(best_row["kmeans_k"]),
            random_state=42,
            n_init=20
        )
        kmeans_labels = kmeans_model.fit_predict(emb_eje)

        df_eje["umap_x"] = coords[:, 0]
        df_eje["umap_y"] = coords[:, 1]
        df_eje["hdbscan_cluster"] = hdbscan_labels
        df_eje["kmeans_cluster"] = kmeans_labels
        df_eje["cluster"] = hdbscan_labels

        valid_mask = hdbscan_labels != -1
        if valid_mask.sum() >= 2 and len(set(kmeans_labels[valid_mask])) >= 2 and len(set(hdbscan_labels[valid_mask])) >= 2:
            ari = adjusted_rand_score(kmeans_labels[valid_mask], hdbscan_labels[valid_mask])
        else:
            ari = np.nan

        output_csv = OUTPUT_DIR / f"validacion_ward_kmeans_{limpiar_nombre_eje(eje)}.csv"
        df_eje.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print("Archivo guardado:", output_csv)

        nombres = df_eje["exponente"].fillna("").tolist() if "exponente" in df_eje.columns else None
        projection_path = generar_proyeccion_umap(coords, hdbscan_labels, eje, OUTPUT_DIR, nombres=nombres)
        print("Proyección UMAP guardada:", projection_path)

        resultados.append({
            "eje_tematico": eje,
            "n_registros": len(df_eje),
            "k_ward_estimada": k_estimada,
            "best_umap_neighbors": int(best_row["n_neighbors"]),
            "best_umap_min_dist": float(best_row["min_dist"]),
            "best_hdbscan_min_cluster_size": int(best_row["min_cluster_size"]),
            "best_hdbscan_min_samples": int(best_row["min_samples"]),
            "best_hdbscan_clusters": int(best_row["n_clusters"]),
            "best_hdbscan_ruido": int(best_row["n_ruido"]),
            "best_hdbscan_silhouette": float(best_row["silhouette"]),
            "best_hdbscan_davies_bouldin": float(best_row["davies_bouldin"]),
            "best_kmeans_k": int(best_row["kmeans_k"]),
            "best_kmeans_silhouette": float(best_row["kmeans_silhouette"]),
            "best_kmeans_davies_bouldin": float(best_row["kmeans_davies_bouldin"]),
            "ari_kmeans_hdbscan": ari,
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_DIR / "metricas_ward_kmeans.csv", index=False, encoding="utf-8-sig")

    print("\nResumen guardado en:", OUTPUT_DIR / "metricas_ward_kmeans.csv")
    print(df_resultados)
