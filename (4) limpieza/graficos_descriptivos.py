import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("graficos")
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================================================
# CARGAR LOS 6 CORPUS SOLO EJES
# ======================================================

archivos = [
    "corpus_debate_1_solo_ejes.csv",
    "corpus_debate_2_solo_ejes.csv",
    "corpus_debate_3_solo_ejes.csv",
    "corpus_debate_4_solo_ejes.csv",
    "corpus_debate_5_solo_ejes.csv",
    "corpus_debate_6_solo_ejes.csv",
]

dfs = []

for archivo in archivos:
    df = pd.read_csv(archivo, encoding="utf-8-sig")
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

df_total["duracion_segmento"] = df_total["end"] - df_total["start"]
df_total["n_palabras"] = df_total["text"].astype(str).apply(lambda x: len(x.split()))


# ======================================================
# 1. TIEMPO TOTAL DE HABLA POR CANDIDATO
# ======================================================

tiempo_candidato = (
    df_total
    .groupby("exponente", as_index=False)
    .agg(
        duracion_total_min=("duracion_segmento", lambda x: x.sum() / 60),
        palabras_total=("n_palabras", "sum"),
        segmentos=("text", "count")
    )
    .sort_values("duracion_total_min", ascending=False)
)

tiempo_candidato.to_csv(
    OUTPUT_DIR / "tiempo_habla_por_candidato.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 8))
plt.barh(
    tiempo_candidato["exponente"],
    tiempo_candidato["duracion_total_min"]
)
plt.gca().invert_yaxis()
plt.xlabel("Duración total aproximada (minutos)")
plt.ylabel("Candidato")
plt.title("Tiempo total de intervención por candidato")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tiempo_total_por_candidato.png", dpi=300)
plt.close()


# ======================================================
# 2. PALABRAS TOTALES POR CANDIDATO
# ======================================================

palabras_candidato = tiempo_candidato.sort_values(
    "palabras_total",
    ascending=False
)

plt.figure(figsize=(12, 8))
plt.barh(
    palabras_candidato["exponente"],
    palabras_candidato["palabras_total"]
)
plt.gca().invert_yaxis()
plt.xlabel("Número aproximado de palabras")
plt.ylabel("Candidato")
plt.title("Cantidad de palabras por candidato")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "palabras_por_candidato.png", dpi=300)
plt.close()


# ======================================================
# 3. TIEMPO DE HABLA POR EJE TEMÁTICO
# ======================================================

tiempo_eje = (
    df_total
    .groupby("eje_tematico", as_index=False)
    .agg(
        duracion_total_min=("duracion_segmento", lambda x: x.sum() / 60),
        palabras_total=("n_palabras", "sum"),
        segmentos=("text", "count")
    )
    .sort_values("duracion_total_min", ascending=False)
)

tiempo_eje.to_csv(
    OUTPUT_DIR / "tiempo_habla_por_eje.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))
plt.barh(
    tiempo_eje["eje_tematico"],
    tiempo_eje["duracion_total_min"]
)
plt.gca().invert_yaxis()
plt.xlabel("Duración total aproximada (minutos)")
plt.ylabel("Eje temático")
plt.title("Tiempo total de intervención por eje temático")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tiempo_total_por_eje.png", dpi=300)
plt.close()


# ======================================================
# 4. TIEMPO POR CANDIDATO Y EJE
# ======================================================

tiempo_candidato_eje = (
    df_total
    .groupby(["exponente", "eje_tematico"], as_index=False)
    .agg(
        duracion_total_min=("duracion_segmento", lambda x: x.sum() / 60),
        palabras_total=("n_palabras", "sum"),
        segmentos=("text", "count")
    )
)

tiempo_candidato_eje.to_csv(
    OUTPUT_DIR / "tiempo_habla_por_candidato_y_eje.csv",
    index=False,
    encoding="utf-8-sig"
)


# ======================================================
# 5. GRÁFICO POR DEBATE: QUIÉN HABLA MÁS
# ======================================================

for debate_id in sorted(df_total["debate_id"].unique()):
    df_debate = df_total[df_total["debate_id"] == debate_id].copy()

    resumen = (
        df_debate
        .groupby("exponente", as_index=False)
        .agg(
            duracion_total_min=("duracion_segmento", lambda x: x.sum() / 60),
            palabras_total=("n_palabras", "sum"),
            segmentos=("text", "count")
        )
        .sort_values("duracion_total_min", ascending=False)
    )

    resumen.to_csv(
        OUTPUT_DIR / f"tiempo_habla_{debate_id}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 7))
    plt.barh(
        resumen["exponente"],
        resumen["duracion_total_min"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Duración aproximada (minutos)")
    plt.ylabel("Candidato")
    plt.title(f"Tiempo de intervención por candidato - {debate_id}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"tiempo_habla_{debate_id}.png", dpi=300)
    plt.close()


print("Gráficos generados en carpeta /graficos")
print("\nResumen tiempo por candidato:")
print(tiempo_candidato.to_string(index=False))