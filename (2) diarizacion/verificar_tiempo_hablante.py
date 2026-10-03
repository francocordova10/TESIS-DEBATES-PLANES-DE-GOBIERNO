import pandas as pd

for debate_id in ["debate_1", "debate_2", "debate_3", "debate_4", "debate_5", "debate_6"]:
    archivo = f"corpus_{debate_id}_solo_ejes.csv"
    df = pd.read_csv(archivo)

    df["duracion_segmento"] = df["end"] - df["start"]
    df["n_palabras"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    resumen = (
        df.groupby("eje_tematico")
        .agg(
            segmentos=("text", "count"),
            duracion_total_min=("duracion_segmento", lambda x: round(x.sum() / 60, 2)),
            duracion_promedio_seg=("duracion_segmento", lambda x: round(x.mean(), 2)),
            palabras_total=("n_palabras", "sum")
        )
        .reset_index()
    )

    print("\n==============================")
    print(debate_id)
    print("==============================")
    print(resumen)