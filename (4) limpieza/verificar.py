import pandas as pd
from pathlib import Path

archivos = [
    "corpus_debate_1_por_candidato_eje.csv",
    "corpus_debate_2_por_candidato_eje.csv",
    "corpus_debate_3_por_candidato_eje.csv",
    "corpus_debate_4_por_candidato_eje.csv",
    "corpus_debate_5_por_candidato_eje.csv",
    "corpus_debate_6_por_candidato_eje.csv",
]

for archivo in archivos:
    path = Path(archivo)

    if not path.exists():
        print("\nNO EXISTE:", archivo)
        continue

    df = pd.read_csv(path, encoding="utf-8-sig")

    print("\n==============================")
    print(archivo)
    print("==============================")
    print("Filas:", len(df))
    print("Candidatos únicos:", df["exponente_id"].nunique())

    print("\nEjes:")
    print(df["eje_tematico"].value_counts())

    print("\nTextos menores a 50 palabras:")
    cortos = df[df["n_palabras_aprox"] < 50]

    if cortos.empty:
        print("No hay textos muy cortos.")
    else:
        print(cortos[["exponente", "eje_tematico", "n_palabras_aprox"]])