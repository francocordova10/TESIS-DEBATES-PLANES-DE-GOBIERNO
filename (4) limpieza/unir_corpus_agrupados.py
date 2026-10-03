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

dfs = []

for archivo in archivos:
    path = Path(archivo)

    if not path.exists():
        print(f"No existe: {archivo}")
        continue

    df = pd.read_csv(path, encoding="utf-8-sig")
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

df_total.to_csv(
    "corpus_total_por_candidato_eje.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Archivo generado: corpus_total_por_candidato_eje.csv")
print("Filas totales:", len(df_total))
print("Candidatos únicos:", df_total["exponente_id"].nunique())

print("\nFilas por debate:")
print(df_total["debate_id"].value_counts())

print("\nFilas por eje:")
print(df_total["eje_tematico"].value_counts())

print("\nCandidatos con más filas:")
print(df_total["exponente"].value_counts())

print("\nTextos muy cortos:")
cortos = df_total[df_total["n_palabras_aprox"] < 50]

if cortos.empty:
    print("No hay textos muy cortos.")
else:
    print(cortos[["debate_id", "exponente", "eje_tematico", "n_palabras_aprox"]])