import pandas as pd

# ======================================================
# CONFIGURA SOLO ESTO
# ======================================================

DEBATE_ID = "debate_6"

INPUT_FILE = f"corpus_{DEBATE_ID}_solo_ejes.csv"
OUTPUT_FILE = f"corpus_{DEBATE_ID}_por_candidato_eje.csv"


# ======================================================
# CARGAR
# ======================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

df["text"] = df["text"].astype(str).str.strip()
df["exponente"] = df["exponente"].astype(str).str.strip()
df["exponente_id"] = df["exponente_id"].astype(str).str.strip()
df["eje_tematico"] = df["eje_tematico"].astype(str).str.strip()

df = df[df["text"].str.len() > 0].copy()


# ======================================================
# AGRUPAR
# 1 fila = 1 candidato + 1 eje
# ======================================================

df_agrupado = (
    df
    .groupby(
        ["debate_id", "exponente_id", "exponente", "eje_tematico"],
        as_index=False
    )
    .agg(
        start=("start", "min"),
        end=("end", "max"),
        n_segmentos=("text", "count"),
        texto_completo=("text", " ".join)
    )
)

df_agrupado["duracion_aprox_seg"] = df_agrupado["end"] - df_agrupado["start"]
df_agrupado["n_caracteres"] = df_agrupado["texto_completo"].str.len()
df_agrupado["n_palabras_aprox"] = df_agrupado["texto_completo"].str.split().str.len()

df_agrupado = df_agrupado.sort_values(
    ["eje_tematico", "exponente"]
).reset_index(drop=True)

df_agrupado.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ======================================================
# RESUMEN
# ======================================================

print("Archivo generado:", OUTPUT_FILE)
print("Filas:", len(df_agrupado))
print("Candidatos únicos:", df_agrupado["exponente_id"].nunique())

print("\nDistribución por eje:")
print(df_agrupado["eje_tematico"].value_counts())

print("\nResumen por candidato y eje:")
print(
    df_agrupado[
        ["exponente", "eje_tematico", "n_segmentos", "n_palabras_aprox"]
    ].to_string()
)

print("\nTextos muy cortos para revisar:")
cortos = df_agrupado[df_agrupado["n_palabras_aprox"] < 50]

if cortos.empty:
    print("No hay textos sospechosamente cortos.")
else:
    print(cortos[["exponente", "eje_tematico", "n_palabras_aprox"]])