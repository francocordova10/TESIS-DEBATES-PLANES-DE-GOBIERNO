import pandas as pd

# ======================================================
# CONFIGURA SOLO ESTO
# ======================================================

DEBATE_ID = "debate_1"

INPUT_FILE = f"corpus_{DEBATE_ID}_empastado.csv"
OUTPUT_FILE = f"corpus_{DEBATE_ID}_solo_ejes.csv"


# ======================================================
# CARGAR CORPUS EMPASTADO
# ======================================================

df = pd.read_csv(INPUT_FILE)

df["exponente"] = df["exponente"].astype(str).str.strip()
df["exponente_id"] = df["exponente_id"].astype(str).str.strip()
df["tipo_exponente"] = df["tipo_exponente"].astype(str).str.strip()
df["eje_tematico"] = df["eje_tematico"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()


# ======================================================
# NORMALIZAR EJES
# ======================================================

MAPEO_EJES = {
    "Seguridad Ciudadana y Lucha contra la Criminalidad": "Seguridad ciudadana y lucha contra la criminalidad",
    "Seguridad ciudadana y lucha contra la criminalidad": "Seguridad ciudadana y lucha contra la criminalidad",

    "Integridad Pública y Lucha contra la Corrupción": "Integridad pública y lucha contra la corrupción",
    "Integridad pública y lucha contra la corrupción": "Integridad pública y lucha contra la corrupción",

    "Educación, Innovación y Tecnología": "Educación, innovación y tecnología",
    "Educación, innovación y tecnología": "Educación, innovación y tecnología",

    "Empleo, Desarrollo y Emprendimiento": "Empleo, desarrollo y emprendimiento",
    "Empleo, desarrollo y emprendimiento": "Empleo, desarrollo y emprendimiento",

    "Participacion Ciudadana": "Participación ciudadana",
    "Participación Ciudadana": "Participación ciudadana",
    "Participación ciudadana": "Participación ciudadana",

    "Mensaje Final": "Mensaje final",
    "Mensaje final": "Mensaje final",
    "Minuto Final": "Mensaje final",

    "Ruido": "Ruido",
    "No identificado": "No identificado"
}

df["eje_tematico"] = df["eje_tematico"].replace(MAPEO_EJES)


# ======================================================
# DEFINIR EJES VÁLIDOS SEGÚN DEBATE
# ======================================================

if DEBATE_ID in ["debate_1", "debate_2", "debate_3"]:
    EJES_VALIDOS = [
        "Seguridad ciudadana y lucha contra la criminalidad",
        "Integridad pública y lucha contra la corrupción"
    ]
else:
    EJES_VALIDOS = [
        "Educación, innovación y tecnología",
        "Empleo, desarrollo y emprendimiento"
    ]


# ======================================================
# FILTRAR SOLO CANDIDATOS Y EJES TEMÁTICOS
# ======================================================

df_limpio = df[
    (df["tipo_exponente"] == "candidato") &
    (df["eje_tematico"].isin(EJES_VALIDOS))
].copy()

# Quitar textos demasiado cortos
df_limpio = df_limpio[df_limpio["text"].str.len() > 10].copy()

# Ordenar
df_limpio = df_limpio.sort_values(["debate_id", "start"]).reset_index(drop=True)

df_limpio.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ======================================================
# RESUMEN
# ======================================================

print("Archivo generado:", OUTPUT_FILE)
print("Filas originales:", len(df))
print("Filas limpias:", len(df_limpio))

print("\nEjes válidos usados:")
print(EJES_VALIDOS)

print("\nDistribución por eje:")
print(df_limpio["eje_tematico"].value_counts())

print("\nDistribución por candidato:")
print(df_limpio["exponente"].value_counts())

print("\nCandidatos únicos:", df_limpio["exponente_id"].nunique())