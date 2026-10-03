import json
import pandas as pd
import re
from pathlib import Path


# ======================================================
# CONFIGURA SOLO ESTO
# ======================================================

DEBATE_ID = "debate_6"

JSON_FILE = f"{DEBATE_ID}.json"
MAPEO_FILE = f"mapeo_{DEBATE_ID}.csv"
OUTPUT_FILE = f"corpus_{DEBATE_ID}_empastado.csv"


# ======================================================
# 1. Cargar JSON de Whisper
# ======================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df_json = pd.DataFrame([
    {
        "debate_id": DEBATE_ID,
        "segment_id": seg.get("id"),
        "start": float(seg["start"]),
        "end": float(seg["end"]),
        "text": str(seg.get("text", "")).strip()
    }
    for seg in data["segments"]
])

df_json = df_json[df_json["text"].str.len() > 0].copy()


# ======================================================
# 2. Cargar CSV mapeado
# ======================================================

def cargar_mapeo(path):
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")

        if len(df.columns) == 1:
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    df.columns = [c.strip() for c in df.columns]

    # Quitar columnas vacías tipo Unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    return df


df_map = cargar_mapeo(MAPEO_FILE)

print("Columnas detectadas:", df_map.columns.tolist())

columnas_necesarias = [
    "debate_id",
    "start_speaker",
    "end_speaker",
    "speaker_id",
    "exponente",
    "eje_tematico"
]

faltantes = [c for c in columnas_necesarias if c not in df_map.columns]

if faltantes:
    raise ValueError(
        f"Faltan columnas en {MAPEO_FILE}: {faltantes}. "
        f"Columnas detectadas: {df_map.columns.tolist()}"
    )

df_map["start_speaker"] = pd.to_numeric(df_map["start_speaker"], errors="coerce")
df_map["end_speaker"] = pd.to_numeric(df_map["end_speaker"], errors="coerce")

df_map = df_map.dropna(
    subset=["start_speaker", "end_speaker", "speaker_id", "exponente", "eje_tematico"]
)

df_map = df_map[df_map["end_speaker"] > df_map["start_speaker"]].copy()

df_map = df_map[df_map["debate_id"].astype(str).str.strip() == DEBATE_ID].copy()

if df_map.empty:
    raise ValueError(f"El mapeo quedó vacío para {DEBATE_ID}. Revisa la columna debate_id.")

df_map["exponente"] = df_map["exponente"].astype(str).str.strip()
df_map["eje_tematico"] = df_map["eje_tematico"].astype(str).str.strip()

print("Filas del JSON:", len(df_json))
print("Filas del mapeo:", len(df_map))


# ======================================================
# 3. Función de solapamiento
# ======================================================

def calcular_solapamiento(a_start, a_end, b_start, b_end):
    inicio = max(a_start, b_start)
    fin = min(a_end, b_end)
    return max(0, fin - inicio)


def asignar_datos_segmento(row):
    posibles = df_map[
        (df_map["start_speaker"] < row["end"]) &
        (df_map["end_speaker"] > row["start"])
    ].copy()

    if posibles.empty:
        return pd.Series({
            "speaker_id": "UNKNOWN",
            "exponente": "No identificado",
            "eje_tematico": "No identificado",
            "overlap_asignado": 0
        })

    posibles["overlap"] = posibles.apply(
        lambda m: calcular_solapamiento(
            row["start"],
            row["end"],
            m["start_speaker"],
            m["end_speaker"]
        ),
        axis=1
    )

    posibles = posibles[posibles["overlap"] > 0].copy()

    if posibles.empty:
        return pd.Series({
            "speaker_id": "UNKNOWN",
            "exponente": "No identificado",
            "eje_tematico": "No identificado",
            "overlap_asignado": 0
        })

    best = (
        posibles
        .groupby(["speaker_id", "exponente", "eje_tematico"], as_index=False)["overlap"]
        .sum()
        .sort_values("overlap", ascending=False)
        .iloc[0]
    )

    return pd.Series({
        "speaker_id": best["speaker_id"],
        "exponente": best["exponente"],
        "eje_tematico": best["eje_tematico"],
        "overlap_asignado": best["overlap"]
    })


# ======================================================
# 4. Empastar JSON + mapeo
# ======================================================

df_asignacion = df_json.apply(asignar_datos_segmento, axis=1)

df_final = pd.concat([df_json, df_asignacion], axis=1)


# ======================================================
# 5. Crear ID estable de exponente
# ======================================================

MODERADORES = [
    "fernando carvallo",
    "claudia chiroque",
    "angelica valdes",
    "angélica valdés",
    "pedro tenorio"
]


def limpiar_texto_id(texto):
    texto = str(texto).upper().strip()
    texto = texto.replace("Á", "A").replace("É", "E").replace("Í", "I")
    texto = texto.replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    texto = re.sub(r"[^A-Z0-9]+", "_", texto)
    texto = texto.strip("_")
    return texto


def crear_exponente_id(nombre):
    n = str(nombre).strip().lower()

    if n in ["no identificado", "unknown", ""]:
        return "NO_IDENTIFICADO"

    if n in [
        "ruido",
        "bot",
        "x",
        "locutor externo",
        "locutores externos",
        "locutor",
        "externo"
    ]:
        return "RUIDO"

    if n in [
        "participacion ciudadana",
        "participación ciudadana",
        "pregunta ciudadana",
        "preguntas ciudadanas"
    ]:
        return "PARTICIPACION_CIUDADANA"

    if "ciudadan" in n or "pregunta" in n:
        return "PARTICIPACION_CIUDADANA"

    if n in MODERADORES:
        return "MOD_" + limpiar_texto_id(nombre)

    return "CAND_" + limpiar_texto_id(nombre)


def clasificar_tipo_exponente(nombre):
    n = str(nombre).strip().lower()

    if n in ["no identificado", "unknown", ""]:
        return "no_identificado"

    if n in [
        "ruido",
        "bot",
        "x",
        "locutor externo",
        "locutores externos",
        "locutor",
        "externo"
    ]:
        return "ruido"

    if n in [
        "participacion ciudadana",
        "participación ciudadana",
        "pregunta ciudadana",
        "preguntas ciudadanas"
    ]:
        return "participacion_ciudadana"

    if "ciudadan" in n or "pregunta" in n:
        return "participacion_ciudadana"

    if n in MODERADORES:
        return "moderador"

    return "candidato"


# ======================================================
# 6. Limpieza de etiquetas antes de crear IDs
# ======================================================

df_final["exponente"] = df_final["exponente"].astype(str).str.strip()
df_final["eje_tematico"] = df_final["eje_tematico"].astype(str).str.strip()

# Si el eje es ruido, forzamos exponente como Ruido
df_final.loc[
    df_final["eje_tematico"].str.lower().eq("ruido"),
    "exponente"
] = "Ruido"

# Si el exponente es X, también será ruido
df_final.loc[
    df_final["exponente"].str.lower().eq("x"),
    "exponente"
] = "Ruido"

# Si el exponente es bot, también será ruido
df_final.loc[
    df_final["exponente"].str.lower().eq("bot"),
    "exponente"
] = "Ruido"


# AQUÍ ESTABA EL ERROR: FALTABA CREAR ESTA COLUMNA
df_final["exponente_id"] = df_final["exponente"].apply(crear_exponente_id)

df_final["tipo_exponente"] = df_final["exponente"].apply(clasificar_tipo_exponente)


# ======================================================
# 7. Orden final
# ======================================================

df_final = df_final[
    [
        "debate_id",
        "segment_id",
        "start",
        "end",
        "speaker_id",
        "exponente_id",
        "exponente",
        "tipo_exponente",
        "eje_tematico",
        "overlap_asignado",
        "text"
    ]
]


# ======================================================
# 8. Guardar corpus empastado del debate
# ======================================================

df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nArchivo generado:", OUTPUT_FILE)
print("Filas:", len(df_final))

print("\nDistribución por exponente:")
print(df_final["exponente"].value_counts())

print("\nDistribución por tipo_exponente:")
print(df_final["tipo_exponente"].value_counts())

print("\nDistribución por eje_tematico:")
print(df_final["eje_tematico"].value_counts())

print("\nSegmentos sin match:")
print((df_final["exponente"] == "No identificado").sum())

print("\nPrimeras filas:")
print(df_final.head(20))