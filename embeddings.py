import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# RUTAS
# ======================================================

INPUT_FILE = Path("corpus_total_por_candidato_eje.csv")

OUTPUT_CORPUS_EMBEDDINGS = Path("corpus_total_con_embeddings.csv")
OUTPUT_EMBEDDINGS_NPY = Path("embeddings_sbert.npy")
OUTPUT_SIMILITUD = Path("matriz_similitud_total.csv")


# ======================================================
# CARGAR CORPUS
# ======================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("Corpus cargado")
print("Filas:", len(df))
print("Columnas:", df.columns.tolist())

# Validación mínima
columnas_necesarias = [
    "debate_id",
    "exponente_id",
    "exponente",
    "eje_tematico",
    "texto_completo"
]

faltantes = [c for c in columnas_necesarias if c not in df.columns]

if faltantes:
    raise ValueError(f"Faltan columnas en el corpus: {faltantes}")


# ======================================================
# LIMPIEZA TEXTUAL BÁSICA
# ======================================================

def limpiar_texto(texto):
    texto = str(texto)

    # Quitar espacios repetidos
    texto = re.sub(r"\s+", " ", texto)

    # Quitar espacios al inicio/final
    texto = texto.strip()

    return texto


df["texto_limpio"] = df["texto_completo"].apply(limpiar_texto)

# Quitar textos vacíos o demasiado cortos
df = df[df["texto_limpio"].str.len() > 30].copy()

df = df.reset_index(drop=True)

print("\nDespués de limpieza:")
print("Filas:", len(df))
print("Candidatos únicos:", df["exponente_id"].nunique())
print("\nEjes:")
print(df["eje_tematico"].value_counts())


# ======================================================
# ESTADÍSTICAS DEL CORPUS
# ======================================================

df["n_palabras"] = df["texto_limpio"].apply(lambda x: len(str(x).split()))
df["tokens_aprox"] = (df["n_palabras"] * 1.3).astype(int)

print("\n===== Estadísticas del corpus =====")
print(df["tokens_aprox"].describe())
print("\nTextos con más de 128 tokens (aprox):")
print((df["tokens_aprox"] > 128).sum())
print("Total documentos:", len(df))


# ======================================================
# CARGAR MODELO SBERT
# ======================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("\nCargando modelo:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)
MAX_TOKENS = getattr(model, "max_seq_length", 128)


# ======================================================
# CONFIGURACIÓN DEL CHUNKING
# ======================================================

CHUNK_TOKENS = 120
OVERLAP_TOKENS = 20

print(f"\nChunking por tokens reales: {CHUNK_TOKENS} tokens | Overlap: {OVERLAP_TOKENS}")
print("Límite del modelo:", MAX_TOKENS, "tokens")


# ======================================================
# FUNCIONES DE CHUNKING Y EMBEDDING
# ======================================================

def contar_tokens(texto):
    if not str(texto).strip():
        return 0

    tokens = model.tokenizer(
        str(texto),
        add_special_tokens=False,
        truncation=False,
        return_tensors=None
    )

    return len(tokens["input_ids"])


def dividir_en_chunks_tokens(texto, max_tokens=128, overlap_tokens=20):
    texto = str(texto).strip()

    if not texto:
        return []

    palabras = re.findall(r"\S+", texto)

    if not palabras:
        return []

    token_counts = [contar_tokens(palabra) for palabra in palabras]

    chunks = []
    start = 0

    while start < len(palabras):
        end = start
        token_total = 0

        while end < len(palabras):
            siguiente = token_counts[end]

            if token_total + siguiente > max_tokens and token_total > 0:
                break

            token_total += siguiente
            end += 1

        if end == start:
            end = start + 1
            token_total = token_counts[start]

        chunk_texto = " ".join(palabras[start:end])
        chunks.append(chunk_texto)

        if end >= len(palabras):
            break

        overlap_total = 0
        overlap_start = end

        while overlap_start > start and overlap_total < overlap_tokens:
            overlap_start -= 1
            overlap_total += token_counts[overlap_start]

        start = overlap_start

    return chunks


def embedding_con_chunking(texto):
    chunks = dividir_en_chunks_tokens(
        texto,
        max_tokens=CHUNK_TOKENS,
        overlap_tokens=OVERLAP_TOKENS
    )

    if not chunks:
        return np.zeros(384, dtype=np.float32), 0, 0

    emb_chunks = model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    emb_chunks = np.array(emb_chunks, dtype=np.float32)
    embedding_final = emb_chunks.mean(axis=0)
    embedding_final = embedding_final / np.linalg.norm(embedding_final)

    return embedding_final, len(chunks), contar_tokens(texto)


# ======================================================
# GENERAR EMBEDDINGS CON CHUNKING
# ======================================================

print("\nGenerando embeddings con chunking...")

embeddings = []
n_chunks = []
tokens_modelo = []
se_truncaria = []

for texto in df["texto_limpio"]:
    emb, cantidad_chunks, tokens_total = embedding_con_chunking(texto)

    embeddings.append(emb)
    n_chunks.append(cantidad_chunks)
    tokens_modelo.append(tokens_total)
    se_truncaria.append(int(tokens_total > MAX_TOKENS))

embeddings = np.array(embeddings, dtype=np.float32)

df["n_chunks"] = n_chunks
df["tokens_modelo"] = tokens_modelo
df["se_truncaria"] = se_truncaria

print("Shape embeddings:", embeddings.shape)
print("Promedio chunks:", np.mean(n_chunks))
print("Máximo chunks:", np.max(n_chunks))
print("Textos que exceden el límite del modelo:", df["se_truncaria"].sum())


# ======================================================
# GUARDAR EMBEDDINGS
# ======================================================

np.save(OUTPUT_EMBEDDINGS_NPY, embeddings)

# Guardamos el corpus sin meter todo el vector en Excel/CSV
df["embedding_index"] = range(len(df))

df.to_csv(OUTPUT_CORPUS_EMBEDDINGS, index=False, encoding="utf-8-sig")

print("\nArchivos guardados:")
print("-", OUTPUT_CORPUS_EMBEDDINGS)
print("-", OUTPUT_EMBEDDINGS_NPY)


# ======================================================
# SIMILITUD COSENO TOTAL
# ======================================================

print("\nCalculando matriz de similitud coseno...")

sim_matrix = cosine_similarity(embeddings)

labels = (
    df["debate_id"].astype(str)
    + " | "
    + df["exponente"].astype(str)
    + " | "
    + df["eje_tematico"].astype(str)
)

df_sim = pd.DataFrame(
    sim_matrix,
    index=labels,
    columns=labels
)

df_sim.to_csv(OUTPUT_SIMILITUD, encoding="utf-8-sig")

print("Matriz total guardada:", OUTPUT_SIMILITUD)


# ======================================================
# MATRICES POR EJE
# ======================================================

for eje in df["eje_tematico"].unique():
    df_eje = df[df["eje_tematico"] == eje].copy()

    indices = df_eje["embedding_index"].tolist()
    emb_eje = embeddings[indices]

    sim_eje = cosine_similarity(emb_eje)

    labels_eje = (
        df_eje["debate_id"].astype(str)
        + " | "
        + df_eje["exponente"].astype(str)
    )

    df_sim_eje = pd.DataFrame(
        sim_eje,
        index=labels_eje,
        columns=labels_eje
    )

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

    output_eje = Path(f"matriz_similitud_{nombre_eje}.csv")

    df_sim_eje.to_csv(output_eje, encoding="utf-8-sig")

    print("Matriz por eje guardada:", output_eje)


print("\nProceso de embeddings terminado.")