from pathlib import Path
import pandas as pd
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline

# ==========================
# CONFIGURACIÓN
# ==========================

HF_TOKEN = "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"

AUDIO_PATTERN = "debate_*_recortado.wav"
OUTPUT_PREFIX = "diarizacion"

# ==========================
# CARGAR PIPELINE
# ==========================

print("Cargando modelo de diarización...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print("Dispositivo usado:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================
# FUNCIÓN PARA CARGAR AUDIO SIN TORCHAUDIO
# ==========================

def cargar_audio_para_pyannote(audio_path):
    audio, sample_rate = sf.read(str(audio_path))

    # Si viene estéreo: convertir a mono
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    # Convertir a float32
    audio = audio.astype(np.float32)

    # Pyannote espera forma: (canales, tiempo)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

# ==========================
# DIARIZAR AUDIOS
# ==========================

audio_files = [Path("debate_6_recortado.wav")]

print(f"Audios encontrados: {len(audio_files)}")
for audio in audio_files:
    print("-", audio.name)

# Para probar primero solo debate 1, descomenta esta línea:
# audio_files = [Path("debate_1_recortado.wav")]

for audio_file in audio_files:
    debate_id = audio_file.stem.replace("_recortado", "")
    output_csv = f"{OUTPUT_PREFIX}_{debate_id}.csv"

    if Path(output_csv).exists():
        print(f"\n{output_csv} ya existe. Saltando.")
        continue

    print(f"\nCargando audio {audio_file.name}...")
    audio_input = cargar_audio_para_pyannote(audio_file)

    print(f"Diarizando {audio_file.name}...")

    diarization = pipeline(audio_input)

    annotation = diarization.speaker_diarization

    rows = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({
            "debate_id": debate_id,
            "start_speaker": round(turn.start, 3),
            "end_speaker": round(turn.end, 3),
            "speaker_id": speaker
        })
    df_diar = pd.DataFrame(rows)
    df_diar.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Guardado: {output_csv}")
    print(df_diar.head())

print("\nDiarización terminada.")