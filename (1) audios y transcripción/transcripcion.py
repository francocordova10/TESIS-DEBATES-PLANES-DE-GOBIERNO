import os

debates = {
    "debate_1": "https://www.youtube.com/watch?v=OikfsWFXkJA",
    "debate_2": "https://www.youtube.com/watch?v=QKW-18R6erI",
    "debate_3": "https://www.youtube.com/watch?v=-mh3zAMpNek",
    "debate_4": "https://www.youtube.com/watch?v=XWjOq-O3Zr0",
    "debate_5": "https://www.youtube.com/watch?v=yGEF69Luvmo",
    "debate_6": "https://www.youtube.com/watch?v=f1YsyEqGE40"
}

for debate_id, url in debates.items():
    output_file = f"{debate_id}.wav"

    print(f"Iniciando descarga de {debate_id}...")
    os.system(f'yt-dlp -x --audio-format wav -o "{output_file}" "{url}"')
    print(f"Descarga completada: {output_file}")

inicios = {
    "debate_1": "00:00:55",
    "debate_2": "00:08:09",
    "debate_3": "01:17:17",
    "debate_4": "01:10:22",
    "debate_5": "01:18:56",
    "debate_6": "01:01:43"
}

for debate_id, inicio in inicios.items():
    input_file = f"{debate_id}.wav"
    output_file = f"{debate_id}_recortado.wav"

    if not os.path.exists(input_file):
        print(f"No existe {input_file}, saltando.")
        continue

    print(f"Recortando {debate_id} desde {inicio}...")

    command = [
        "ffmpeg",
        "-y",
        "-ss", inicio,
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        output_file
    ]

    subprocess.run(command)

print("Recortes terminados.")

from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

from pathlib import Path
import json

audio_files = sorted(Path(".").glob("debate_*_recortado.wav"))

print("Audios encontrados:")
for audio in audio_files:
    print(audio.name)

for audio_path in audio_files:
    debate_id = audio_path.stem.replace("_recortado", "")
    output_json = f"{debate_id}.json"

    if Path(output_json).exists():
        print(f"{output_json} ya existe. Saltando.")
        continue

    print(f"\nTranscribiendo {debate_id}...")

    segments, info = model.transcribe(
        str(audio_path),
        language="es",
        vad_filter=True,
        beam_size=5
    )

    segments_list = []

    for i, segment in enumerate(segments):
        segments_list.append({
            "id": i,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    result = {
        "debate_id": debate_id,
        "language": info.language,
        "duration": info.duration,
        "segments": segments_list
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Guardado: {output_json}")


import pandas as pd
import json
from pathlib import Path

json_files = sorted(Path(".").glob("debate_*.json"))

print("JSON encontrados:")
for f in json_files:
    print(f.name)

rows = []

for json_file in json_files:
    debate_id = json_file.stem

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for seg in data["segments"]:
        rows.append({
            "debate_id": debate_id,
            "segment_id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

df = pd.DataFrame(rows)

print("Filas y columnas:", df.shape)
df.head()