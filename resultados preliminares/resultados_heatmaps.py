import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MATRICES = [
    "matriz_similitud_seguridad_ciudadana_y_lucha_contra_la_criminalidad.csv",
    "matriz_similitud_integridad_publica_y_lucha_contra_la_corrupcion.csv",
    "matriz_similitud_educacion_innovacion_y_tecnologia.csv",
    "matriz_similitud_empleo_desarrollo_y_emprendimiento.csv",
]

OUTPUT_DIR = Path("graficos")
OUTPUT_DIR.mkdir(exist_ok=True)

for matriz_file in MATRICES:
    path = Path(matriz_file)

    if not path.exists():
        print("No existe:", matriz_file)
        continue

    df_sim = pd.read_csv(path, index_col=0, encoding="utf-8-sig")

    plt.figure(figsize=(14, 12))
    plt.imshow(df_sim.values, aspect="auto")
    plt.colorbar(label="Similitud coseno")

    plt.xticks(
        range(len(df_sim.columns)),
        df_sim.columns,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(df_sim.index)),
        df_sim.index,
        fontsize=6
    )

    titulo = matriz_file.replace("matriz_similitud_", "").replace(".csv", "")
    plt.title(f"Matriz de similitud semántica - {titulo}")
    plt.tight_layout()

    output_img = OUTPUT_DIR / matriz_file.replace(".csv", ".png")
    plt.savefig(output_img, dpi=300)
    plt.close()

    print("Heatmap generado:", output_img)