# -*- coding: utf-8 -*-
"""
Análisis complementarios de la Fase 9 (Sección 6.2–6.3 del artículo).
Se ejecuta DESPUÉS de fase9_contraste_planes.py, sobre la carpeta resultados_fase9:
  1. Prueba de permutación global (cuatro ejes, permutación conjunta candidato–plan).
  2. Comparación pareada de rangos: mejor coincidencia vs. promedio (Wilcoxon).
  3. Relación entre extensión del plan y rango del plan propio (Spearman).
Uso:  .\\venv_embeddings\\Scripts\\python.exe fase9_analisis_complementario.py
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = Path("resultados_fase9")
rng = np.random.default_rng(42)
B = 10000
salida = []

for agg in ["promedio", "mejor_coincidencia"]:
    mats = []
    for f in sorted(glob.glob(str(R / f"9_1_similitud_{agg}_*.csv"))):
        S = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
        mats.append((S - S.mean(axis=0)) / S.std(axis=0, ddof=0))   # z por plan (Ec. 3)
    planes = list(mats[0].columns)
    pos = {p: i for i, p in enumerate(planes)}

    def rango_medio(asig):
        r = []
        for Z in mats:
            for c in Z.index:
                z = Z.loc[c].values
                r.append(1 + (z > z[pos[asig[c]]]).sum())
        return np.mean(r)

    obs = rango_medio({c: c for c in planes})
    cont = sum(rango_medio(dict(zip(planes, rng.permutation(planes)))) <= obs for _ in range(B))
    salida.append({"analisis": "permutacion_global", "agregacion": agg,
                   "valor": obs, "p": (1 + cont) / (1 + B)})

d = pd.read_csv(R / "9_3_correspondencia_por_candidato.csv", encoding="utf-8-sig")
t = d.pivot_table(index=["eje_tematico", "exponente_id"], columns="agregacion", values="rango_plan_propio")
w = wilcoxon(t["mejor_coincidencia"], t["promedio"])
salida.append({"analisis": "wilcoxon_mejor_vs_promedio", "agregacion": "ambas",
               "valor": w.statistic, "p": w.pvalue})

pl = pd.read_csv(R / "p2_planes_descriptivo.csv", encoding="utf-8-sig")[["exponente_id", "palabras"]]
for agg, g in d.merge(pl, on="exponente_id").groupby("agregacion"):
    r = g.groupby("exponente_id").agg(rango=("rango_plan_propio", "mean"), palabras=("palabras", "first"))
    rho = spearmanr(r["rango"], r["palabras"])
    salida.append({"analisis": "spearman_extension_vs_rango", "agregacion": agg,
                   "valor": rho.correlation, "p": rho.pvalue})

out = pd.DataFrame(salida)
out.to_csv(R / "9_6_analisis_complementario.csv", index=False, encoding="utf-8-sig")
print(out.round(4).to_string(index=False))
