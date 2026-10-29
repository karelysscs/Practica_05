"""Crea la escalera L0-L3 de las llamadas (tarea 2) y anota la SNR medida.

    python scripts/degradar.py

Salida: data/audio_degradado/t2_llamada_XX_L{0..3}.wav y resultados/degradacion.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registro.config import cfg, ruta  # noqa: E402
from registro.degradar import escalera  # noqa: E402
from scripts.transcribir import audios  # noqa: E402

filas = []
for a in audios(ruta("audio"), "t2_"):
    filas += escalera(a, ruta("audio_degradado"), cfg())
    print(a.stem, "listo")
df = pd.DataFrame(filas)
df.to_csv(ruta("resultados") / "degradacion.csv", index=False)
print(df.to_string(index=False))
