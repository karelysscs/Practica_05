"""Arma la transcripción de referencia de cada clip a partir de los guiones.

    python scripts/referencias.py

La referencia es lo que dice el guion, sin las etiquetas A/B. Si al grabar alguien dijo algo
distinto, corrige el .txt a mano en data/verdad/transcripciones/ (ese archivo es la verdad).
No sobrescribe un .txt que ya existe, para no borrar esas correcciones (usa --forzar para rehacerlos).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registro.config import ruta  # noqa: E402

GUIONES = {"tarea1_entrevistas.md": ("Entrevista", "t1_entrevista"),
           "tarea2_llamadas.md": ("Llamada", "t2_llamada")}


def turnos_por_clip(md: str, titulo: str) -> dict[int, list[str]]:
    clips, actual = {}, None
    for linea in md.splitlines():
        if m := re.match(rf"## {titulo} (\d+)", linea):
            actual = int(m[1])
            clips[actual] = []
        elif actual and (m := re.match(r"\*\*[AB]:\*\*\s*(.+)", linea)):
            clips[actual].append(m[1].strip())
        elif linea.startswith("## "):
            actual = None
    return clips


def main(forzar=False):
    destino = ruta("verdad") / "transcripciones"
    destino.mkdir(parents=True, exist_ok=True)
    for archivo, (titulo, prefijo) in GUIONES.items():
        md = (ruta("guiones") / archivo).read_text(encoding="utf-8")
        for n, turnos in turnos_por_clip(md, titulo).items():
            txt = destino / f"{prefijo}_{n:02d}.txt"
            if txt.exists() and not forzar:
                continue
            txt.write_text("\n".join(turnos) + "\n", encoding="utf-8")
            print(f"{txt.name}: {sum(len(t.split()) for t in turnos)} palabras")


if __name__ == "__main__":
    main("--forzar" in sys.argv)
