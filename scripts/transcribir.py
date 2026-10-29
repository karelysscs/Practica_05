"""Transcribe los audios y guarda la caché en resultados/transcripciones/<experimento>/<clip>.json.

Pensado para Colab con GPU (ver notebooks/colab_gpu.ipynb):

    python scripts/transcribir.py --tarea1            # todos los experimentos de config.yaml
    python scripts/transcribir.py --tarea1 --solo large_cuda_fp16 small_cpu_int8
    python scripts/transcribir.py --escalera          # tarea 2: L0-L3 con asr.defecto

Cada experimento corre en su propio proceso: así la VRAM medida es solo la de ese modelo.
Si el .json ya existe no se repite (borra la carpeta para rehacerlo).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registro.config import cfg, ruta  # noqa: E402

EXTENSIONES = {".wav", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".flac", ".mp4", ".webm", ".3gp", ".amr"}


def audios(carpeta: Path, prefijo: str) -> list[Path]:
    return sorted(p for p in carpeta.glob(f"{prefijo}*") if p.suffix.lower() in EXTENSIONES)


def correr_uno(exp: dict, archivos: list[Path], glosario: list[str], destino: Path):
    from registro.asr import transcribir
    destino.mkdir(parents=True, exist_ok=True)
    c = cfg()["asr"]
    for a in archivos:
        salida = destino / f"{a.stem}.json"
        if salida.exists():
            continue
        r = transcribir(str(a), exp["modelo"], exp["dispositivo"], exp["precision"], c["idioma"],
                        glosario if exp.get("glosario") else None, c["beam_size"], c["vad_filter"])
        r |= {"experimento": exp["nombre"], "clip": a.stem}
        salida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        gpu = f", VRAM pico {r['vram_pico_mb']} MB" if r["vram_pico_mb"] else ""
        print(f"  {a.stem}: {r['segundos']} s ({r['seg_por_minuto']} s/min) en {r['dispositivo_real']}{gpu}", flush=True)
        if exp["dispositivo"] == "cuda" and not r["dispositivo_real"].startswith("cuda"):
            raise SystemExit(f"Se pidió GPU pero corrió en {r['dispositivo_real']}: revisa el entorno")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tarea1", action="store_true")
    p.add_argument("--escalera", action="store_true")
    p.add_argument("--solo", nargs="*", help="nombres de experimentos")
    p.add_argument("--_uno", help=argparse.SUPPRESS)
    p.add_argument("--_nivel", help=argparse.SUPPRESS)
    a = p.parse_args()
    c = cfg()
    base = ruta("transcripciones")

    if a._uno:  # proceso hijo: un solo experimento
        if a._nivel:
            exp = {**c["asr"]["defecto"], "nombre": a._uno}
            archivos = sorted(ruta("audio_degradado").glob(f"t2_*_{a._nivel}.wav"))
            correr_uno(exp, archivos, c["glosario"]["tarea2"], base / a._uno)
        else:
            exp = next(e for e in c["experimentos"] if e["nombre"] == a._uno)
            archivos = audios(ruta("audio"), "t1_")
            if exp["clips"] != "todos":
                archivos = [x for x in archivos if x.stem in exp["clips"]]
            correr_uno(exp, archivos, c["glosario"]["tarea1"], base / a._uno)
        return

    tareas = []
    if a.tarea1:
        if not audios(ruta("audio"), "t1_"):
            raise SystemExit(f"No hay audios t1_* en {ruta('audio')}")
        tareas += [[e["nombre"]] for e in c["experimentos"] if not a.solo or e["nombre"] in a.solo]
    if a.escalera:
        tareas += [[f"escalera_{n}", "--_nivel", n] for n in c["degradacion"]["niveles"]]
    for t in tareas:
        print(f"== {t[0]}", flush=True)
        subprocess.run([sys.executable, __file__, "--_uno", *t], check=True)


if __name__ == "__main__":
    main()
