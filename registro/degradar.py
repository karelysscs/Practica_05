"""Escalera de degradación de la tarea 2: el mismo audio, cada vez peor.

L0  tal cual se grabó.
L1  nota de voz tipo WhatsApp: Opus mono a bajo bitrate (config: degradacion.L1_bitrate).
L2  línea telefónica: solo 300-3400 Hz y remuestreado a 8 kHz.
L3  L1 + ruido de fondo a una SNR fija (config: degradacion.L3_snr_db), medida después de mezclar.

L1 necesita ffmpeg (Colab ya lo trae).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16_000


def leer(ruta: str | Path, sr: int = SR) -> np.ndarray:
    from faster_whisper import decode_audio  # PyAV: lee m4a, mp3, ogg, wav...
    return decode_audio(str(ruta), sampling_rate=sr)


def whatsapp(x: np.ndarray, bitrate: str) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("L1 necesita ffmpeg (en Colab ya está; en Windows: winget install ffmpeg)")
    with tempfile.TemporaryDirectory() as d:
        wav, opus = Path(d) / "x.wav", Path(d) / "x.ogg"
        sf.write(wav, x, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-c:a", "libopus",
                        "-b:a", bitrate, "-application", "voip", str(opus)], check=True)
        return leer(opus)


def telefono(x: np.ndarray, banda: tuple[int, int], sr_tel: int) -> np.ndarray:
    sos = butter(6, banda, btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x)
    y = resample_poly(y, sr_tel, SR)          # a 8 kHz: así suena una llamada
    return resample_poly(y, SR, sr_tel).astype(np.float32)  # de vuelta a 16 kHz, que es lo que espera Whisper


def ruido_rosa(n: int, rng) -> np.ndarray:
    blanco = rng.standard_normal(n)
    f = np.fft.rfft(blanco)
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    return np.fft.irfft(f, n).astype(np.float32)


def mezclar_ruido(x: np.ndarray, ruido: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    """Escala el ruido para la SNR pedida y devuelve (mezcla, SNR medida)."""
    ruido = np.resize(ruido, len(x))
    p_voz, p_ruido = np.mean(x ** 2), np.mean(ruido ** 2)
    ruido = ruido * np.sqrt(p_voz / (p_ruido * 10 ** (snr_db / 10)))
    y = x + ruido
    pico = np.max(np.abs(y))
    if pico > 1:
        y, x, ruido = y / pico, x / pico, ruido / pico
    medida = 10 * np.log10(np.mean(x ** 2) / np.mean(ruido ** 2))
    return y.astype(np.float32), float(medida)


def escalera(ruta: str | Path, destino: Path, cfg: dict) -> list[dict]:
    """Escribe <clip>_L0..L3.wav en `destino` y devuelve lo que se midió en cada nivel."""
    c = cfg["degradacion"]
    destino.mkdir(parents=True, exist_ok=True)
    nombre = Path(ruta).stem
    x0 = leer(ruta)
    x1 = whatsapp(x0, c["L1_bitrate"])
    x2 = telefono(x0, tuple(c["L2_banda"]), c["L2_frecuencia"])
    rng = np.random.default_rng(c["semilla"])
    archivo_ruido = Path(c["ruido"])
    if not archivo_ruido.is_absolute():
        from .config import RAIZ
        archivo_ruido = RAIZ / archivo_ruido
    if archivo_ruido.exists():
        ruido, tipo_ruido = leer(archivo_ruido), f"grabado ({archivo_ruido.name})"
    else:
        ruido, tipo_ruido = ruido_rosa(len(x1), rng), "rosa sintético (no se encontró ruido grabado)"
    x3, snr = mezclar_ruido(x1, ruido, c["L3_snr_db"])
    filas = []
    for nivel, x, detalle in [("L0", x0, "original"), ("L1", x1, f"Opus {c['L1_bitrate']} mono"),
                              ("L2", x2, f"banda {c['L2_banda'][0]}-{c['L2_banda'][1]} Hz, {c['L2_frecuencia']} Hz"),
                              ("L3", x3, f"L1 + ruido {tipo_ruido}")]:
        sf.write(destino / f"{nombre}_{nivel}.wav", x, SR)
        filas.append({"clip": nombre, "nivel": nivel, "detalle": detalle,
                      "snr_db_medida": round(snr, 2) if nivel == "L3" else None})
    return filas
