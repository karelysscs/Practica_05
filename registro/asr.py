"""Transcripción con faster-whisper, midiendo dónde corrió de verdad y cuánta VRAM usó.

faster-whisper puede caer en CPU sin avisar (~40 veces más lento). Por eso no se confía en lo que
pedimos: se lee en tiempo de ejecución el dispositivo del modelo cargado y, con NVML, cuánta memoria
de la GPU usa este proceso mientras transcribe.
"""
from __future__ import annotations

import inspect
import os
import threading
import time
from functools import lru_cache


@lru_cache(maxsize=2)
def cargar(modelo: str, dispositivo: str, precision: str):
    from faster_whisper import WhisperModel
    return WhisperModel(modelo, device=dispositivo, compute_type=precision,
                        cpu_threads=min(8, os.cpu_count() or 4))


class MedidorVRAM:
    """Muestrea cada 50 ms la memoria de GPU que usa este proceso (pico en MB)."""

    def __init__(self):
        self.pico_mb, self.disponible, self._parar = 0.0, False, threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu = pynvml.nvmlDeviceGetName(self._h)
            self.gpu = self.gpu.decode() if isinstance(self.gpu, bytes) else self.gpu
            self.disponible = True
        except Exception:
            self.gpu = None

    def _uso_proceso(self) -> float:
        pid = os.getpid()
        for p in self._nv.nvmlDeviceGetComputeRunningProcesses(self._h):
            if p.pid == pid and p.usedGpuMemory:
                return p.usedGpuMemory / 2**20
        return 0.0

    def _bucle(self):
        while not self._parar.is_set():
            try:
                self.pico_mb = max(self.pico_mb, self._uso_proceso())
            except Exception:
                pass
            time.sleep(0.05)

    def __enter__(self):
        if self.disponible:
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        if self.disponible:
            self._hilo.join(timeout=1)
            self.pico_mb = max(self.pico_mb, self._uso_proceso())


def dispositivo_real(m) -> str:
    """Lo que dice el modelo ya cargado (CTranslate2), no lo que pedimos."""
    return f"{m.model.device}:{m.model.device_index[0]}" if hasattr(m.model, "device_index") else str(m.model.device)


def transcribir(ruta_audio: str, modelo: str, dispositivo: str, precision: str, idioma: str = "es",
                glosario: list[str] | None = None, beam_size: int = 5, vad_filter: bool = True) -> dict:
    import ctranslate2
    import faster_whisper
    m = cargar(modelo, dispositivo, precision)
    kw = {"language": idioma, "beam_size": beam_size, "vad_filter": vad_filter}
    mecanismo = None
    if glosario:  # hotwords existe desde faster-whisper 1.0.2; en versiones viejas solo hay initial_prompt
        if "hotwords" in inspect.signature(m.transcribe).parameters:
            kw["hotwords"], mecanismo = ", ".join(glosario), "hotwords"
        else:
            kw["initial_prompt"], mecanismo = "Glosario: " + ", ".join(glosario) + ".", "initial_prompt"
    with MedidorVRAM() as vram:
        t0 = time.perf_counter()
        segmentos, info = m.transcribe(ruta_audio, **kw)
        segmentos = list(segmentos)  # el generador es perezoso: aquí corre de verdad
        segundos = time.perf_counter() - t0
    return {
        "texto": " ".join(s.text.strip() for s in segmentos),
        "segmentos": [{"inicio": round(s.start, 2), "fin": round(s.end, 2), "texto": s.text.strip()} for s in segmentos],
        "modelo": modelo, "precision": precision, "dispositivo_pedido": dispositivo,
        "dispositivo_real": dispositivo_real(m), "gpus_cuda_visibles": ctranslate2.get_cuda_device_count(),
        "gpu": vram.gpu, "vram_pico_mb": round(vram.pico_mb, 1) if vram.disponible else None,
        "segundos": round(segundos, 2), "duracion_audio": round(info.duration, 2),
        "seg_por_minuto": round(segundos / (info.duration / 60), 2) if info.duration else None,
        "glosario": mecanismo, "faster_whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__,
    }
