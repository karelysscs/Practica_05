"""El motor: UNA función para las dos tareas.

    procesar(ruta_audio, esquema="tarea1" | "tarea2") -> dict

El esquema decide qué campos se buscan, con qué extractor y con qué tipo se verifica cada uno.
La tarea 2 no copia código: usa este mismo motor con otro esquema y, además, una validación
aritmética (validar.py) que el esquema declara como paso final.

Verificación rápida desde la terminal:
    python -m registro.motor data/audio/t1_entrevista_01.m4a --esquema tarea1
    python -m registro.motor --texto data/verdad/transcripciones/t2_llamada_03.txt --esquema tarea2
"""
from __future__ import annotations

import time
import traceback

from . import llm, reglas
from .config import cfg as leer_cfg
from .validar import validar_pedido
from .verificar import verificar

ESQUEMAS = {
    "tarea1": {
        "reglas": reglas.extraer_entrevista,
        "tipos": {"anios_experiencia": "numero", "ultimo_puesto": "texto", "tecnologias": "lista",
                  "pretension_salarial": "monto", "disponibilidad": "disponibilidad", "nivel_ingles": "categoria"},
        "glosario": "tarea1",
        "post": None,
    },
    "tarea2": {
        "reglas": reglas.extraer_pedido,
        "tipos": {"items": "items", "promocion": "promocion", "direccion_entrega": "texto",
                  "metodo_pago": "categoria", "total_dicho": "numero"},
        "glosario": "tarea2",
        "post": "validar_pedido",
    },
}


def procesar(ruta_audio: str | None = None, esquema: str = "tarea1", metodo: str = "reglas",
             asr: dict | None = None, transcripcion: str | None = None, cfg: dict | None = None) -> dict:
    """Audio (o transcripción ya hecha) → transcripción, campos, campos verificados, dispositivo, tiempo y error.

    - `asr`: {modelo, dispositivo, precision, glosario}. Por defecto, config.yaml → asr.defecto.
    - `transcripcion`: si se pasa, no se vuelve a transcribir (lo usan la app y la evaluación con caché).
    - `metodo`: "reglas" o "llm".
    """
    cfg = cfg or leer_cfg()
    e = ESQUEMAS[esquema]
    salida = {"esquema": esquema, "metodo": metodo, "transcripcion": transcripcion, "campos": None,
              "campos_verificados": None, "estados": None, "dispositivo": None, "segundos": None, "error": None}
    t0 = time.perf_counter()
    try:
        if transcripcion is None:
            from .asr import transcribir
            a = {**cfg["asr"]["defecto"], **(asr or {})}
            r = transcribir(str(ruta_audio), a["modelo"], a["dispositivo"], a["precision"], cfg["asr"]["idioma"],
                            cfg["glosario"][e["glosario"]] if a.get("glosario") else None,
                            cfg["asr"]["beam_size"], cfg["asr"]["vad_filter"])
            salida |= {"transcripcion": r["texto"], "dispositivo": r["dispositivo_real"], "asr": r}
        texto = salida["transcripcion"]
        campos = e["reglas"](texto, cfg) if metodo == "reglas" else llm.extraer(esquema, texto, cfg)
        valores, estados = verificar(campos, e["tipos"], texto)
        salida |= {"campos": campos, "campos_verificados": valores, "estados": estados}
        if e["post"] == "validar_pedido":
            salida["validacion"] = validar_pedido(valores, estados, cfg["validacion"]["tolerancia_soles"])
    except Exception as ex:  # el motor nunca se cae: devuelve el error para que lo muestre quien llama
        salida["error"] = f"{type(ex).__name__}: {ex}"
        salida["traza"] = traceback.format_exc(limit=3)
    salida["segundos"] = round(time.perf_counter() - t0, 2)
    return salida


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Procesa un audio (o un .txt) con el motor")
    p.add_argument("audio", nargs="?")
    p.add_argument("--texto", help="usar esta transcripción (.txt) en vez de transcribir")
    p.add_argument("--esquema", default="tarea1", choices=list(ESQUEMAS))
    p.add_argument("--metodo", default="reglas", choices=["reglas", "llm"])
    p.add_argument("--laptop", action="store_true", help="usar la configuración liviana de CPU (asr.laptop)")
    a = p.parse_args()
    texto = Path(a.texto).read_text(encoding="utf-8") if a.texto else None
    r = procesar(a.audio, a.esquema, a.metodo, leer_cfg()["asr"]["laptop"] if a.laptop else None, texto)
    r.pop("asr", None)
    print(json.dumps({k: v for k, v in r.items() if k != "transcripcion"}, ensure_ascii=False, indent=2, default=str))
