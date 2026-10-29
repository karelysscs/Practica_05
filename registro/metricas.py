"""Dos capas de medición.

1. Transcripción: WER y CER contra la transcripción de referencia (ambas normalizadas con
   texto.normalizar, la misma regla para las dos).
2. Registro: si cada campo extraído es igual al de la tabla de respuestas correctas
   (data/verdad/verdad_tarea*.csv), F1 macro del nivel de inglés y error absoluto en los números.

Con 4 clips por tarea los intervalos son anchos: se reportan con bootstrap (por clip) o Wilson
(proporciones) para que se vea cuánto NO se puede concluir.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .reglas import comparable_direccion
from .texto import clave, normalizar
from .verificar import SIN_VERIFICAR


# ───────────────────────────────────────────────────────── capa 1: transcripción
def wer(ref: str, hip: str) -> float:
    import jiwer
    return jiwer.wer(normalizar(ref), normalizar(hip) or "∅")


def cer(ref: str, hip: str) -> float:
    import jiwer
    return jiwer.cer(normalizar(ref), normalizar(hip) or "∅")


def bootstrap_media(valores, reps=2000, confianza=0.95, semilla=7) -> tuple[float, float, float]:
    """Media e intervalo de confianza remuestreando clips (con 4 clips, el IC es muy ancho)."""
    v = np.asarray(valores, dtype=float)
    if len(v) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(reps, len(v)), replace=True).mean(axis=1)
    a = (1 - confianza) / 2
    return float(v.mean()), float(np.quantile(medias, a)), float(np.quantile(medias, 1 - a))


def wilson(aciertos: int, n: int, confianza=0.95) -> tuple[float, float, float]:
    if n == 0:
        return (math.nan, math.nan, math.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confianza) / 2)
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centro - margen), min(1.0, centro + margen)


# ───────────────────────────────────────────────────────── capa 2: registro
def cargar_verdad(ruta_csv) -> dict[str, dict]:
    df = pd.read_csv(ruta_csv, dtype=str, keep_default_na=False)
    return {f["clip"]: f for f in df.to_dict("records")}


def comparar_entrevista(pred: dict, verdad: dict) -> dict:
    """{campo: {"ok": bool, "error_abs": float|None}} para la tarea 1."""
    def num(x):
        return None if x in (None, SIN_VERIFICAR) else float(x)
    r = {}
    n = num(pred.get("anios_experiencia"))
    r["anios_experiencia"] = {"ok": n == float(verdad["anios_experiencia"]),
                              "error_abs": None if n is None else abs(n - float(verdad["anios_experiencia"]))}
    p = pred.get("ultimo_puesto")
    r["ultimo_puesto"] = {"ok": p not in (None, SIN_VERIFICAR) and clave(p) == clave(verdad["ultimo_puesto"])}
    t = pred.get("tecnologias")
    real = {clave(x) for x in verdad["tecnologias"].split("|")}
    hall = {clave(x) for x in t} if isinstance(t, list) else set()
    r["tecnologias"] = {"ok": hall == real, "f1": _f1_conjuntos(hall, real)}
    s = pred.get("pretension_salarial")
    ok = isinstance(s, dict) and s["monto"] == float(verdad["pretension_salarial"]) and s["moneda"] == verdad["moneda"]
    r["pretension_salarial"] = {"ok": ok, "error_abs": abs(s["monto"] - float(verdad["pretension_salarial"]))
                                if isinstance(s, dict) else None}
    d = pred.get("disponibilidad")
    r["disponibilidad"] = {"ok": d not in (None, SIN_VERIFICAR) and clave(d) == clave(verdad["disponibilidad"])}
    i = pred.get("nivel_ingles")
    r["nivel_ingles"] = {"ok": i not in (None, SIN_VERIFICAR) and clave(i) == clave(verdad["nivel_ingles"]),
                         "pred": clave(i) if i not in (None, SIN_VERIFICAR) else "ninguno",
                         "real": clave(verdad["nivel_ingles"])}
    return r


def comparar_pedido(pred: dict, verdad: dict) -> dict:
    r = {}
    items = pred.get("items")
    real = sorted((i["producto"], i["tamano"] or None, int(i["cantidad"])) for i in json.loads(verdad["items"]))
    hall = sorted((i["producto"], i["tamano"] or None, int(i["cantidad"])) for i in items) \
        if isinstance(items, list) else None
    r["items"] = {"ok": hall == real}
    promo = pred.get("promocion")
    r["promocion"] = {"ok": isinstance(promo, dict) and clave(promo["nombre"]) == clave(verdad["promocion"])}
    # dirección: mismos números y mismas dos primeras palabras (tipo de vía + nombre)
    d, v = comparable_direccion(pred.get("direccion_entrega") if pred.get("direccion_entrega") != SIN_VERIFICAR
                                else None), comparable_direccion(verdad["direccion_entrega"])
    import re
    r["direccion_entrega"] = {"ok": bool(d) and re.findall(r"\d+", d) == re.findall(r"\d+", v)
                              and d.split()[:2] == v.split()[:2]}
    m = pred.get("metodo_pago")
    r["metodo_pago"] = {"ok": m not in (None, SIN_VERIFICAR) and clave(m) == clave(verdad["metodo_pago"])}
    t = pred.get("total_dicho")
    t = None if t in (None, SIN_VERIFICAR) else float(t)
    r["total_dicho"] = {"ok": t == float(verdad["total_dicho"]),
                        "error_abs": None if t is None else abs(t - float(verdad["total_dicho"]))}
    return r


def _f1_conjuntos(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    tp = len(a & b)
    return 0.0 if tp == 0 else 2 * tp / (len(a) + len(b))


def f1_macro(reales: list[str], predichos: list[str], etiquetas: list[str]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(reales, predichos, labels=etiquetas, average="macro", zero_division=0))
