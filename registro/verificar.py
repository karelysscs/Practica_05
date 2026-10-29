"""Regla de oro: un valor solo se acepta si aparece literalmente en la transcripción.

"Literalmente" = después de normalizar (texto.clave: minúsculas, sin tildes ni signos, números en
dígitos). Cada campo trae una evidencia (el fragmento de donde salió). Se exigen dos cosas:
1. la evidencia está en la transcripción, y
2. el valor se puede leer en la evidencia (no basta con citar algo cualquiera).
Si falla cualquiera de las dos, el campo queda como "sin_verificar".
"""
from __future__ import annotations

import re

from .texto import clave

SIN_VERIFICAR = "sin_verificar"

SINONIMOS_DISPONIBILIDAD = {
    "inmediata": ["inmediato", "inmediata", "inmediatamente", "ya mismo"],
    "2 semanas": ["2 semanas", "15 dias"],
    "1 mes": ["1 mes", "un mes", "30 dias"],
}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
CLAVES_PAGO = {"efectivo": "efectivo", "yape": "yape", "plin": "plin", "tarjeta": "tarjeta"}
CLAVES_PROMO = {"duo familiar": "duo familiar", "cupon BIENVENIDA": "bienvenida"}
PALABRA_PRODUCTO = {"gaseosa 1.5L": "gaseosa", "pan al ajo": "ajo", "pepperoni": "pep"}  # "pep": Whisper a veces escribe "peperoni"


def _contiene(aguja: str, pajar: str) -> bool:
    a, p = clave(aguja), clave(pajar)
    return bool(a) and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", p) is not None


def _num_en(n, ev: str) -> bool:
    if isinstance(n, float) and n % 1 == 0.5:
        base = int(n)
        return _contiene(f"{base}", ev) or _contiene("un", ev) and base == 1 and _contiene("y medio", ev)
    return _contiene(str(int(n)), ev)


def valor_en_evidencia(tipo: str, valor, ev: str) -> bool:
    if valor is None:
        return False
    if tipo == "numero":
        return _num_en(valor, ev)
    if tipo in ("texto", "categoria"):
        return _contiene(CLAVES_PAGO.get(valor, valor), ev)
    if tipo == "monto":
        moneda = "soles" if valor["moneda"] == "PEN" else "dolares"
        return _contiene(str(valor["monto"]), ev) and _contiene(moneda, ev)
    if tipo == "disponibilidad":
        if re.match(r"\d{4}-\d{2}-\d{2}$", valor):
            _, mes, dia = valor.split("-")
            return (_contiene(str(int(dia)), ev) or int(dia) == 1 and _contiene("primero", ev)) \
                and _contiene(MESES[int(mes) - 1], ev)
        return any(_contiene(s, ev) for s in SINONIMOS_DISPONIBILIDAD.get(valor, [valor]))
    raise ValueError(tipo)


def verificar_campo(tipo: str, campo: dict, transcripcion: str) -> tuple[object, str]:
    """Devuelve (valor aceptado, estado). estado ∈ {verificado, sin_verificar, vacio}."""
    valor, ev = campo.get("valor"), campo.get("evidencia") or ""
    if valor in (None, [], ""):
        return None, "vacio"
    if tipo == "lista":  # cada elemento tiene que estar en la transcripción
        ok = all(_contiene(v, transcripcion) for v in valor)
        return (valor, "verificado") if ok else (SIN_VERIFICAR, SIN_VERIFICAR)
    if tipo == "items":  # cada ítem: su evidencia está en la transcripción y nombra el producto y la cantidad
        if campo.get("no_reconocido"):  # el vendedor nombró algo que no se pudo leer: el pedido está incompleto
            return SIN_VERIFICAR, SIN_VERIFICAR
        ok = all(_contiene(i.get("evidencia", ""), transcripcion)
                 and PALABRA_PRODUCTO.get(i["producto"], i["producto"]) in clave(i["evidencia"])
                 and (_num_en(i["cantidad"], i["evidencia"]) or i["cantidad"] == 1)
                 for i in valor)
        return (valor, "verificado") if ok else (SIN_VERIFICAR, SIN_VERIFICAR)
    if tipo == "promocion":
        if valor["nombre"] == "ninguna":
            # "ninguna" es creíble si no se nombra ninguna promoción, o si cada vez que se nombra
            # alguien dice que no aplica ("ninguno", "no me vale")
            t = clave(transcripcion)
            menciones = [m.start() for p in CLAVES_PROMO.values() for m in re.finditer(p, t)]
            ok = all(re.search(r"ningun|no me vale|no aplica|no tengo|no vale", t[max(0, k - 80):k + 80])
                     for k in menciones)
        else:
            ok = _contiene(ev, transcripcion) and _contiene(CLAVES_PROMO[valor["nombre"]], ev)
        return (valor, "verificado") if ok else (SIN_VERIFICAR, SIN_VERIFICAR)
    if _contiene(ev, transcripcion) and valor_en_evidencia(tipo, valor, ev):
        return valor, "verificado"
    return SIN_VERIFICAR, SIN_VERIFICAR


def verificar(campos: dict, tipos: dict, transcripcion: str) -> tuple[dict, dict]:
    """(valores verificados, estado de cada campo)."""
    valores, estados = {}, {}
    for nombre, tipo in tipos.items():
        valores[nombre], estados[nombre] = verificar_campo(tipo, campos.get(nombre, {}), transcripcion)
    return valores, estados
