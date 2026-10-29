"""Tarea 2: el pedido se valida solo. Σ cantidad × precio unitario − promoción debe dar el total dicho.

Si no cuadra, una regla decide de quién es la culpa:
- error_asr: el registro no es confiable (algún campo quedó sin_verificar, algún ítem no tiene
  precio en la carta, o no se encontró el total). Lo más probable es que el modelo haya oído mal.
- error_hablante: el registro está completo y verificado, pero la cuenta no da. Entonces quien
  habló sumó mal (el vendedor dijo un total equivocado).
"""
from __future__ import annotations

from .verificar import SIN_VERIFICAR


def suma_pedido(items: list[dict], descuento: float) -> float | None:
    if not items or any(i.get("precio_unitario") is None for i in items):
        return None
    return sum(i["cantidad"] * i["precio_unitario"] for i in items) - (descuento or 0)


def validar_pedido(valores: dict, estados: dict, tolerancia: float = 0.5) -> dict:
    items = valores.get("items")
    promo = valores.get("promocion")
    total = valores.get("total_dicho")
    descuento = promo["descuento"] if isinstance(promo, dict) else 0
    suma = suma_pedido(items if isinstance(items, list) else [], descuento)
    if suma is None or total in (None, SIN_VERIFICAR):
        cuadra = None
    else:
        cuadra = abs(suma - total) <= tolerancia
    confiable = all(e == "verificado" for c, e in estados.items() if c in ("items", "promocion", "total_dicho"))
    if cuadra:
        diagnostico = "ok"
    elif not confiable or cuadra is None:
        diagnostico = "error_asr"
    else:
        diagnostico = "error_hablante"
    return {"suma_calculada": suma, "total_dicho": total if total != SIN_VERIFICAR else None,
            "cuadra": cuadra, "diagnostico": diagnostico, "alerta": diagnostico != "ok"}
