"""Camino 2 de extracción: un LLM devuelve JSON con valor + evidencia por campo.

El modelo, el esfuerzo y el prompt están en config.yaml (llm). La clave va en .env
(ANTHROPIC_API_KEY). Las respuestas se guardan en resultados/llm/ para no volver a pagar
ni depender de internet al evaluar (y para que GitHub Actions use solo la caché).
"""
from __future__ import annotations

import hashlib
import json

from .config import ruta

_EVID = {"type": "string", "description": "Fragmento exacto de la transcripción de donde sale el valor."}


def _campo(tipo_valor) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["valor", "evidencia"],
            "properties": {"valor": tipo_valor, "evidencia": _EVID}}


def esquema_json(esquema: str, cfg: dict) -> dict:
    nulo = lambda t: {"anyOf": [t, {"type": "null"}]}  # noqa: E731
    if esquema == "tarea1":
        props = {
            "anios_experiencia": _campo(nulo({"type": "number"})),
            "ultimo_puesto": _campo(nulo({"type": "string"})),
            "tecnologias": _campo({"type": "array", "items": {"type": "string"}}),
            "pretension_salarial": _campo(nulo({
                "type": "object", "additionalProperties": False, "required": ["monto", "moneda"],
                "properties": {"monto": {"type": "number"}, "moneda": {"type": "string", "enum": ["PEN", "USD"]}}})),
            "disponibilidad": _campo(nulo({"type": "string", "description":
                                           "'inmediata', 'N semanas', 'N mes' o una fecha AAAA-MM-DD"})),
            "nivel_ingles": _campo(nulo({"type": "string", "enum": cfg["extraccion"]["niveles_ingles"]})),
        }
    else:
        carta = cfg["carta"]
        productos = list(carta["pizzas"]) + list(carta["extras"])
        item = {"type": "object", "additionalProperties": False,
                "required": ["producto", "tamano", "cantidad", "evidencia"],
                "properties": {"producto": {"type": "string", "enum": productos},
                               "tamano": nulo({"type": "string", "enum": ["personal", "mediana", "familiar"]}),
                               "cantidad": {"type": "integer"}, "evidencia": _EVID}}
        props = {
            "items": _campo({"type": "array", "items": item}),
            "promocion": _campo({"type": "string", "enum": list(carta["promociones"]) + ["ninguna"]}),
            "direccion_entrega": _campo(nulo({"type": "string"})),
            "metodo_pago": _campo(nulo({"type": "string", "enum": carta["metodos_pago"]})),
            "total_dicho": _campo(nulo({"type": "number"})),
        }
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _pedir(esquema: str, transcripcion: str, cfg: dict) -> dict:
    import anthropic
    c = cfg["llm"]
    tarea = ("la ficha del candidato de esta entrevista de trabajo" if esquema == "tarea1"
             else "el pedido final de esta llamada a una pizzería (con las correcciones aplicadas)")
    respuesta = anthropic.Anthropic().messages.create(
        model=c["modelo"], max_tokens=c["max_tokens"], system=c["prompt_sistema"],
        output_config={"effort": c["esfuerzo"], "format": {"type": "json_schema", "schema": esquema_json(esquema, cfg)}},
        messages=[{"role": "user", "content": f"Llena {tarea}.\n\nTranscripción:\n{transcripcion}"}],
    )
    if respuesta.stop_reason == "refusal":
        raise RuntimeError("El modelo rechazó la solicitud")
    return json.loads(next(b.text for b in respuesta.content if b.type == "text"))


def extraer(esquema: str, transcripcion: str, cfg: dict) -> dict:
    """Mismo formato que reglas.py: {campo: {"valor", "evidencia"}}."""
    huella = hashlib.sha1(f"{cfg['llm']['modelo']}|{esquema}|{cfg['llm']['prompt_sistema']}|{transcripcion}"
                          .encode("utf-8")).hexdigest()[:16]
    cache = ruta("resultados") / "llm" / f"{esquema}_{huella}.json"
    if cache.exists():
        crudo = json.loads(cache.read_text(encoding="utf-8"))
    else:
        crudo = _pedir(esquema, transcripcion, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(crudo, ensure_ascii=False, indent=1), encoding="utf-8")
    return _a_formato_comun(esquema, crudo, cfg)


def _a_formato_comun(esquema: str, crudo: dict, cfg: dict) -> dict:
    if esquema == "tarea1":
        nivel = crudo["nivel_ingles"]
        return crudo | {"nivel_ingles": nivel}
    carta = cfg["carta"]
    items = crudo["items"]["valor"] or []
    for it in items:
        it["precio_unitario"] = (carta["pizzas"][it["producto"]].get(it["tamano"]) if it["producto"] in carta["pizzas"]
                                 else carta["extras"].get(it["producto"]))
    promo = crudo["promocion"]["valor"]
    return crudo | {
        "items": {"valor": items, "evidencia": crudo["items"]["evidencia"]},
        "fuente_items": "llm",
        "promocion": {"valor": {"nombre": promo, "descuento": carta["promociones"].get(promo, 0)},
                      "evidencia": crudo["promocion"]["evidencia"]},
    }
