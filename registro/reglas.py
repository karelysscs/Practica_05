"""Camino 1 de extracción: expresiones regulares sobre el texto de trabajo (ver texto.texto_trabajo).

Cada extractor devuelve {campo: {"valor": ..., "evidencia": "fragmento del texto"}}.
La evidencia es lo que después se verifica contra la transcripción (verificar.py).
"""
from __future__ import annotations

import re

from .texto import clave, sin_tildes, texto_trabajo

CANT = r"(\d+|una|un|uno)"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


def _cantidad(s: str) -> int:
    return 1 if s in ("un", "una", "uno") else int(s)


def _campo(valor, evidencia=""):
    return {"valor": valor, "evidencia": evidencia}


def _primero(patrones, texto, flags=0):
    for p in patrones:
        if m := re.search(p, texto, flags):
            return m
    return None


# ───────────────────────────────────────────────────────── tarea 1: entrevistas
def extraer_entrevista(transcripcion: str, cfg: dict) -> dict:
    t = texto_trabajo(transcripcion)
    ex = cfg["extraccion"]
    return {
        "anios_experiencia": _experiencia(t),
        "ultimo_puesto": _puesto(t),
        "tecnologias": _tecnologias(t, ex["tecnologias"]),
        "pretension_salarial": _pretension(t),
        "disponibilidad": _disponibilidad(t, ex["anio_referencia"]),
        "nivel_ingles": _ingles(t),
    }


def _experiencia(t):
    m = _primero([r"(?:tengo|son|llevo|cuento con) ((\d+|un|una) años?( y medio)?) de experiencia",
                  r"en total (?:tengo|son|llevo) ((\d+|un|una) años?( y medio)?)",
                  r"(?:ya son|llevo) ((\d+) años?( y medio)?)",
                  r"(?:tengo) ((\d+|un|una) años?( y medio)?)[.,]",
                  r"(?:tengo|llevo) ((\d+|un|una) años?( y medio)?) (?:de|en) (?:prácticas|trabajo|el área)"], t)
    if not m:
        return _campo(None)
    valor = _cantidad(m[2]) + (0.5 if m[3] else 0)
    return _campo(valor, m[1])


def _puesto(t):
    fin = r"(?= en (?:una|un|la|el)\b|[.,;]|$)"
    m = _primero([rf"(?:mi )?último puesto fue (?:como )?(?:de )?([^.,;]+?){fin}",
                  rf"(?:mi )?último trabajo fue (?:como )?([^.,;]+?){fin}",
                  rf"actualmente soy ([^.,;]+?){fin}",
                  rf"\bsoy ((?:analista|líder|lider|especialista|jefe|jefa|desarrollador|desarrolladora|practicante|diseñador|diseñadora)[^.,;]*?){fin}"], t)
    return _campo(m[1].strip(), m[1].strip()) if m else _campo(None)


def _tecnologias(t, vocabulario):
    halladas, primera, ultima = [], None, None
    for tec in vocabulario:
        for m in re.finditer(rf"(?<!\w){re.escape(tec.lower())}(?!\w)", t):
            halladas.append((m.start(), tec))
            primera = m.start() if primera is None else min(primera, m.start())
            ultima = m.end() if ultima is None else max(ultima, m.end())
            break
    if not halladas:
        return _campo([])
    return _campo([tec for _, tec in sorted(halladas)], ", ".join(tec.lower() for _, tec in sorted(halladas)))


def _pretension(t):
    zona = _zona(t, r"pretensi[oó]n|ganar|sueldo|salari|pido|pidiendo|buscando")
    candidatos = list(re.finditer(r"(\d{3,}) (soles|dólares|dolares)", zona))
    if not candidatos:
        return _campo(None)
    m = candidatos[-1]  # si la persona se corrige, vale lo último que dijo
    moneda = "PEN" if m[2] == "soles" else "USD"
    return _campo({"monto": int(m[1]), "moneda": moneda}, m[0])


def _zona(t, ancla):
    """Desde la primera mención del tema hasta el final (las respuestas vienen después de la pregunta)."""
    m = re.search(ancla, t)
    return t[m.start():] if m else t


def _disponibilidad(t, anio):
    verbo = r"(?:empezar|comenzar|incorporarme|incorporarse|unirme|unirse|empiezo)"
    m = re.search(rf"{verbo}[^.?]*?(?:de inmediato|inmediatamente|ya mismo)", t) or \
        re.search(r"(?:de inmediato|inmediatamente|ya mismo)", _zona(t, verbo))
    if m:
        return _campo("inmediata", m[0])
    m = re.search(rf"{verbo}[^.?]*?(?:a partir del?|desde el) (\d+|primero) de ({'|'.join(MESES)})", t)
    if m:
        dia = 1 if m[1] == "primero" else int(m[1])
        return _campo(f"{anio}-{MESES.index(m[2]) + 1:02d}-{dia:02d}", m[0])
    m = re.search(rf"{verbo}[^.?]*? en (\d+|un|una) (semanas?|mes|meses|días)", t)
    if m:
        n, unidad = _cantidad(m[1]), m[2]
        if unidad.startswith("día"):  # 15 días = 2 semanas, 30 días = 1 mes
            n, unidad = (n // 7, "semanas") if n < 30 else (n // 30, "mes")
        unidad = "semanas" if unidad.startswith("semana") else "mes"
        return _campo(f"{n} {unidad}", m[0])
    return _campo(None)


def _ingles(t):
    zona = _zona(t, r"inglés|ingles")
    m = re.search(r"\b(básico|basico|intermedio|avanzado)\b", zona)
    return _campo(sin_tildes(m[1]), m[0]) if m else _campo(None)


# ───────────────────────────────────────────────────────── tarea 2: pedidos de pizza
PIZZA = (rf"{CANT}\s+(?:pizzas?\s+)?(?:(personales|personal|medianas|mediana|familiares|familiar)\s+)?(?:de\s+)?"
         r"(americana|pepp?eroni|hawaiana|suprema)s?(?:\s+(personales|personal|medianas|mediana|familiares|familiar))?")
GASEOSA = rf"{CANT}\s+gaseosas?(?:\s+de\s+(?:litro y medio|1[.,]5 litros?))?"
PAN = rf"{CANT}\s+(?:panes|pan)\s+al\s+ajo"


def _tamano(s):
    if not s:
        return None
    return {"personal": "personal", "mediana": "mediana", "familiar": "familiar"}[re.sub(r"(es|s)$", "", s)
                                                                                 if s not in ("personal",) else s]


def _items_en(segmento: str) -> list[dict]:
    """Ítems mencionados en un tramo de texto, en orden de aparición."""
    hallados = []
    for m in re.finditer(PIZZA, segmento):
        prod = "pepperoni" if m[3].startswith("pep") else m[3]
        hallados.append((m.start(), {"producto": prod, "tamano": _tamano(m[2] or m[4]), "cantidad": _cantidad(m[1]),
                                     "evidencia": m[0]}))
    for patron, prod in ((GASEOSA, "gaseosa 1.5L"), (PAN, "pan al ajo")):
        for m in re.finditer(patron, segmento):
            hallados.append((m.start(), {"producto": prod, "tamano": None, "cantidad": _cantidad(m[1]),
                                         "evidencia": m[0]}))
    return [i for _, i in sorted(hallados, key=lambda x: x[0])]


def _items_dialogo(t: str) -> list[dict]:
    """Sin confirmación del vendedor: recorre la llamada aplicando las correcciones.

    - Cada mención de un mismo producto y tamaño fija su cantidad (no suma): así el vendedor
      que repite el pedido no lo duplica.
    - "cambie la X por (otra|una) Y" reemplaza el ítem que coincide con X.
    - "mejor que sean N" cambia la cantidad del último ítem mencionado.
    """
    pedido, ultimo = {}, None
    for oracion in re.split(r"(?<=[.?!])\s+", t):
        if m := re.search(r"cambi\w*(?:la|lo)?\s+(?:la |el )?(\w+) por (?:otra|otro|una|un) ([^.?!]+)", oracion):
            viejo, nuevo = m[1], _items_en("1 " + m[2])
            clave_vieja = next((k for k in pedido if viejo.startswith(k[0][:5]) or (k[1] and viejo.startswith(k[1][:5]))),
                               None)
            if clave_vieja:
                anterior = pedido.pop(clave_vieja)
                prod, tam = clave_vieja
                tam_nuevo = _tamano(re.search(r"personal|mediana|familiar", m[2])[0]) \
                    if re.search(r"personal|mediana|familiar", m[2]) else tam
                prod_nuevo = nuevo[0]["producto"] if nuevo else prod
                k = (prod_nuevo, tam_nuevo)
                pedido[k] = {**anterior, "producto": prod_nuevo, "tamano": tam_nuevo,
                             "cantidad": pedido.get(k, {"cantidad": 0})["cantidad"] + anterior["cantidad"],
                             "evidencia": m[0]}
                ultimo = k
            continue
        for item in _items_en(oracion):
            k = (item["producto"], item["tamano"])
            pedido[k] = item
            ultimo = k
        if (m := re.search(r"mejor que sean (\d+)", oracion)) and ultimo in pedido:
            pedido[ultimo] = {**pedido[ultimo], "cantidad": int(m[1]), "evidencia": m[0]}
    return list(pedido.values())


def extraer_pedido(transcripcion: str, cfg: dict) -> dict:
    t = texto_trabajo(transcripcion)
    carta = cfg["carta"]
    # El vendedor suele repetir el pedido final ("le confirmo..."): si está, es la fuente más limpia.
    conf = list(re.finditer(r"(?:le )?confirmo(?: el pedido)?:?", t))
    if conf:
        segmento = t[conf[-1].end():]
        segmento = segmento[:m.start()] if (m := re.search(r"(?:el total|serían|sería)", segmento)) else segmento
        items, fuente = _items_en(segmento), "confirmacion"
    else:
        segmento, fuente = t, "dialogo"
    if not conf or not items:
        items, fuente = _items_dialogo(t), "dialogo"
    for it in items:
        it["precio_unitario"] = _precio(it, carta)
    zona_promo = segmento if fuente == "confirmacion" else t
    items_campo = _campo(items, " | ".join(i["evidencia"] for i in items))
    if fuente == "confirmacion":
        items_campo["no_reconocido"] = _no_reconocido(segmento)
    return {
        "items": items_campo,
        "fuente_items": fuente,
        "promocion": _promocion(zona_promo, carta),
        "direccion_entrega": _direccion(t),
        "metodo_pago": _pago(t, carta["metodos_pago"]),
        "total_dicho": _total(t),
    }


def _no_reconocido(segmento: str) -> list[str]:
    """Menciones con cantidad y tamaño ("3 avallanas medianas") que ningún ítem explica.

    Si el modelo escribió mal un producto, el ítem se pierde en silencio y la suma sale baja:
    sin esta revisión, el sistema culparía a quien habló de un error que es del modelo.
    """
    cubiertos = [m.span() for p in (PIZZA, GASEOSA, PAN) for m in re.finditer(p, segmento)]
    sueltos = []
    for m in re.finditer(rf"{CANT}\s+(?:\w+\s+){{0,2}}?(?:personal|mediana|familiar)\w*", segmento):
        if not any(a <= m.start() < b for a, b in cubiertos):
            sueltos.append(m[0])
    return sueltos


def _precio(item, carta):
    if item["producto"] in carta["pizzas"]:
        return carta["pizzas"][item["producto"]].get(item["tamano"]) if item["tamano"] else None
    return carta["extras"].get(item["producto"])


def _promocion(t, carta):
    if m := re.search(r"d[uú]o familiar", t):
        return _campo({"nombre": "duo familiar", "descuento": carta["promociones"]["duo familiar"]}, m[0])
    if m := re.search(r"cup[oó]n bienvenida", t):
        return _campo({"nombre": "cupon BIENVENIDA", "descuento": carta["promociones"]["cupon BIENVENIDA"]}, m[0])
    return _campo({"nombre": "ninguna", "descuento": 0})


VIAS = r"(?:avenida|av\.?|jirón|jiron|jr\.?|calle|manzana|mz\.?|pasaje|psje\.?)"


def _direccion(t):
    for m in re.finditer(rf"\b{VIAS}\s[^.?!]{{1,80}}", t):
        if re.search(r"\d", m[0]):
            return _campo(m[0].strip(" ,"), m[0].strip(" ,"))
    return _campo(None)


def _pago(t, metodos):
    zona = _zona(t, r"pag[ao]r?\b|pago|medio de pago")
    claves = {"efectivo": r"efectivo", "yape": r"\byape", "plin": r"\bplin", "tarjeta": r"tarjeta"}
    for texto in (zona, t):
        hallados = [(m.start(), metodo, m[0]) for metodo, p in claves.items() if metodo in metodos
                    for m in [re.search(p, texto)] if m]
        if hallados:
            _, metodo, ev = min(hallados)
            return _campo(metodo, ev)
    return _campo(None)


def _total(t):
    patrones = [r"total (?:es|son|sería|serían|seria|serian|queda en|de) (?:de )?(\d+)",
                r"(\d+) soles en total", r"serían (\d+) soles", r"total[:,]? (\d+)"]
    hallados = [m for p in patrones for m in re.finditer(p, t)]
    if not hallados:
        return _campo(None)
    m = max(hallados, key=lambda m: m.start())
    return _campo(int(m[1]), m[0])


def comparable_direccion(d: str | None) -> str:
    """Dirección en forma comparable: abreviaturas expandidas y sin tildes ni signos."""
    if not d:
        return ""
    d = clave(d)
    for corto, largo in {"av": "avenida", "jr": "jiron", "mz": "manzana", "dpto": "departamento",
                         "urb": "urbanizacion", "psje": "pasaje", "of": "oficina"}.items():
        d = re.sub(rf"\b{corto}\b", largo, d)
    return d
