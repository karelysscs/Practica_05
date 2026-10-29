"""Limpieza de texto y números dichos en español → dígitos.

Regla de normalización (la misma para la transcripción de referencia y la del modelo):
1. minúsculas y Unicode NFC;
2. números escritos en palabras → dígitos ("cuatro mil quinientos" → "4500") y
   separadores de miles quitados ("4.500" → "4500"), para que "4500" y "cuatro mil quinientos"
   no cuenten como error;
3. símbolos de moneda a palabras ("S/" → "soles", "$" → "dólares");
4. sin puntuación ni espacios repetidos. Las tildes se quedan.
"""
from __future__ import annotations

import re
import unicodedata

UNIDADES = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9,
}
ESPECIALES = {  # 10-29: se escriben en una palabra
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintiún": 21, "veintiuna": 21, "veintidós": 22, "veintidos": 22, "veintitrés": 23, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}
DECENAS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
           "noventa": 90}
CENTENAS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
            "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500,
            "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700, "setecientas": 700,
            "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900}
# "un", "una" y "uno" solos no se convierten: "una pizza" o "un año y medio" deben quedar como están.
# Solo cuentan dentro de un número más largo ("noventa y un", "un mil").
AMBIGUOS = {"un", "una", "uno"}


def _clase(p: str) -> int:
    if p in CENTENAS:
        return 3
    if p in DECENAS:
        return 2
    if p in ESPECIALES:
        return 1
    if p in UNIDADES:
        return 0
    return -1


def _valor(p: str) -> int:
    return CENTENAS.get(p) or DECENAS.get(p) or ESPECIALES.get(p) or UNIDADES[p]


def numeros_a_digitos(texto: str) -> str:
    """'mil doscientos cuarenta y cinco' → '1245'; 'noventa y un soles' → '91 soles'."""
    partes = re.findall(r"\w+|\W+", texto)
    salida, i = [], 0
    while i < len(partes):
        p = partes[i].lower()
        siguiente = _siguiente_palabra(partes, i)
        empieza = (_clase(p) >= 0 and p not in AMBIGUOS) or p == "mil" or (p in AMBIGUOS and siguiente == "mil")
        if not empieza:
            salida.append(partes[i])
            i += 1
            continue
        total, bloque, ultima_clase, j, fin = 0, 0, 9, i, i
        while j < len(partes):
            q = partes[j].lower()
            if not re.match(r"\w", q):  # espacios entre palabras del número
                if q.strip():
                    break
                j += 1
                continue
            if q == "mil":
                total += (bloque or 1) * 1000
                bloque, ultima_clase, fin = 0, 9, j
            elif q == "y" and ultima_clase == 2 and _clase(_siguiente_palabra(partes, j)) == 0:
                pass  # "cuarenta y cinco": la "y" une decena y unidad
            elif _clase(q) >= 0 and _clase(q) < ultima_clase and not (ultima_clase == 2 and _clase(q) == 1):
                if ultima_clase == 2 and partes[j - 2].lower() != "y" and _clase(q) == 0:
                    break  # "treinta cinco" sin "y" no es un número
                bloque += _valor(q)
                ultima_clase, fin = _clase(q), j
            else:
                break
            j += 1
        salida.append(str(total + bloque))
        i = fin + 1
    return "".join(salida)


def _siguiente_palabra(partes, i):
    for p in partes[i + 1:]:
        if re.match(r"\w", p):
            return p.lower()
        if p.strip():
            return ""
    return ""


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def texto_trabajo(texto: str) -> str:
    """Texto para extraer campos: minúsculas, números en dígitos y monedas en palabras, con puntuación."""
    t = unicodedata.normalize("NFC", texto.lower())
    t = t.replace("…", "...")
    t = re.sub(r"\bus\s?\$\s?", "$", t)
    t = re.sub(r"\bs/\.?\s?(\d[\d.,]*)", r"\1 soles", t)
    t = re.sub(r"\$\s?(\d[\d.,]*)", r"\1 dólares", t)
    t = re.sub(r"(\d)[.,]00\b", r"\1", t)                    # 4500.00 → 4500
    t = re.sub(r"\b(\d{1,3})(?:[.,](\d{3}))+\b", lambda m: m.group(0).replace(".", "").replace(",", ""), t)
    t = re.sub(r"(\d+)\s?%", r"\1 por ciento", t)
    t = re.sub(r"(\d+)\.?º", r"\1", t)                         # 1.º de diciembre → 1 de diciembre
    t = numeros_a_digitos(t.replace("por ciento", "por_ciento")).replace("por_ciento", "por ciento")
    return re.sub(r"[ \t]+", " ", t).strip()


def normalizar(texto: str) -> str:
    """Forma canónica para comparar textos (WER, CER y verificación literal)."""
    t = texto_trabajo(texto)
    t = re.sub(r"(\d)[.,](\d)", r"\1_\2", t)                   # 1.5 no se parte en "1 5"
    t = re.sub(r"[^\w\s]", " ", t).replace("_", ".")
    return re.sub(r"\s+", " ", t).strip()


def clave(texto: str) -> str:
    """Para comparar valores sin importar tildes ni mayúsculas: 'Dúo Familiar' == 'duo familiar'."""
    return sin_tildes(normalizar(texto))
