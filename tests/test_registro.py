"""Pruebas sin GPU ni audio: corren en la laptop y en GitHub Actions.

    pytest -q
"""
import re

import pytest

from registro import metricas as M
from registro.config import cfg, ruta
from registro.motor import procesar
from registro.texto import normalizar, numeros_a_digitos, texto_trabajo

VERDAD1 = M.cargar_verdad(ruta("verdad") / "verdad_tarea1.csv")
VERDAD2 = M.cargar_verdad(ruta("verdad") / "verdad_tarea2.csv")


def ref(clip):
    return (ruta("verdad") / "transcripciones" / f"{clip}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("dicho, digitos", [
    ("cuatro mil quinientos soles", "4500 soles"),
    ("mil doscientos cuarenta y cinco", "1245"),
    ("noventa y un soles", "91 soles"),
    ("ciento trece", "113"),
    ("una pizza y dos gaseosas", "una pizza y 2 gaseosas"),
    ("un año y medio", "un año y medio"),
])
def test_numeros(dicho, digitos):
    assert numeros_a_digitos(dicho) == digitos


def test_formatos_de_whisper():
    assert texto_trabajo("Pido S/ 4,500.00 al mes") == "pido 4500 soles al mes"
    assert texto_trabajo("US$ 2.800") == "2800 dólares"
    assert texto_trabajo("un veinte por ciento") == "un 20 por ciento"
    assert normalizar("¡Cuatro mil quinientos, entendido!") == normalizar("4500 entendido")


@pytest.mark.parametrize("clip", sorted(VERDAD1))
def test_entrevista_con_texto_perfecto(clip):
    r = procesar(transcripcion=ref(clip), esquema="tarea1")
    assert r["error"] is None
    assert all(c["ok"] for c in M.comparar_entrevista(r["campos_verificados"], VERDAD1[clip]).values())
    assert set(r["estados"].values()) == {"verificado"}


@pytest.mark.parametrize("clip", sorted(VERDAD2))
def test_pedido_con_texto_perfecto(clip):
    r = procesar(transcripcion=ref(clip), esquema="tarea2")
    assert r["error"] is None
    assert all(c["ok"] for c in M.comparar_pedido(r["campos_verificados"], VERDAD2[clip]).values())
    esperado = "ok" if VERDAD2[clip]["trampa_total_mal"] == "no" else "error_hablante"
    assert r["validacion"]["diagnostico"] == esperado


@pytest.mark.parametrize("clip", sorted(VERDAD2))
def test_pedido_sin_confirmacion_aplica_correcciones(clip):
    """Si el vendedor no repite el pedido, se recorre la llamada aplicando los cambios del cliente."""
    texto = "\n".join(l for l in ref(clip).splitlines() if "confirmo" not in l.lower())
    r = procesar(transcripcion=texto, esquema="tarea2")
    assert r["campos"]["fuente_items"] == "dialogo"
    assert M.comparar_pedido(r["campos_verificados"], VERDAD2[clip])["items"]["ok"]


def test_valor_inventado_queda_sin_verificar():
    from registro.verificar import verificar_campo
    t = "Mi pretensión es de seis mil soles."
    assert verificar_campo("monto", {"valor": {"monto": 6000, "moneda": "PEN"}, "evidencia": "6000 soles"}, t)[1] == \
        "verificado"
    assert verificar_campo("monto", {"valor": {"monto": 7000, "moneda": "PEN"}, "evidencia": "6000 soles"}, t)[1] == \
        "sin_verificar"
    assert verificar_campo("monto", {"valor": {"monto": 7000, "moneda": "PEN"}, "evidencia": "7000 soles"}, t)[1] == \
        "sin_verificar"


def test_total_cambiado_a_mano_dispara_alerta():
    texto = re.sub(r"noventa y un soles", "ochenta soles", ref("t2_llamada_01"))
    r = procesar(transcripcion=texto, esquema="tarea2")
    assert r["validacion"]["alerta"] and r["validacion"]["diagnostico"] == "error_hablante"


def test_producto_mal_oido_es_error_del_modelo():
    """Caso real de Whisper small en CPU: oyó "avallanas" en vez de "hawaianas"."""
    texto = ref("t2_llamada_03").replace("hawaianas", "avallanas")
    r = procesar(transcripcion=texto, esquema="tarea2")
    assert r["estados"]["items"] == "sin_verificar"
    assert r["validacion"]["diagnostico"] == "error_asr"


def test_config_sin_credenciales():
    """config.yaml no debe tener claves: esas van en .env."""
    texto = (ruta("verdad").parents[1] / "config.yaml").read_text(encoding="utf-8")
    assert not re.search(r"sk-ant-\w|api_key\s*:", texto, re.IGNORECASE)
    assert cfg()["llm"]["modelo"]
