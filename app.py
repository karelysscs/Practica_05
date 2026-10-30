"""App de la práctica 05: de la voz al registro.

    streamlit run app.py

Carga resultados ya calculados (resultados/*.csv y la caché de transcripciones): no transcribe al abrir.
Solo la pestaña "Probar" llama al motor, y únicamente si tú se lo pides.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from registro.config import cfg, ruta
from registro.motor import ESQUEMAS, procesar

st.set_page_config(page_title="De la voz al registro", page_icon="🎙️", layout="wide")
C = cfg()
RES = ruta("resultados")


@st.cache_data
def csv(nombre):
    p = RES / nombre
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def resumen():
    p = RES / "resumen.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def referencia(clip):
    p = ruta("verdad") / "transcripciones" / f"{clip}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def transcripcion_cache(clip):
    """La mejor transcripción guardada del clip: asr.defecto para la tarea 1, L0 para la tarea 2."""
    base = ruta("transcripciones")
    candidatos = [base / "large_cuda_fp16_glos" / f"{clip}.json", base / "escalera_L0" / f"{clip}_L0.json"]
    for p in candidatos + sorted(base.glob(f"*/{clip}*.json")):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


# Lo que ve la persona: los nombres internos (sin tildes) no cambian, solo cómo se muestran.
ETIQUETAS = {
    "clip": "Clip", "items": "Ítems", "promocion": "Promoción", "descuento": "Descuento (S/)",
    "direccion_entrega": "Dirección de entrega", "metodo_pago": "Método de pago", "total_dicho": "Total dicho",
    "total_correcto": "Total correcto", "cuadra": "¿Cuadra?", "trampa_cambio_pedido": "Trampa: cambio de pedido",
    "trampa_total_mal": "Trampa: total mal dicho", "trampa_direccion_digitos": "Trampa: dirección con números",
    "anios_experiencia": "Años de experiencia", "ultimo_puesto": "Último puesto", "tecnologias": "Tecnologías",
    "pretension_salarial": "Pretensión salarial", "moneda": "Moneda", "disponibilidad": "Disponibilidad",
    "nivel_ingles": "Nivel de inglés", "trampas": "Trampas", "experimento": "Experimento", "modelo": "Modelo",
    "dispositivo_real": "Dispositivo real", "precision": "Precisión", "glosario": "Glosario", "clips": "Clips",
    "wer": "WER", "wer_ic_bajo": "WER (IC bajo)", "wer_ic_alto": "WER (IC alto)", "cer": "CER",
    "seg_por_minuto": "Segundos por min de audio", "vram_pico_mb": "VRAM pico (MB)", "fuente": "Transcripción",
    "metodo": "Método", "acierto_global": "Campos correctos", "ic_bajo": "IC bajo", "ic_alto": "IC alto",
    "f1_macro_ingles": "F1 macro de inglés", "error_abs_anios": "Error absoluto en años",
    "error_abs_pretension": "Error absoluto en pretensión", "sin_verificar": "Sin verificar", "nivel": "Nivel",
    "acierto_campos": "Campos correctos", "rotos": "Registros rotos", "rotos_atrapados": "Rotos atrapados",
    "rotos_no_atrapados": "Rotos no atrapados", "falsas_alarmas": "Falsas alarmas",
    "diagnostico_acierto": "Diagnóstico correcto", "campos_ok": "Campos correctos", "suma_calculada": "Suma calculada",
    "diagnostico": "Diagnóstico", "causa_real": "Causa real", "roto": "Roto", "atrapado": "Atrapado",
    "detalle": "Detalle", "snr_db_medida": "SNR medida (dB)", "campo": "Campo", "valor": "Valor", "estado": "Estado",
    "evidencia": "Evidencia",
}
VALORES = {
    "si": "sí", "duo familiar": "dúo familiar", "cupon BIENVENIDA": "cupón BIENVENIDA", "basico": "básico",
    "yape": "Yape", "plin": "Plin", "PEN": "soles", "USD": "dólares", "error_hablante": "error de quien habló",
    "error_asr": "error del modelo", "sin_verificar": "sin verificar", "vacio": "vacío", "llm": "LLM",
    "referencia": "texto perfecto", True: "sí", False: "no",
}


def texto_items(items) -> str:
    """Lista de ítems en lenguaje normal: '2 pepperoni familiar a S/ 46; 1 gaseosa 1.5L a S/ 9'."""
    if isinstance(items, str):
        items = json.loads(items)
    partes = []
    for i in items:
        nombre = " ".join(x for x in (i["producto"], i.get("tamano")) if x)
        precio = i.get("precio_unitario")
        partes.append(f"{i['cantidad']} {nombre}" + (f" a S/ {precio}" if precio is not None else ""))
    return "; ".join(partes)


def bonito(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla lista para mostrar: nombres de columna y valores con ortografía correcta."""
    d = df.copy()
    for c in d.columns:
        if c == "items":
            d[c] = d[c].map(texto_items)
        elif c == "tecnologias" and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].str.replace("|", ", ", regex=False)
        elif d[c].dtype == bool or not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v: VALORES.get(v, v))
    return d.rename(columns=ETIQUETAS)


def valor_legible(campo, v) -> str:
    """Valor de un campo extraído (pestaña Probar), como lo leería una persona."""
    if campo == "items" and isinstance(v, list):
        return texto_items(v)
    if campo == "promocion" and isinstance(v, dict):
        nombre = VALORES.get(v["nombre"], v["nombre"])
        return f"{nombre} (−S/ {v['descuento']})" if v["descuento"] else nombre
    if campo == "pretension_salarial" and isinstance(v, dict):
        return f"{v['monto']} {VALORES[v['moneda']]}"
    if campo == "tecnologias" and isinstance(v, list):
        return ", ".join(v)
    if campo == "direccion_entrega" and isinstance(v, str) and v != "sin_verificar":
        return v.title()
    return "—" if v is None else str(VALORES.get(v, v))


def falta(nombre):
    st.info(f"Todavía no hay `{nombre}`. Corre el notebook de Colab y luego `python scripts/evaluar.py`.")


st.title("🎙️ De la voz al registro")
st.caption("Entrevistas de trabajo → ficha del candidato · Llamadas a una pizzería → pedido que se valida solo")

t_problema, t_corpus, t_resultados, t_pedidos, t_probar = st.tabs(
    ["1 · Problema y método", "2 · Corpus", "3 · Resultados", "4 · Pedidos (tarea 2)", "5 · Probar"])

# ───────────────────────────────────────────────────────── 1
with t_problema:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("El problema")
        st.markdown("""
Las herramientas de dictado (WhatsApp, Zoom, Otter...) entregan **texto**. Un área de recursos humanos
o un delivery necesita otra cosa: **campos en una tabla** y saber **cuánto confiar** en cada uno.

- **Tarea 1:** de una entrevista grabada sale la ficha del candidato (6 campos).
- **Tarea 2:** de una llamada sale el pedido, que además **se valida solo**: la suma de
  cantidad × precio, menos la promoción, tiene que dar el total que se dijo.

**Regla de oro:** un valor solo se acepta si aparece literalmente en la transcripción.
Si no, queda como `sin_verificar` y lo revisa una persona.
""")
    with c2:
        st.subheader("El flujo (un solo motor para las dos tareas)")
        st.graphviz_chart("""
digraph {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#f3f4f6", fontname="Helvetica"];
  audio [label="Audio (.m4a)"]; asr [label="faster-whisper\\n(GPU en Colab)"];
  cache [label="Caché de transcripciones\\n(resultados/transcripciones)"];
  norm [label="Normalizar\\n(números → dígitos)"]; ext [label="Extraer campos\\nreglas | LLM"];
  ver [label="Verificar contra\\nla transcripción"]; reg [label="Registro\\n+ estados"];
  val [label="Tarea 2: validar suma\\n¿error del modelo o de quien habló?"];
  audio -> asr -> cache -> norm -> ext -> ver -> reg -> val;
}""")
    st.subheader("Qué corre una vez y qué corre por cada audio")
    st.markdown("""
| Una vez | Por cada audio |
|---|---|
| Escribir guiones y tablas de respuestas correctas | Transcribir (GPU) y guardar en caché |
| Descargar el modelo de Whisper | Normalizar, extraer, verificar |
| Elegir el modelo con los experimentos de la fase 2 | Tarea 2: validar la suma y diagnosticar |
""")

# ───────────────────────────────────────────────────────── 2
with t_corpus:
    tarea = st.radio("Tarea", ["tarea1", "tarea2"], horizontal=True,
                     format_func=lambda t: "Tarea 1 · Entrevistas" if t == "tarea1" else "Tarea 2 · Llamadas")
    verdad = pd.read_csv(ruta("verdad") / f"verdad_{tarea}.csv")
    st.markdown("**Todo el contenido es inventado** y las personas grabadas dieron su consentimiento.")
    clip = st.selectbox("Clip", verdad["clip"].tolist())
    audio = next((p for p in sorted(ruta("audio").glob(f"{clip}.*"))), None)
    if audio:
        st.audio(str(audio))
    else:
        st.caption("El audio no está en esta computadora (no se sube a GitHub).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Transcripción de referencia** (lo que se dijo)")
        st.text_area("ref", referencia(clip), height=320, label_visibility="collapsed")
    with c2:
        r = transcripcion_cache(clip)
        st.markdown(f"**Transcripción del modelo** ({r['experimento'] if r else 'sin caché'})")
        st.text_area("asr", r["texto"] if r else "", height=320, label_visibility="collapsed")
    st.markdown("**Respuestas correctas de este clip**")
    st.dataframe(bonito(verdad[verdad["clip"] == clip]).T.rename(columns=lambda _: "Valor").astype(str),
                 use_container_width=True)
    st.markdown("**Distribución de los campos en el corpus**")
    cols = [c for c in verdad.columns if c not in ("clip", "items", "trampas")]
    distribucion = {ETIQUETAS.get(c, c): verdad[c].map(lambda v: VALORES.get(v, v)).value_counts().to_dict()
                    for c in cols if c != "tecnologias"}
    st.dataframe(pd.DataFrame(distribucion).T.fillna("").astype(str), use_container_width=True)

# ───────────────────────────────────────────────────────── 3
with t_resultados:
    st.subheader("Capa 1 · Transcripción (tarea 1)")
    asr = csv("tarea1_asr_resumen.csv")
    if asr is None:
        falta("resultados/tarea1_asr_resumen.csv")
    else:
        st.dataframe(bonito(asr), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.markdown("**Segundos de cómputo por minuto de audio** (menos es mejor)")
        c1.bar_chart(asr.set_index("experimento")["seg_por_minuto"])
        c2.markdown("**WER** (menos es mejor)")
        c2.bar_chart(asr.set_index("experimento")["wer"])
        R = resumen()
        if "cpu_vs_gpu_small" in R:
            x = R["cpu_vs_gpu_small"]
            st.success(f"Con el mismo modelo (small), la GPU fue **{x['veces_mas_rapido']}× más rápida**: "
                       f"{x['seg_por_min_gpu']} vs {x['seg_por_min_cpu']} s por minuto de audio.")
        if "precision" in R:
            x = R["precision"]
            st.write(f"float16 vs int8: el texto cambió en **{x['clips_con_texto_distinto']} de {x['clips']}** clips "
                     f"(WER entre ambas: {x['wer_entre_precisiones']}).")
        if "glosario" in R:
            x = R["glosario"]
            st.write(f"Glosario por `{x['mecanismo']}`: cambio de WER {x['dif_wer_media']:+.4f}, "
                     f"IC 95% [{x['ic'][0]:+.4f}, {x['ic'][1]:+.4f}] con {x['clips']} clips.")
    st.subheader("Capa 2 · Registro (tarea 1)")
    reg = csv("tarea1_registro_resumen.csv")
    if reg is None:
        falta("resultados/tarea1_registro_resumen.csv")
    else:
        st.dataframe(bonito(reg), use_container_width=True, hide_index=True)
        det = csv("tarea1_registro.csv")
        asr_det = det[det.fuente != "referencia"] if det is not None else None
        if asr_det is not None and len(asr_det):
            st.markdown("**¿El WER predice si la ficha sirve?** Cada punto es un clip.")
            st.scatter_chart(asr_det, x="wer", y="campos_ok", color="metodo")
        x = resumen().get("wer_vs_registro")
        if x:
            st.write("WER alto con ficha perfecta:", x.get("wer_alto_registro_perfecto"))
            st.write("WER bajo con ficha rota:", x.get("wer_bajo_registro_roto"))

# ───────────────────────────────────────────────────────── 4
with t_pedidos:
    st.subheader("Escalera de degradación")
    st.markdown("L0 original · L1 nota de voz (Opus a bajo bitrate) · L2 línea telefónica (300-3400 Hz, 8 kHz) · "
                "L3 = L1 + ruido de fondo")
    deg = csv("degradacion.csv")
    if deg is not None:
        st.dataframe(bonito(deg[deg.nivel == "L3"]), use_container_width=True, hide_index=True)
    res2 = csv("tarea2_resumen.csv")
    if res2 is None:
        falta("resultados/tarea2_resumen.csv")
    else:
        st.dataframe(bonito(res2), use_container_width=True, hide_index=True)
        niveles = res2[res2.nivel.str.startswith("L")]
        curva = niveles[niveles.metodo == "reglas"].set_index("nivel")
        c1, c2 = st.columns(2)
        c1.markdown("**WER por nivel**")
        c1.line_chart(curva["wer"])
        c2.markdown("**Campos correctos por nivel**")
        c2.line_chart(niveles.pivot(index="nivel", columns="metodo", values="acierto_campos")
                      .rename(columns=lambda m: "LLM" if m == "llm" else "Reglas"))
        st.subheader("Validación: registros rotos atrapados vs. no atrapados")
        st.bar_chart(curva[["rotos_atrapados", "rotos_no_atrapados", "falsas_alarmas"]].rename(columns=ETIQUETAS))
    esc = csv("tarea2_escalera.csv")
    if esc is not None:
        st.subheader("Detalle por llamada")
        st.dataframe(bonito(esc[["nivel", "metodo", "clip", "wer", "campos_ok", "suma_calculada", "total_dicho",
                                 "total_correcto", "diagnostico", "causa_real", "roto", "atrapado", "sin_verificar"]]),
                     use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────── 5
with t_probar:
    st.subheader("Probar el motor")
    st.markdown("Elige un clip o pega una transcripción. Para la demo: cambia el total dicho en una llamada "
                "y mira cómo salta la alerta.")
    c1, c2, c3 = st.columns(3)
    esquema = c1.selectbox("Esquema", list(ESQUEMAS), format_func=lambda e: {"tarea1": "Entrevista",
                                                                              "tarea2": "Pedido"}[e])
    metodo = c2.selectbox("Método", ["reglas", "llm"])
    prefijo = "t1_" if esquema == "tarea1" else "t2_"
    clips = sorted(p.stem for p in (ruta("verdad") / "transcripciones").glob(f"{prefijo}*.txt"))
    base = c3.selectbox("Empezar desde", ["(vacío)"] + clips)
    origen = st.radio("Texto", ["referencia", "modelo (caché)"], horizontal=True)
    inicial = ""
    if base != "(vacío)":
        r = transcripcion_cache(base)
        inicial = referencia(base) if origen == "referencia" or not r else r["texto"]
    texto = st.text_area("Transcripción", inicial, height=240, key=f"txt_{esquema}_{base}_{origen}")
    subido = st.file_uploader("…o sube un audio (se transcribe en esta computadora con asr.laptop: lento)",
                              type=["m4a", "mp3", "wav", "ogg"])
    if st.button("Procesar", type="primary"):
        with st.spinner("Procesando..."):
            if subido:
                tmp = RES / f"_subido{Path(subido.name).suffix}"
                tmp.write_bytes(subido.getvalue())
                r = procesar(str(tmp), esquema, metodo, C["asr"]["laptop"])
            else:
                r = procesar(transcripcion=texto, esquema=esquema, metodo=metodo)
        if r["error"]:
            st.error(r["error"])
        else:
            if r.get("dispositivo"):
                st.caption(f"Transcrito en {r['dispositivo']} en {r['segundos']} s")
                st.text_area("Transcripción obtenida", r["transcripcion"], height=150)
            filas = [{"Campo": ETIQUETAS.get(k, k), "Valor": valor_legible(k, v),
                      "Estado": VALORES.get(r["estados"][k], r["estados"][k]),
                      "Evidencia": r["campos"][k].get("evidencia", "")}
                     for k, v in r["campos_verificados"].items()]
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
            if "validacion" in r:
                v = r["validacion"]
                if v["diagnostico"] == "ok":
                    st.success(f"La cuenta cuadra: {v['suma_calculada']} = {v['total_dicho']}")
                elif v["diagnostico"] == "error_hablante":
                    st.error(f"No cuadra: la suma da {v['suma_calculada']} pero se dijo {v['total_dicho']}. "
                             "El registro está verificado, así que el error es de quien habló.")
                else:
                    st.warning("No se puede confirmar la cuenta: hay campos sin verificar. "
                               "Probablemente el modelo oyó mal. Revisar a mano.")
