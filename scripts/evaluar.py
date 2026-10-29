"""Calcula todas las métricas desde la caché de transcripciones. No usa GPU ni audio.

    python scripts/evaluar.py          # solo reglas
    python scripts/evaluar.py --llm    # además el camino LLM (usa resultados/llm/ como caché)

Escribe en resultados/:
  tarea1_asr.csv             WER, CER, s/min, VRAM y dispositivo por experimento y clip
  tarea1_asr_resumen.csv     lo mismo resumido por experimento, con IC bootstrap del WER
  tarea1_registro.csv        acierto por campo, clip, método y fuente de la transcripción
  tarea1_registro_resumen.csv
  tarea2_escalera.csv        por clip y nivel: WER, campos, validación, roto/atrapado
  tarea2_resumen.csv         por nivel
  resumen.json               cifras para el README y el video
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registro import metricas as M  # noqa: E402
from registro.config import cfg, ruta  # noqa: E402
from registro.motor import ESQUEMAS, procesar  # noqa: E402
from registro.texto import normalizar  # noqa: E402

C = cfg()
MB = C["metricas"]
REF = ruta("verdad") / "transcripciones"


def referencia(clip: str) -> str | None:
    p = REF / f"{clip}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def cache(experimento: str) -> dict[str, dict]:
    carpeta = ruta("transcripciones") / experimento
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob("*.json"))}


def nombre_defecto() -> str:
    d = C["asr"]["defecto"]
    for e in C["experimentos"]:
        if all(e[k] == d[k] for k in ("modelo", "dispositivo", "precision", "glosario")):
            return e["nombre"]
    return C["experimentos"][0]["nombre"]


# ───────────────────────────────────────────────────────── tarea 1, capa 1
def tarea1_asr(resumen: dict):
    filas = []
    for e in C["experimentos"]:
        for clip, r in cache(e["nombre"]).items():
            ref = referencia(clip)
            filas.append({"experimento": e["nombre"], "modelo": e["modelo"], "dispositivo_pedido": e["dispositivo"],
                          "dispositivo_real": r["dispositivo_real"], "gpu": r.get("gpu"), "precision": e["precision"],
                          "glosario": r.get("glosario") or "no", "clip": clip,
                          "wer": M.wer(ref, r["texto"]) if ref else None, "cer": M.cer(ref, r["texto"]) if ref else None,
                          "segundos": r["segundos"], "seg_por_minuto": r["seg_por_minuto"],
                          "vram_pico_mb": r.get("vram_pico_mb")})
    df = pd.DataFrame(filas)
    if df.empty:
        print("tarea 1: no hay transcripciones en caché todavía")
        return df
    df.to_csv(ruta("resultados") / "tarea1_asr.csv", index=False)
    res = []
    for exp, g in df.groupby("experimento", sort=False):
        w, lo, hi = M.bootstrap_media(g["wer"].dropna(), MB["bootstrap_reps"], MB["confianza"], MB["semilla"])
        res.append({"experimento": exp, "modelo": g["modelo"].iat[0], "dispositivo_real": g["dispositivo_real"].iat[0],
                    "precision": g["precision"].iat[0], "glosario": g["glosario"].iat[0], "clips": len(g),
                    "wer": round(w, 4), "wer_ic_bajo": round(lo, 4), "wer_ic_alto": round(hi, 4),
                    "cer": round(g["cer"].mean(), 4), "seg_por_minuto": round(g["seg_por_minuto"].mean(), 2),
                    "vram_pico_mb": g["vram_pico_mb"].max()})
    pd.DataFrame(res).to_csv(ruta("resultados") / "tarea1_asr_resumen.csv", index=False)

    # ¿cambia el texto entre precisiones? (mismo modelo y glosario, solo cambia float16 → int8)
    a, b = cache("large_cuda_fp16"), cache("large_cuda_int8")
    comunes = sorted(set(a) & set(b))
    if comunes:
        resumen["precision"] = {
            "clips": len(comunes),
            "clips_con_texto_distinto": sum(normalizar(a[c]["texto"]) != normalizar(b[c]["texto"]) for c in comunes),
            "wer_entre_precisiones": round(sum(M.wer(a[c]["texto"], b[c]["texto"]) for c in comunes) / len(comunes), 4),
        }
    # glosario vs sin glosario: diferencia pareada de WER por clip, con IC bootstrap
    s, g_ = cache("large_cuda_fp16"), cache("large_cuda_fp16_glos")
    comunes = [c for c in sorted(set(s) & set(g_)) if referencia(c)]
    if comunes:
        dif = [M.wer(referencia(c), g_[c]["texto"]) - M.wer(referencia(c), s[c]["texto"]) for c in comunes]
        m, lo, hi = M.bootstrap_media(dif, MB["bootstrap_reps"], MB["confianza"], MB["semilla"])
        resumen["glosario"] = {"mecanismo": next(iter(g_.values())).get("glosario"), "clips": len(comunes),
                               "dif_wer_media": round(m, 4), "ic": [round(lo, 4), round(hi, 4)],
                               "lectura": "negativo = el glosario baja el WER; si el IC cruza 0 no se puede afirmar"}
    # CPU vs GPU con el mismo modelo
    cpu, gpu = df[df.experimento == "small_cpu_int8"], df[df.experimento == "small_cuda_fp16"]
    if len(cpu) and len(gpu):
        resumen["cpu_vs_gpu_small"] = {"seg_por_min_cpu": round(cpu.seg_por_minuto.mean(), 2),
                                       "seg_por_min_gpu": round(gpu.seg_por_minuto.mean(), 2),
                                       "veces_mas_rapido": round(cpu.seg_por_minuto.mean() / gpu.seg_por_minuto.mean(), 1)}
    return df


# ───────────────────────────────────────────────────────── tarea 1, capa 2
def tarea1_registro(metodos: list[str], df_asr: pd.DataFrame, resumen: dict):
    verdad = M.cargar_verdad(ruta("verdad") / "verdad_tarea1.csv")
    mejor = cache(nombre_defecto())
    fuentes = {"referencia": {c: referencia(c) for c in verdad if referencia(c)},
               f"asr:{nombre_defecto()}": {c: r["texto"] for c, r in mejor.items()}}
    filas = []
    for fuente, textos in fuentes.items():
        for metodo in metodos:
            for clip, texto in textos.items():
                if clip not in verdad:
                    continue
                r = procesar(transcripcion=texto, esquema="tarea1", metodo=metodo, cfg=C)
                if r["error"]:
                    print(f"  {clip} {metodo}: {r['error']}")
                    continue
                cmp = M.comparar_entrevista(r["campos_verificados"], verdad[clip])
                bruto = M.comparar_entrevista({k: v["valor"] for k, v in r["campos"].items()}, verdad[clip])
                fila = {"fuente": fuente, "metodo": metodo, "clip": clip,
                        "wer": M.wer(referencia(clip), texto) if referencia(clip) else None}
                for campo, c in cmp.items():
                    fila[f"{campo}_ok"] = c["ok"]
                    fila[f"{campo}_ok_sin_verificar"] = bruto[campo]["ok"]
                    fila[f"{campo}_estado"] = r["estados"][campo]
                    if "error_abs" in c:
                        fila[f"{campo}_error_abs"] = c["error_abs"]
                fila["ingles_pred"], fila["ingles_real"] = cmp["nivel_ingles"]["pred"], cmp["nivel_ingles"]["real"]
                fila["tecnologias_f1"] = cmp["tecnologias"]["f1"]
                fila["campos_ok"] = sum(c["ok"] for c in cmp.values())
                filas.append(fila)
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df.to_csv(ruta("resultados") / "tarea1_registro.csv", index=False)
    campos = list(ESQUEMAS["tarea1"]["tipos"])
    res = []
    for (fuente, metodo), g in df.groupby(["fuente", "metodo"]):
        fila = {"fuente": fuente, "metodo": metodo, "clips": len(g)}
        aciertos = int(sum(g[f"{c}_ok"].sum() for c in campos))
        p, lo, hi = M.wilson(aciertos, len(g) * len(campos), MB["confianza"])
        fila |= {"acierto_global": round(p, 3), "ic_bajo": round(lo, 3), "ic_alto": round(hi, 3)}
        for c in campos:
            fila[c] = round(g[f"{c}_ok"].mean(), 3)
        fila["f1_macro_ingles"] = round(M.f1_macro(g["ingles_real"].tolist(), g["ingles_pred"].tolist(),
                                                   C["extraccion"]["niveles_ingles"]), 3)
        fila["error_abs_anios"] = g["anios_experiencia_error_abs"].mean()
        fila["error_abs_pretension"] = g["pretension_salarial_error_abs"].mean()
        fila["sin_verificar"] = int(sum((g[f"{c}_estado"] == "sin_verificar").sum() for c in campos))
        res.append(fila)
    pd.DataFrame(res).to_csv(ruta("resultados") / "tarea1_registro_resumen.csv", index=False)

    # Clips para mostrar que el WER no predice si el registro sirve
    asr = df[(df.fuente != "referencia") & (df.metodo == "reglas")].dropna(subset=["wer"])
    if len(asr):
        perfectos, rotos = asr[asr.campos_ok == len(campos)], asr[asr.campos_ok < len(campos)]
        resumen["wer_vs_registro"] = {
            "wer_alto_registro_perfecto": perfectos.sort_values("wer").iloc[-1][["clip", "wer", "campos_ok"]].to_dict()
            if len(perfectos) else None,
            "wer_bajo_registro_roto": rotos.sort_values("wer").iloc[0][["clip", "wer", "campos_ok"]].to_dict()
            if len(rotos) else None,
        }
    return df


# ───────────────────────────────────────────────────────── tarea 2
def tarea2(metodos: list[str], resumen: dict):
    verdad = M.cargar_verdad(ruta("verdad") / "verdad_tarea2.csv")
    filas = []
    fuentes = [("referencia", "texto perfecto", {c: referencia(c) for c in verdad if referencia(c)})]
    for n in C["degradacion"]["niveles"]:
        textos = {r["clip"].rsplit(f"_{n}", 1)[0]: r["texto"] for r in cache(f"escalera_{n}").values()}
        if textos:
            fuentes.append((n, f"escalera_{n}", textos))
    for nivel, origen, textos in fuentes:
        for metodo in metodos:
            for clip, texto in textos.items():
                if clip not in verdad:
                    continue
                v = verdad[clip]
                r = procesar(transcripcion=texto, esquema="tarea2", metodo=metodo, cfg=C)
                if r["error"]:
                    print(f"  {clip} {nivel} {metodo}: {r['error']}")
                    continue
                cmp = M.comparar_pedido(r["campos_verificados"], v)
                val = r["validacion"]
                campos_ok = sum(c["ok"] for c in cmp.values())
                extraccion_ok = campos_ok == len(cmp)
                total_mal = v["trampa_total_mal"] == "si"
                roto = (not extraccion_ok) or total_mal  # el registro no sirve tal cual: hay que revisarlo
                sin_verif = [k for k, e in r["estados"].items() if e == "sin_verificar"]
                atrapado = val["alerta"] or bool(sin_verif)
                if val["alerta"] and val["cuadra"] is not None:
                    causa_real = "error_hablante" if (cmp["items"]["ok"] and cmp["promocion"]["ok"]
                                                      and cmp["total_dicho"]["ok"] and total_mal) else "error_asr"
                elif val["alerta"]:
                    causa_real = "error_asr"
                else:
                    causa_real = None
                fila = {"nivel": nivel, "metodo": metodo, "clip": clip,
                        "wer": M.wer(referencia(clip), texto) if referencia(clip) else None,
                        "cer": M.cer(referencia(clip), texto) if referencia(clip) else None,
                        **{f"{k}_ok": c["ok"] for k, c in cmp.items()}, "campos_ok": campos_ok,
                        "fuente_items": r["campos"].get("fuente_items"), "sin_verificar": ",".join(sin_verif),
                        "suma_calculada": val["suma_calculada"], "total_dicho": val["total_dicho"],
                        "total_correcto": float(v["total_correcto"]), "cuadra": val["cuadra"],
                        "diagnostico": val["diagnostico"], "causa_real": causa_real,
                        "diagnostico_ok": None if causa_real is None else val["diagnostico"] == causa_real,
                        "roto": roto, "atrapado": atrapado,
                        "trampa_total_mal": total_mal, "trampa_cambio_pedido": v["trampa_cambio_pedido"] == "si"}
                filas.append(fila)
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df.to_csv(ruta("resultados") / "tarea2_escalera.csv", index=False)
    res = []
    for (nivel, metodo), g in df.groupby(["nivel", "metodo"], sort=False):
        p, lo, hi = M.wilson(int(g.campos_ok.sum()), len(g) * len(ESQUEMAS["tarea2"]["tipos"]), MB["confianza"])
        d = g.dropna(subset=["diagnostico_ok"])
        res.append({"nivel": nivel, "metodo": metodo, "clips": len(g),
                    "wer": round(g.wer.mean(), 4) if g.wer.notna().any() else None,
                    "acierto_campos": round(p, 3), "ic_bajo": round(lo, 3), "ic_alto": round(hi, 3),
                    "rotos": int(g.roto.sum()), "rotos_atrapados": int((g.roto & g.atrapado).sum()),
                    "rotos_no_atrapados": int((g.roto & ~g.atrapado).sum()),
                    "falsas_alarmas": int((~g.roto & g.atrapado).sum()),
                    "diagnostico_acierto": f"{int(d.diagnostico_ok.sum())}/{len(d)}" if len(d) else "-"})
    rdf = pd.DataFrame(res)
    rdf.to_csv(ruta("resultados") / "tarea2_resumen.csv", index=False)
    resumen["tarea2"] = rdf.to_dict("records")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", action="store_true", help="incluir el camino LLM (necesita ANTHROPIC_API_KEY o caché)")
    a = p.parse_args()
    metodos = ["reglas"] + (["llm"] if a.llm else [])
    ruta("resultados").mkdir(parents=True, exist_ok=True)
    resumen = {}
    df = tarea1_asr(resumen)
    tarea1_registro(metodos, df, resumen)
    tarea2(metodos, resumen)
    (ruta("resultados") / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")
    for f in ["tarea1_asr_resumen.csv", "tarea1_registro_resumen.csv", "tarea2_resumen.csv"]:
        if (ruta("resultados") / f).exists():
            print(f"\n== {f}\n" + pd.read_csv(ruta("resultados") / f).to_string(index=False))


if __name__ == "__main__":
    main()
