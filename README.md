# Práctica 05 · De la voz al registro

Reconocimiento de voz (ASR) aplicado a dos tareas, con **un solo motor**:

- **Tarea 1 · Entrevistas:** de una entrevista de trabajo grabada sale la ficha del candidato
  (años de experiencia, último puesto, tecnologías, pretensión salarial, disponibilidad, nivel de inglés).
- **Tarea 2 · Llamadas a una pizzería:** de una llamada sale el pedido, que **se valida solo**:
  Σ cantidad × precio − promoción tiene que dar el total dicho. Si no cuadra, el sistema dice si
  el error es del modelo (oyó mal) o de quien habló (sumó mal).

**Regla de oro:** un valor solo entra al registro si aparece literalmente en la transcripción.
Si no, queda como `sin_verificar`.

- 📄 Documento del pipeline (público): **[enlace a Notion — pendiente]** · borrador en [docs/pipeline.md](docs/pipeline.md)
- 🎬 Video: **[enlace — pendiente]**

## Datos inventados y consentimiento

- **Todo el contenido de los audios es inventado**: no hay sueldos, direcciones, empresas ni datos
  personales reales. La pizzería "Tono" y sus precios no existen.
- **Todas las personas grabadas dieron su consentimiento** para ser grabadas y para que los audios se
  usen en esta práctica. No aparece nadie que no haya aceptado.

## Estructura

```
config.yaml               rutas, modelos, umbrales, carta y prompts (nada de esto está en el código)
.env.example              nombres de las credenciales (el .env real no se sube)
registro/                 el motor y sus piezas
  motor.py                procesar(audio, esquema) → transcripción, campos, verificados, dispositivo, tiempo, error
  asr.py                  faster-whisper + medición real de dispositivo y VRAM (NVML)
  texto.py                normalización y números dichos → dígitos
  reglas.py               extracción con expresiones regulares (camino 1)
  llm.py                  extracción con LLM, JSON con evidencia (camino 2)
  verificar.py            verificación literal → sin_verificar
  validar.py              tarea 2: validación aritmética y diagnóstico
  metricas.py             WER, CER, acierto por campo, F1 macro, IC bootstrap y Wilson
  degradar.py             escalera L0-L3
scripts/                  referencias.py · transcribir.py · degradar.py · evaluar.py
notebooks/colab_gpu.ipynb la parte que necesita GPU
app.py                    app de Streamlit (lee resultados ya calculados)
data/guiones/             guiones de grabación (.md y .pdf)
data/verdad/              respuestas correctas y transcripciones de referencia
resultados/               caché de transcripciones (sí se sube) y métricas
tests/                    pruebas (corren sin GPU)
```

## Instalación (laptop, sin GPU)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (en Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # opcional: solo para el camino LLM
```

## Cómo se corre

**1. Transcribir con GPU (Colab).** Abre [notebooks/colab_gpu.ipynb](notebooks/colab_gpu.ipynb) en Colab,
activa la GPU T4 y ejecuta todo. Al final baja `resultados.zip` y descomprímelo aquí.

**2. Métricas (laptop):**

```bash
python scripts/evaluar.py          # reglas
python scripts/evaluar.py --llm    # reglas + LLM (necesita ANTHROPIC_API_KEY en .env, o la caché de resultados/llm/)
```

**3. App:**

```bash
streamlit run app.py
```

**Comando de verificación del motor** (sin GPU, con una transcripción ya hecha):

```bash
python -m registro.motor --texto data/verdad/transcripciones/t2_llamada_03.txt --esquema tarea2
```

Con un audio (en CPU, modelo liviano):

```bash
python -m registro.motor data/audio/t1_entrevista_01.m4a --esquema tarea1 --laptop
```

**Pruebas:** `pytest -q`

## Audios y modelos

Los audios **no se suben** a GitHub (`.gitignore`). Para regenerar los resultados:
grabar con los guiones de `data/guiones/`, dejar los archivos en `data/audio/` como
`t1_entrevista_01.m4a … t2_llamada_04.m4a` (o en Google Drive para Colab) y correr el notebook.
Los pesos de Whisper se descargan solos de Hugging Face la primera vez. Las transcripciones en
caché (`resultados/transcripciones/`) sí están en el repo: con ellas se reproducen todas las métricas.

## Normalización (la misma para referencia y modelo)

1. minúsculas; 2. números en palabras → dígitos ("cuatro mil quinientos" → 4500) y sin separador de miles;
3. "S/" → soles, "$" → dólares; 4. sin puntuación. **Las tildes se mantienen.**
Para comparar valores de campos además se ignoran tildes y mayúsculas.

## Resultados

Todo sale de `python scripts/evaluar.py` sobre la caché de transcripciones (ver `resultados/*.csv`).
Transcripción hecha en Google Colab con una **Tesla T4** (faster-whisper 1.2.1).

**Tarea 1 · Transcripción** (`resultados/tarea1_asr_resumen.csv`, 4 entrevistas)

| Experimento | Dispositivo real | WER (IC 95%) | CER | s de cómputo por min de audio | VRAM pico |
|---|---|---|---|---|---|
| small, float16 | cuda:0 | 0.104 (0.063–0.148) | 0.049 | 1.7 | 884 MB |
| large-v3, float16 | cuda:0 | **0.061** (0.052–0.068) | 0.041 | 6.6 | 4948 MB |
| large-v3, int8_float16 | cuda:0 | 0.066 (0.042–0.084) | 0.046 | 7.4 | 3412 MB |
| large-v3, float16 + glosario (hotwords) | cuda:0 | 0.088 (0.041–0.168) | 0.071 | 5.9 | 5044 MB |
| small, int8 · CPU | cpu:0 | 0.103 (0.071–0.133) | 0.048 | 48.9 | — |
| large-v3, int8 · CPU (1 clip) | cpu:0 | 0.048 | 0.036 | 126.3 | — |

- **Prueba de GPU:** el dispositivo es el que reporta el modelo ya cargado (CTranslate2), no el que se pidió,
  y la VRAM se mide con NVML para el proceso que transcribe. Si se pide GPU y corre en CPU, el script se detiene.
- **CPU vs GPU** (mismo modelo small): 48.9 vs 1.7 s por minuto de audio → la GPU es **~29 veces más rápida**.
  Nota: la medición de small en CPU se hizo en la laptop (Intel i5-1235U), no en la CPU de Colab.
- **float16 vs int8:** el texto cambió en los 4 clips (WER entre ambas versiones: 0.032); int8 usa ~30% menos VRAM
  con un WER parecido.
- **Glosario** (mecanismo `hotwords`, porque faster-whisper 1.2.1 lo soporta): cambio de WER +0.028,
  IC 95% [−0.015, +0.098]. El IC cruza 0: con 4 clips **no se puede afirmar** que empeore el texto.

**Tarea 1 · Registro** (`resultados/tarea1_registro_resumen.csv`, 6 campos × 4 entrevistas, extracción por reglas)

| Transcripción usada | Campos correctos (IC 95% Wilson) | F1 macro inglés | Sin verificar |
|---|---|---|---|
| texto perfecto (referencia) | 1.00 (0.86–1.00) | 1.00 | 0 |
| large-v3 float16 + glosario | **0.83** (0.64–0.93) | 0.89 | 0 |
| large-v3 int8 | 0.79 (0.60–0.91) | 0.89 | 0 |
| large-v3 float16 | 0.75 (0.55–0.88) | 0.89 | 0 |
| small CPU | 0.67 (0.47–0.82) | 0.89 | 0 |
| small GPU | 0.63 (0.43–0.79) | 0.89 | 0 |

- **El WER no predice si la ficha sirve.** Con glosario el WER sube (0.061 → 0.088) pero la ficha mejora
  (75% → 83%), porque el glosario acierta justo en las palabras que importan (tecnologías, puestos). Por eso
  el motor usa glosario por defecto: se eligió por la ficha, no por el WER.
- Clip con WER alto y ficha perfecta: entrevista 1 con large-v3 int8 (WER 0.073, 6/6 campos).
  Clip con WER bajo y ficha rota: entrevista 4 con large-v3 int8 (WER 0.032, 4/6): el modelo escribió
  "experiencias" en vez de "experiencia" y "Adobe" sin "XD".
- Errores típicos del modelo: "Desamanzado" por "avanzado", "bucket" por "backend", "Git" no reconocido.

**Tarea 2 · Escalera de degradación** (`resultados/tarea2_resumen.csv`, 5 campos × 4 llamadas, large-v3 float16 + glosario)

| Nivel | WER | Campos correctos (IC 95%) | Registros rotos | Atrapados | No atrapados | Diagnóstico correcto |
|---|---|---|---|---|---|---|
| texto perfecto | 0 | 1.00 (0.84–1.00) | 3 | 3 | 0 | 3/3 |
| L0 original | 0.047 | 0.90 (0.70–0.97) | 3 | 3 | 0 | 3/3 |
| L1 nota de voz (Opus 12k) | 0.080 | 0.90 (0.70–0.97) | 3 | 3 | 0 | 3/3 |
| L2 teléfono (300–3400 Hz, 8 kHz) | 0.090 | 0.85 (0.64–0.95) | 3 | 3 | 0 | 3/3 |
| L3 = L1 + ruido a 5 dB SNR | 0.101 | 0.85 (0.64–0.95) | 4 | 4 | 0 | 3/4 |

- "Roto" = el registro no se puede usar tal cual (algún campo mal extraído o un total mal dicho).
  "Atrapado" = la validación dio alerta o algún campo quedó `sin_verificar`.
- **Lo que la validación atrapa:** los 3 totales mal dichos a propósito, en todos los niveles.
- **Lo que no distingue bien:** en L3 el modelo oyó "90" en vez de "91" en la llamada 1. La suma no cuadró
  y saltó la alerta, pero el diagnóstico culpó a quien habló, porque el número mal oído igual "aparecía" en la
  transcripción. Un número mal oído y verificado se ve igual que un error de quien habló.
- La escalera no llega a un registro perfecto en L3, así que no hizo falta endurecerla más.
- El ruido de L3 es **ruido rosa sintético** (no se grabó ruido ambiente); la SNR medida después de mezclar es 5.0 dB.

## Qué NO se puede concluir con este corpus

Son 4 clips por tarea (8 en total), con 1-2 voces. Los intervalos de confianza son anchos: una
diferencia de WER entre modelos menor a lo que cubre el IC no se puede afirmar. No dice nada sobre
otros acentos, ruido real distinto al probado, ni conversaciones sin guion. El experimento large-v3 en CPU
tiene un solo clip, así que su F1 de inglés (0.33) no significa nada.

## Limitaciones

- Las reglas de la tarea 2 dependen de que el vendedor confirme el pedido; si no lo hace, se usa
  un recorrido de la llamada que entiende "cambie X por Y" y "mejor que sean N", pero no cualquier corrección.
- Las tecnologías se buscan en un vocabulario cerrado (config.yaml).
- Sin separación de hablantes: no se sabe quién dijo cada cosa.

## Agentes de IA y servidores MCP

| Herramienta | Uso |
|---|---|
| Claude Code (app de escritorio de Claude) | Ayudó a escribir guiones, código, pruebas y documentación. Revisé y entiendo cada módulo. |
| (completar) | (servidores MCP conectados en Claude Code, una línea cada uno) |
