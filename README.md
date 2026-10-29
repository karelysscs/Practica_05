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

> Se llenan con `python scripts/evaluar.py` cuando estén las transcripciones. Ver `resultados/*.csv`.

**Tarea 1 · Transcripción** (`resultados/tarea1_asr_resumen.csv`)

| Experimento | Dispositivo real | WER (IC 95%) | CER | s/min | VRAM pico |
|---|---|---|---|---|---|
| pendiente | | | | | |

**Tarea 1 · Registro** (`resultados/tarea1_registro_resumen.csv`)

| Fuente | Método | Acierto global (IC 95%) | F1 macro inglés | Sin verificar |
|---|---|---|---|---|
| texto perfecto | reglas | 1.00 (0.86–1.00) | 1.00 | 0 |
| pendiente | | | | |

**Tarea 2 · Escalera** (`resultados/tarea2_resumen.csv`)

| Nivel | WER | Acierto campos | Rotos | Atrapados | No atrapados |
|---|---|---|---|---|---|
| texto perfecto | 0 | 1.00 | 3 | 3 | 0 |
| pendiente | | | | | |

## Qué NO se puede concluir con este corpus

Son 4 clips por tarea (8 en total), con 1-2 voces. Los intervalos de confianza son anchos: una
diferencia de WER entre modelos menor a lo que cubre el IC no se puede afirmar. No dice nada sobre
otros acentos, ruido real distinto al probado, ni conversaciones sin guion.

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
