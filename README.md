# 🤖 Curso de Inteligencia Artificial: LangChain, RAG y LangGraph

Curso práctico para aprender a construir aplicaciones con Modelos de Lenguaje (LLMs), desde los fundamentos teóricos hasta la construcción de agentes con **LangChain**, sistemas de recuperación de información (**RAG**) y flujos multi-agente con **LangGraph**.

Todo el código se ejecuta con [**uv**](https://docs.astral.sh/uv/), el gestor de entornos y dependencias de Python.

---

## 📚 Contenido del curso

1. **Introducción a la Inteligencia Artificial**
   - Historia y modelos fundacionales
   - Arquitectura Transformers, tokenización y embeddings
2. **Ingeniería de Prompt**
   - Componentes de un prompt efectivo (rol, tarea, contexto, formato)
   - Técnicas: Zero-Shot, Few-Shot, Chain of Thought, Least-to-Most, ReAct, Self-Consistency
3. **LangChain**
   - LangChain Expression Language (LCEL), `Runnables` y `RunnableParallel`
   - Prompt Templates (`ChatPromptTemplate`, `MessagesPlaceholder`)
   - Output Parsers y estructuras de salida con Pydantic
4. **RAG (Retrieval-Augmented Generation)**
   - Document Loaders, Text Splitters
   - Embeddings, Vector Stores y Retrievers
5. **LangGraph**
   - Componentes de un grafo y primer programa
   - Agentes y uso de herramientas (Tools) con LLMs

---

## 🗂️ Estructura del repositorio

```
curso_ia/
├── notebooks/                  # Notebooks teóricos y de práctica por módulo
│   ├── modulo_2.ipynb          # LangChain, LCEL, Prompt Templates, Output Parsers
│   ├── modulo_3_RAG.ipynb      # Document loaders, embeddings, vector stores, retrievers
│   ├── langgraph.ipynb         # Grafos y agentes con LangGraph
│   └── recursos/               # PDFs de ejemplo usados en los notebooks (contratos, Quijote)
├── src/
│   └── modulo_1/                # Proyectos y ejercicios prácticos
│       ├── 00.proyecto_modulo_2/   # Chatbot básico con Streamlit
│       ├── 01.proyecto_modulo_2/   # Ejercicio de Prompt Templates
│       └── 03.cv_analyzer/         # Proyecto: analizador de hojas de vida (Streamlit)
├── main.py
├── pyproject.toml              # Dependencias del proyecto
└── uv.lock                     # Lockfile de dependencias (uv)
```

---

## ✅ Requisitos previos

- **Python 3.12** (definido en `.python-version`)
- **uv** instalado ([guía oficial de instalación](https://docs.astral.sh/uv/getting-started/installation/))
- Claves de API de los proveedores de LLM que se usarán en el curso:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Google AI Studio](https://aistudio.google.com/app/apikey) (Gemini)

### Instalar uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verifica la instalación:
```bash
uv --version
```

---

## 🚀 Paso a paso: puesta en marcha

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd curso_ia
```

### 2. Instalar las dependencias con uv

`uv` crea automáticamente el entorno virtual (`.venv`) e instala todo lo definido en `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

### 3. Configurar las variables de entorno

Crea un archivo `.env` en la raíz del proyecto (puedes copiar este bloque) con tus propias claves:

```env
OPENAI_API_KEY=tu_api_key_de_openai
GOOGLE_API_KEY=tu_api_key_de_google
```

> ⚠️ El archivo `.env` contiene información sensible: nunca lo subas al repositorio (ya está ignorado en `.gitignore`).

### 4. Seleccionar el intérprete / kernel del entorno

Si vas a trabajar con los notebooks en VS Code o Jupyter, selecciona como kernel el intérprete generado por `uv` en `.venv` (`.venv\Scripts\python.exe` en Windows).

Para abrir Jupyter Lab directamente con uv:

```bash
uv run jupyter lab
```

---

## ▶️ Cómo ejecutar cada parte del curso

### Notebooks (teoría y práctica guiada)

Abre cualquiera de los notebooks en `notebooks/` (`modulo_2.ipynb`, `modulo_3_RAG.ipynb`, `langgraph.ipynb`) con el kernel del `.venv` y ejecútalos celda a celda.

### Chatbot básico con Streamlit

```bash
uv run streamlit run "src/modulo_1/00.proyecto_modulo_2/01_chatbot.py"
```

### Proyecto CV Analyzer

```bash
cd "src/modulo_1/03.cv_analyzer"
uv run streamlit run app.py
```

### Script principal

```bash
uv run main.py
```

---

## 🛠️ Comandos útiles de uv

| Acción | Comando |
|---|---|
| Instalar/sincronizar dependencias | `uv sync` |
| Añadir una nueva dependencia | `uv add <paquete>` |
| Ejecutar un script dentro del entorno | `uv run <archivo.py>` |
| Ejecutar Jupyter Lab | `uv run jupyter lab` |
| Activar el entorno manualmente (Windows) | `.venv\Scripts\activate` |
