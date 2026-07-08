# ⚖️ Asistente Legal RAG

Un chatbot que responde preguntas sobre **contratos de arrendamiento** (vivienda, locales de negocio, plazas de garaje) leyendo directamente los PDFs de esos contratos y citando la información encontrada. Está construido con **RAG (Retrieval-Augmented Generation)**, LangChain, OpenAI y Streamlit.

Si estás empezando en IA, piensa en este proyecto como un ejemplo completo y pequeño de "cómo hacer que un modelo de lenguaje conteste preguntas sobre TUS documentos" en vez de solo lo que aprendió durante su entrenamiento.

---

## 🧠 ¿Qué es RAG y por qué lo necesitamos?

Un modelo como GPT-4o **no ha leído tus contratos de arrendamiento**. Si le preguntas directamente "¿cuál es la dirección del local arrendado en el contrato X?", no tiene forma de saberlo (o peor, se lo inventa: esto se llama **alucinación**).

**RAG (Retrieval-Augmented Generation)** resuelve esto en dos pasos:

1. **Retrieval (recuperación):** buscamos, dentro de nuestros propios documentos, los fragmentos de texto más relacionados con la pregunta del usuario.
2. **Augmented Generation (generación aumentada):** le pasamos esos fragmentos al modelo de lenguaje como "contexto" y le pedimos que responda **basándose únicamente en ellos**.

De esta forma el modelo no adivina: responde con datos reales extraídos de los PDFs.

```
Pregunta del usuario
        │
        ▼
 1) Buscar fragmentos relevantes en la base de datos vectorial (Retrieval)
        │
        ▼
 2) Insertar esos fragmentos + la pregunta en un prompt
        │
        ▼
 3) El LLM genera la respuesta usando SOLO esa información (Generation)
        │
        ▼
      Respuesta + fragmentos usados (para verificar)
```

---

## 📁 Estructura del proyecto

```
asistente_legal_RAG/
├── app.py          # Interfaz de usuario (Streamlit) - lo que ve el usuario
├── rag_system.py   # El "cerebro": construye y ejecuta la cadena RAG
├── prompts.py       # Las instrucciones (prompts) que se le dan al LLM
├── config.py        # Parámetros configurables (modelos, tamaños de búsqueda, etc.)
└── README.md         # Este archivo
```

Los PDFs de contratos que alimentan al sistema están en `src/modulo_3/contratos/`. Para poder consultarlos, primero deben haberse convertido en una base de datos vectorial (Chroma) guardada en la carpeta `chroma_db` (ver sección [Puesta en marcha](#-puesta-en-marcha)).

---

## 🔍 Conceptos clave usados en el proyecto

Antes de ver el código, conviene entender estas piezas. Todas aparecen en `rag_system.py`.

### 1. Embeddings (`OpenAIEmbeddings`)
Un **embedding** convierte un texto en una lista de números (un vector) que representa su "significado". Dos textos con significados parecidos tienen vectores parecidos. Esto es lo que permite *buscar por significado* en vez de buscar palabras exactas.

- Modelo usado: `text-embedding-3-large` (ver `config.py`).

### 2. Vector Store (`Chroma`)
Es la "base de datos" donde se guardan esos vectores (uno por cada fragmento de los contratos PDF). Cuando llega una pregunta, también se convierte en vector y se comparan matemáticamente para encontrar los fragmentos más parecidos.

- Ruta local: `CHROMA_DB_PATH = "chroma_db"`.

### 3. Retrievers: las distintas formas de "buscar"
El proyecto no usa una única estrategia de búsqueda, sino que combina varias (esto es más avanzado que un RAG básico, ¡buen ejemplo para aprender!):

| Retriever | ¿Qué hace? | ¿Para qué sirve aquí? |
|---|---|---|
| **Similarity** | Devuelve los fragmentos matemáticamente más parecidos a la pregunta. | Es la búsqueda "directa", rápida y simple. |
| **MMR** (*Maximal Marginal Relevance*) | Busca fragmentos relevantes mientras **evita traer resultados repetidos/similares entre sí**. | Da respuestas más completas cuando la información está repartida en varios contratos. |
| **MultiQueryRetriever** | Usa un LLM (`gpt-4o-mini`) para **reformular la pregunta original en 3 variantes distintas** y busca con cada una. | Compensa que el usuario pueda preguntar "¿quién es el arrendatario?" cuando el PDF dice "inquilino". Aumenta las probabilidades de encontrar el fragmento correcto. |
| **EnsembleRetriever** (búsqueda híbrida) | Combina los resultados de MMR+MultiQuery (peso 0.7) y Similarity (peso 0.3) en una sola lista final. | Aprovecha lo mejor de ambos enfoques: diversidad + precisión. |

Todo esto se controla con `ENABLE_HYBRID_SEARCH` en `config.py`.

### 4. Dos modelos de lenguaje, cada uno con un rol distinto
- `QUERY_MODEL = "gpt-4o-mini"` → un modelo pequeño y económico que **solo reformula preguntas** para el MultiQueryRetriever.
- `GENERATION_MODEL = "gpt-4o"` → un modelo más potente que **redacta la respuesta final** al usuario.

Es una buena práctica: usar modelos baratos para tareas simples y modelos más caros solo donde aportan más calidad.

### 5. Prompts (`prompts.py`)
Los prompts son las "instrucciones" escritas que guían al LLM. Aquí hay varios, aunque solo dos se usan activamente en la app:

- `RAG_TEMPLATE`: el prompt final que arma la respuesta. Le dice al modelo "responde SOLO con la información de estos fragmentos, cita literalmente cuando sea posible, indica si falta información...".
- `MULTI_QUERY_PROMPT`: instruye al modelo pequeño sobre cómo generar las 3 variantes de la pregunta.
- `RELEVANCE_PROMPT` y `ENTITY_EXTRACTION_PROMPT`: están definidos pero **no se usan actualmente** en `rag_system.py` — quedan como ejemplos/plantillas para extender el proyecto (por ejemplo, filtrar documentos irrelevantes o extraer entidades como nombres y fechas de forma estructurada).

### 6. La cadena RAG (LangChain Expression Language)
En `rag_system.py`, la lógica completa se define de forma declarativa con el operador `|` (parecido a una tubería/pipe de Unix):

```python
rag_chain = (
    {
        "context": final_retriever | format_docs,   # 1. Busca y formatea los documentos
        "question": RunnablePassthrough()             # 2. Deja pasar la pregunta tal cual
    }
    | prompt              # 3. Rellena la plantilla RAG_TEMPLATE
    | llm_generation      # 4. El LLM genera la respuesta
    | StrOutputParser()   # 5. Convierte la respuesta en texto plano
)
```

Cada `|` conecta la salida de un paso con la entrada del siguiente. Es la forma "moderna" de LangChain de construir flujos de IA.

---

## 🖥️ La interfaz (`app.py`)

Construida con [Streamlit](https://streamlit.io/), una librería que convierte scripts de Python en aplicaciones web sin necesidad de HTML/CSS/JS.

- **Barra lateral:** muestra qué tipo de retriever y modelos están activos, y un botón para limpiar el chat.
- **Columna izquierda:** el chat en sí (`st.chat_input`, `st.chat_message`), con el historial guardado en `st.session_state`.
- **Columna derecha:** muestra los fragmentos de documentos usados para responder la última pregunta (fuente, página y contenido), para que el usuario pueda **verificar** de dónde salió la respuesta — algo esencial en un contexto legal.

Flujo cuando el usuario escribe una pregunta:
1. Se guarda el mensaje del usuario en el historial.
2. Se llama a `query_rag(prompt)` (definido en `rag_system.py`).
3. Se guarda la respuesta del asistente junto con los documentos usados.
4. `st.rerun()` refresca la página para mostrar todo.

---

## ⚙️ `config.py`: los "controles" del sistema

```python
EMBEDDING_MODEL = "text-embedding-3-large"  # convierte texto → vectores
QUERY_MODEL = "gpt-4o-mini"                 # reformula preguntas (barato)
GENERATION_MODEL = "gpt-4o"                 # redacta la respuesta final (potente)

CHROMA_DB_PATH = "chroma_db"                # dónde vive la base de datos vectorial

SEARCH_TYPE = "mmr"        # estrategia de búsqueda principal
MMR_DIVERSITY_LAMBDA = 0.7 # 0 = máxima diversidad, 1 = máxima relevancia
MMR_FETCH_K = 20           # candidatos iniciales que evalúa MMR
SEARCH_K = 2                # fragmentos finales que se devuelven

ENABLE_HYBRID_SEARCH = True  # combina MMR+MultiQuery con Similarity
SIMILARITY_THRESHOLD = 0.70
```

Cambiar estos valores es la forma más fácil de experimentar sin tocar el código: por ejemplo, subir `SEARCH_K` para traer más contexto, o bajar `MMR_DIVERSITY_LAMBDA` para priorizar diversidad sobre relevancia exacta.

---

## 🚀 Puesta en marcha

### 1. Requisitos previos
- Python 3.12+ (el proyecto usa [`uv`](https://docs.astral.sh/uv/) como gestor de dependencias, ver `pyproject.toml` en la raíz del repo).
- Una **API Key de OpenAI** válida, con acceso a `gpt-4o`, `gpt-4o-mini` y `text-embedding-3-large`.
- Un archivo `.env` en la raíz del proyecto con:
  ```
  OPENAI_API_KEY=tu_api_key_aqui
  ```

### 2. Generar la base de datos vectorial (`chroma_db`)
Este proyecto **espera encontrar** una carpeta `chroma_db` ya construida en el mismo directorio desde donde se ejecuta `app.py`, con los embeddings de los PDFs de `src/modulo_3/contratos/`. Si no existe todavía, hay que indexar los contratos primero (cargar los PDFs con un `PyPDFLoader`, dividirlos en fragmentos y guardarlos en `Chroma` con `OpenAIEmbeddings`) antes de lanzar la app.

### 3. Ejecutar la aplicación
Desde esta carpeta (`src/modulo_3/asistente_legal_RAG`):

```bash
uv run streamlit run app.py
```

Streamlit abrirá una pestaña del navegador (normalmente en `http://localhost:8501`).

### 4. Probarlo
Escribe una pregunta como:

> ¿Qué personas participan en los contratos de arrendamiento de locales comerciales?

y observa en la columna derecha qué fragmentos de los PDFs se usaron para construir la respuesta.

---

## 🧩 Resumen del flujo completo (de punta a punta)

```
Usuario escribe una pregunta en Streamlit
        │
        ▼
MultiQueryRetriever (gpt-4o-mini) genera 3 variantes de la pregunta
        │
        ▼
Cada variante se busca en Chroma con MMR (y también con Similarity, si el
modo híbrido está activo) → se recuperan fragmentos de los contratos PDF
        │
        ▼
EnsembleRetriever combina y pondera todos los resultados
        │
        ▼
Los fragmentos se formatean con format_docs() (fuente + página + texto)
        │
        ▼
Se insertan en RAG_TEMPLATE junto con la pregunta original
        │
        ▼
gpt-4o genera la respuesta final, basada solo en esos fragmentos
        │
        ▼
Streamlit muestra la respuesta + los fragmentos usados como evidencia
```

---

## 💡 Ideas para seguir aprendiendo/extendiendo el proyecto

- Usar `RELEVANCE_PROMPT` para filtrar automáticamente fragmentos poco relevantes antes de generar la respuesta.
- Usar `ENTITY_EXTRACTION_PROMPT` para mostrar en la barra lateral las entidades clave (nombres, fechas, importes) de cada contrato.
- Añadir memoria conversacional real (que el asistente recuerde preguntas anteriores del mismo chat al buscar contexto).
- Mostrar métricas de relevancia (score) de cada fragmento recuperado.
- Sustituir Chroma por otro vector store (FAISS, Pinecone, etc.) sin cambiar el resto del sistema, gracias a que LangChain abstrae esa capa.
