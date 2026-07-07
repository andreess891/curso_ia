
import httpx

from dotenv import load_dotenv,find_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

#  +--------------------------------------------------+
#  | Para levantar la aplicación, ejecutar el comando:|
#  | streamlit run 5.1.streamlit_chatbot_tarea.py     |
#  +--------------------------------------------------+

#Configurar la pagina de la aplicación
st.set_page_config(page_title="Chatbot básico", page_icon="🤖")
st.title("Chatbot básico con Streamlit usando LangChain")
st.markdown("Este es un ejemplo de un *chatbot básico* utilizando Streamlit y LangChain con el modelo de lenguaje de OpenAI.")

# Configuración del modelo y la temperatura en una barra lateral
with st.sidebar:
    st.header("Configuración del modelo")
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=2)

    # ¡Nuevo! Personalidad configurable
    personalidad = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal", 
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )

    # Recrear el modelo con nuevos parámetros
    chat_model = ChatOpenAI(
        model=model_name, 
        temperature=temperature,
        http_client=http_client
    )

# Incialización del historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

system_messages = {
    "Útil y amigable": "Eres un asistente útil y amigable llamado ChatBot Pro. Responde de manera clara y concisa.",
    "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
    "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y amigable, como un buen amigo.",
    "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas con precisión técnica.",
    "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías, ejemplos creativos y mantén un tono alegre."
}

# Creamos la plantilla de prompt para el chatbot.
chat_prompt = ChatPromptTemplate([
    ("system", system_messages[personalidad]),
    ("human", "Historial de conversación:\n{historial}\n\nPregunta actual: {mensaje}")
])

# Usamos el operador "|" para encadenar la plantilla con el modelo de lenguaje, pasando el mensaje del usuario y el historial de mensajes como variables de entrada a la plantilla.
chain = chat_prompt | chat_model

# Mostar mensajes previos en la interfaz de streamlit
for msg in st.session_state.messages:
   if isinstance(msg, SystemMessage):
       continue #no muestro el mensaje por pantalla
   
   role = "assistant" if isinstance(msg, AIMessage) else "user"

   with st.chat_message(role):
       st.markdown(msg.content)

if st.button("🗑️ Nueva conversación"):
    st.session_state.messages = []
    st.rerun()

# Cuadtro de entrada de texto de usuario
pregunta = st.chat_input("Escribe tu mensaje: ")

if pregunta:
    # Mostrar inmediatamente el mensaje del usuario en la interfaz de streamlit
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Streaming de la respuesta
            for chunk in chain.stream({"mensaje": pregunta, "historial": st.session_state.messages}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")