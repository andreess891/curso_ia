import httpx

from dotenv import load_dotenv,find_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Para levantar la aplicación, ejecutar el comando:
# streamlit run 5.streamlit_chatbot.py

#Configurar la pagina de la aplicación
st.set_page_config(page_title="Chatbot básico", page_icon="🤖")
st.title("Chatbot básico con Streamlit usando LangChain")
st.markdown("Este es un ejemplo de un *chatbot básico* utilizando Streamlit y LangChain con el modelo de lenguaje de OpenAI.")

chat_model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.5,
    http_client=http_client
)

# Incialización del historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostar mensajes previos en la interfaz de streamlit
for msg in st.session_state.messages:
   if isinstance(msg, SystemMessage):
       #no muestro el mensaje por pantalla
       continue
   
   role = "assistant" if isinstance(msg, AIMessage) else "user"

   with st.chat_message(role):
       st.markdown(msg.content)

# Cuadtro de entrada de texto de usuario
pregunta = st.chat_input("Escribe tu mensaje: ")

if pregunta:
    # Mostrar inmediatamente el mensaje del usuario en la interfaz de streamlit
    with st.chat_message("user"):
        st.markdown(pregunta)

    # almacenamos el mensaje del usuario en la memoria de la sesión
    st.session_state.messages.append(HumanMessage(content=pregunta))

    # Generar una respuesta usando el modelo de lenguaje
    respuesta = chat_model.invoke(st.session_state.messages)

    # Mostrar la respuesta del modelo en la interfaz de streamlit
    with st.chat_message("assistant"):
        st.markdown(respuesta.content)

    # Almacenar la respuesta del modelo en la memoria de la sesión
    st.session_state.messages.append(AIMessage(content=respuesta.content))