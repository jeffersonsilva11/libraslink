# app.py
import streamlit as st
import cv2
import numpy as np
import time
from reconhecedor import ReconhecedorGestosEEmocao

# 1) Página e título
st.set_page_config(page_title="👋 Libras Live", layout="wide")
st.title("🎥 Libras Live – Chat de Gestos e Emoções")
st.markdown("Clique em **Iniciar Detecção** para abrir a câmera e começar a detecção.")

# 2) Carrega recursos
@st.cache_resource
def load_resources():
    return ReconhecedorGestosEEmocao(
        model_dir="modelos",
        sequence_length=10,
        threshold=0.6,
        display_mode="sequence"
    )
recognizer = load_resources()

# 3) Placeholders para vídeo, chat e status
video_placeholder = st.empty()
chat_placeholder = st.empty()
status_placeholder = st.empty()

# 4) Botões de controle alinhados
col1, col2 = st.columns(2)
start = col1.button("Iniciar Detecção")
stop  = col2.button("Parar Detecção")

# 5) Lista de mensagens em memória
messages = []

# 6) Loop de captura/OpenCV
# Nota: em simulação de dispositivo (DevTools mobile), cv2.VideoCapture roda no servidor e pode não acessar a câmera do cliente.
def run_detection():
    cap = cv2.VideoCapture(0)
    status_placeholder.info("Status: Capturando 📹")
    while cap.isOpened():
        if stop:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # Detecção leve: a cada 5 frames
        if int(time.time() * 10) % 5 == 0:
            keypoints, _ = recognizer.detectar_keypoints(frame)
            gesture, conf = recognizer.prever_gesto(keypoints)
            emotion, _ = recognizer.detectar_emocao(frame)
            if gesture and conf > recognizer.threshold:
                if not messages or messages[-1]["text"] != gesture:
                    timestamp = time.strftime("%H:%M:%S")
                    messages.append({
                        "text": gesture,
                        "emotion": emotion,
                        "timestamp": timestamp
                    })
        # Exibe vídeo responsivo
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        # Exibe chat abaixo do vídeo com texto preto
        chat_html = (
            "<div style='height:40vh; overflow-y:auto; border:1px solid #ddd;"
            " padding:8px; border-radius:8px; background:#f8f9fa;'>"
        )
        for msg in messages:
            emoji = "😊" if msg["emotion"] == "feliz" else "😐"
            chat_html += (
                f"<p style='margin:0; padding:4px; color:#000;'>"
                f"<strong>{msg['text']}</strong> {emoji} <em>{msg['timestamp']}</em></p>"
            )
        chat_html += "</div>"
        chat_placeholder.markdown(chat_html, unsafe_allow_html=True)
    cap.release()
    video_placeholder.empty()
    status_placeholder.success("Status: Parado 🛑")

if start:
    run_detection()
else:
    status_placeholder.info("Status: Aguardando início ⏳")

# Observação: Para testar em dispositivos móveis sem travar, use a versão com Streamlit-WebRTC.