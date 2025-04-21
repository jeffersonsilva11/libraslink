import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
import json
import os
from pathlib import Path
from collections import deque

class ReconhecedorGestosEEmocao:
    def __init__(self, model_dir="../modelos", sequence_length=10, threshold=0.6, display_mode="sequence"):
        """
        Inicializa o reconhecedor de gestos e emoções em tempo real
        
        Args:
            model_dir: Diretório contendo modelos treinados
            sequence_length: Tamanho da sequência para suavização
            threshold: Limiar de confiança para detecção de gestos
            display_mode: Modo de exibição ('single' ou 'sequence')
        """
        self.model_dir = Path(model_dir)
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.display_mode = display_mode
        
        # Carregar o modelo
        model_path = self.model_dir / "best_model.pkl"
        if not model_path.exists():
            # Tentar encontrar qualquer modelo disponível
            models = list(self.model_dir.glob("*_model.pkl"))
            if not models:
                raise FileNotFoundError(f"Nenhum modelo encontrado em {self.model_dir}")
            model_path = models[0]
            print(f"Modelo 'best_model.pkl' não encontrado. Usando {model_path.name} em vez disso.")
        
        # Carregar o modelo
        print(f"Carregando modelo de {model_path}...")
        self.model = joblib.load(model_path)
        
        # Carregar o normalizador (scaler)
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            print("Aviso: Scaler não encontrado. As características não serão normalizadas.")
        
        # Carregar o mapeamento de labels
        mapping_path = self.model_dir / "label_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                # Converter chaves para inteiros
                self.label_mapping = {int(k): v for k, v in mapping.items()}
        else:
            raise FileNotFoundError(f"Mapeamento de labels não encontrado em {mapping_path}")
        
        print(f"Mapeamento de gestos: {self.label_mapping}")
        
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Inicializar MediaPipe Face Mesh para detecção de expressões faciais
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pontos-chave para detecção de sorriso
        # Mapeamento dos índices relevantes para o rosto no MediaPipe Face Mesh
        self.face_landmarks_indices = {
            'left_mouth': 61,    # Canto esquerdo da boca
            'right_mouth': 291,  # Canto direito da boca
            'top_lip_center': 13,  # Centro do lábio superior
            'bottom_lip_center': 14,  # Centro do lábio inferior
            'left_cheek': 117,   # Bochecha esquerda
            'right_cheek': 346,  # Bochecha direita
            'nose_tip': 1        # Ponta do nariz (para referência)
        }
        
        # Fila para suavização de previsões
        self.predictions = deque(maxlen=self.sequence_length)
        self.last_recognized_gestures = deque(maxlen=5)  # Para sequências de gestos
        self.last_recognition_time = 0
        
        # Adicionar suavização para emoções
        self.emotion_predictions = deque(maxlen=self.sequence_length)
        self.current_emotion = None
        
    def detectar_keypoints(self, image, num_keypoints=21):
        """Detecta os pontos-chave das mãos na imagem"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        keypoints = []
        
        # Contar o número de mãos detectadas
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extrair informações de cada ponto
                # Calcular valores de referência para normalização
                landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] 
                                          for landmark in hand_landmarks.landmark])
                min_vals = np.min(landmarks_array, axis=0)
                max_vals = np.max(landmarks_array, axis=0)
                range_vals = max_vals - min_vals
                
                # Normalizar coordenadas
                for landmark in hand_landmarks.landmark:
                    # Normalização para range [0,1]
                    if np.any(range_vals > 0):
                        norm_x = (landmark.x - min_vals[0]) / range_vals[0] if range_vals[0] > 0 else 0
                        norm_y = (landmark.y - min_vals[1]) / range_vals[1] if range_vals[1] > 0 else 0
                        norm_z = (landmark.z - min_vals[2]) / range_vals[2] if range_vals[2] > 0 else 0
                    else:
                        norm_x, norm_y, norm_z = 0, 0, 0
                    
                    keypoints.extend([norm_x, norm_y, norm_z])
        
        # Padronizar o número de características
        while len(keypoints) < 3 * num_keypoints:
            keypoints.extend([0, 0, 0])
        
        # Truncar excesso
        keypoints = keypoints[:3 * num_keypoints]
        
        # Adicionar características derivadas (similar ao treinamento)
        if len(keypoints) >= 15:  # Certifique-se de que há pontos suficientes
            # Exemplo: distância entre a ponta do polegar e a ponta do indicador (simplificado)
            thumb_tip_idx = 4 * 3
            index_tip_idx = 8 * 3
            
            if thumb_tip_idx + 2 < len(keypoints) and index_tip_idx + 2 < len(keypoints):
                thumb_tip = np.array(keypoints[thumb_tip_idx:thumb_tip_idx+3])
                index_tip = np.array(keypoints[index_tip_idx:index_tip_idx+3])
                
                if np.any(thumb_tip) and np.any(index_tip):  # Se ambos os pontos forem diferentes de zero
                    dist = np.linalg.norm(thumb_tip - index_tip)
                    keypoints.append(dist)
                else:
                    keypoints.append(0)
            else:
                keypoints.append(0)
        
        # Adicionar número de mãos como característica
        keypoints.append(num_hands)
        
        # Garantir comprimento correto final
        while len(keypoints) < 3 * num_keypoints + 6:  # +6 para características adicionais incluindo num_hands
            keypoints.append(0)
        
        keypoints = keypoints[:3 * num_keypoints + 6]
        
        return np.array(keypoints), results
    
    def detectar_emocao(self, image):
        """Detecta emoção a partir das expressões faciais"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        emotion = "neutro"
        face_results = results  # Salvar para desenho
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Pegar o primeiro rosto
            
            # Extrair pontos-chave relevantes para detecção de sorriso
            landmarks = {}
            for name, idx in self.face_landmarks_indices.items():
                landmarks[name] = (
                    face_landmarks.landmark[idx].x,
                    face_landmarks.landmark[idx].y,
                    face_landmarks.landmark[idx].z
                )
            
            # Calcular elevação dos cantos da boca em relação ao centro
            mouth_center_y = (landmarks['top_lip_center'][1] + landmarks['bottom_lip_center'][1]) / 2
            left_elevation = mouth_center_y - landmarks['left_mouth'][1]
            right_elevation = mouth_center_y - landmarks['right_mouth'][1]
            
            # Calcular distância entre lábios (para sorriso aberto)
            lip_distance = landmarks['bottom_lip_center'][1] - landmarks['top_lip_center'][1]
            
            # Definir limiares para detecção de sorriso
            # Valores positivos indicam que os cantos estão acima do centro (sorriso)
            # Definir limiares para detecção de expressões
            sorriso_threshold = 0.005  # Reduzido para detectar sorrisos mais sutis
            tristeza_threshold = 0.008  # Adicionado threshold específico para tristeza

            # Verificar se há sorriso
            if (left_elevation > sorriso_threshold and right_elevation > sorriso_threshold):
                if lip_distance > 0.04:  # Sorriso aberto
                    emotion = "feliz"
                else:  # Sorriso fechado
                    emotion = "feliz"
        
        # Suavizar previsão de emoção
        self.emotion_predictions.append(emotion)
        
        # Determinar a emoção mais frequente
        if len(self.emotion_predictions) >= 5:
            emotions = {}
            for e in self.emotion_predictions:
                emotions[e] = emotions.get(e, 0) + 1
            self.current_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        else:
            self.current_emotion = emotion
        
        return self.current_emotion, face_results
    
    def prever_gesto(self, keypoints):
        """Prediz o gesto com base nos keypoints"""
        if len(keypoints) == 0:
            return None, 0
        
        # Normalizar se o scaler estiver disponível
        if self.scaler is not None:
            keypoints = self.scaler.transform(keypoints.reshape(1, -1))
        
        # Obter previsão e probabilidade
        prediction = self.model.predict(keypoints.reshape(1, -1))[0]
        
        # Tentar obter probabilidades para o threshold
        confidence = 0.0
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(keypoints.reshape(1, -1))[0]
                confidence = proba[prediction]
            else:
                # Se não tivermos probabilidades, confie na previsão
                confidence = 1.0
        except Exception as e:
            print(f"Aviso: Não foi possível obter probabilidade: {e}")
            confidence = 0.9  # Default
        
        # Mapear o índice para o nome do gesto
        gesture = self.label_mapping.get(prediction, "Desconhecido")
        
        return gesture, confidence
    
    def suavizar_previsoes(self, gesture, confidence):
        """Suaviza as previsões para reduzir variações rápidas"""
        if confidence >= self.threshold:
            self.predictions.append(gesture)
        
        if len(self.predictions) < self.sequence_length // 2:
            return None
        
        # Contar ocorrências de cada gesto na sequência
        counts = {}
        for g in self.predictions:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        
        # Encontrar o gesto mais frequente
        if not counts:
            return None
        
        most_common = max(counts.items(), key=lambda x: x[1])
        
        # Só retornar se for suficientemente frequente
        if most_common[1] >= self.sequence_length // 2:
            return most_common[0]
        
        return None
    
    def desenhar_ui(self, image, hand_results, face_results, gesture=None, emotion=None, confidence=0):
        """Desenha a interface de usuário com os resultados do reconhecimento"""
        # Desenhar landmarks das mãos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )
        
        # Desenhar landmarks faciais (opcionalmente, para debug)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Desenhar pontos-chave específicos usados para detecção de emoção
                for name, idx in self.face_landmarks_indices.items():
                    pos = face_landmarks.landmark[idx]
                    h, w, _ = image.shape
                    cx, cy = int(pos.x * w), int(pos.y * h)
                    cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)
        
        h, w, _ = image.shape
        
        # Painel de informações
        info_height = 120  # Aumentado para incluir informação de emoção
        info_panel = np.zeros((info_height, w, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(info_panel, "Reconhecimento de Gestos e Emoções em LIBRAS", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar gesto reconhecido
        if gesture:
            color = (0, 255, 0) if confidence > self.threshold else (120, 120, 120)
            text = f"Gesto: {gesture} ({confidence:.2f})"
            cv2.putText(info_panel, text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(info_panel, "Nenhum gesto detectado", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        
        # Mostrar emoção detectada
        emotion_color = (120, 120, 120)
        if emotion == "feliz":
            emotion_color = (0, 255, 255)  # Amarelo para feliz
            emotion_text = "Emoção: Feliz 😊"
        elif emotion == "triste":
            emotion_color = (255, 0, 0)    # Azul para triste
            emotion_text = "Emoção: Triste 😢"
        else:
            emotion_text = "Emoção: Neutro 😐"
        
        cv2.putText(info_panel, emotion_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # Modo sequência: mostrar gestos reconhecidos recentemente
        if self.display_mode == "sequence" and self.last_recognized_gestures:
            seq_text = " ".join(self.last_recognized_gestures)
            
            # Fundo escuro para o texto da sequência
            overlay = image.copy()
            cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            # Texto da sequência
            cv2.putText(image, seq_text, (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combinar o painel de informações com a imagem
        combined_image = np.vstack((info_panel, image))
        
        return combined_image
    
    def executar(self, recognition_interval=1.0):
        """Inicia o reconhecimento em tempo real a partir da webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir a câmera!")
            return False
        
        print("=== Reconhecimento de Gestos e Emoções em Tempo Real ===")
        print(f"Gestos reconhecíveis: {', '.join(self.label_mapping.values())}")
        print("Emoções detectáveis: Feliz, Neutro, Triste")
        print("Pressione 'q' para sair, 'c' para limpar a sequência")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Espelhar o frame horizontalmente para melhor experiência do usuário
                frame = cv2.flip(frame, 1)
                
                # Detectar pontos-chave das mãos
                keypoints, hand_results = self.detectar_keypoints(frame)
                
                # Detectar emoção pelo rosto
                emotion, face_results = self.detectar_emocao(frame)
                
                # Prever gesto
                current_time = time.time()
                gesture, confidence = self.prever_gesto(keypoints)
                
                # Suavizar previsões
                smoothed_gesture = self.suavizar_previsoes(gesture, confidence)
                
                # Atualizar a sequência se necessário
                if (smoothed_gesture and 
                    (current_time - self.last_recognition_time) > recognition_interval and
                    confidence > self.threshold):
                    # Evitar repetições consecutivas do mesmo gesto
                    if (not self.last_recognized_gestures or 
                        smoothed_gesture != self.last_recognized_gestures[-1]):
                        self.last_recognized_gestures.append(smoothed_gesture)
                        self.last_recognition_time = current_time
                        
                        # Imprimir gesto e emoção para log
                        print(f"Gesto: {smoothed_gesture}, Emoção: {emotion}")
                
                # Desenhar interface
                display_frame = self.desenhar_ui(
                    frame.copy(), 
                    hand_results, 
                    face_results, 
                    smoothed_gesture, 
                    emotion, 
                    confidence
                )
                
                # Mostrar frame
                cv2.imshow('Reconhecimento de Gestos e Emoções em LIBRAS', display_frame)
                
                # Tratar comandos
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Limpar sequência
                    self.last_recognized_gestures.clear()
                
        finally:
            # Limpar recursos
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.face_mesh.close()
            
            print("Reconhecimento finalizado.")
            return True

if __name__ == "__main__":
    # Diretórios podem ser ajustados conforme necessário
    model_dir = "../modelos"
    
    recognizer = ReconhecedorGestosEEmocao(
        model_dir=model_dir, 
        sequence_length=10, 
        threshold=0.6,
        display_mode="sequence"
    )
    recognizer.executar(recognition_interval=1.0)