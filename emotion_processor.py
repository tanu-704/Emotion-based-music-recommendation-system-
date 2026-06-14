import cv2
import numpy as np
import av
import mediapipe as mp
from keras.models import load_model
import streamlit as st

# ---------------- Load Model ----------------
model = load_model("model.h5", compile=False)
labels = np.load("labels.npy")

# ---------------- Mediapipe ----------------
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

holis = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class EmotionProcessor:

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        data = []

        # ---------------- FACE ----------------
        if res.face_landmarks:
            for lm in res.face_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)
        else:
            data.extend([0.0] * 468 * 2)

        # ---------------- LEFT HAND ----------------
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)
        else:
            data.extend([0.0] * 21 * 2)

        # ---------------- RIGHT HAND ----------------
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)
        else:
            data.extend([0.0] * 21 * 2)

        data = np.array(data).reshape(1, -1)

        prediction = labels[np.argmax(model.predict(data, verbose=0))]

        # Save emotion in Streamlit session
        st.session_state["emotion"] = prediction

        # Show text
        cv2.putText(frm, str(prediction), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # ---------------- DRAW LANDMARKS ----------------
        if res.face_landmarks:
            drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        if res.left_hand_landmarks:
            drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if res.right_hand_landmarks:
            drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")
