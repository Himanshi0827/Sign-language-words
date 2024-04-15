
import cv2
import os
import numpy as np
from utility import mp_holistic, mp_drawing
import utility
import mediapipe as mp
from datetime import datetime
from tensorflow.keras.models import load_model
from googletrans import Translator
from gtts import gTTS
import base64
import streamlit as st

class RealtimeRecognition:
    def __init__(self, model_path='action.h5'):
        self.model = load_model(model_path)
        self.model.summary()
        self.actions = np.array(['hello', 'thanks', 'iloveyou'])
        self.sequence = []
        self.sentence = []
        self.threshold = 0.80
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # self.target_language = input("Enter the target language code: ")

    def _prob_viz(self, res, input_frame, colors):
        output_frame = input_frame.copy()
        max_prob_index = np.argmax(res)
        prob_percentage = int(res[max_prob_index] * 100)

        cv2.rectangle(output_frame, (0, 60 + max_prob_index * 40), (prob_percentage, 90 + max_prob_index * 40),
                      colors[max_prob_index], -1)
        cv2.putText(output_frame, self.actions[max_prob_index], (0, 85 + max_prob_index * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return output_frame

    def _show_sentence(self, image):
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Realtime Recognition', image)

    def _text_to_speech(self, sentence, target_language):
        translator = Translator()
        translation = translator.translate(sentence, dest=target_language)
        tts = gTTS(text=translation.text, lang=target_language)
        tts.save('output.mp3')
        os.system('mpg123 output.mp3')

    def base64_to_image(self, base64_string):
        # Decode base64 string into bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode numpy array into image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    


    def run(self):
        cap = cv2.VideoCapture(0)
        image_placeholder = st.empty()  # Create an empty placeholder for the image

        while cap.isOpened():
            ret, frame = cap.read()
            image, results = utility.mediapipe_detection(frame, self.holistic)

            keypoints = utility.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                print(self.actions[np.argmax(res)])
                print(res)
                print(res[np.argmax(res)])
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence) > 0:
                        if self.actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(self.actions[np.argmax(res)])
                            #self._text_to_speech(self.sentence[-1], self.target_language)
                            self._text_to_speech("".join(self.sentence), self.target_language)
                    else:
                        self.sentence.append(self.actions[np.argmax(res)])
                        #self._text_to_speech(self.sentence[-1], self.target_language)
                        self._text_to_speech("".join(self.sentence), self.target_language)

                if len(self.sentence) > 5:
                    self.sentence = self.sentence[-5:]

                image = self._prob_viz(res, image, [(0, 0, 255), (0, 255, 0), (255, 0, 0)])

            image_placeholder.image(image, channels="BGR")  # Display the image in the Streamlit app

        cap.release() 



    def stop_recognition(self):
        # Destroy the window and stop the recognition
        cv2.destroyAllWindows()
        # st.audio('output.mp3', format='audio/mp3')
        if os.path.exists('output.mp3'):
             st.audio('output.mp3', format='audio/mp3')

def main():
    st.title("Realtime Recognition with Streamlit")

    target_language = st.text_input("Enter the target language code:")
    start_button = st.button("Start Recognition")
    stop_button = st.button("Stop Recognition")
    if start_button:
        realtime_recognition = RealtimeRecognition()
        realtime_recognition.target_language = target_language
        realtime_recognition.run()
    if stop_button:
         realtime_recognition = RealtimeRecognition()
         realtime_recognition.stop_recognition()
   
    

if __name__ == "__main__":
    main()