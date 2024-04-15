import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime
from keras.models import load_model
from googletrans import Translator
from gtts import gTTS
import base64
import streamlit as st
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



# Function to detect poses

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    

    
# function to extract keypoint values from the results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    
#------------------------------------------------------------------------------

# Function to draw landmarks with modified circle and line thickness

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    


# Realtime Recognition class

class RealtimeRecognition:

# Initialize the RealtimeRecognition class

    def __init__(self, model_path='action.h5'):
        self.model = load_model(model_path)
        self.model.summary()
        self.actions = np.array(['hello', 'thanks', 'iloveyou'])
        self.sequence = []
        self.sentence = []
        self.threshold = 0.80
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.target_language = ''  # Initialize target_language attribute



# Function to detect poses

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    


# Function to extract keypoint values from the results

    def _prob_viz(self, res, input_frame, colors):
        output_frame = input_frame.copy()
        max_prob_index = np.argmax(res)
        prob_percentage = int(res[max_prob_index] * 100)

        cv2.rectangle(output_frame, (0, 60 + max_prob_index * 40), (prob_percentage, 90 + max_prob_index * 40),
                      colors[max_prob_index], -1)
        cv2.putText(output_frame, self.actions[max_prob_index], (0, 85 + max_prob_index * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return output_frame
    


# Function to visualize the probability of the actions

    def _show_sentence(self, image):
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Realtime Recognition', image)



# Function to convert text to speech

    def _text_to_speech(self, sentence, target_language):
        translator = Translator()
        translation = translator.translate(sentence, dest=target_language)
        tts = gTTS(text=translation.text, lang=target_language)
        tts.save('output.mp3')
        os.system('mpg123 output.mp3')



# Function to convert base64 string to image

    def base64_to_image(self, base64_string):
        # Decode base64 string into bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode numpy array into image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    


# Function to run the Realtime Recognition

    def run(self):
        cap = cv2.VideoCapture(0)
        image_placeholder = st.empty()  # Create an empty placeholder for the image

        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, self.holistic)

            keypoints = extract_keypoints(results)
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



# Function to stop the recognition

    def stop_recognition(self):
        # Destroy the window and stop the recognition
        cv2.destroyAllWindows()
        # st.audio('output.mp3', format='audio/mp3')
        if os.path.exists('output.mp3'):
             st.audio('output.mp3', format='audio/mp3')



# Function to run the Streamlit app

def main():

    language_options = {
    'en': 'English',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'ru': 'Russian',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tl': 'Filipino',
    'ur': 'Urdu'
    }


    st.title("Realtime Recognition with Streamlit")

    # Display dropdown list to select target language
    target_language = st.selectbox("Select the target language:", list(language_options.values()))

    # Target language code
    target_language = [key for key, value in language_options.items() if value == target_language][0]

    # Display the selected language code
    st.write(f"Selected language code: {target_language}")

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