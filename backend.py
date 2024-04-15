# from flask import Flask, jsonify, request, send_file
# from flask_socketio import SocketIO, emit
# import base64
# from realtime_recognition import RealtimeRecognition
# from googletrans import Translator
# from gtts import gTTS
# from flask_cors import CORS, cross_origin

# app = Flask(__name__)
# CORS(app)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests
# realtime_recognition = RealtimeRecognition()

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
# @socketio.on('imageFrame')
# def handle_frame(data):
#     image_data_base64 = data['image']

#     if image_data_base64 is not None:
#         result = realtime_recognition.run(image_data_base64)
#         emit('signResult', {'sign': result['predicted_sign']})  # Assuming 'predicted_sign' is the key for the sign prediction
#     else:
#         print("Invalid image received")

# # @socketio.on('imageFrame')
# # def handle_frame(data):
# #     # Extract base64-encoded image data from the received data
# #     image_data_base64 = data['image']

# #     # Check if the image is valid
# #     if image_data_base64 is not None:
# #         # Process the image using sign detection code
# #         result = realtime_recognition.run(image_data_base64)
# #         print("result")

# #         # Emit the sign detection result back to the frontend
# #         emit('signResult', {'sign': result})
# #     else:
# #         print("Invalid image received")

# @app.route('/lan', methods=['POST'])
# @cross_origin()  # Apply CORS to this route
# def translate_sentence():
#     # Receive the sentence and language from the frontend
#     data = request.json
#     sentence = data.get('sentence')
#     source_language = data.get('lan')
#     target_language = data.get('lan2')
    
#     # Translate the sentence
#     translator = Translator()
#     translation = translator.translate(sentence, src=source_language, dest=target_language)
#     print(translation.text)
    
#     # Return the translated sentence
#     return jsonify({'translation': translation.text})

# @app.route('/tts', methods=['POST'])
# @cross_origin()  # Apply CORS to this route
# def text_to_speech():
#     data = request.json
#     sentence = data.get('sentence')
#     target_language = data.get('target_language')  # Update to match the frontend field name
#     print(sentence, target_language)

#     if not sentence or not target_language:
#         return jsonify({'error': 'Missing sentence or target_language'}), 400

#     # Translate the sentence
#     translator = Translator()
#     translation = translator.translate(sentence, dest=target_language)
#     translated_text = translation.text

#     # Generate TTS audio
#     tts = gTTS(text=translated_text, lang=target_language)
#     tts.save('output.mp3')

#     # Send the audio file back to the frontend
#     return send_file('output.mp3', as_attachment=True)


# if __name__ == "__main__":
#     socketio.run(app, host='0.0.0.0', debug=True)


























import streamlit as st
import io
import base64
import requests
import cv2
import numpy as np

# Function to send image frame to the backend
def send_image_to_backend(image_data):
    # API endpoint for sending image data to the backend
    endpoint = 'http://localhost:5000/imageFrame'

    # Data to be sent in the request body
    data = {'image': image_data}

    try:
        # Send POST request to the backend
        response = requests.post(endpoint, json=data)
        
        # Check if request was successful
        response.raise_for_status()

        # Check if response contains JSON data
        if response.headers['content-type'] == 'application/json':
            return response.json()
        else:
            return {'error': 'Invalid response format'}
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Error:", err)

    return {'error': 'Unknown error occurred'}

# Function to send stop signal to the backend
def stop_recognition():
    url = 'http://localhost:5000/stopRecognition'
    response = requests.get(url)
    return response.json()

# Main Streamlit app
def main():
    st.title('Realtime Sign Language Recognition')

    # Start recognition button
    if st.button('Start Recognition'):
        st.write('Recognition started...')
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = io.BytesIO(buffer)
            image_data = base64.b64encode(frame_bytes.read()).decode()

            # Send image frame to backend for recognition
            response = send_image_to_backend(image_data)
            if 'sign' in response:
                predicted_sign = response['sign']
                st.write('Predicted Sign:', predicted_sign)
            else:
                st.write('Error: Sign not found in response')
                predicted_sign = None

            # Stop recognition button with a unique key
            if st.button('Stop Recognition', key='stop_button'):
                stop_response = stop_recognition()
                st.write('Recognition stopped...')
                if stop_response['success']:
                    st.write('Text-to-speech processing...')
                    # You can add text-to-speech processing here if needed
                break

            st.image(frame, channels='BGR', use_column_width=True)

if __name__ == '__main__':
    main()


