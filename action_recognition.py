# action_recognition.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime

class ActionRecognitionModel:
    def __init__(self, data_path='MP_Data', actions=['hello', 'thanks', 'iloveyou'],
                 no_sequences=30, sequence_length=30):
        self.data_path = data_path
        self.actions = np.array(actions)
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.label_map = {label: num for num, label in enumerate(actions)}
        self.model = None

    def _load_data(self):
        sequences, labels = [], []

        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.data_path, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)

        return X, y

    def _create_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, epochs=2000):
        X, y = self._load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

        self.model = self._create_model()

        log_dir = os.path.join('Logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb_callback = TensorBoard(log_dir=log_dir)

        try:
            self.model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback], validation_data=(X_test, y_test))
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current model...")
            self.model.save('action.h5')
            print("Model saved.")

if __name__ == "__main__":
    # Example usage
    model_instance = ActionRecognitionModel()
    model_instance.train_model()
