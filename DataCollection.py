# data_collector.py

import cv2
import os
import numpy as np
import utility
from utility import mp_holistic, mp_drawing

class DataCollector:
    def __init__(self, data_path='MP_Data', actions=['hello','iloveyou', 'thanks'], 
                 no_sequences=30, sequence_length=30):
        self.data_path = data_path
        self.actions = actions
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.label_map = {label: num for num, label in enumerate(actions)}

        self._create_data_directories()

    def _create_data_directories(self):
        for action in self.actions:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(self.data_path, action, str(sequence)))
                except:
                    pass

    def collect_data(self):
        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in self.actions:
                for sequence in range(self.no_sequences):
                    for frame_num in range(self.sequence_length):
                        ret, frame = cap.read()
                        image, results = utility.mediapipe_detection(frame, holistic)

                        utility.draw_styled_landmarks(image, results)

                        if frame_num == 0:
                            utility.show_start_collection_message(image, action, sequence)
                        else:
                            utility.show_collection_message(image, action, sequence)

                        self._export_keypoints(results, action, sequence, frame_num)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()

    def _export_keypoints(self, results, action, sequence, frame_num):
        keypoints = utility.extract_keypoints(results)
        npy_path = os.path.join(self.data_path, action, str(sequence), str(frame_num))
        np.save(npy_path, keypoints)


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()
