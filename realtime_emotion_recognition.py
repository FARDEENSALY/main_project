import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

class RealTimePetEmotionRecognition:
    def __init__(self, model_path='400epoch_train.h5'):
        print("--- Initializing System ---")
        try:
            # Load model with compile=False to avoid custom layer errors
            self.model = load_model(model_path, compile=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            return

        # IMPORTANT: Keras sorts classes alphabetically. 
        # ['Angry', 'Other', 'Sad', 'happy'] -> 'Other' usually comes after 'Angry'
        # Double check your training folder names!
        self.class_labels = ['Angry', 'happy', 'Other', 'Sad']

        # Detectors
        self.face_cascades = [
            ('cat', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')),
            ('human', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
        ]

        # Config
        self.target_size = (224, 224)
        self.confidence_threshold = 0.5 
        self.frame_skip = 2 # Reduced for smoother real-time feel
        self.frame_count = 0
        
        # Tracking for smoothing
        self.last_emotion = "Searching..."
        self.last_confidence = 0.0
        self.last_faces = []

    def preprocess_roi(self, roi):
        """Processes the cropped face area specifically"""
        # Convert BGR to RGB
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize using INTER_AREA for better quality downsampling
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img_array = np.expand_dims(img, axis=0)
        # EfficientNet specific normalization
        return preprocess_input(img_array)

    def detect_pet_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for name, cascade in self.face_cascades:
            if cascade.empty(): continue
            # Detect faces
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            if len(faces) > 0:
                return faces # Returns first found type
        return []

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found")
            return

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            # Only process every Nth frame to save CPU/GPU
            if self.frame_count % self.frame_skip == 0:
                faces = self.detect_pet_face(frame)
                self.last_faces = faces

                if len(faces) > 0:
                    # Get the largest detected face
                    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                    
                    # Add 10% padding so we don't cut off ears/chin
                    pad_w, pad_h = int(w*0.1), int(h*0.1)
                    y1, y2 = max(0, y-pad_h), min(frame.shape[0], y+h+pad_h)
                    x1, x2 = max(0, x-pad_w), min(frame.shape[1], x+w+pad_w)
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    # Predict
                    processed = self.preprocess_roi(roi)
                    preds = self.model.predict(processed, verbose=0)[0]
                    idx = np.argmax(preds)
                    
                    self.last_emotion = self.class_labels[idx]
                    self.last_confidence = preds[idx]
                else:
                    self.last_emotion = "No Face"
                    self.last_confidence = 0.0

            # --- Drawing Logic ---
            display_frame = frame.copy()
            for (x, y, w, h) in self.last_faces:
                color = (0, 255, 0) if self.last_confidence > 0.6 else (0, 165, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                text = f"{self.last_emotion} ({self.last_confidence:.2f})"
                cv2.putText(display_frame, text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Pet Emotion Monitor', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimePetEmotionRecognition()
    if app.model:
        app.run()