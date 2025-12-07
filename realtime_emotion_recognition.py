import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

class RealTimePetEmotionRecognition:
    def __init__(self, model_path='efficientnet_pet_emotion_proper.h5'):
        """Initialize the real-time emotion recognition system"""
        print("Loading model...")
        try:
            self.model = load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            return

        # Class labels (adjust based on your dataset, sorted alphabetically)
        # These should match the training data classes
        self.class_labels = ['Angry', 'happy', 'Other', 'Sad']

        # Initialize face detector (you can use pet-specific detector if available)
        # For now, using Haar cascades for general face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Parameters - match training input size
        self.target_size = (300, 300)
        self.confidence_threshold = 0.7  # Increased threshold for better accuracy
        self.frame_skip = 5  # Process every 5th frame for stability
        self.frame_count = 0

        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 10  # Keep last 10 predictions for smoothing
        self.min_confidence = 0.3  # Minimum confidence to consider prediction

        print("🎥 Real-time pet emotion recognition initialized!")
        print("Press 'q' to quit, 'c' to capture screenshot")

    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Convert to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, self.target_size)

        # Convert to array and preprocess
        frame_array = np.expand_dims(frame_resized, axis=0)
        frame_preprocessed = preprocess_input(frame_array)

        return frame_preprocessed

    def smooth_prediction(self, emotion, confidence):
        """Smooth predictions using history"""
        if confidence < self.min_confidence:
            # If confidence is too low, return the most common recent prediction
            if self.prediction_history:
                # Count occurrences of each emotion in history
                emotion_counts = {}
                for hist_emotion, _ in self.prediction_history[-5:]:  # Last 5 predictions
                    emotion_counts[hist_emotion] = emotion_counts.get(hist_emotion, 0) + 1

                # Return most common emotion if it appears more than once
                most_common = max(emotion_counts.items(), key=lambda x: x[1])
                if most_common[1] > 1:
                    return most_common[0], 0.5  # Return with moderate confidence
            return emotion, confidence

        # Add current prediction to history
        self.prediction_history.append((emotion, confidence))

        # Keep only recent history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        # If we have enough history, check for consistency
        if len(self.prediction_history) >= 3:
            recent_emotions = [e for e, c in self.prediction_history[-3:]]
            # If last 3 predictions are the same, boost confidence
            if len(set(recent_emotions)) == 1:
                return emotion, min(confidence + 0.2, 0.95)

        return emotion, confidence

    def predict_emotion(self, frame):
        """Predict emotion from frame"""
        if self.model is None:
            return "Model not loaded", 0.0

        try:
            preprocessed = self.preprocess_frame(frame)
            predictions = self.model.predict(preprocessed, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]

            predicted_label = self.class_labels[predicted_class_idx]

            # Apply smoothing
            smoothed_emotion, smoothed_confidence = self.smooth_prediction(predicted_label, float(confidence))

            return smoothed_emotion, smoothed_confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            print(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'No shape'}")
            print(f"Model input shape: {self.model.input_shape}")
            return "Error", 0.0

    def detect_faces(self, frame):
        """Detect faces in the frame using multiple scales for better pet detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try different parameters for better pet face detection
        faces = []
        # Standard detection
        faces.extend(self.face_cascade.detectMultiScale(gray, 1.1, 4))
        # More sensitive detection for smaller faces
        faces.extend(self.face_cascade.detectMultiScale(gray, 1.2, 3))
        # Less sensitive for larger faces
        faces.extend(self.face_cascade.detectMultiScale(gray, 1.05, 5))

        # Remove duplicates (overlapping detections)
        if faces:
            # Simple non-maximum suppression
            faces = self._non_max_suppression(faces, 0.3)

        return faces

    def _non_max_suppression(self, boxes, overlap_thresh):
        """Apply non-maximum suppression to remove overlapping boxes"""
        if len(boxes) == 0:
            return []

        # Convert to numpy array
        boxes = np.array(boxes)

        # Initialize the list of picked indexes
        pick = []

        # Grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # Compute the area of the bounding boxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort the bounding boxes by the bottom-right y-coordinate
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            # Grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        # Return only the bounding boxes that were picked
        return boxes[pick].tolist()

    def draw_results(self, frame, faces, emotion, confidence):
        """Draw detection results on frame"""
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Add emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def run_realtime_recognition(self):
        """Run real-time emotion recognition"""
        print("🎬 Starting real-time recognition...")
        print("📹 Opening camera...")

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        print("✅ Camera opened successfully!")
        print("🎯 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 'c' to capture screenshot")
        print("   - Press 's' to save current frame")

        fps_counter = 0
        fps_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect faces
            faces = self.detect_faces(frame)

            # Predict emotion (use whole frame if no faces detected)
            if len(faces) > 0:
                # Use the largest face detected
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence = self.predict_emotion(face_roi)
            else:
                # Use whole frame if no faces detected
                emotion, confidence = self.predict_emotion(frame)

            # Draw results
            frame = self.draw_results(frame, faces, emotion, confidence)

            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display frame
            cv2.imshow('Pet Emotion Recognition - Real Time', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pet_emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved as {filename}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pet_emotion_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as {filename}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    print("🐾 Pet Facial Expression Recognition - Real Time")
    print("=" * 55)

    # Initialize the recognition system
    recognizer = RealTimePetEmotionRecognition()

    if recognizer.model is None:
        print("❌ Failed to initialize. Please check model file.")
        return

    # Run real-time recognition
    recognizer.run_realtime_recognition()

if __name__ == "__main__":
    main()
