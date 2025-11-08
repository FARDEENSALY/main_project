import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Pet_Facial_Expression_Recognition-main', 'efficientNetB5_pet_emotion_model.h5')

print(f"Looking for model at: {model_path}")

# Load the trained model
model = load_model(model_path)

# Define emotion labels
emotion_labels = ['Angry', 'Happy', 'Other', 'Sad']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pet face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')

def preprocess_image(img):
    # Resize image to match model's expected sizing
    img = cv2.resize(img, (224, 224))
    # Convert to array and normalize
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

print("Starting real-time pet emotion detection...")
print("Press 'q' to quit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect pet faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # Preprocess the face
            processed_face = preprocess_image(face_roi)
            
            # Make prediction
            prediction = model.predict(processed_face)
            emotion_idx = np.argmax(prediction[0])
            emotion = emotion_labels[emotion_idx]
            confidence = prediction[0][emotion_idx] * 100
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with emotion and confidence
            text = f"{emotion}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    # Display the frame
    cv2.imshow('Pet Emotion Detection', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()