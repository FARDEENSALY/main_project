import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# Define constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
EMOTION_LABELS = ['Angry', 'Happy', 'Other', 'Sad']

def create_model():
    # Create base model with EfficientNetB5
    base_model = EfficientNetB5(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Create new model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pet face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')

# Create and compile model
print("Creating model...")
model = create_model()
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

# Load trained weights
print("Loading weights...")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'Pet_Facial_Expression_Recognition-main', 
                         'efficientNetB5_pet_emotion_model.h5')
print(f"Looking for model at: {model_path}")
model.load_weights(model_path)

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

print("Starting real-time pet emotion detection...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect pet faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        try:
            # Extract and preprocess face
            face_roi = frame[y:y+h, x:x+w]
            processed_face = preprocess_image(face_roi)
            
            # Make prediction
            prediction = model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = prediction[0][emotion_idx] * 100
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
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