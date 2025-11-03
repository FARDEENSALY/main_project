import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'Pet_Facial_Expression_Recognition-main', 
                         'efficientNetB5_pet_emotion_model.h5')
print(f"Looking for model at: {MODEL_PATH}")
# Infer class label order from dataset folders (keeps mapping consistent with training)
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'Pet_Facial_Expression_Recognition-main',
                           'pets_facial_expression_dataset')
try:
    EMOTIONS = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != 'master']
except Exception:
    # Fallback to a sensible default if dataset folder isn't present
    EMOTIONS = ['Angry', 'Other', 'Sad', 'happy']
print('Using class labels:', EMOTIONS)

def create_model():
    """Create and compile the model architecture"""
    base_model = tf.keras.applications.EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='max'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.016),
                        activity_regularizer=keras.regularizers.l1(0.006),
                        bias_regularizer=keras.regularizers.l1(0.006),
                        activation='relu'),
        keras.layers.Dropout(rate=0.45, seed=123),
        keras.layers.Dense(len(EMOTIONS), activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def preprocess_image(img):
    """Preprocess image for model input"""
    # Match training preprocessing: do NOT scale to [0,1] if the model was trained on raw 0-255 images
    resized = cv2.resize(img, IMG_SIZE)
    arr = resized.astype('float32')
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    # Load model
    print("Loading model...")
    model = create_model()
    model.load_weights(MODEL_PATH)
    
    # Initialize webcam and face detector
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
    
    print("Ready! Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each face
        for (x, y, w, h) in faces:
            try:
                # Get and process face region
                face = frame[y:y+h, x:x+w]
                processed_face = preprocess_image(face)
                
                # Predict emotion
                prediction = model.predict(processed_face, verbose=0)[0]
                emotion = EMOTIONS[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Draw results
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, 
                          f"{emotion}: {confidence:.0f}%",
                          (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.9,
                          color,
                          2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Show frame
        cv2.imshow('Pet Emotion Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()