import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

def predict_pet_emotion(image_path, model_path='400epoch_train.h5'):
    """
    Predict pet facial emotion from an image
    """
    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully!")
    except:
        print("Model not found. Please train the model first by running: python run_efficientnet.py")
        return None

    # Load and preprocess the image
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)

        # Class labels (based on the dataset structure, sorted alphabetically)
        class_labels = ['Angry', 'Other', 'Sad', 'happy']

        predicted_class_label = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        # Display result
        plt.imshow(img)
        plt.axis('off')
        if predicted_class_label == 'Other':
            plt.title(f"The pet appears normal\nConfidence: {confidence:.2f}%")
        else:
            plt.title(f"The pet appears {predicted_class_label}\nConfidence: {confidence:.2f}%")
        plt.show()

        return predicted_class_label, confidence

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    # Test with sample images
    test_images = [
        'pets_facial_expression_dataset/Angry/02.jpg',
        'pets_facial_expression_dataset/Sad/031.jpg',
        'pets_facial_expression_dataset/happy/032.jpg',
        'pets_facial_expression_dataset/Other/20.jpg'
    ]

    print("Pet Facial Expression Recognition Demo")
    print("=" * 50)

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nPredicting emotion for: {img_path}")
            result = predict_pet_emotion(img_path)
            if result:
                emotion, confidence = result
                print(f"Predicted emotion: {emotion} (Confidence: {confidence:.2f}%)")
        else:
            print(f"Image not found: {img_path}")

    print("\nDemo completed!")
