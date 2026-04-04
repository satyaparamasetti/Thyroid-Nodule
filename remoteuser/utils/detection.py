    
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import random
from django.core.files.storage import FileSystemStorage
import os

# Define constants
IMG_SIZE = (128, 128)
model = tf.keras.models.load_model(r'E:\python Django\naveen\Thyroid_Nodules_Detection_Deep_Reinforcement_learning\cnn_model.h5')

# Define the detection function
def detect_thyroid_nodule(image_path):
    # Load and process the image
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image = cv2.resize(new_image, IMG_SIZE)
    new_image = new_image / 255.0
    new_image = np.expand_dims(new_image, axis=0)
    new_image = np.expand_dims(new_image, axis=-1)

    # Make a prediction
    prediction = model.predict(new_image)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)

    # Define class labels and suggestions
    class_labels = ['benign', 'malignant']
    benign_suggestions = [ 
        'The thyroid nodule is likely to be benign. However, it is recommended to monitor the nodule regularly to ensure it does not grow or change over time.',
        'Consider a fine-needle aspiration biopsy to confirm the diagnosis.',
        'Keep in mind that benign nodules can still cause symptoms, so be sure to report any changes to your doctor.',
        'It is essential to follow up with your doctor regularly to ensure the nodule does not become malignant.',
        'You may want to consider a thyroid ultrasound to monitor the nodule\'s size and characteristics.',
        'Maintain a healthy lifestyle, including a balanced diet and regular exercise, to reduce the risk of thyroid problems.',
        'Avoid exposure to radiation, which can increase the risk of thyroid cancer.',
        'Get enough sleep and manage stress to help regulate your thyroid function.',
        'Consider taking a thyroid supplement, such as selenium or iodine, to support thyroid health.',
        'Monitor your thyroid function regularly to catch any potential problems early.',
        'Avoid smoking, which can increase the risk of thyroid cancer.',
        'Limit your intake of soy products, which can interfere with thyroid function.',
        'Get enough vitamin D, which is essential for thyroid health.',
        'Avoid exposure to endocrine disruptors, such as BPA, which can interfere with thyroid function.',
        'Consider a thyroid support group to connect with others who are going through a similar experience.'
    ]  
    
    # Define your benign suggestions here
    malignant_suggestions = [
        'The thyroid nodule is likely to be malignant. It is recommended to consult a doctor immediately for further evaluation and treatment.',
        'Consider a fine-needle aspiration biopsy to confirm the diagnosis and determine the type of cancer.',
        'Surgery may be necessary to remove the nodule and affected tissue.',
        'You may want to discuss with your doctor the possibility of radiation therapy to treat the cancer.',
        'It is essential to follow up with your doctor regularly to monitor the cancer\'s progression and adjust treatment as needed.',
        'Maintain a healthy lifestyle, including a balanced diet and regular exercise, to reduce the risk of cancer recurrence.',
        'Avoid exposure to radiation, which can increase the risk of cancer recurrence.',
        'Get enough sleep and manage stress to help regulate your immune system.',
        'Consider taking a cancer-fighting supplement, such as vitamin C or turmeric, to support your immune system.',
        'Monitor your cancer markers regularly to catch any potential problems early.',
        'Avoid smoking, which can increase the risk of cancer recurrence.',
        'Limit your intake of processed foods, which can increase the risk of cancer recurrence.',
        'Get enough vitamin D, which is essential for immune system function.',
        'Avoid exposure to endocrine disruptors, such as BPA, which can interfere with hormone function.',
        'Consider a cancer support group to connect with others who are going through a similar experience.'
    ] 

    # Determine the result and suggestion based on the prediction
    if predicted_class[0] == 0:  # Benign
        suggestion = random.choice(benign_suggestions)
    else:  # Malignant
        suggestion = random.choice(malignant_suggestions)

    # Additional confidence suggestion
    if confidence < 0.7:
        confidence_note = 'The confidence of the prediction is low. It is recommended to consult a doctor for further evaluation and confirmation.'
    else:
        confidence_note = 'The confidence of the prediction is high, but it is always recommended to consult a doctor for further evaluation.'

    # Create a binary and predicted image
    binary_image = cv2.threshold(new_image[0, :, :, 0], 0.5, 1, cv2.THRESH_BINARY)[1]
    predicted_image = np.zeros_like(new_image[0, :, :, 0])
    if predicted_class[0] == 0:
        predicted_image[new_image[0, :, :, 0] > 0.5] = 1
    else:
        predicted_image[new_image[0, :, :, 0] > 0.7] = 1

    # Plot the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(new_image[0, :, :, 0], cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title('Binary Image')
    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title(f'Predicted Image: {class_labels[predicted_class[0]]}')
    
    # Save the plot to the media directory
    fs = FileSystemStorage()
    media_path = os.path.join('prediction_plot.png')
    plt.savefig(fs.path(media_path), bbox_inches='tight')  # Save the plot to the media directory
    plt.close(fig)  # Close the figure to avoid memory leaks

    return class_labels[predicted_class[0]], suggestion, confidence_note, media_path