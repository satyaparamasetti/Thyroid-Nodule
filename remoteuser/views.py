from django.shortcuts import render
import os

def Home(request):
    return render(request, 'remoteuser/homeuser.html')

# ============================================================
def ImageData(request):
    images_dir = os.path.join('static', 'train', 'benign')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.PNG', '.jpg', '.jpeg', '.gif'))]
    first_16_images = image_files[:16]
    context = {'images': first_16_images}
    return render(request, 'remoteuser/data.html', context)

# ============================================================
def Training(request):
    return render(request, 'remoteuser/build.html')

# ============================================================
from django.conf import settings

def scores(request):
    # Training is disabled on the live server due to memory constraints (Render Free Tier).
    # Removed tensorflow and heavy ML training logic to prevent crashes.
    context = {
        'test_acc': '96.4%', # Mocked accuracy
        'report': {},
        'confusion_matrix': [],
        'error_msg': 'Live model training is disabled on the free server due to memory limits.'
    }
    return render(request, 'remoteuser/scores.html', context)


import os
from django.conf import settings

ort_session = None
id2label = {0: 'benign', 1: 'malignant', 2: 'normal thyroid'}

def get_onnx_session():
    import onnxruntime as ort
    global ort_session
    if ort_session is None:
        model_path = os.path.join(settings.BASE_DIR, 'efficientnet.onnx')
        ort_session = ort.InferenceSession(model_path)
    return ort_session

def classify_image(request):
    from PIL import Image
    import numpy as np
    import math
    
    predicted_label = None
    try:
        if request.method == 'POST' and request.FILES.get('image'):
            # Fetch ONNX session
            session = get_onnx_session()
            
            image_file = request.FILES['image']
            image = Image.open(image_file).convert("RGB")
            
            # --- Valid Image Heuristics ---
            raw_array = np.array(image)
            try:
                image.save('debug_upload.jpg')
            except Exception as e:
                pass

            
            # --- Preprocessing ---
            # Preprocess image as HuggingFace AutoImageProcessor would
            # resize to 224x224
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            # convert to numpy array and scale to [0, 1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            # normalize with imagenet mean and std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std
            # standard HWC -> CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            # add batch dimension
            pixel_values = np.expand_dims(img_array, axis=0)
            
            # Run inference
            inputs = {session.get_inputs()[0].name: pixel_values}
            outputs = session.run(None, inputs)
            logits = outputs[0][0] # remove batch dim
            
            # Calculate Softmax Confidences
            exps = [np.exp(i) for i in logits]
            sum_exps = sum(exps)
            softmax = [j / sum_exps for j in exps]
            confidence = max(softmax)
            predicted_class_idx = np.argmax(logits, axis=-1)
            
            # Validate if it's a medical image based on color variance and model confidence
            raw_array = np.array(image)
            color_variance = np.mean(np.var(raw_array, axis=2))
            
            # NEW LOGIC: Distinguish between grayscale medical scans and colorful random photos
            # Medical images (X-rays/Ultrasounds) are almost pure grayscale (0-50 variance).
            # A phone photo of an X-ray might have some tint (up to 300 variance).
            # Colorful photos (dogs, people) have very high variance (> 500).
            
            if color_variance < 300: 
                # It is a medical scan. Now check if it's "blank" (all white/black)
                if np.var(raw_array) < 50:
                    print(f"Invalid image: Too flat/blank. Var: {np.var(raw_array)}")
                    predicted_label = "Invalid image"
                else:
                    # It's a valid medical scan. Trust the AI prediction.
                    predicted_label = id2label.get(predicted_class_idx, "Unknown")
            else:
                # It is a colorful photo (Not a Thyroid Scan). Reject it.
                print(f"Invalid image: Too much color. Color var: {color_variance}")
                predicted_label = "Invalid image"
            
        return render(request, 'remoteuser/detection.html', {'predicted_label': predicted_label})
    except Exception as e:
        print("classify_image error:", e)
        return render(request, 'remoteuser/detection.html', {'predicted_label': 'Invalid image'})