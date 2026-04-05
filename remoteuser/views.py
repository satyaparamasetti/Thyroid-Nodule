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
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import cv2

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    dataset_dir = os.path.join(settings.BASE_DIR, 'static', 'train')
    classes = ['benign', 'malignant']

    # ── Load images ──
    images, labels = [], []
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(classes.index(class_name))

    images = np.array(images)
    labels = np.array(labels)

    # ── Split ──
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # ── Normalise ──
    train_images = train_images / 255.0
    test_images  = test_images  / 255.0

    # ── One-hot encode ──
    num_classes  = len(classes)
    train_labels_oh = np.eye(num_classes)[train_labels]
    test_labels_oh  = np.eye(num_classes)[test_labels]

    # ── Add channel dim ──
    train_images = np.expand_dims(train_images, axis=-1)
    test_images  = np.expand_dims(test_images,  axis=-1)

    # ── Build model ──
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ── Augmentation ──
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )

    # ── Train ──
    model.fit(
        datagen.flow(train_images, train_labels_oh, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(test_images, test_labels_oh)
    )

    # ── Evaluate ──
    test_loss, test_acc = model.evaluate(test_images, test_labels_oh)

    predictions      = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    # test_labels is still the integer array (not one-hot) — use it directly
    true_labels      = test_labels  # shape (n,), values 0 or 1

    # ── Classification report ──
    # target_names makes sklearn use 'benign'/'malignant' as keys instead of '0'/'1'
    report = classification_report(
        true_labels, predicted_labels,
        target_names=classes,
        output_dict=True
    )

    # ── Fix keys so Django templates can access them with dot notation ──
    # 1. rename 'f1-score' → 'f1_score' inside every sub-dict
    for key in list(report.keys()):
        if isinstance(report[key], dict) and 'f1-score' in report[key]:
            report[key]['f1_score'] = report[key].pop('f1-score')

    # 2. rename spaced keys → underscored keys
    if 'macro avg' in report:
        report['macro_avg'] = report.pop('macro avg')
    if 'weighted avg' in report:
        report['weighted_avg'] = report.pop('weighted avg')

    # ── Confusion matrix ──
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    # ── Save model ──
    model.save('new_cnn_model.h5', save_format='tf')

    context = {
        'test_acc':        test_acc,
        'report':          report,
        'confusion_matrix': confusion_mat.tolist(),
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
    
    predicted_label = None
    try:
        if request.method == 'POST' and request.FILES.get('image'):
            # Fetch ONNX session
            session = get_onnx_session()
            
            image_file = request.FILES['image']
            image = Image.open(image_file).convert("RGB")
            
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
            logits = outputs[0]
            
            predicted_class_idx = np.argmax(logits, axis=-1)[0]
            predicted_label = id2label.get(predicted_class_idx, "Unknown")
            
        return render(request, 'remoteuser/detection.html', {'predicted_label': predicted_label})
    except Exception as e:
        print("classify_image error:", e)
        return render(request, 'remoteuser/detection.html', {'predicted_label': 'Invalid image'})