import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ---------------------------
# 1️⃣ DATASET PATHS
# ---------------------------
train_dir = 'dataset/TRAIN'   # make sure folder name matches
test_dir = 'dataset/TEST'

# ---------------------------
# 2️⃣ PREPROCESSING
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Add this line here — to confirm what classes are found
print("📁 Classes found:", train_data.class_indices)

# ---------------------------
# 3️⃣ MODEL CREATION
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 4️⃣ TRAINING
# ---------------------------
print("🚀 Training the model...")
history = model.fit(train_data, validation_data=test_data, epochs=10)

# ---------------------------
# 5️⃣ SAVE MODEL
# ---------------------------
model.save('waste_sorter_model.h5')
print("✅ Model saved as waste_sorter_model.h5")

# ---------------------------
# 6️⃣ PLOT TRAINING RESULTS
# ---------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.show()

# ---------------------------
# 7️⃣ TEST SINGLE IMAGE (OPTIONAL)
# ---------------------------
def predict_waste(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_names = list(train_data.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    print(f"♻ Predicted: {predicted_class} ({confidence*100:.2f}%)")
    return predicted_class

# Test with any image (optional)
test_image = 'sample.jpg'  # change this to your test image
if os.path.exists(test_image):
    predict_waste(test_image)
else:
    print("⚠ No sample.jpg found — place a test image in the folder.")

# ---------------------------
# 8️⃣ REAL-TIME DETECTION (Add this part at END of the code)
# ---------------------------
print("📸 Starting real-time waste sorting... Press 'q' to quit.")
cap = cv2.VideoCapture(0)
class_names = list(train_data.class_indices.keys())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    cv2.putText(frame, f'{label} ({confidence*100:.1f}%)', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Smart Waste Sorter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()