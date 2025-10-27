import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import cv2

# Parameters
img_size = (64, 64)
batch_size = 32
epochs = 10
data_dir = "data"  # expects data/train/<class> and data/validation/<class>

# Data generators (use ImageDataGenerator for augmentation)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical')

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical')

num_classes = train_gen.num_classes

# Simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*img_size,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# Save model
model.save('sign_cnn.h5')

# Real-time prediction using webcam
# Map indices to labels:
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret: break
    # Preprocess: crop/resize, convert BGR->RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, img_size)
    x = im.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    label = idx_to_label[int(np.argmax(preds))]
    prob = float(np.max(preds))
    # Overlay
    cv2.putText(frame, f"{label} ({prob:.2f})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

