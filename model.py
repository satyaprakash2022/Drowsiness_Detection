import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization

def load_data(directory, target_size=(24, 24), class_mode='categorical'):
    data = []
    labels = []
    classes = os.listdir(directory)
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(directory, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, target_size)  # Resize to target size
            data.append(img)
            labels.append(class_indices[cls])

    data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values
    data = np.expand_dims(data, axis=-1)  # Add channel dimension
    labels = np.array(labels)
    if class_mode == 'categorical':
        labels = to_categorical(labels, num_classes=len(classes))
    return data, labels

# Load training and validation data
BS = 32
TS = (24, 24)
train_data, train_labels = load_data('data/train', target_size=TS)
valid_data, valid_labels = load_data('data/valid', target_size=TS)

# Calculate steps per epoch
SPE = len(train_data) // BS
VS = len(valid_data) // BS
print(SPE, VS)

# Define the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    x=train_data, y=train_labels,
    validation_data=(valid_data, valid_labels),
    epochs=15, batch_size=BS, steps_per_epoch=SPE, validation_steps=VS
)

# Save the model
model.save('models/cnnCat2.h5', overwrite=True)
