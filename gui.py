import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def load_images(folder_path):
    images = []
    labels = []
    emotion_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

    for emotion_folder in os.listdir(folder_path):
        if emotion_folder in emotion_map:
            label = emotion_map[emotion_folder]
            for file_name in os.listdir(os.path.join(folder_path, emotion_folder)):
                image_path = os.path.join(folder_path, emotion_folder, file_name)
                image = Image.open(image_path)
                images.append(np.array(image))
                labels.append(label)

    return np.array(images), np.array(labels)

train_images, train_labels = load_images('train/')
test_images, test_labels = load_images('test/')

def preprocess_images(images):
    images = images.astype('float32') / 255.0
    images = np.repeat(images[:, :, :, np.newaxis], 3, axis=-1)
    return images


train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

model_json = model.to_json()
with open("model_a1_.json", "w") as json_file:
    json_file.write(model_json)