import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.image.resize(img_array, target_size) / 255.0
    return img_array


def process_image(img, image_size=224):
    image = np.squeeze(img)
    image = tf.image.resize(image, (image_size, image_size)) / 255.0
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('model_path', type=str, help='Path to saved model')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names')
    return parser.parse_args()

def predict(image_path, model, top_k, class_names):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    top_classes = [class_names[str(value)] for value in top_indices.numpy()[0]]
    return top_values.numpy()[0], top_classes

def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    if model is None:
        print("Loading model failed, recreating manually...")
        base_model = hub.KerasLayer(args.model_path, trainable=False)
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(32,activation= 'relu'),
            tf.keras.layers.Dense(102, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        classes = {int(i): class_names[str(i)] for i in classes}

    top_values, top_classes = predict(args.image_path, model, args.top_k, classes)

    print("These are the top probabilities:", top_values)
    print("Of these top classes:", top_classes)

if __name__ == '__main__':
    main()