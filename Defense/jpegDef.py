import numpy as np
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import tensorflow as tf
from PIL import Image
import io

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_data()

# Define a simple CNN model for classification
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model

# Compile the model
input_shape = (28, 28, 1)
model = create_model(input_shape)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Wrap the model using ART's TensorFlowV2Classifier for adversarial robustness
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    nb_classes=10,
    input_shape=input_shape,
    clip_values=(min_pixel_value, max_pixel_value)
)

# Train the model (simplified for brevity)
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=1)

# Apply feature squeezing using bit depth reduction and spatial smoothing
bit_depth_reduction = FeatureSqueezing(bit_depth=5, clip_values=(0, 1))
smoothing = SpatialSmoothing(window_size=3)

def apply_jpeg_compression(x, quality=75):
    compressed_images = []
    for img in x:
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img_pil = Image.fromarray(np.squeeze(img), mode='L')  # Convert to grayscale PIL image
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        img_compressed = Image.open(buffer)
        img_compressed = np.array(img_compressed).astype(np.float32) / 255.0  # Convert back to float32
        compressed_images.append(img_compressed)
    return np.expand_dims(np.array(compressed_images), axis=-1)

def detect_adversarial(x, classifier):
    # Original prediction
    predictions_original = classifier.predict(x)
    

    x_squeezed_bit, _ = bit_depth_reduction(x)
    predictions_squeezed_bit = classifier.predict(x_squeezed_bit)
    

    x_squeezed_smooth, _ = smoothing(x)
    predictions_squeezed_smooth = classifier.predict(x_squeezed_smooth)

    x_jpeg_compressed = apply_jpeg_compression(x, quality=75)
    predictions_jpeg_compressed = classifier.predict(x_jpeg_compressed)
    

    difference_bit = np.mean(np.abs(predictions_original - predictions_squeezed_bit), axis=1)
    difference_smooth = np.mean(np.abs(predictions_original - predictions_squeezed_smooth), axis=1)
    difference_jpeg = np.mean(np.abs(predictions_original - predictions_jpeg_compressed), axis=1)
    

    detection_threshold = 0.1
    
    # Detect if the input is adversarial based on the differences
    is_adversarial = (difference_bit > detection_threshold) | (difference_smooth > detection_threshold) | (difference_jpeg > detection_threshold)
    
    return is_adversarial


adversarial_detection = detect_adversarial(x_test[:10], classifier)

# Print the results
for i, is_adv in enumerate(adversarial_detection):
    print(f"Input {i} is {'adversarial' if is_adv else 'normal'}")
