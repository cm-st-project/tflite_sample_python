import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_unquant1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load labels
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


label_path = "labels.txt"  # Path to your labels file
labels = load_labels(label_path)

# Load and preprocess input image
input_shape = input_details[0]['shape']
input_image = np.array(Image.open("perch.jpg").resize((input_shape[1], input_shape[2])))
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension if needed
input_image = input_image.astype(np.float32)  # Convert to float32 if needed

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Interpret results with labels
top_prediction_index = np.argmax(output_data, axis=1)

top_prediction_label = labels[top_prediction_index[0]]

print("Top prediction label:", top_prediction_label)
