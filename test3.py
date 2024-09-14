import tensorflow as tf

# Path to the frozen graph
frozen_graph_path = "/Users/ethanxu/hack-the-north/garbage-classification/ssd_mobilenet_v2_taco_2018_03_29.pb"

# Input and output tensor names
input_arrays = ["image_tensor"]
output_arrays = ["detection_boxes", "detection_scores", "detection_classes"]

# Convert the model
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    frozen_graph_path, input_arrays, output_arrays
)
tflite_model = converter.convert()

# Save the converted model
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)