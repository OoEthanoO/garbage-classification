import tensorflow as tf
import cv2
import numpy as np
import csv

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph("/Users/ethanxu/hack-the-north/garbage-classification/ssd_mobilenet_v2_taco_2018_03_29.pb")

input_tensor = graph.get_tensor_by_name("image_tensor:0")
output_tensor = graph.get_tensor_by_name("detection_boxes:0")

with tf.compat.v1.Session(graph=graph) as sess:
    # Example input
    input_data = np.random.rand(1, 300, 300, 3).astype(np.float32)
    output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})

print(output_data)

import cv2
image = cv2.imread('/Users/ethanxu/hack-the-north/garbage-classification/IMG_0086.jpeg')
image_resized = cv2.resize(image, (300, 300))
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

with tf.compat.v1.Session(graph=graph) as sess:
    boxes, scores, classes = sess.run(
        [graph.get_tensor_by_name('detection_boxes:0'),
         graph.get_tensor_by_name('detection_scores:0'),
         graph.get_tensor_by_name('detection_classes:0')],
        feed_dict={graph.get_tensor_by_name('image_tensor:0'): input_data}
    )

def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        for idx, row in enumerate(reader):
            class_names[idx] = row[0]
    return class_names

class_names = load_class_names('/Users/ethanxu/hack-the-north/garbage-classification/TACO/detector/taco_config/map_1.csv')

h, w, _ = image.shape

for i in range(len(boxes[0])):
    if scores[0][i] > 0.40:  # Confidence threshold
        box = boxes[0][i] * [h, w, h, w]
        class_id = int(classes[0][i])
        label = str(class_names.get(class_id, 'Unknown')) + ': ' + str(scores[0][i])
        cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
        cv2.putText(image, label, (int(box[1]), int(box[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()