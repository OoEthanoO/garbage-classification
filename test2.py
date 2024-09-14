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

def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        for idx, row in enumerate(reader):
            class_names[idx] = row[0]
    return class_names

graph = load_graph("/Users/ethanxu/hack-the-north/garbage-classification/ssd_mobilenet_v2_taco_2018_03_29.pb")
class_names = load_class_names('/Users/ethanxu/hack-the-north/garbage-classification/TACO/detector/taco_config/map_1.csv')

input_tensor = graph.get_tensor_by_name("image_tensor:0")
output_boxes = graph.get_tensor_by_name("detection_boxes:0")
output_scores = graph.get_tensor_by_name("detection_scores:0")
output_classes = graph.get_tensor_by_name("detection_classes:0")

cap = cv2.VideoCapture(0)

with tf.compat.v1.Session(graph=graph) as sess:
    while True:
        ret, image = cap.read()
        if not ret:
            break

        h, w, _ = image.shape
        input_data = cv2.resize(image, (300, 300))
        input_data = np.expand_dims(input_data, axis=0)

        boxes, scores, classes = sess.run(
            [output_boxes, output_scores, output_classes],
            feed_dict={input_tensor: input_data}
        )

        for i in range(len(scores[0])):
            if scores[0][i] > 0.4:  # Threshold for detection
                box = boxes[0][i]
                class_name = class_names[int(classes[0][i])]
                y1, x1, y2, x2 = box
                y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(class_name) + " " + str(scores[0][i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Object Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()