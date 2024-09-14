import cv2
import yolov5
from flask import Flask, request, jsonify, send_file
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load pretrained model
model = yolov5.load('best-32e.pt')

# Set model parameters
model.conf = 0.33  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Define class mapping
class_mapping = {
    'Aluminium_Foil': 'recyclable',
    'Bottle': 'recyclable',
    'Bottle_cap': 'recyclable',
    'Broken_glass': 'garbage',
    'Can': 'recyclable',
    'Carton': 'garbage',
    'Cigarette': 'garbage',
    'Cup': 'recyclable',
    'Lid': 'recyclable',
    'Other_Litter': 'garbage',
    'Other_plastic': 'garbage',
    'Paper': 'recyclable',
    'Plastic_bag_wrapper': 'garbage',
    'Plastic_container': 'recyclable',
    'Pop_tab': 'recyclable',
    'Straw': 'garbage',
    'Styrofoam_piece': 'garbage',
    'Unlabeled_litter': 'garbage'
}

# def draw_boxes(img, boxes, labels):
#     for box, label in zip(boxes, labels):
#         x1, y1, x2, y2 = map(int, box)
#         color = (0, 255, 0) if label == 'recyclable' else (0, 0, 255)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_boxes(img, boxes, labels, class_names):
    for box, label, class_name in zip(boxes, labels, class_names):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if label == 'recyclable' else (0, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {label}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    frame = np.array(image)

    # Perform inference
    results = model(frame)

    # Parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    # scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Update labels based on class mapping
    class_names = [model.names[int(cls)] for cls in categories]
    updated_labels = [class_mapping[class_name] for class_name in class_names]

    # Annotate image with updated labels
    # draw_boxes(frame, boxes, updated_labels)

    draw_boxes(frame, boxes, updated_labels, class_names)

    # Convert image back to BGR for OpenCV display
    annotated_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Encode image to send as response
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)