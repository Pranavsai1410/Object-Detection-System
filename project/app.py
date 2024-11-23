import os
import cv2
import numpy as np
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pixellib.torchbackend.instance import instanceSegmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Load YOLO model
def load_yolo(yolo_cfg, yolo_weights, yolo_names):
    if not os.path.exists(yolo_cfg):
        raise FileNotFoundError(f"YOLO configuration file not found: {yolo_cfg}")
    if not os.path.exists(yolo_weights):
        raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights}")
    if not os.path.exists(yolo_names):
        raise FileNotFoundError(f"YOLO names file not found: {yolo_names}")

    net = cv2.dnn.readNet(yolo_cfg, yolo_weights)
    with open(yolo_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Perform instance segmentation using PixelLib
def instance_segmentation(image_path, output_image_name, ins):
    ins.segmentImage(
        image_path,
        text_thickness=2,
        show_bboxes=True,
        output_image_name=output_image_name
    )
    return cv2.imread(output_image_name)  # Return the processed image

# Perform object detection using YOLO
def yolo_object_detection(frame, net, output_layers, classes):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

# Process media
def process_media(path, yolo_cfg=None, yolo_weights=None, yolo_names=None, seg_model_path=None):
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()

    # Load YOLO model if processing a video
    if file_extension == ".mp4":
        net, classes, output_layers = load_yolo(yolo_cfg, yolo_weights, yolo_names)
    else:
        net, classes, output_layers = None, None, None

    # Load PixelLib model if processing an image
    if file_extension in [".jpg", ".jpeg", ".png"]:
        ins = instanceSegmentation()
        ins.load_model(seg_model_path)
    else:
        ins = None

    # Process image
    if file_extension in [".jpg", ".jpeg", ".png"] and os.path.isfile(path):
        output_image_name = "static/uploads/output_image.jpg"
        processed_frame = instance_segmentation(path, output_image_name, ins)
    
    # Process video
    elif file_extension == ".mp4" and os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {path}")
        
        processed_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = yolo_object_detection(frame, net, output_layers, classes)
            processed_frames.append(processed_frame)
        
        cap.release()

        # Save processed frames as video
        output_video_path = "static/uploads/output_video.mp4"
        height, width, _ = processed_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

        for frame in processed_frames:
            out.write(frame)
        out.release()

        # Return the path to the processed video
        return output_video_path

    return processed_frame

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            yolo_cfg = r"C:\Users\chand\OneDrive\Documents\Siddhu\yolov3.cfg"
            yolo_weights = r"C:\Users\chand\OneDrive\Documents\Siddhu\yolov3.weights"
            yolo_names = r"C:\Users\chand\OneDrive\Documents\Siddhu\coco.names"
            seg_model_path = r"C:\Users\chand\OneDrive\Documents\Siddhu\pointrend_resnet50.pkl"

            processed_media = process_media(file_path, yolo_cfg, yolo_weights, yolo_names, seg_model_path)

            if isinstance(processed_media, str):
                return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(processed_media))
            else:
                output_image_path = "static/uploads/output_image.jpg"
                cv2.imwrite(output_image_path, processed_media)
                return send_from_directory(app.config['UPLOAD_FOLDER'], "output_image.jpg")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)