import sys
import cv2
import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation

# Load YOLO
def load_yolo(yolo_cfg, yolo_weights, yolo_names):
    net = cv2.dnn.readNet(yolo_cfg, yolo_weights)
    with open(yolo_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Perform instance segmentation using PixelLib
def instance_segmentation(image_path, output_image_name):
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")
    ins.segmentImage(
        image_path,                
        text_thickness=2,           
        show_bboxes=True,           
        output_image_name=output_image_name  
    )

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

# Main function to process video based on the mode
def process_video(video_path, mode, yolo_cfg=None, yolo_weights=None, yolo_names=None):
    # Load YOLO model if mode is 'yolo'
    if mode == "yolo":
        net, classes, output_layers = load_yolo(yolo_cfg, yolo_weights, yolo_names)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Initialize frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize a list to store the processed frames
    processed_frames = []

    # Loop over the frames of the video
    while True:
        # Read a frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Perform processing based on the mode
        if mode == "segmentation":
            # Save frame as image for PixelLib processing
            cv2.imwrite("temp_frame.jpg", frame)
            instance_segmentation("temp_frame.jpg", "temp_output_frame.jpg")
            frame = cv2.imread("temp_output_frame.jpg")
        elif mode == "yolo":
            frame = yolo_object_detection(frame, net, output_layers, classes)

        # Add the processed frame to the list
        processed_frames.append(frame)

    # Release the video capture object
    cap.release()

    # Return the processed frames and dimensions
    return processed_frames, frame_width, frame_height

# Example usage
video_path = r"C:\Users\padal\OneDrive\Desktop\hello\rtrp\hello2.mp4"
mode = "yolo"  # Change to "segmentation" for instance segmentation

if mode == "yolo":
    yolo_cfg = r"C:\Users\padal\OneDrive\Desktop\hello\rtrp\yolov3.cfg"
    yolo_weights = r"C:\Users\padal\OneDrive\Desktop\hello\rtrp\yolov3.weights"
    yolo_names = r"C:\Users\padal\OneDrive\Desktop\hello\rtrp\coco.names"
    processed_frames, frame_width, frame_height = process_video(video_path, mode, yolo_cfg, yolo_weights, yolo_names)
else:
    processed_frames, frame_width, frame_height = process_video(video_path, mode)

# Save the processed frames as a video
out = cv2.VideoWriter('output3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

for frame in processed_frames:
    out.write(frame)

out.release()