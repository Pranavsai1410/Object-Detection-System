# 🖼️ Image Segmentation and Object Detection System

## 🚀 Overview

Welcome to the **Image Segmentation and Object Detection System**!  
This project combines the power of YOLOv3 and PointRend to deliver advanced image segmentation and object detection capabilities. It identifies objects in images and accurately segments them to highlight key areas with precision.
This allows users to upload images and videos for object detection and instance segmentation using:

YOLO (You Only Look Once) for object detection in videos.
PixelLib's Instance Segmentation for images.
After the media files are uploaded, they are processed, and the application returns the processed media (segmented images or object-detected videos) back to the user.

---

## 🏗️ Features

- **Image Processing**: Performs instance segmentation on uploaded images (.jpg, .jpeg, .png) using PixelLib.
- **Object Detection**: Detects objects in images using YOLOv3 with bounding box overlays.
- **Video Processing**: Detects objects in videos (.mp4) using the YOLO model.
- **Advanced Segmentation**: Uses PointRend for high-quality instance segmentation masks.
- **Processed Output**: Returns processed media (image/video) to the user with detected objects or segmentations.

---

## 📸 Example Outputs

### Input Image:
![Screenshot 2024-06-17 163228](https://github.com/user-attachments/assets/38d49c16-a573-45de-ad3c-17588af8fe98)


### Output Image with Segmentation and Detection:
![Screenshot 2024-06-17 163507](https://github.com/user-attachments/assets/5810e09f-498d-4986-ab11-5c48d2bd00b7)

---
# Usage
### Upload Image:
- Select and upload an image file (.jpg, .jpeg, or .png).
- The system will process the image and display the object detection bounding boxes and segmentation masks.

### Upload Video:
- Upload a video file (.mp4).
- The system processes the video, detects objects, and returns frames with the objects detected.
### View Results:
- After processing, the system displays the result on the web page with bounding boxes or segmented masks for images.
- Videos are returned as processed media, where you can view object detection results frame by frame.
