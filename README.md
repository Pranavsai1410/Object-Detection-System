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


Apologies for the confusion earlier! Here's the complete README file written fully in Markdown code format for you:

markdown
Copy code
# Image Segmentation and Object Detection System

## Overview

This project is an advanced **Image Segmentation** and **Object Detection** system that leverages cutting-edge technologies like **YOLOv3**, **PointRend**, and **PixelLib** to perform high-quality, real-time image and video processing. The system allows users to upload images and videos, which are then processed to detect objects, segment them, and return the processed media with bounding boxes or masks overlaid.

## Features

### 1. **Object Detection**
   - Identifies objects in images using the **YOLOv3** model, a state-of-the-art algorithm known for its speed and accuracy.
   - Outputs bounding boxes around detected objects in real-time.

### 2. **Advanced Segmentation**
   - **PointRend** is used to produce high-quality masks for objects in images, providing a more detailed and accurate segmentation.
   - Suitable for tasks like instance segmentation where precise object boundaries are important.

### 3. **Real-time Processing**
   - Supports fast, efficient image and video processing.
   - Designed to handle large files with minimal delay for user-friendly performance.

### 4. **Customizable Models**
   - Easily integrate new datasets into the system or improve existing models to fine-tune performance.
   - Modular design allows for seamless updates and customizations.

### 5. **Visualization Tools**
   - Output includes **bounding boxes** for object detection and **segmented masks** overlaid on the input image.
   - Helps visualize results effectively, making the process intuitive and easy to interpret.

### 6. **Image Processing**
   - Performs **instance segmentation** on uploaded images in formats like `.jpg`, `.jpeg`, `.png` using **PixelLib**.
   - The segmentation output can be displayed with fine boundaries of each object.

### 7. **Video Processing**
   - Detects objects in **videos** (.mp4 format) using the **YOLOv3** model.
   - Processes each frame of the video and returns frames with detected objects.

### 8. **File Upload**
   - Provides a **user-friendly web interface** for easy image and video uploads.
   - Supports common image and video formats.

### 9. **Processed Output**
   - After processing, the system returns the processed image or video to the user.
   - The output includes detected objects with bounding boxes for images and segmented masks for both images and videos.

## Installation

### Prerequisites

1. **Python 3.x**  
2. **Required libraries**:  
   Install the necessary libraries with the following command:
   ```bash
   pip install -r requirements.txt
Additional Dependencies:
YOLOv3 and PointRend for object detection and segmentation.
PixelLib for image processing.
OpenCV for real-time video processing.
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/image-segmentation-object-detection.git
cd image-segmentation-object-detection
Run the Web Interface
To start the application, run:

bash
Copy code
python app.py
Visit http://127.0.0.1:5000 in your web browser to upload images or videos and view processed outputs.

Usage
Upload Image:

Select and upload an image file (.jpg, .jpeg, or .png).
The system will process the image and display the object detection bounding boxes and segmentation masks.
Upload Video:

Upload a video file (.mp4).
The system processes the video, detects objects, and returns frames with the objects detected.
View Results:

After processing, the system displays the result on the web page with bounding boxes or segmented masks for images.
Videos are returned as processed media, where you can view object detection results frame by frame.
Example
Input Image

Output Image (Object Detection + Segmentation)

Input Video

Output Video (Object Detection in Video)

Contributing
If you wish to contribute to this project, feel free to fork the repository and submit pull requests. Make sure to follow the code style and write meaningful commit messages.

Steps to Contribute:
Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes and push them to your forked repository.
Open a pull request to the original repository.
License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- YOLOv3 for its high-performance object detection capabilities.
- PointRend for high-quality segmentation.
- PixelLib for image segmentation and instance segmentation tasks.
- OpenCV for video processing.

# Cloning
To clone the repository, use the following command in your terminal or command line interface:
git clone https://github.com/Pranavsai1410/Object-Detection-System.git
This command will download the entire project, including the codebase, model files, and other resources from the GitHub repository to your local machine. After cloning, you can navigate to the project directory and start using the code or running the web interface!
