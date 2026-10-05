# Weeks 1-3 Image Processing Lab

## Project Overview
This project processes a user image using OpenCV and Matplotlib to perform basic image analysis and image enhancement tasks.

The script is implemented in [weeks01-03_image_lab.py](weeks01-03_image_lab.py) and reads the image from the project folder.

## Image Used
The project image is stored at:
- Images/original.jpg.jpeg

## Setup
From PowerShell in the project folder, create the virtual environment if needed,
install dependencies, and run the lab:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\weeks01-03_image_lab.py
```

## Tasks Covered
1. Image inspection
2. Pixel channel and grayscale views
3. Brightness, contrast, and threshold adjustments
4. Blur and edge detection

## Measured Results
The script was run successfully and the following values were measured:

- Width: 963 pixels
- Height: 1280 pixels
- Channels: 3
- Shape: [1280, 963, 3]
- Pixel count: 1,232,640
- Estimated memory size: 3,697,920 bytes
- Color order: RGB

## Generated Output Files
The following files were generated in the outputs folder:

- pixel_views.png
- adjustments.png
- blur_and_edges.png

## Observations
- The original image is a color image with three channels (RGB).
- The grayscale conversion reduces the color information into one channel.
- Brightness and contrast adjustments changed the luminance and visual clarity of the image.
- Thresholding produced a black-and-white segmentation effect.
- Blur reduced sharpness, and Sobel edge detection emphasized object boundaries.

## Conclusion
This lab helped us understand how images are stored, measured, and transformed using basic computer vision techniques. The image was successfully loaded and processed using OpenCV, and the output visualizations were generated correctly.
