# Panorama Generation from Videos

## Coursework: Panorama Generation from Videos

**Course:** Computer Vision (COMP3065)
**Author:** Ruiqi Zhang (20320529)

### Overview

This project is a coursework assignment focused on creating panoramic images from video files. It leverages computer vision techniques such as feature detection and image stitching to generate high-quality panoramas by analyzing video frames.

### Features

- **Video Upload and Processing:** Upload a video file to be processed into a panorama.
- **Key Frame Precision:** Choose the precision of keyframe sampling using a slider.
- **Image Stitching:** Stitch left and right frames into a seamless panorama using feature matching and homography.
- **Progress Dialog:** Monitor the real-time progress of the processing with a dialog.
- **Image Preview:** Preview the final generated panorama within the application.

### Technologies Used

- Python 3.x
- OpenCV
- PyQt5

### Installation

1. Clone the Repository:

   Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/R1tchey/CoureseWork_CV.git
   ```

2. Install Dependencies:

   Make sure Python 3 is installed. Then, install the required packages using `pip`

   ```bash
   pip install opencv-python PyQt5 numpy
   ```

### How to Run

1. Navigate to the project directory.

2. Run the main application file using:

   ```bash
   python main.py
   ```

3. The application window will appear. Follow these steps:

   - Click "Upload Video" to upload your video file.
   - Adjust the precision slider to set the keyframe sampling rate.
   - Click "Process Panorama" to start generating the panorama.
   - View the generated panorama once the process is complete.

### Notes

- Make sure the video file is in `.mp4` or `.avi` format for compatibility.
- The processing time will vary depending on the precision selected.

### Contact

For any inquiries, feel free to reach out to the author:

- **Ruiqi Zhang** ([scyrz5@nottingham.edu.cn](scyrz5@nottingham.edu.cn))
