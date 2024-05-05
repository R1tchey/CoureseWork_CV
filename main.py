# COMPUTER VISION
# COMP3065
# Ruiqi Zhang (20320529)

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QProgressDialog, \
    QHBoxLayout, QSpinBox, QSlider, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class VideoProcessor(QThread):
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)

    def __init__(self, file_path, percent_frame):
        """
        Initialize VideoProcessor with file path and precision percentage.
        
        Args:
            file_path (str): Path to the video file.
            percent_frame (float): Percentage of frames to use in processing.
        """
        super().__init__()
        self.file_path = file_path
        self.percent_frame = percent_frame

    def compute_homography(self, frame, sift, bf, prev_kp, prev_des):
        """
        Compute the homography matrix between the current and previous frame.

        Args:
            frame (ndarray): Current frame to match.
            sift (cv2.SIFT): SIFT object for feature detection.
            bf (cv2.BFMatcher): Brute Force Matcher for feature matching.
            prev_kp (list): Keypoints of the previous frame.
            prev_des (ndarray): Descriptors of the previous frame.

        Returns:
            tuple: The computed homography matrix and the corresponding mask.
        """
        kp, des = sift.detectAndCompute(frame, None)  # Detect keypoints and compute descriptors for the current frame.
        matches = bf.knnMatch(des, prev_des, k=2)  # Find matches using k-Nearest Neighbors.

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Apply Lowe's ratio test to filter good matches.
                good_matches.append(m)

        # Ensure there are enough good matches to calculate a reliable homography matrix.
        if len(good_matches) > 4:
            src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Compute the homography matrix.

        return H, mask

    def erosion_optimization_blend(self, transformed_frame, panorama):
        """
        Apply erosion optimization to blend the transformed frame into the panorama.

        Args:
            transformed_frame (ndarray): Frame after homography transformation.
            panorama (ndarray): Current panorama image.

        Returns:
            ndarray: The updated panorama image.
        """
        valid_mask = (transformed_frame.sum(axis=2) != 0)  # Identify non-black pixels in the frame.
        valid_mask_uint8 = valid_mask.astype(np.uint8) * 255  # Convert the mask to uint8.

        # Apply erosion operation to remove near-black noise around the edges.
        kernel = np.ones((3, 3), np.uint8)
        valid_mask_eroded = cv2.erode(valid_mask_uint8, kernel, iterations=1)

        # Update the panorama using the eroded mask.
        panorama[valid_mask_eroded == 255] = transformed_frame[valid_mask_eroded == 255]
        return panorama

    def run(self):
        """
        Process the video to generate a panorama image.
        This method splits frames into left and right groups, stitches them individually, and finally merges them.
        """
        print(f"Processing video with percent_frame: {self.percent_frame}")
        print('VideoProcessor started')

        # Open the video file and initialize frame groups.
        cap = cv2.VideoCapture(self.file_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_num = int(total_frame_num / 2)

        # Initialize left and right frame groups.
        left_frames = []
        right_frames = []

        # Check if the video is opened successfully.
        if cap.isOpened():
            # Read all frames and distribute them into left and right groups.
            frames = [cap.read() for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
            left_frames = [frame for ret, frame in frames[:middle_num + 1] if ret]
            right_frames = [frame for ret, frame in frames[middle_num + 1:] if ret]

            # Reverse the left frames for correct ordering.
            left_frames.reverse()

            # Release the video resource.
            cap.release()

        # Calculate the number of frames needed based on the precision.
        percent_frame = self.percent_frame
        total_frames = len(right_frames)
        frames_needed = max(1, int(total_frames * percent_frame))

        # Extract indices based on the desired percentage.
        indices = np.linspace(0, total_frames - 1, num=frames_needed, dtype=int)

        # Extract relevant frames.
        reduced_frames_right = [right_frames[idx] for idx in indices]
        reduced_frames_left = [left_frames[idx] for idx in indices]

        # Initialize the panorama with the first right-side frame.
        prev_frame = reduced_frames_right[0]
        height, width = prev_frame.shape[:2]

        # Define dimensions for the right panorama.
        PANORAMA_WIDTH = int(width * 2)
        PANORAMA_HEIGHT = int(height * 2.5)

        pixel_height_0 = int((PANORAMA_HEIGHT - height) / 2)
        pixel_height_1 = int((PANORAMA_HEIGHT + height) / 2)

        ## ------------------- Stitch the Right Part ------------------- ##

        # Create an empty panorama image for the right side.
        panorama_right = np.zeros((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3), dtype=np.uint8)

        # Embed the first frame in the center of the right panorama.
        start_y = int((PANORAMA_HEIGHT - height) / 2)
        end_y = start_y + height
        for row in range(start_y, end_y):
            panorama_right[row, :width] = prev_frame[row - start_y]

        # Update the previous frame for subsequent stitching.
        prev_frame = panorama_right.copy()

        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

        current_frame_index = 0
        for frame in reduced_frames_right:
            current_frame_index += 1
            print("Right: %s" % current_frame_index)
            progress_percent = int((current_frame_index) / (frames_needed * 2) * 100)
            self.progressChanged.emit(progress_percent)

            # Compute homography and transform the current frame.
            MM, mask = self.compute_homography(frame, sift, bf, prev_kp, prev_des)
            transformed_frame = cv2.warpPerspective(frame, MM, (PANORAMA_WIDTH, PANORAMA_HEIGHT))

            panorama_right = self.erosion_optimization_blend(transformed_frame, panorama_right)

            # Update the previous frame for the next iteration.
            prev_frame = transformed_frame
            prev_kp, prev_des = sift.detectAndCompute(panorama_right, None)

        # Detect black pixels in the right panorama to determine the cropping area.
        panorama_right_gray = cv2.cvtColor(panorama_right, cv2.COLOR_BGR2GRAY)
        min_x_first_row = PANORAMA_WIDTH
        min_x_last_row = PANORAMA_WIDTH

        for x in range(width, PANORAMA_WIDTH):
            if (panorama_right_gray[pixel_height_0, x] == 0 or x == PANORAMA_WIDTH):
                min_x_first_row = x
                break

        for x in range(width, PANORAMA_WIDTH):
            if (panorama_right_gray[pixel_height_1 - 1, x] == 0 or x == PANORAMA_WIDTH):
                min_x_last_row = x
                break

        # Determine the minimum x-coordinate to crop.
        min_x = min(min_x_first_row, min_x_last_row)

        # Crop the right panorama to remove black borders.
        cropped_image_right = panorama_right[
                              int(PANORAMA_HEIGHT / 2 - height / 2):int(PANORAMA_HEIGHT / 2 + height / 2), 0:int(min_x)]

        ## --------------- The Right Part is Completed ---------------- ##

        ## ------------------- Stitch the Left Part ------------------- ##

        # Initialize the left panorama with the first left-side frame.
        prev_frame = reduced_frames_left[0]
        panorama_left = np.zeros((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3), np.uint8)
        start_y = int((PANORAMA_HEIGHT - height) / 2)
        start_x = int(PANORAMA_WIDTH - width)

        for i in range(height):
            for j in range(width):
                panorama_left[start_y + i, start_x + j] = prev_frame[i, j]

        prev_frame = panorama_left
        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

        current_frame = 0
        for frame in reduced_frames_left:
            current_frame_index += 1

            progress_percent = int((current_frame_index) / (frames_needed * 2) * 100)
            self.progressChanged.emit(progress_percent)
            current_frame += 1
            print("Left: %s" % current_frame)

            MM, mask = self.compute_homography(frame, sift, bf, prev_kp, prev_des)
            transformed_frame = cv2.warpPerspective(frame, MM, (PANORAMA_WIDTH, PANORAMA_HEIGHT))

            panorama_left = self.erosion_optimization_blend(transformed_frame, panorama_left)

            prev_kp, prev_des = sift.detectAndCompute(panorama_left, None)

        # Detect black pixels in the left panorama to determine the cropping area.
        panorama_left_gray = cv2.cvtColor(panorama_left, cv2.COLOR_BGR2GRAY)
        max_x_first_row = 0
        max_x_last_row = 0

        for x in range(PANORAMA_WIDTH - 1, -1, -1):
            if panorama_left_gray[int(pixel_height_0 + 1), x] == 0 or x == 0:
                max_x_first_row = x
                break

        for x in range(PANORAMA_WIDTH - 1, -1, -1):
            if panorama_left_gray[int(pixel_height_1 - 1), x] == 0 or x == 0:
                max_x_last_row = x
                break

        max_x = max(max_x_first_row, max_x_last_row)

        cropped_image_left = panorama_left[
                             int(PANORAMA_HEIGHT / 2 - height / 2):int(PANORAMA_HEIGHT / 2 + height / 2),
                             int(max_x):PANORAMA_WIDTH]

        ## --------------- The Left Part is Completed ---------------- ##


        ## -------------------- Stitch Two Parts --------------------- ##
        # Initialize the final panorama and embed the cropped left side.
        height_final, width_final = cropped_image_left.shape[:2]
        FINAL_PANORAMA_WIDTH = int(width_final * 2)
        FINAL_PANORAMA_HEIGHT = int(height_final * 2.5)

        panorama_final = np.zeros((FINAL_PANORAMA_HEIGHT, FINAL_PANORAMA_WIDTH, 3), dtype=np.uint8)
        start_y = int((FINAL_PANORAMA_HEIGHT - height_final) / 2)
        end_y = start_y + height_final

        for row in range(start_y, end_y):
            panorama_final[row, :width_final] = cropped_image_left[row - start_y]

        prev_kp, prev_des = sift.detectAndCompute(panorama_final, None)
        MM, mask = self.compute_homography(cropped_image_right, sift, bf, prev_kp, prev_des)
        transformed_frame = cv2.warpPerspective(cropped_image_right, MM, (FINAL_PANORAMA_WIDTH, FINAL_PANORAMA_HEIGHT))

        panorama_final = self.erosion_optimization_blend(transformed_frame, panorama_final)

        panorama_final_gray = cv2.cvtColor(panorama_final, cv2.COLOR_BGR2GRAY)
        min_x = FINAL_PANORAMA_WIDTH

        for x in range(width_final, FINAL_PANORAMA_WIDTH):
            if panorama_final_gray[int(FINAL_PANORAMA_HEIGHT / 2), x] == 0 or x == FINAL_PANORAMA_WIDTH:
                min_x = x
                break

        cropped_panorama_final = panorama_final[
                                 int(FINAL_PANORAMA_HEIGHT / 2 - height_final / 2):int(
                                     FINAL_PANORAMA_HEIGHT / 2 + height_final / 2),
                                 0:int(min_x)]

        if cropped_panorama_final is not None:
            self.finished.emit(cropped_panorama_final)

        print("End of run() method")


class PanoramaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface, including layout, buttons, labels, and sliders.
        """
        self.tip_label = QLabel("Please select a video file (MP4, AVI)")
        self.upload_button = QPushButton("Upload Video")
        self.process_button = QPushButton("Process Panorama")
        self.reset_button = QPushButton("Reset Parameters")
        self.slider = QSlider(Qt.Horizontal)
        self.precision_label = QLabel("Precision: 0.05")
        self.upload_status = QLabel("")
        self.image_label = QLabel()
        self.tip_precision_label = QLabel("Please note that higher accuracy requires longer processing times!")

        # Configure the slider for frame precision selection.
        self.slider.setMinimum(5)
        self.slider.setMaximum(50)
        self.slider.setValue(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        # Layout for all widgets.
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tip_label)
        main_layout.addWidget(self.upload_button)
        main_layout.addWidget(self.upload_status)
        main_layout.addWidget(self.process_button)
        main_layout.addWidget(self.reset_button)
        main_layout.addWidget(QLabel("Select Precision (Key Frame Sampling Rate):"))
        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.precision_label)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.tip_precision_label)

        self.setLayout(main_layout)

        # Set process button style to be more prominent.
        self.process_button.setStyleSheet("font-size: 16px; background-color: #4CAF50; color: white;")

        # Connect button and slider signals to their respective slots.
        self.upload_button.clicked.connect(self.upload_video)
        self.process_button.clicked.connect(self.process_panorama)
        self.reset_button.clicked.connect(self.reset_parameters)
        self.slider.valueChanged.connect(self.update_precision_label)

    def update_precision_label(self):
        """
        Update the precision label based on the current slider value.
        """
        precision_value = self.slider.value() / 100
        self.precision_label.setText(f"Precision: {precision_value:.2f}")

    def upload_video(self):
        """
        Trigger a file dialog for the user to select a video file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        self.file_path = file_path
        if file_path:
            self.upload_status.setText("Video uploaded successfully!")
            self.upload_status.setStyleSheet("color: green;")
        else:
            self.upload_status.setText("No video uploaded.")
            self.upload_status.setStyleSheet("color: red;")

    def process_panorama(self):
        """
        Start processing the video into a panorama by creating a VideoProcessor instance.
        Also initialize a progress dialog to show processing status.
        """
        if not self.file_path:
            self.upload_status.setText("Please upload a video first.")
            self.upload_status.setStyleSheet("color: red;")
            return

        percent_frame = self.slider.value() / 100
        print(f"percent_frame set to: {percent_frame}")

        self.progress_dialog = QProgressDialog("Processing Video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        self.processor = VideoProcessor(self.file_path, percent_frame)
        self.processor.progressChanged.connect(self.progress_dialog.setValue)
        self.processor.finished.connect(self.show_panorama)
        self.processor.start()

    def reset_parameters(self):
        """
        Reset the precision slider to its default value and clear the upload status label.
        """
        self.slider.setValue(5)
        self.precision_label.setText("Precision: 0.05")
        self.upload_status.setText("")

    def show_panorama(self, panorama):
        """
        Display the generated panorama image in the main application window.

        Args:
            panorama (ndarray): The final stitched panorama image.
        """
        self.progress_dialog.close()
        panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        h, w, ch = panorama.shape
        bytes_per_line = ch * w
        q_img = QImage(panorama.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication([])
    window = PanoramaApp()
    window.show()
    app.exec_()
