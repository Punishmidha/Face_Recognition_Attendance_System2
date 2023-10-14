# Face_Recognition_Attendance_System2

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load known face encodings and names
punish_image = face_recognition.load_image_file("/content/drive/MyDrive/CaoProject/sample1.jpg")
punish_encoding = face_recognition.face_encodings(punish_image)[0]

sukanyag_image = face_recognition.load_image_file("/content/drive/MyDrive/CaoProject/sample2.jpg")
sukanyag_encoding = face_recognition.face_encodings(sukanyag_image)[0]

known_face_encodings = [punish_image, sukanyag_encoding]
known_face_names = ["punish", "sukanya_g"]

students = known_face_names.copy()

# Prepare CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = open(current_date + '.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Continue with the rest of your code
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # The rest of your code continues here

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()

