import cv2
import numpy as np
from sklearn.svm import SVC
from flask import Flask, render_template, Response
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import os

dataset_path = 'dataset/'
print("Dataset path exists:", os.path.exists(dataset_path))
print("Contents of dataset folder:", os.listdir(dataset_path) if os.path.exists(dataset_path) else "Folder not found")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print("Created dataset folder:", dataset_path)
def load_dataset(dataset_path):
    X = []
    y = []
    labels = {}
    label_count = 0

    if not os.path.exists(dataset_path):
        print("Dataset path does not exist:", dataset_path)
        return np.array(X), np.array(y), labels  # Return empty arrays

    people = os.listdir(dataset_path)
    if not people:
        print("Dataset folder is empty!")
        return np.array(X), np.array(y), labels  # Return empty arrays

    for person_name in people:
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    face = face_cascade.detectMultiScale(img, 1.3, 5)
                    if len(face) > 0:
                        (x, y, w, h) = face[0]
                        face_image = img[y:y+h, x:x+w]
                        lbp_hist = get_lbp_histogram(face_image)
                        X.append(lbp_hist)
                        y.append(label_count)
            labels[label_count] = person_name
            label_count += 1
    return np.array(X), np.array(y), labels
dataset_path = 'dataset/'

# Ensure dataset folder exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

person_name = input("Enter the person's name: ")
person_folder = os.path.join(dataset_path, person_name)

# Create folder for the person
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

video_capture = cv2.VideoCapture(0)
count = 0

while count < 5:  # Capture 5 images
    ret, frame = video_capture.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        file_name = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(file_name, face)
        print(f"Saved: {file_name}")
        count += 1

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("Image capture completed!")
