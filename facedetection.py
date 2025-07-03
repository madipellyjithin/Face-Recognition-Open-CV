import cv2
import os

# Dataset path
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Enter multiple person names (comma-separated)
person_names = input("Enter names of persons (comma-separated): ").split(",")

for person_name in person_names:
    person_name = person_name.strip()  # Remove extra spaces
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"\nCapturing images for {person_name}...")

    while count < 50:  # Capture 50 images per person
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            file_name = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(file_name, face)
            print(f"Saved: {file_name}")
            count += 1

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 50:
            break

    cap.release()
    print(f"Face capture completed for {person_name}!\n")

cv2.destroyAllWindows()
print("All face captures completed!")
