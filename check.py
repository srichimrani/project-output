# check.py
import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def count_faces(folder):
    detected  = 0
    not_found = 0

    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img      = cv2.imread(img_path)

        if img is None:
            continue

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            detected += 1
        else:
            not_found += 1
            print(f"No face found in: {img_file}")

    return detected, not_found

print("-- Checking ME folder --")
d, n = count_faces('dataset/ME')
print(f"Faces detected : {d}")
print(f"No face found  : {n}")

print("\n-- Checking NOT_ME folder --")
d, n = count_faces('dataset/NOT_ME')
print(f"Faces detected : {d}")
print(f"No face found  : {n}")