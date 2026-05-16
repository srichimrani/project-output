# train_knn.py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import pickle

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
IMG_SIZE = (64, 64)

def detect_face(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img[y:y+h, x:x+w]

def extract_features(img):
    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.concatenate([hog_feat, lbp_hist])

def augment(img):
    h, w = img.shape[:2]
    M_l  = cv2.getRotationMatrix2D((w//2, h//2),  10, 1.0)
    M_r  = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
    return [
        img,
        cv2.flip(img, 1),
        cv2.convertScaleAbs(img, alpha=1.2, beta=20),
        cv2.convertScaleAbs(img, alpha=0.8, beta=-20),
        cv2.warpAffine(img, M_l, (w, h)),
        cv2.warpAffine(img, M_r, (w, h)),
    ]

def load_dataset(data_dir):
    X, y = [], []
    for label, folder in enumerate(['NOT_ME', 'ME']):
        path  = os.path.join(data_dir, folder)
        count = 0
        print(f"Loading {folder}...")
        for f in os.listdir(path):
            img  = cv2.imread(os.path.join(path, f))
            if img is None: continue
            face = detect_face(img)
            if face is None: continue
            for v in augment(face):
                X.append(extract_features(v))
                y.append(label)
                count += 1
        print(f"  {count} images loaded")
    return np.array(X), np.array(y)

# ── Load ──
print("=" * 40)
print("  kNN MODEL")
print("=" * 40)
X, y = load_dataset('dataset')

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ──
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Fine-tune ──
print("\nFine-tuning...")
params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan', 'minkowski']
}
grid = GridSearchCV(
    KNeighborsClassifier(),
    params,
    cv=5,
    scoring='accuracy',
    verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest settings : {grid.best_params_}")
model = grid.best_estimator_

# ── Evaluate ──
y_pred = model.predict(X_test)
print("\n-- Results --")
print(f"Accuracy     : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"K value used : {model.n_neighbors}")
print(classification_report(
    y_test, y_pred,
    target_names=['NOT ME', 'Brad Pitt']
))

# ── Save ──
with open('model.pkl',  'wb') as f: pickle.dump(model,  f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
print("Saved model.pkl and scaler.pkl")