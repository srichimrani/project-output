from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import json
import os

# ✅ MUST be first (disable GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from skimage.feature import hog, local_binary_pattern
import tf_keras
import threadpoolctl

# Safe threadpool init
try:
    threadpoolctl.threadpool_info()
except Exception as e:
    print("threadpool warning:", e)

app = Flask(__name__)

# ───────────────────────────────
# SAFE PATH LOADING
# ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    model_scaler = pickle.load(f)

print("MODEL LOADED OK")
print("SCALER LOADED OK")

# ───────────────────────────────
# TFJS MODEL LOADER
# ───────────────────────────────
def load_tfjs_model(model_json_path):
    model_dir = os.path.dirname(os.path.abspath(model_json_path))

    with open(model_json_path, 'r') as f:
        model_json = json.load(f)

    topology = json.dumps(model_json['modelTopology'])
    keras_model = tf_keras.models.model_from_json(topology)

    tfjs_weights = {}

    for group in model_json['weightsManifest']:
        binary = b''
        for shard_path in group['paths']:
            with open(os.path.join(model_dir, shard_path), 'rb') as f:
                binary += f.read()

        offset = 0
        for entry in group['weights']:
            shape = entry['shape']
            dtype = np.float32 if entry['dtype'] == 'float32' else np.int32

            n_elements = int(np.prod(shape)) if shape else 1
            n_bytes = n_elements * 4

            arr = np.frombuffer(binary[offset:offset + n_bytes], dtype=dtype)
            tfjs_weights[entry['name']] = arr.reshape(shape) if shape else arr

            offset += n_bytes

    keras_weights = []
    for w in keras_model.weights:
        base_name = w.name.split(':')[0]
        if base_name in tfjs_weights:
            keras_weights.append(tfjs_weights[base_name])
        else:
            raise ValueError(f"Missing weight: {base_name}")

    keras_model.set_weights(keras_weights)
    return keras_model


tm_model = load_tfjs_model(os.path.join(BASE_DIR, 'model.json'))
TM_LABELS = ["ME", "NOT ME"]
TM_SIZE = (224, 224)

# ───────────────────────────────
# FACE DETECTION (SAFE)
# ───────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

IMG_SIZE = (64, 64)

def detect_face(img):
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

    face = img[y:y+h, x:x+w]

    if face is None or face.size == 0:
        return None

    return face

# ───────────────────────────────
# FEATURE EXTRACTION (SAFE)
# ───────────────────────────────
def extract_features(img):
    if img is None or img.size == 0:
        return None

    try:
        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        return np.concatenate([hog_feat, lbp_hist]).reshape(1, -1)

    except Exception as e:
        print("Feature error:", e)
        return None

# ───────────────────────────────
# PREPROCESS TM
# ───────────────────────────────
def preprocess_for_tm(img):
    img = cv2.resize(img, TM_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def decode_image(file_storage):
    img_arr = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# ───────────────────────────────
# ROUTES
# ───────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = decode_image(request.files['image'])

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        face = detect_face(img)

        if face is None:
            return jsonify({'error': 'No face detected'}), 400

        features = extract_features(face)

        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        scaled = model_scaler.transform(features)

        prediction = model.predict(scaled)[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(scaled)[0]
        else:
            confidence = np.array([0.5, 0.5])

        is_me = bool(prediction == 1)

        return jsonify({
            'label': 'ME' if is_me else 'NOT ME',
            'confidence': round(float(max(confidence)) * 100, 2),
            'color': 'green' if is_me else 'red',
            'is_me': is_me
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_tm', methods=['POST'])
def predict_tm():
    try:
        img = decode_image(request.files['image'])

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        face = detect_face(img)

        if face is None:
            return jsonify({'error': 'No face detected'}), 400

        processed = preprocess_for_tm(face)
        preds = tm_model.predict(processed)[0]

        idx = int(np.argmax(preds))
        is_me = idx == 0

        return jsonify({
            'label': TM_LABELS[idx],
            'confidence': round(float(np.max(preds)) * 100, 2),
            'color': 'green' if is_me else 'red',
            'is_me': is_me,
            'all_scores': {
                'ME': round(float(preds[0]) * 100, 2),
                'NOT ME': round(float(preds[1]) * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    try:
        file_bytes = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        face = detect_face(img)

        if face is None:
            return jsonify({'error': 'No face detected'}), 400

        features = extract_features(face)

        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        scaled = model_scaler.transform(features)

        knn_pred = model.predict(scaled)[0]

        if hasattr(model, "predict_proba"):
            knn_proba = model.predict_proba(scaled)[0]
            knn_conf = round(float(max(knn_proba)) * 100, 2)
        else:
            knn_conf = 50.0

        knn_is_me = bool(knn_pred == 1)

        tm_preds = tm_model.predict(preprocess_for_tm(face))[0]
        tm_idx = int(np.argmax(tm_preds))
        tm_is_me = tm_idx == 0
        tm_conf = round(float(np.max(tm_preds)) * 100, 2)

        return jsonify({
            'knn': {
                'label': 'ME' if knn_is_me else 'NOT ME',
                'confidence': knn_conf
            },
            'tm': {
                'label': TM_LABELS[tm_idx],
                'confidence': tm_conf
            },
            'agreement': knn_is_me == tm_is_me,
            'winner': 'kNN' if knn_conf > tm_conf else 'Teachable Machine'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
