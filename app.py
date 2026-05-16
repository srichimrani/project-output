# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from skimage.feature import hog, local_binary_pattern
import tf_keras
import threadpoolctl

try:
    threadpoolctl.threadpool_info()
except Exception as e:
    print("Warning: threadpoolctl initialization failed:", e)

app = Flask(__name__)

# ── Conventional ML Model (HOG + LBP + kNN) ──────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    model_scaler = pickle.load(f)

# ── TF.js model loader ────────────────────────────────────────────────────────
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
            raise ValueError(f"Weight {base_name} not found in TFJS weights!")
    keras_model.set_weights(keras_weights)
    return keras_model

tm_model  = load_tfjs_model('model.json')
TM_LABELS = ["ME", "NOT ME"]   # Class 1 = ME, Class 2 = NOT ME
TM_SIZE   = (224, 224)

# ── Shared utilities ──────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
IMG_SIZE = (64, 64)

def detect_face(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img[y:y+h, x:x+w]

def extract_features(img):
    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return np.concatenate([hog_feat, lbp_hist]).reshape(1, -1)

def preprocess_for_tm(img):
    img = cv2.resize(img, TM_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def decode_image(file_storage):
    img_arr = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = decode_image(request.files['image'])
        if img is None:
            return jsonify({'error': 'Invalid image format.'}), 400
        face = detect_face(img)
        if face is None:
            return jsonify({'error': 'No face detected.'}), 400
        features        = extract_features(face)
        scaled_features = model_scaler.transform(features)
        prediction      = model.predict(scaled_features)[0]
        confidence      = model.predict_proba(scaled_features)[0]
        is_me           = bool(prediction == 1)
        return jsonify({
            'label': 'ME' if is_me else 'NOT ME',
            'confidence': round(float(max(confidence)) * 100, 2),
            'color': 'green' if is_me else 'red',
            'is_me': is_me
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_tm', methods=['POST'])
def predict_tm():
    try:
        img = decode_image(request.files['image'])
        if img is None:
            return jsonify({'error': 'Invalid image format.'}), 400
        face = detect_face(img)
        if face is None:
            return jsonify({'error': 'No face detected.'}), 400
        processed = preprocess_for_tm(face)
        preds     = tm_model.predict(processed)[0]
        class_idx = int(np.argmax(preds))
        is_me     = class_idx == 0
        return jsonify({
            'label': TM_LABELS[class_idx],
            'confidence': round(float(np.max(preds)) * 100, 2),
            'color': 'green' if is_me else 'red',
            'is_me': is_me,
            'all_scores': {'ME': round(float(preds[0])*100,2), 'NOT ME': round(float(preds[1])*100,2)}
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Run BOTH models and return unified comparison result."""
    try:
        file_bytes = request.files['image'].read()
        img_arr    = np.frombuffer(file_bytes, np.uint8)
        img        = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format.'}), 400
        face = detect_face(img)
        if face is None:
            return jsonify({'error': 'No face detected in the image.'}), 400

        # kNN
        features        = extract_features(face)
        scaled_features = model_scaler.transform(features)
        knn_pred        = model.predict(scaled_features)[0]
        knn_proba       = model.predict_proba(scaled_features)[0]
        knn_is_me       = bool(knn_pred == 1)
        knn_conf        = round(float(max(knn_proba)) * 100, 2)

        # Teachable Machine
        processed = preprocess_for_tm(face)
        tm_preds  = tm_model.predict(processed)[0]
        tm_idx    = int(np.argmax(tm_preds))
        tm_is_me  = tm_idx == 0
        tm_conf   = round(float(np.max(tm_preds)) * 100, 2)

        return jsonify({
            'knn': {
                'label':      'ME' if knn_is_me else 'NOT ME',
                'confidence': knn_conf,
                'is_me':      knn_is_me,
                'color':      'green' if knn_is_me else 'red'
            },
            'tm': {
                'label':      TM_LABELS[tm_idx],
                'confidence': tm_conf,
                'is_me':      tm_is_me,
                'color':      'green' if tm_is_me else 'red',
                'all_scores': {'ME': round(float(tm_preds[0])*100,2), 'NOT ME': round(float(tm_preds[1])*100,2)}
            },
            'agreement': knn_is_me == tm_is_me,
            'winner':    'kNN (HOG+LBP)' if knn_conf > tm_conf else 'Teachable Machine'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
