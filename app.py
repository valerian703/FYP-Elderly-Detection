import os
import cv2
import dlib
import base64
import numpy as np
import json
import time
import threading
import traceback
import re
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, Response, jsonify, stream_with_context)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError

# --- Basic Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_very_secret_key_auto_capture')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///security_system_auto_capture.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Model Paths & Config ---
DLIB_LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DNN_MODEL_FILE = "deploy.prototxt"
DNN_WEIGHTS_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.5
POSES = ["front", "left", "right", "chin_up", "chin_down", "mouth_open"]
MAX_IMAGES_TO_CAPTURE = 5
CAPTURE_DELAY_SECONDS = 3
MOUTH_OPEN_THRESHOLD = 15

# --- Global Variables ---
landmark_detector = None
face_recognizer = None
dnn_detector = None
known_face_data = {"embeddings": [], "labels": []}
login_recognition_state = {"status": "initializing", "username": None, "timestamp": 0}
registration_process_state = {
    "status": "idle", "current_pose_index": 0, "total_images_captured": 0,
    "instruction": "Please start the camera.", "error_message": None,
    "face_detected_for_auto_capture": False,
    "auto_capture_countdown": None,
    "auto_capture_start_time": None
}
state_lock = threading.Lock()

# --- Database Model (User - Same as before) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    hashed_pin = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    birthday = db.Column(db.String(10), nullable=True)
    embedding = db.Column(db.Text, nullable=True)
    def __repr__(self): return f'<User {self.username} ({self.role})>'

# --- Model Loading, Embedding Helpers, DB Load ---
def load_all_models():
    global landmark_detector, dnn_detector, face_recognizer
    print("Loading face recognition models...")
    models_loaded = True; error_messages = []
    if os.path.exists(DLIB_LANDMARK_MODEL_PATH):
        try: landmark_detector = dlib.shape_predictor(DLIB_LANDMARK_MODEL_PATH); print("  - Dlib Landmark model loaded.")
        except Exception as e: error_messages.append(f"Failed Landmark: {e}"); traceback.print_exc()
    else: error_messages.append(f"Dlib Landmark model not found: {DLIB_LANDMARK_MODEL_PATH}")
    if os.path.exists(FACE_RECOGNITION_MODEL_PATH):
        try: face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH); print("  - Dlib Recognition model loaded.")
        except Exception as e: error_messages.append(f"Failed Recognition: {e}"); traceback.print_exc()
    else: error_messages.append(f"Dlib Face Recognition model not found: {FACE_RECOGNITION_MODEL_PATH}")
    if os.path.exists(DNN_MODEL_FILE) and os.path.exists(DNN_WEIGHTS_FILE):
        try: dnn_detector = cv2.dnn.readNetFromCaffe(DNN_MODEL_FILE, DNN_WEIGHTS_FILE); print("  - OpenCV DNN detector loaded.")
        except Exception as e: error_messages.append(f"Failed DNN: {e}"); traceback.print_exc()
    else: error_messages.append(f"OpenCV DNN model files not found ({DNN_MODEL_FILE}, {DNN_WEIGHTS_FILE}).")
    if error_messages: print("--- MODEL LOADING ERRORS ---"); [print(f" - {msg}") for msg in error_messages]; models_loaded = False
    if not all([landmark_detector, face_recognizer, dnn_detector]): print("CRITICAL: One or more essential models failed to load."); models_loaded = False
    if models_loaded: print("All essential models loaded successfully.")
    return models_loaded

def normalize_embedding(embedding):
    if isinstance(embedding, dlib.vector): embedding = np.array(embedding)
    return embedding

def compute_face_embedding(image, rect):
    if not landmark_detector or not face_recognizer: return None
    try:
        shape = landmark_detector(image, rect)
        embedding_vector = face_recognizer.compute_face_descriptor(image, shape, 1)
        return normalize_embedding(embedding_vector)
    except Exception as e: print(f"Error computing embedding: {e}"); return None

def is_mouth_open(image_bgr, rect):
    if not landmark_detector: return False
    try:
        shape = landmark_detector(image_bgr, rect)
        top_lip_y = shape.part(62).y
        bottom_lip_y = shape.part(66).y
        distance = abs(bottom_lip_y - top_lip_y)
        return distance > MOUTH_OPEN_THRESHOLD
    except Exception as e: print(f"DEBUG: Server-side is_mouth_open error: {e}"); return False

# REVISED FUNCTION load_embeddings_from_db
def load_embeddings_from_db(force_reload=False):
    global known_face_data
    if not force_reload and known_face_data.get("embeddings") and known_face_data.get("labels"):
        print("Using cached known face data for login.")
        return True

    print("Loading known embeddings from database...")
    local_embeddings = []
    local_labels = []
    loaded_count = 0
    try:
        with app.app_context():
            users_with_embeddings = User.query.filter(User.embedding.isnot(None)).all()
        
        for user_db in users_with_embeddings:
            try:
                embedding_list = json.loads(user_db.embedding)
                embedding_np = np.array(embedding_list)
                if embedding_np.ndim == 1 and embedding_np.size > 0 : # Basic validation
                    local_embeddings.append(embedding_np)
                    local_labels.append(user_db.username)
                    loaded_count += 1
                else:
                    print(f"Warning: Invalid embedding structure for user {user_db.username}. Skipping.")
            except Exception as e_parse: # More specific variable name for inner exception
                print(f"Warning: Could not parse embedding for user {user_db.username}: {e_parse}")
        
        # Successfully processed all users (or no users with embeddings found)
        with state_lock:
            known_face_data["embeddings"] = local_embeddings
            known_face_data["labels"] = local_labels
        print(f"Loaded {loaded_count} embeddings from database.")
        return True # Moved here: indicates successful completion of try block

    except Exception as e_db: # Outer exception for major DB query/load issues (e.g., DB connection)
        print(f"ERROR querying database for embeddings: {e_db}")
        traceback.print_exc()
        with state_lock:
            # Reset global state to a safe default on critical error
            known_face_data = {"embeddings": [], "labels": []}
        return False # Indicates failure to load from DB


def decode_image(data_url):
    try:
        encoded_data = data_url.split(',')[1]; nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR); return img
    except Exception as e: print(f"Error decoding image: {e}"); return None

# --- Video Generators ---
def generate_frames_simple(): # (Same as before)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened(): print("Error: Could not open video stream for simple feed."); return
    while True:
        success, frame = camera.read()
        if not success: break
        if dnn_detector:
            (h, w) = frame.shape[:2]; blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_detector.setInput(blob); detections = dnn_detector.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > DNN_CONFIDENCE_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        frame_bytes = buffer.tobytes(); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

def login_video_generator(): # (Same as in previous corrected version)
    global login_recognition_state
    cap = cv2.VideoCapture(0)
    last_recognition_time = 0
    recognition_interval = 0.5
    if not cap.isOpened():
        print("CRITICAL ERROR: Cannot open camera for login feed.")
        return
    print("Login video generator started.")

    while True:
        ret, frame = cap.read()
        current_time = time.time()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        (h, w) = display_frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_detector.setInput(blob)
        detections = dnn_detector.forward()

        best_conf = 0.0
        best_box = None
        best_rect = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE_THRESHOLD and confidence > best_conf:
                best_conf = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                if startX < endX and startY < endY:
                    best_box = (startX, startY, endX, endY)
                    best_rect = dlib.rectangle(startX, startY, endX, endY)

        if best_rect and (current_time - last_recognition_time > recognition_interval):
            last_recognition_time = current_time
            embedding = compute_face_embedding(frame, best_rect)

            if embedding is not None:
                with state_lock:
                    current_known_embeddings = list(known_face_data.get("embeddings", []))
                    current_known_labels = list(known_face_data.get("labels", []))

                if current_known_embeddings:
                    try:
                        known_embeddings_np = np.array(current_known_embeddings)
                        if known_embeddings_np.ndim != 2 or embedding.ndim != 1 or \
                           known_embeddings_np.shape[1] != embedding.shape[0]:
                            print(f"Shape mismatch error in login_video_generator. Known: {known_embeddings_np.shape}, Current: {embedding.shape}")
                            raise ValueError("Embedding shape mismatch for distance calculation.")

                        distances = np.linalg.norm(known_embeddings_np - embedding, axis=1)
                        min_distance_idx = np.argmin(distances)
                        min_distance = distances[min_distance_idx]

                        recognized_username_val = None
                        recognition_status_val = "unknown"

                        if min_distance < SIMILARITY_THRESHOLD:
                            recognized_username_val = current_known_labels[min_distance_idx]
                            recognition_status_val = "recognized"
                        
                        with state_lock:
                            login_recognition_state.update({
                                "status": recognition_status_val,
                                "username": recognized_username_val,
                                "timestamp": current_time
                            })
                    except Exception as e_recog:
                        print(f"Error during login face recognition processing: {e_recog}")
                        traceback.print_exc()
                        with state_lock:
                            login_recognition_state.update({
                                "status": "embed_error",
                                "username": None,
                                "timestamp": current_time
                            })
                else: 
                    with state_lock:
                        login_recognition_state.update({
                            "status": "no_known_faces",
                            "username": None,
                            "timestamp": current_time
                        })
            else: 
                with state_lock:
                    login_recognition_state.update({
                        "status": "embed_error",
                        "username": None,
                        "timestamp": current_time
                    })
        elif not best_rect: 
            with state_lock:
                if current_time - login_recognition_state.get("timestamp", 0) > 2.0:
                    login_recognition_state.update({
                        "status": "detecting",
                        "username": None
                    })

        status_text = ""
        box_color = (0, 0, 255)
        with state_lock:
            current_status = login_recognition_state["status"]
            current_rec_user = login_recognition_state["username"]

        if current_status == "recognized" and current_rec_user:
            status_text = f"Recognized: {current_rec_user}"
            box_color = (0, 255, 0)
        elif current_status == "unknown":
            status_text = "Unknown Face"
        elif current_status == "detecting":
            status_text = "Looking for face..."
            box_color = (255, 165, 0)
        elif current_status == "no_known_faces":
            status_text = "No faces registered"
        elif current_status == "embed_error":
            status_text = "Recognition Error"
        elif current_status == "initializing":
            status_text = "Initializing..."

        if best_box:
            (startX, startY, endX, endY) = best_box
            cv2.rectangle(display_frame, (startX, startY), (endX, endY), box_color, 2)
            y_text_pos = startY - 15 if startY > 15 else startY + 25
            cv2.putText(display_frame, status_text, (startX, y_text_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        else:
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    print("Releasing login camera.")
    cap.release()
    print("Login video generator stopped.")

# --- Routes (Rest of the routes remain the same as in the previous complete code block) ---
@app.route('/')
def home():
    if 'username' in session: return redirect(url_for('private_data'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session: return redirect(url_for('private_data'))
    if request.method == 'POST':
        username = request.form.get('username'); pin = request.form.get('pin'); role = request.form.get('role')
        if not username or not pin or not role: flash('Username, PIN, and Role are required.', 'error'); return render_template('login.html')
        user_db = User.query.filter_by(username=username).first()
        if user_db and user_db.role == role and check_password_hash(user_db.hashed_pin, pin):
            session['username'] = username; session['role'] = role; flash(f'Welcome back, {username}!', 'success'); return redirect(url_for('private_data'))
        else: flash('Invalid credentials or role.', 'error'); return render_template('login.html')
    with state_lock:
        login_recognition_state["status"] = "detecting"
        login_recognition_state["username"] = None
        login_recognition_state["timestamp"] = time.time()
    return render_template('login.html')

@app.route('/check-user-exists', methods=['POST'])
def check_user_exists():
    data = request.get_json(); username = data.get('username')
    user_exists = User.query.filter_by(username=username).first() is not None
    return jsonify({'exists': user_exists})

@app.route('/register-user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        username = request.form.get('username'); pin = request.form.get('pin'); confirm_pin = request.form.get('confirmPin')
        phone = request.form.get('phone'); birthday = request.form.get('birthday'); role = request.form.get('role')
        error = None
        if not all([username, pin, confirm_pin, phone, birthday, role]): error = "All fields are required."
        elif pin != confirm_pin: error = "PINs do not match."
        elif len(pin) != 6 or not pin.isdigit(): error = "PIN must be exactly 6 digits."
        elif User.query.filter_by(username=username).first(): error = f"Username '{username}' is already taken."
        elif not re.fullmatch(r'[a-zA-Z0-9_.-]+', username): error = "Username contains invalid characters."
        if error: flash(error, 'error'); return render_template('register.html')
        session['temp_registration_data'] = {
            'username': username, 'hashed_pin': generate_password_hash(pin),
            'phone': phone, 'birthday': birthday, 'role': role
        }
        session['temp_face_embeddings'] = []
        session.modified = True
        return redirect(url_for('face_registration'))
    return render_template('register.html')

@app.route('/face-registration', methods=['GET'])
def face_registration():
    if 'temp_registration_data' not in session:
        flash('Please complete registration details first.', 'error'); return redirect(url_for('register_user'))
    if not all([dnn_detector, landmark_detector, face_recognizer]):
         flash('Face processing models not ready.', 'error'); return redirect(url_for('register_user'))
    with state_lock:
        current_embeddings_in_session = session.get('temp_face_embeddings', [])
        initial_total_images = len(current_embeddings_in_session) if isinstance(current_embeddings_in_session, list) else 0
        current_pose_index_for_state = 0
        initial_instruction = f"Get ready for: {POSES[current_pose_index_for_state % len(POSES)].upper() if POSES else 'First Pose'}"

        registration_process_state.update({
            "status": "active",
            "current_pose_index": current_pose_index_for_state,
            "total_images_captured": initial_total_images,
            "auto_capture_countdown": None,
            "auto_capture_start_time": None,
            "face_detected_for_auto_capture": False,
            "instruction": initial_instruction,
            "error_message": None
        })
    return render_template('face_registration.html',
                           poses_json=json.dumps(POSES),
                           max_images=MAX_IMAGES_TO_CAPTURE,
                           capture_delay_ms = CAPTURE_DELAY_SECONDS * 1000)

@app.route('/upload-face', methods=['POST'])
def upload_face():
    if 'temp_registration_data' not in session or 'temp_face_embeddings' not in session:
        return jsonify({'status': 'error', 'message': 'Session expired.'}), 400

    current_embeddings_in_session = session.get('temp_face_embeddings', [])
    if not isinstance(current_embeddings_in_session, list):
        current_embeddings_in_session = []; session['temp_face_embeddings'] = current_embeddings_in_session

    if len(current_embeddings_in_session) >= MAX_IMAGES_TO_CAPTURE:
         return jsonify({'status': 'error', 'message': 'Max images captured.'}), 400

    try:
        image_data_url = request.data.decode('utf-8'); img = decode_image(image_data_url)
        if img is None:
            with state_lock:
                registration_process_state.update({
                    "error_message": "Failed to decode image.",
                    "instruction": "Capture failed: Image could not be decoded.",
                    "face_detected_for_auto_capture": False,
                    "auto_capture_start_time": None,
                    "auto_capture_countdown": None
                })
            return jsonify({'status': 'error', 'message': 'Failed to decode image.'}), 400

        (h, w) = img.shape[:2]; blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_detector.setInput(blob); detections = dnn_detector.forward()
        best_conf = 0.0; best_rect = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE_THRESHOLD and confidence > best_conf:
                best_conf = confidence; box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int"); startX, startY = max(0, startX), max(0, startY); endX, endY = min(w - 1, endX), min(h - 1, endY)
                if startX < endX and startY < endY: best_rect = dlib.rectangle(startX, startY, endX, endY)

        if not best_rect:
            with state_lock:
                registration_process_state.update({
                    "error_message": "No face detected clearly.",
                    "instruction": "Capture failed: No face detected clearly. Try again.",
                    "face_detected_for_auto_capture": False, 
                    "auto_capture_start_time": None,
                    "auto_capture_countdown": None
                })
            return jsonify({'status': 'error', 'message': 'No face detected clearly.'}), 400
        
        embedding = compute_face_embedding(img, best_rect)
        if embedding is None:
            with state_lock:
                registration_process_state.update({
                    "error_message": "Could not compute face embedding.",
                    "instruction": "Capture failed: Could not process face. Try again.",
                    "face_detected_for_auto_capture": False,
                    "auto_capture_start_time": None,
                    "auto_capture_countdown": None
                })
            return jsonify({'status': 'error', 'message': 'Could not compute face embedding.'}), 400

        current_embeddings_in_session.append(embedding.tolist())
        session['temp_face_embeddings'] = current_embeddings_in_session
        session.modified = True

        with state_lock:
            new_total_captured = len(current_embeddings_in_session)
            registration_process_state["total_images_captured"] = new_total_captured
            registration_process_state["error_message"] = None 

            registration_process_state["face_detected_for_auto_capture"] = False
            registration_process_state["auto_capture_start_time"] = None
            registration_process_state["auto_capture_countdown"] = None

            if new_total_captured < MAX_IMAGES_TO_CAPTURE:
                registration_process_state["instruction"] = f"Image {new_total_captured} captured successfully. Prepare for next pose."
            else: 
                registration_process_state["instruction"] = "All images captured! Please submit."
                registration_process_state["status"] = "pending_submit"

        return jsonify({'status': 'success',
                        'message': f'Image {len(current_embeddings_in_session)} processed.',
                        'images_captured': len(current_embeddings_in_session)})
    except Exception as e:
        print(f"Error processing face upload: {e}"); traceback.print_exc()
        with state_lock:
            registration_process_state.update({
                "status":"error", 
                "error_message": "Server error during upload.",
                "instruction": "Server error during upload. Please try again or restart registration.",
                "face_detected_for_auto_capture": False,
                "auto_capture_start_time": None,
                "auto_capture_countdown": None
            })
        return jsonify({'status': 'error', 'message': 'Server error processing image.'}), 500

@app.route('/complete-registration', methods=['POST'])
def complete_registration():
    if 'temp_registration_data' not in session or 'temp_face_embeddings' not in session:
        return jsonify({'status': 'error', 'message': 'Invalid session state.'}), 400
    embeddings_from_session = session.get('temp_face_embeddings', [])
    if not isinstance(embeddings_from_session, list) or len(embeddings_from_session) < MAX_IMAGES_TO_CAPTURE:
        return jsonify({'status': 'error', 'message': f'Need {MAX_IMAGES_TO_CAPTURE} face captures.'}), 400
    try:
        reg_data = session['temp_registration_data']
        embeddings_np = [np.array(e) for e in embeddings_from_session];
        avg_embedding = np.mean(embeddings_np, axis=0)
        avg_embedding_json = json.dumps(avg_embedding.tolist())
        new_user = User(username=reg_data['username'], hashed_pin=reg_data['hashed_pin'], role=reg_data['role'],
                        phone_number=reg_data['phone'], birthday=reg_data['birthday'], embedding=avg_embedding_json)
        db.session.add(new_user); db.session.commit()
        print(f"User registered to DB: {reg_data['username']}")
        load_embeddings_from_db(force_reload=True)
        with state_lock: registration_process_state.update({"status": "complete", "instruction":"Registration successful!"})
        flash('Registration successful! Please login.', 'success'); status_code = 200
    except IntegrityError:
        db.session.rollback(); print(f"DB IntegrityError: Username {reg_data.get('username', 'UNKNOWN')} likely exists.")
        with state_lock: registration_process_state.update({"status":"error", "error_message": "Username already exists.", "instruction": "Registration failed: Username already exists."})
        flash('Username already exists.', 'error'); return jsonify({'status': 'error', 'message': 'Username already exists.'}), 409
    except Exception as e:
        db.session.rollback(); print(f"Error completing registration: {e}"); traceback.print_exc()
        with state_lock: registration_process_state.update({"status":"error", "error_message": "Server error on completion.", "instruction": "Registration failed: Server error."})
        flash('An error occurred.', 'error');
        return jsonify({'status': 'error', 'message': 'Server error during completion.'}), 500
    session.pop('temp_registration_data', None); session.pop('temp_face_embeddings', None)
    return jsonify({'status': 'success', 'message': 'Registration complete.'}), status_code

@app.route('/private-data')
def private_data():
    if 'username' not in session: flash('Please login.', 'error'); return redirect(url_for('login'))
    user_db = User.query.filter_by(username=session['username']).first()
    if not user_db: flash('User data not found.', 'error'); session.clear(); return redirect(url_for('login'))
    template_data = {'phone_number': user_db.phone_number, 'birthday': user_db.birthday}
    return render_template('private_data.html', username=user_db.username, user_data=template_data)

# --- Video and SSE Feeds ---
@app.route('/video_feed')
def video_feed():
    if 'username' not in session: return "Access Denied", 401
    return Response(generate_frames_simple(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login_video_feed')
def login_video_feed():
    if not all([dnn_detector, landmark_detector, face_recognizer]): return "Models not loaded", 503
    return Response(login_video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login_status_feed')
def login_status_feed():
    @stream_with_context
    def event_stream():
        print("Login Status SSE client connected."); last_pushed_status = None; last_pushed_user = None; last_pushed_ts = 0
        try:
            while True:
                current_status = None; current_user = None; current_ts = 0
                with state_lock:
                    current_status = login_recognition_state["status"]
                    current_user = login_recognition_state["username"]
                    current_ts = login_recognition_state["timestamp"]

                if current_status != last_pushed_status or \
                   (current_status == "recognized" and current_user != last_pushed_user) or \
                   current_ts != last_pushed_ts :
                    message_data = {"status": current_status, "username": current_user}
                    yield f"data: {json.dumps(message_data)}\n\n"
                    last_pushed_status = current_status
                    last_pushed_user = current_user
                    last_pushed_ts = current_ts
                time.sleep(0.3)
        except GeneratorExit: print("Login Status SSE client disconnected.")
        finally: print("Login Status SSE stream closed.")
    response = Response(event_stream(), mimetype="text/event-stream"); response.headers['Cache-Control'] = 'no-cache'; response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/registration_status_feed')
def registration_status_feed():
    @stream_with_context
    def event_stream():
        print("Registration Status SSE client connected."); last_pushed_state_json = None
        try:
            while True:
                time.sleep(0.25) 
                current_state_snapshot = {}
                action_hint = "continue_posing" 

                with state_lock:
                    for key in ["status", "current_pose_index", "total_images_captured",
                                "instruction", "error_message", "face_detected_for_auto_capture",
                                "auto_capture_countdown", "auto_capture_start_time"]:
                        current_state_snapshot[key] = registration_process_state.get(key)

                    pose_idx = current_state_snapshot.get("current_pose_index", 0)
                    current_pose_name = POSES[pose_idx % len(POSES)] if POSES and pose_idx < len(POSES) else "current"

                    if current_state_snapshot.get("status") == "active" and \
                       current_state_snapshot.get("face_detected_for_auto_capture") and \
                       current_state_snapshot.get("auto_capture_start_time") is not None:
                        elapsed_time = time.time() - current_state_snapshot["auto_capture_start_time"]
                        countdown_value = CAPTURE_DELAY_SECONDS - int(elapsed_time)
                        
                        if countdown_value <= 0:
                            current_state_snapshot["auto_capture_countdown"] = 0
                            current_state_snapshot["instruction"] = f"Attempting capture for {current_pose_name.upper()}..."
                            action_hint = "trigger_capture"
                            registration_process_state["auto_capture_start_time"] = None 
                        else:
                            current_state_snapshot["auto_capture_countdown"] = countdown_value
                            current_state_snapshot["instruction"] = f"Hold {current_pose_name.upper()}! Capturing in {countdown_value}..."
                    
                    elif current_state_snapshot.get("status") == "active" and \
                         not current_state_snapshot.get("face_detected_for_auto_capture") and \
                         current_state_snapshot.get("auto_capture_start_time") is None:
                        current_state_snapshot["auto_capture_countdown"] = None
                        is_error = current_state_snapshot.get("error_message")
                        instr = current_state_snapshot.get("instruction", "")
                        is_post_upload_instr = "captured" in instr.lower() or "failed" in instr.lower() or "error" in instr.lower()

                        if not is_error and not is_post_upload_instr and \
                           current_state_snapshot.get("status") not in ["pending_submit", "complete", "error"]:
                            current_state_snapshot["instruction"] = f"Align for {current_pose_name.upper()}. Face not clear."
                    
                    elif current_state_snapshot.get("status") == "pending_submit":
                        current_state_snapshot["instruction"] = "All images captured! Please submit."
                    elif current_state_snapshot.get("status") == "complete":
                        current_state_snapshot["instruction"] = "Registration successful!"
                    
                    if current_state_snapshot.get("error_message") and \
                       current_state_snapshot.get("status") in ["error", "active"]:
                        current_state_snapshot["instruction"] = current_state_snapshot["error_message"]

                    current_state_snapshot["action_hint"] = action_hint
                    
                    registration_process_state["auto_capture_countdown"] = current_state_snapshot.get("auto_capture_countdown")
                    registration_process_state["instruction"] = current_state_snapshot.get("instruction")

                current_state_json = json.dumps(current_state_snapshot)
                if current_state_json != last_pushed_state_json:
                    yield f"data: {current_state_json}\n\n"
                    last_pushed_state_json = current_state_json

        except GeneratorExit: print("Registration Status SSE client disconnected.")
        except Exception as e: print(f"Error in reg status feed: {e}"); traceback.print_exc()
        finally: print("Registration Status SSE stream closed.")
    response = Response(event_stream(), mimetype="text/event-stream"); response.headers['Cache-Control'] = 'no-cache'; response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/logout')
def logout():
    session.clear(); flash('You have been logged out.', 'success'); return redirect(url_for('login'))

def create_database():
    print("Checking/creating database tables..."); db.create_all(); print("DB tables checked/created.")

@app.route('/face-detected-registration', methods=['POST'])
def face_detected_registration():
    if 'temp_registration_data' not in session:
        return jsonify({"error": "Not in registration process"}), 403

    data = request.get_json()
    face_detected_on_client = data.get('detected', False)
    client_current_pose_index = data.get('pose_index', 0)

    response_payload = {"action": "continue_posing", "instruction": "Processing..."}

    with state_lock:
        if registration_process_state["status"] != "active":
            response_payload["action"] = "wait"
            response_payload["instruction"] = "Server not ready for active registration."
            return jsonify(response_payload)

        if registration_process_state["total_images_captured"] >= MAX_IMAGES_TO_CAPTURE:
            response_payload["action"] = "submit_now"
            response_payload["instruction"] = registration_process_state.get("instruction", "All images captured. Please submit.")
            return jsonify(response_payload)

        registration_process_state["current_pose_index"] = client_current_pose_index
        current_pose_name = POSES[client_current_pose_index % len(POSES)] if POSES and client_current_pose_index < len(POSES) else "current"

        if face_detected_on_client:
            if not registration_process_state.get("face_detected_for_auto_capture") or \
               registration_process_state.get("auto_capture_start_time") is None:
                registration_process_state["face_detected_for_auto_capture"] = True
                registration_process_state["auto_capture_start_time"] = time.time()
                if "Face not clear" in str(registration_process_state.get("error_message")) or \
                   "Align for" in str(registration_process_state.get("error_message")) or \
                   "Face lost" in str(registration_process_state.get("error_message")): # Clear transient messages
                    registration_process_state["error_message"] = None
                
                response_payload["instruction"] = f"Face detected for {current_pose_name.upper()}. Hold still..."
        else: 
            if registration_process_state.get("face_detected_for_auto_capture"): 
                registration_process_state["face_detected_for_auto_capture"] = False
                registration_process_state["auto_capture_start_time"] = None 
                registration_process_state["auto_capture_countdown"] = None
                response_payload["instruction"] = f"Align for {current_pose_name.upper()}. Face lost."
        
        response_payload["instruction"] = registration_process_state.get("instruction", response_payload["instruction"])

    return jsonify(response_payload)


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Application Starting ---")
    if not load_all_models(): print("--- Exiting: model loading failure. ---"); exit(1)
    with app.app_context():
        create_database()
        print("Performing initial load of known faces from DB...")
        load_embeddings_from_db(force_reload=True)
    with state_lock:
        login_recognition_state["status"] = "detecting"
        login_recognition_state["timestamp"] = time.time()
    print("\n--- Starting Flask App (DB, Face Login, Auto-Capture Reg Guidance) ---")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)