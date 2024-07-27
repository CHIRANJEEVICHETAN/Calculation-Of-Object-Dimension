import os
import cv2
import MySQLdb
from flask import Flask, request, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import imutils
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import torch
import uuid
from azure.storage.blob import BlobServiceClient
from ultralytics import YOLO
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Database configuration from environment variables
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

# Initialize MySQL
mysql = MySQLdb.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB']
)

# Initialize YOLOv8 model
# model = YOLO('yolov8m.pt')
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Azure Custom Vision configuration
ENDPOINT = os.getenv('AZURE_ENDPOINT')
PREDICTION_KEY = os.getenv('AZURE_PREDICTION_KEY')
PROJECT_ID = os.getenv('AZURE_PROJECT_ID')
PUBLISHED_NAME = os.getenv('AZURE_PUBLISHED_NAME')

credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
custom_vision_client = CustomVisionPredictionClient(ENDPOINT, credentials)

# Azure Blob Storage configuration
blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_BLOB_CONNECTION_STRING'))
container_client = blob_service_client.get_container_client(os.getenv('CLIENT_NAME'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def detect_objects_yolo_v5(model, frame):
    model = model
    results = model(frame)
    return results.pandas().xyxy[0].to_numpy()

def draw_labels(frame, detected_objects):
    for obj in detected_objects:
        x, y, x2, y2, confidence, class_id, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], int(obj[5]), obj[6]
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def upload_image_to_blob(image_path, image_name):
    # Ensure filename is secure
    secure_name = secure_filename(image_name)
    
    # Retrieve the CLIENT_NAME from the environment variables
    client_name = os.getenv('CLIENT_NAME')
    
    # Check if CLIENT_NAME is defined
    if not client_name:
        raise ValueError("CLIENT_NAME environment variable is not defined.")

    # Check and modify the container name to follow Azure's naming rules
    container_name = re.sub(r'[^a-z0-9-]', '', client_name.lower())
    container_name = container_name.strip('-')
    
    if not (3 <= len(container_name) <= 63):
        raise ValueError("Invalid container name. It must be between 3 and 63 characters long.")
    
    unique_name = str(uuid.uuid4()) + "_" + secure_name
    blob_client = container_client.get_blob_client(unique_name)
    
    with open(image_path, "rb") as data:
        blob_client.upload_blob(data)
    
    return blob_client.url


def process_image(image_path, ref_width, mode):
    global model
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image at {image_path}")
        return None

    # YOLO object detection
    detected_objects = detect_objects_yolo_v5(model, image)
    draw_labels(image, detected_objects)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
    refObj = None

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        if refObj is None:
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / ref_width)
            continue

        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        if mode == 'size':
            for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
                cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
                cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
                cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
                D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if abs(xA - xB) > abs(yA - yB):
                    cv2.putText(orig, "L", (int(mX), int(mY)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                else:
                    cv2.putText(orig, "B", (int(mX), int(mY)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif mode == 'distance':
            D = dist.euclidean(refObj[1], (cX, cY)) / refObj[2]
            (mX, mY) = midpoint(refObj[1], (cX, cY))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        image = orig

    with open(image_path, "rb") as image_data:
        results = custom_vision_client.detect_image(PROJECT_ID, PUBLISHED_NAME, image_data)

    for prediction in results.predictions:
        if prediction.probability > 0.5:
            left = prediction.bounding_box.left * image.shape[1]
            top = prediction.bounding_box.top * image.shape[0]
            width = prediction.bounding_box.width * image.shape[1]
            height = prediction.bounding_box.height * image.shape[0]
            cv2.rectangle(image, (int(left), int(top)), (int(left + width), int(top + height)), (255, 0, 0), 2)
            cv2.putText(image, "{}: {:.2f}%".format(prediction.tag_name, prediction.probability * 100),
                        (int(left), (int(top - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            L = max(width, height)
            B = min(width, height)
            cv2.putText(image, "L: {:.1f}in".format(L), (int(left), int(top + height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "B: {:.1f}in".format(B), (int(left), int(top + height + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def save_image_to_db(filename, url):
    cursor = mysql.cursor()
    cursor.execute("INSERT INTO images1 (filename, url) VALUES (%s, %s)", (filename, url))
    mysql.commit()
    cursor.close()

def process_frame(frame, ref_width, mode):
    global model
    # YOLO object detection
    detected_objects = detect_objects_yolo_v5(model, frame)
    draw_labels(frame, detected_objects)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    refObj = None

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        if refObj is None:
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / ref_width)
            continue

        orig = frame.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        if mode == 'size':
            for ((xA, yA), (xB, yB)) in zip(refCoords, objCoords):
                cv2.circle(orig, (int(xA), int(yA)), 5, (0, 0, 255), -1)
                cv2.circle(orig, (int(xB), int(yB)), 5, (0, 0, 255), -1)
                cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), (0, 0, 255), 2)
                D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        elif mode == 'distance':
            D = dist.euclidean(refObj[1], (cX, cY)) / refObj[2]
            (mX, mY) = midpoint(refObj[1], (cX, cY))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        frame = orig

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        ref_width = float(request.form['width'])
        mode = request.form['mode']
        processed_image = process_image(filepath, ref_width, mode)
        if processed_image is None:
            return "Error: Unable to process the image."

        processed_filename = 'processed_' + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, processed_image)
        
        # Upload processed image to Azure Blob Storage
        blob_url = upload_image_to_blob(processed_filepath, processed_filename)
        
        # Save processed image information to the database
        save_image_to_db(processed_filename, blob_url)

        return render_template('result.html', original_image=filepath, processed_image=processed_filepath, blob_url=blob_url)

    return redirect(request.url)

# @app.route('/video_feed')
# def video_feed():
#     return render_template('video_feed.html')

@app.route('/video_feed')
def video_stream():
    ref_width = float(request.args.get('ref_width', 1.0))
    mode = request.args.get('mode', 'size')
    video_source = request.args.get('video_source', 0)

    def generate_frames():
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        while True:
            success, frame = cap.read()
            if not success:
                break
            processed_frame = process_frame(frame, ref_width, mode)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
