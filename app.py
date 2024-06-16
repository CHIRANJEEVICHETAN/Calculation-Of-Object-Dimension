import os
import cv2
import MySQLdb
from flask import Flask, request, render_template, redirect, url_for
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
from dotenv import load_dotenv

 # Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Database configuration
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

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Corrected Azure Custom Vision configuration
ENDPOINT = os.getenv("ENDPOINT")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
PUBLISHED_NAME = os.getenv("PUBLISHED_NAME")

credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
custom_vision_client = CustomVisionPredictionClient(ENDPOINT, credentials)

# Azure Blob Storage configuration
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client("PROJECT_TYPE")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def upload_image_to_blob(image_path, image_name):
    unique_name = str(uuid.uuid4()) + "_" + image_name
    blob_client = container_client.get_blob_client(unique_name)
    with open(image_path, "rb") as data:
        blob_client.upload_blob(data)
    return blob_client.url

def process_image(image_path, ref_width, mode):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image at {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
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
                
                # Label length 'L' and breadth 'B'
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

    # Azure Custom Vision detection
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
                        (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Label length 'L' and breadth 'B'
            L = max(width, height)
            B = min(width, height)
            cv2.putText(image, "L: {:.1f}in".format(L), (int(left), int(top + height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "B: {:.1f}in".format(B), (int(left), int(top + height + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def save_image_to_db(filename, url):
    cursor = mysql.cursor()
    cursor.execute("INSERT INTO images (filename, url) VALUES (%s, %s)", (filename, url))
    mysql.commit()
    cursor.close()

def process_frame(frame, ref_width, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
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

        orig = frame.copy()
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
        elif mode == 'distance':
            # Correctly call dist.euclidean with two points
            D = dist.euclidean(refObj[1], (cX, cY)) / refObj[2]
            (mX, mY) = midpoint(refObj[1], (cX, cY))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    

        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
     # YOLOv5 detection
    results = model(frame)

    # Process YOLOv5 results
    for det in results.xyxy[0]:  # detections per frame
        if det is not None and len(det):
            # xyxy format: [x_min, y_min, x_max, y_max, confidence, class]
            x_min, y_min, x_max, y_max, confidence, cls = det
            if confidence > 0.5:  # Confidence threshold
                label = f'{model.names[int(cls)]} {confidence:.2f}'
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        return False

    return True


@app.route('/')
def index():
    return render_template('index.html')

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
        if processed_image is not None:
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, processed_image)

            # Upload original and processed images to Azure Blob Storage
            original_url = upload_image_to_blob(filepath, filename)
            result_url = upload_image_to_blob(result_path, 'result_' + filename)

            # Save the processed image URL to the database
            save_image_to_db(filename, result_url)

            return render_template('index.html', result=url_for('static', filename='uploads/result_' + filename))

    return redirect(request.url)

@app.route('/video')
def video():
    url = request.args.get('url')
    ref_width = float(request.args.get('width'))
    mode = request.args.get('mode')

    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        process_frame(frame, ref_width, mode)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return "Video stream ended."

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=8000)
