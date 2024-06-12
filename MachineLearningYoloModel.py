import cv2
import torch
from scipy.spatial import distance as dist
import detect_object_size as obs
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def load_yolo_v5():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    return model

def detect_objects_yolo_v5(model, frame):
    results = model(frame)
    return results.pandas().xyxy[0].to_numpy()

def draw_labels(frame, detected_objects):
    for obj in detected_objects:
        x, y, x2, y2, confidence, class_id, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], int(obj[5]), obj[6]
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_image(image_path, ref_width, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image at {image_path}")
        return

    detected_objects = detect_objects_yolo_v5(model, image)
    draw_labels(image, detected_objects)

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

        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Image", orig)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def process_frame(frame, ref_width, model):
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

        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame

def main():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["image", "video", "dimension"],
        default="video",
        help="mode of the operation: 'image', 'video' or 'dimension' (default is 'video')")
    ap.add_argument("-w",
                    "--width",
                    type=float,
                    required=True,
                    help="width of the left-most object in the image (in inches)")
    ap.add_argument("-u",
                    "--url",
                    type=str,
                    required=False,
                    help="URL of the IP Webcam stream (required for 'video' mode)")
    
    ap.add_argument("-c",
                    "--capture",
                    type=str,
                    default="image",
                    help="mode of the operation ('image' or 'video') in object_size program")

    ap.add_argument("-i",
                    "--image",
                    type=str,
                    required=False,
                    help="path to the input image (required for 'image' mode)")
    args = vars(ap.parse_args())

    model = load_yolo_v5()

    if args["mode"] == "image":
        if args.get("image", None) is None:
            print("Image path is required for image mode.")
            return
        process_image(args["image"], args["width"], model)
    elif args["mode"] == "dimension":
        obs.main()
    elif args["mode"] == "video":
        if args.get("url", None) is None:
            print("URL is required for video mode.")
            return
        cap = cv2.VideoCapture(args["url"])

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = process_frame(frame, args["width"], model)
            cv2.imshow("Frame", frame)
            time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
