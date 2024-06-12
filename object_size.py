import numpy as np
import argparse
import imutils
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import torch

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def measure_objects(image, width, pixelsPerMetric):
    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left-to-right and initialize the pixels per metric calibration variable
    (cnts, _) = contours.sort_contours(cnts)

    # Loop over the contours individually
    for c in cnts:
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # Compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv3() or imutils.is_cv4() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Unpack the ordered bounding box, then compute the midpoint between points
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If the pixels per metric has not been initialized, compute it
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        # Compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Draw the object sizes on the image
        cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return orig, pixelsPerMetric

def detect_objects_with_yolov5(image):
    # Load the YOLOv5 model (change to yolov5m or yolov5l for more accuracy)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Use 'yolov5x' for a more accurate model
    
    # Detect objects
    results = model(image)
    
    # Process the results
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1].numpy()
    label_names = results.names  # Get the names of the classes
    
    return labels, cords, label_names

def process_video(stream_url, width):
    # Capture video from IP Webcam stream
    cap = cv2.VideoCapture(stream_url)
    pixelsPerMetric = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        labels, cords, label_names = detect_objects_with_yolov5(frame)
        
        # Draw bounding boxes and labels on the detected objects
        for label, cord in zip(labels, cords):
            x1, y1, x2, y2, conf = cord
            if conf < 0.5:  # Lower confidence threshold to include more objects
                continue
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = label_names[int(label)]
            cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Measure objects
        frame, pixelsPerMetric = measure_objects(frame, width, pixelsPerMetric)
        
        # Display the frame
        cv2.imshow("Video", frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the input image")
    ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
    ap.add_argument("-c", "--capture", type=str, default="image", help="mode of the operation ('image' or 'video')")
    ap.add_argument("-m", "--mode", type=str, default="image", help="mode of the operation: 'image' or 'video' (default is 'video')")
    ap.add_argument("-u", "--url", type=str, help="URL of the IP Webcam stream")
    args = vars(ap.parse_args())

    if args["capture"] == "image":
        if args["image"] is None:
            raise ValueError("Image path must be provided for image mode.")
        
        # Load the image
        image = cv2.imread(args["image"])
        
        # Detect objects using YOLOv5
        labels, cords, label_names = detect_objects_with_yolov5(image)
        
        # Draw bounding boxes and labels on the detected objects
        for label, cord in zip(labels, cords):
            x1, y1, x2, y2, conf = cord
            if conf < 0.5:  # Lower confidence threshold to include more objects
                continue
            x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = label_names[int(label)]
            cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Measure objects
        image, _ = measure_objects(image, args["width"], pixelsPerMetric=None)
        
        # Display the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args["capture"] == "video":
        if args["url"] is None:
            raise ValueError("URL must be provided for video mode.")
        
        # Process the video stream
        process_video(args["url"], args["width"])

if __name__ == "__main__":
    main()
