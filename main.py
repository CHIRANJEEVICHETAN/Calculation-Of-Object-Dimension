import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time
import os

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_frame(frame, ref_width):
    # Convert the frame to grayscale and blur it slightly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left-to-right and then initialize the distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
    refObj = None

    # Loop over the contours individually
    for c in cnts:
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)

        # Compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # If this is the first contour we are examining (i.e., the left-most contour), we presume this is the reference object
        if refObj is None:
            # Unpack the ordered bounding box, then compute the midpoint between the top-left and bottom-left points, followed by the midpoint between the top-right and bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Compute the Euclidean distance between the midpoints, then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / ref_width)
            continue

        # Draw the contours on the frame
        orig = frame.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

        # Stack the reference coordinates and the object coordinates to include the object center
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        # Loop over the original points
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            # Draw circles corresponding to the current points and connect them with a line
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)

            # Compute the Euclidean distance between the coordinates, and then convert the distance in pixels to distance in units
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Show the output frame
        cv2.imshow("Frame", orig)

def process_video(url, ref_width):
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, ref_width)
        time.sleep(3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path, ref_width):
    frame = cv2.imread(image_path)
    process_frame(frame, ref_width)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
    ap.add_argument("-m", "--mode", type=str, required=True, choices=["image", "video"], help="mode of input: 'image' or 'video'")
    ap.add_argument("-u", "--url", type=str, help="URL of the IP Webcam stream (required if mode is 'video')")
    ap.add_argument("-i", "--image", type=str, help="Path to the input image (required if mode is 'image')")
    args = vars(ap.parse_args())

    if args["mode"] == "video":
        if args["url"] is None:
            print("Error: URL must be provided for video mode.")
            return
        process_video(args["url"], args["width"])
    elif args["mode"] == "image":
        if args["image"] is None:
            print("Error: Image path must be provided for image mode.")
            return
        if not os.path.isfile(args["image"]):
            print("Error: Image file does not exist.")
            return
        process_image(args["image"], args["width"])

if __name__ == "__main__":
    main()
