import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_image(image_path, ref_width):
    image = cv2.imread(image_path)
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
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
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

def process_frame(frame, ref_width):
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


        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

def main():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        default="video",
        help="mode of the operation: 'image' or 'video' (default is 'video')")
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
    ap.add_argument("-i",
                    "--image",
                    type=str,
                    required=False,
                    help="path to the input image (required for 'image' mode)")
    args = vars(ap.parse_args())

    if args["mode"] == "image":
        process_image(args["image"], args["width"])
    elif args["mode"] == "video":
        if args.get("url", None) is None:
            print("URL is required for video mode.")
            return
        cap = cv2.VideoCapture(args["url"])

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            process_frame(frame, args["width"])
            time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
