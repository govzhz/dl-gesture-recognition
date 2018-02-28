#!/usr/bin/env python
# encoding: utf-8
"""

@author:nikan

@file: track.py

@time: 27/02/2018 10:25 AM
"""

import cv2
import numpy as np
import time

# Open Camera object
cap = cv2.VideoCapture(0)

# Decrease frame size

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


def nothing(x):
    pass

def drawl_contour(contour):
    cnts = contour

    # Find convex hull
    hull = cv2.convexHull(cnts)

    # Find convex defects
    hull2 = cv2.convexHull(cnts, returnPoints=False)
    defects = cv2.convexityDefects(cnts, hull2)

    # Get defect points and draw them in the original image
    FarDefect = []
    if not hasattr(defects, 'shape'):
        return
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame, start, end, [0, 255, 0], 1)
        cv2.circle(frame, far, 10, [100, 255, 255], 3)

    # Find moments of the largest contour
    moments = cv2.moments(cnts)

    # Central mass of first order moments
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)

    # Draw center mass
    cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Center', tuple(centerMass), font, 2, (255, 255, 255), 2)

    # Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0, len(FarDefect)):
        x = np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
        distanceBetweenDefectsToCenter.append(distance)

    # Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

    # Get fingertip points from contour hull
    # If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0, len(hull) - 1):
        if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 80) or (
                    np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 80):
            if hull[i][0][1] < 500:
                finger.append(hull[i][0])

    # The fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])
    fingers = finger[0:5]

    # Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0, len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0] - centerMass[0], 2) + np.power(fingers[i][1] - centerMass[0], 2))
        fingerDistance.append(distance)

    # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    # than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0, len(fingers)):
        if fingerDistance[i] > AverageDefectDistance + 130:
            result = result + 1

    # Print number of pointed fingers
    cv2.putText(frame, str(result), (100, 100), font, 2, (255, 255, 255), 2)

    # show height raised fingers
    # cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    # cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)

    # Print bounding rectangle
    x, y, w, h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2)
    
    
def check_face(img, face_cascade, contours):
    new_contours = []
    for contour in contours:
        # mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
        # cv2.drawContours(mask, [contour], 0, 255, -1)  # Draw filled contour in mask
        # out = np.zeros_like(img)  # Extract out the object and place into output image
        # out[mask == 255] = img[mask == 255]
        #
        # # Now crop
        # _tuple = np.where(mask == 255)
        # x = _tuple[0]
        # y = _tuple[1]
        # (topx, topy) = (np.min(x), np.min(y))
        # (bottomx, bottomy) = (np.max(x), np.max(y))
        # out = out[topx:bottomx + 1, topy:bottomy + 1]
        #
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # faces = None
        if not faces:
            new_contours.append(contour)
        else:
            print('yes!')
    return new_contours

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h, s, v = 100, 100, 100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar', 0, 179, nothing)
cv2.createTrackbar('s', 'HSV_TrackBar', 0, 255, nothing)
cv2.createTrackbar('v', 'HSV_TrackBar', 0, 255, nothing)

while 1:
    
    # Measure execution time
    start_time = time.time()
    
    # Capture frames from the camera
    _, frame = cap.read()
    
    # Blur the image
    blur = cv2.blur(frame, (3, 3))
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)
    
    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    face_cascade = cv2.CascadeClassifier('lbp.xml')
    sorted_contours = check_face(frame, face_cascade, sorted_contours)
    for ci in range(min(4, len(sorted_contours))):
        drawl_contour(sorted_contours[ci])
    
    ##### Show final image ########
    cv2.imshow('Dilation', frame)
    ###############################
    
    # Print execution time
    # print time.time()-start_time
    
    # close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
