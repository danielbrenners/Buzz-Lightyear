# USAGE
# python track.py

# import the necessary packages
import numpy as np
import time
import cv2
import imutils

# define the upper and lower boundaries for a color
# to be considered "blue"
colorDark = np.array([98, 98, 48], dtype = "uint8")
colorLight = np.array([247,247,122], dtype = "uint8")


# initialize the camera and grab a reference to the raw camera
# capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image
    frame = f.array

    frame = imutils.resize(frame, width=480)

    # determine which pixels fall within the blue boundaries
    # and then blur the binary image
    blue = cv2.inRange(frame, colorDark, colorLight)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)

    # find contours in the image
    (_, cnts, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # check to see if any contours were found
    if len(cnts) > 0:
        # sort the contours and find the largest one -- we
        # will assume this contour correspondes to the area
        # of my phone
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        # compute the (rotated) bounding box around then
        # contour and then draw it
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (255, 0, 0), 2)

    # show the frame and the binary image
    cv2.imshow("Dat Post-It Tho", frame)
    cv2.imshow("Binary", blue)

    # if your machine is fast, it may display the frames in
    # what appears to be 'fast forward' since more than 32
    # frames per second are being displayed -- a simple hack
    # is just to sleep for a tiny bit in between frames;
    # however, if your computer is slow, you probably want to
    # comment out this line
    time.sleep(0.025)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()