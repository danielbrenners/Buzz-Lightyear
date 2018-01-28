# USAGE

# python emotion_detector_test.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5

# 'camera = cv2.VideoCapture(0)'
# model from checkpoints, not output

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
    help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
    "neutral"]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # initialize the canvas for the visualization, then clone
    # the frame so we can draw on it
    canvas_width = 500
    canvas_height = 250
    bar_bottom_height = 40
    bar_width = 75

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # ensure at least one face was found before continuing
    if len(rects) > 0:
        # determine the largest face area
        rect = sorted(rects, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        # extract the face ROI from the image, then pre-process
        # it for the network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        # label
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        emotion_color = (255,255,255)
        if label is "angry":
            emotion_color = (0,0,255)
        elif label is "scared":
            emotion_color = (0,255,0)
        elif label is "happy":
            emotion_color = (0,255,255)
        elif label is "sad":
            emotion_color = (255,0,0)
        elif label is "surprised":
            emotion_color = (255,0,255)
        else:
            emotion_color = (255,255,255)


        # loop over the labels + probabilities and draw them
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            #text = "{}: {:.2f}%".format(emotion, prob * 100)
            emotion_text = emotion
            prob_text = str(round(prob, 2))
            # draw the label + probability bar on the canvas
            w = int(prob * 300)


            cv2.rectangle(
                canvas,
                (i * bar_width, canvas_height - bar_bottom_height),
                ((i * bar_width) + bar_width, min(canvas_height - bar_bottom_height, canvas_height - w)),
                (0, 255, 0),
                 -1)

            """
            cv2.rectangle(
                canvas,
                (5, (i * 35) + 5),
                (w, (i * 35) + 35),
                (0, 255, 0),
                 -1
            )
            """

            cv2.putText(canvas,
                emotion_text,
                ((i * (bar_width)) + 10, canvas_height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255),
                2)

            cv2.putText(canvas,
                prob_text,
                ((i * (bar_width)) + 10, canvas_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255),
                2)


        # draw the label on the frame
        cv2.putText(frameClone,
            label,
            (fX + 25, fY + fH + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            emotion_color,
            2)

        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
            emotion_color, 1)

    # show our classifications + probabilities
    cv2.imshow("Video Capture", frameClone)
    cv2.imshow("Probability Distribution", canvas)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()