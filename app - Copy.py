# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from flask import Flask, render_template, request
import json
from flask_cors import CORS
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
#from tensorflow.keras.models import load_model
from keras.models import load_model

import pyttsx3
app = Flask(__name__)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
##ap.add_argument("-i", "--input", required=True,
##	help="path to input video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


def tts():
    engine = pyttsx3.init()

    engine.say(className)

    engine.runAndWait()


# initialize the video stream, pointer to output video file, and
# frame dimensions
#vs=cv2.VideoCapture("C:/Users/saiv8/OneDrive/Desktop/detection/queda.mp4")

#vs= cv2.VideoCapture(0)
vs=cv2.VideoCapture(0)



# time.sleep(2.0)
def gen_frames():

    # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        
        
         # Read each frame from the webcam
        grabbed, frame = vs.read()
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)
        
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
                
        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0,0,255), 2, cv2.LINE_AA)
            #key = cv2.waitKey(1) & 0xFF
        #tts()
        if not grabbed:
            
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)




    


