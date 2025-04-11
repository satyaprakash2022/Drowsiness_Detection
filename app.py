from flask import Flask, render_template, Response
import cv2
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
from pygame import mixer
import time
import os

app = Flask(__name__)
mixer.init()
sound = mixer.Sound('static/alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# model = load_model('models/cnncat2.h5')
try:
    model = load_model('models/cnnCat2.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

score = 0
thicc = 2
path = os.getcwd()
lbl=['Close','Open']
def generate_frames():
    global score, thicc
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            left_eye = leye.detectMultiScale(gray)
            right_eye = reye.detectMultiScale(gray)

            rpred = [45]
            lpred = [45]

            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y+h, x:x+w]
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
                r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
                rpred = np.argmax(model.predict(r_eye), axis=-1)
                if(rpred[0]==1):
                    lbl='Open'
                if(rpred[0]==0):
                    lbl='Closed'
                break
                

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y+h, x:x+w]
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
                l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
                lpred = np.argmax(model.predict(l_eye), axis=-1)
                if(lpred[0]==1):
                    lbl='Open'
                if(lpred[0]==0):
                    lbl='Closed'
                break

             # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
            if rpred[0] == 0 and lpred[0] == 0:
                score += 1
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                score -= 1
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score < 0:
                score = 0
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score > 15:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                if thicc < 16:
                    thicc += 2
                else:
                    thicc = max(thicc - 2, 2)
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host = '0.0.0.0' , port = '5000' , debug=True)
