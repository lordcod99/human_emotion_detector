import cv2
import numpy as np
# import tensorflow
# import keras
from keras.models import load_model
from keras.utils import img_to_array


model = load_model('trained_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

while video.isOpened():
    ret , frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.03,6)
    try:
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (127,0,255),thickness =  2)
            roi_gray = gray[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(frame,emotion_prediction,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(125, 246, 55),2)
    except:
        pass
    cv2.imshow('How are you felling', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()

cv2.destroyAllWindows

