import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# path to saved facetracker model
facetracker = load_model(
    '/Users/rileyoest/VS_Code/FaceFinder/facetracker.h5')
# facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.9:
    #if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [
                            640, 480]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [
                            640, 480]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [
                            640, 480]).astype(int), [0, -20])),
                      tuple(np.add(np.multiply(sample_coords[:2], [
                            640, 480]).astype(int), [70, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [
                            640, 480]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
