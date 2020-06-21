import dlib
import numpy as np
import pandas as pd
import time


predictor_path = 'res/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'res/dlib_face_recognition_resnet_model_v1.dat'
face="test/Neymar_search.jpg"
data=pd.read_csv("facedata/fifadb.csv")
Threshold = 0.5
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

img = dlib.load_rgb_image(face)
dets = detector(img, 1)
shape = sp(img, dets[0])
face_descriptor_search = facerec.compute_face_descriptor(img, shape)

def main(face):
##    start = time.time()

    for i in range(len(data)):
        face_descriptor_db=eval(data['embedding'][i])
        minDistance=np.linalg.norm(np.asarray(face_descriptor_search)\
                                   - np.asarray(face_descriptor_db))
        if minDistance<Threshold:
            found = data['name'].get(i)
##    print('Duration: {} seconds'.format(time.time() - start))

    return found

