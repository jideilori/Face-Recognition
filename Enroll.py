## USAGE: Run add_people() to add everyface found in "players/" to the database
## Run add("people/NAME_OF_IMAGE") to add a specific image

import os
import dlib
import csv
predictor_path = 'res/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'res/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def add_people(faces_path="players"):
    '''
    This adds every face found in the directory to
    the database using the name of the image as the
    individuals name.
    '''
    with open("facedata/fifadb.csv", "w",newline='' ) as out_file:
        fieldnames = ['name', 'embedding']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for f in os.listdir(faces_path):
            print("Processing file: {}".format(f))
            img = dlib.load_rgb_image(faces_path+'/'+f)
            dets = detector(img, 1)
            p_name=f.rsplit(".")[0]

            for k, d in enumerate(dets):
                shape = sp(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                face_descriptor = [x for x in face_descriptor]
                writer.writerow({'name': f'{p_name}',
                                 'embedding': f'{face_descriptor}'})
            
def add(face):
    '''
    This adds only a single individual to the database.
    The image must be in people/
    '''
    with open("facedata/fifadb.csv", "a",newline='' ) as out_file:
        fieldnames = ['name', 'embedding']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        img = dlib.load_rgb_image(face)
        dets = detector(img, 1)
        p_name=face.split("/")[1][:-4]
        print(p_name)

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            face_descriptor = [x for x in face_descriptor]
            writer.writerow({'name': f'{p_name}',
                             'embedding': f'{face_descriptor}'})
        
