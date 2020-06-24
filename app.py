import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import numpy as np
import dlib
import pandas as pd
import time


predictor_path = 'res/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'res/dlib_face_recognition_resnet_model_v1.dat'
data=pd.read_csv("facedata/fifadb.csv")
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','.jpeg'}
app = Flask(__name__, static_url_path="/static")


# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 5mb
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            threshold = request.form['threshold_value']
            name=face_recognition(os.path.join(UPLOAD_FOLDER, filename),threshold)
            data={
                "uploaded_img":'static/uploads/'+filename,
                "name":name
            }
            return render_template("index.html",data=data)  
    return render_template('index.html')

    

def face_recognition(path,thresh): 
    Threshold=float(thresh)
    found_list=[]  
    img = dlib.load_rgb_image(path)
    dets = detector(img, 1)
    shape = sp(img, dets[0])
    face_descriptor_search = facerec.compute_face_descriptor(img, shape)
    # start = time.time()
    for i in range(len(data)):
        face_descriptor_db=eval(data['embedding'][i])
        minDistance=np.linalg.norm(np.asarray(face_descriptor_search)\
                                   - np.asarray(face_descriptor_db))
        if minDistance<Threshold:
            found = data['name'].get(i)
            found_list.append(found)

    # print('Duration: {} seconds'.format(time.time() - start))
    return found_list
  

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
