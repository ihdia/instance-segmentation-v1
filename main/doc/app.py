import os

from flask import Flask, render_template, request, jsonify,redirect
from werkzeug import secure_filename
from modeltest import *


# create flask instance
app = Flask(__name__)

UPLOAD_FOLDER = '/home/abhishek/prusty/Instance-segmentation/main/doc/static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# main route
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], "1.jpg"))
		return redirect("http://localhost:5000/")
		

@app.after_request
def add_header(r):

	r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	r.headers["Pragma"] = "no-cache"
	r.headers["Expires"] = "0"
	r.headers['Cache-Control'] = 'public, max-age=0'
	return r

@app.route('/background_process_test')
def background_process_test():
	filepath="/home/abhishek/prusty/Instance-segmentation/main/doc/static/images/1.jpg"
	img=cv2.imread(filepath,1)
	runtest(img)
	return jsonify({"a":1,"b":2,"c":3})	

if __name__ == '__main__':
	app.run('0.0.0.0', debug=True)
