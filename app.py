from flask import Flask, request, jsonify, render_template
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image



UPLOAD_FOLDER = 'static/images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model('monkey_breed_mobilNet2.h5')

class_labels = [
	'Mantled Howler',
	'Patas Monkey',
	'Bald Uakari',
	'Japanese Macaque',
	'Pygmy Marmoset',
	'White Headed Capuchin',
	'Silvery Marmoset',
	'Common Squirrel Monkey',
	'Black Headed Night Monkey',
	'Nilgiri Langur'
	]

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/classifier.html')
def classifier_page():
    return render_template("classifier.html")


@app.route('/classify', methods = ['GET', 'POST'])
def classify():
    if request.method == 'POST':
        pic = request.files['pic']
        pic_name = str(secure_filename(pic.filename))
        pic.save(os.path.join(app.config['UPLOAD_FOLDER'],pic_name))
        pic_path = "../static/images//"+pic_name

        # prediction
        img = image.load_img("static/images//"+pic_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255
        pred = np.argmax(model.predict(x))  # if it predicts 8 => index 8
        prediction = class_labels[pred]
        return render_template("classifier.html", pic_path=pic_path, prediction=prediction)

if __name__ == "__main__":
    app.run()
