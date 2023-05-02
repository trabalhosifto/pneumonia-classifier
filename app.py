import os
from flask import Flask, render_template, request, send_from_directory
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import io

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded file
    img_file = request.files['image']

    # Check if the 'images' directory exists, and create it if it doesn't
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.exists(target):
        os.makedirs('images')

    # Save the image to a temporary location
    image_path = 'images/' + img_file.filename
    name_figure = img_file.filename
    img_file.save(image_path)

    # Load the image using Keras
    # image = load_img(io.BytesIO(img_file.read()), target_size=(64, 64))
    image = load_img(image_path, target_size=(64, 64))
    # image = image.resize((64, 64))

    model = load_model('pneumonia.h5')

    image = img_to_array(image)
    image /= 255
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    print("RESULT =========> ", prediction[0][0])

    if prediction.shape == (1, 1):
        if prediction[0][0] < 0.5:
            class_label = 'negativo_pneumonia'
        else:
            class_label = 'positivo_pneumonia'
    else:
        class_label = 'Desconhecido'

    # Remove the temporary image file
    # os.remove(image_path)

    # Return the classification result to the user
    return render_template('result.html', result=class_label, image_name=name_figure)


@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == '__main__':
    app.run(debug=True)
