from flask import Flask,render_template,request


import re

import os
import base64

from tl import init

import PIL
from matplotlib.pyplot import imread
from skimage.transform import resize
from PIL import Image
import numpy as np
app = Flask(__name__)
global model
model=init()

def convertImage(imgData1):
    encoding = 'utf-8'
    i=imgData1.decode(encoding)
    imgstr = re.search(r'base64,(.*)',i).group(1)

    print(imgstr)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(bytes(imgstr, 'utf-8')))

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	print("debug")
	#read the image into memory
	x = Image.open("output.png").convert("L")
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = resize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	print("debug2")
	#in our computation graph

	#perform the prediction
	out = model.predict(x)
	print(out)
	print(np.argmax(out,axis=1))
	print("debug3")
	#convert the response to a string
	response = np.array_str(np.argmax(out,axis=1))
	return response

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
