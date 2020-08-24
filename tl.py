
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D,MaxPool2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers
from keras.layers.normalization import BatchNormalization


def init():
	model = Sequential()

	# model.add(Lambda(standardize,input_shape=(28,28,1)))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512, activation="relu"))

	model.add(Dense(10, activation="softmax"))

	model.load_weights("model_weights.h5")
	print("Loaded Model from disk")
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


	return model

