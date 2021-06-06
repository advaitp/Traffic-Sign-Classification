import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

classes = 43 

# getting the labels and x values
def load_data(image_path) :
	labels = []
	data = []
	

	for i in range(classes) :
		image_folder = os.path.join(image_path, 'Train', str(i))
		images = os.listdir(image_folder) 
		try :
			for image in images :
				curr_image = Image.open(image_folder + "\\" + str(image)) 
				curr_image = curr_image.resize((30,30)) 
				curr_image = np.array(curr_image)
				data.append(curr_image)
				labels.append(i)
		except :
			print(f'Error in loading image no : {i}')

	data = np.array(data)
	labels = np.array(labels)	
	return data, labels	

def create_dataset(data, labels) :
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)	
	print(f'Size of X_train is {X_train.shape}')
	print(f'Size of X_test is {X_test.shape}')
	print(f'Size of y_train is {y_train.shape}')
	print(f'Size of y_test is {y_test.shape}')
	return X_train, X_test, y_train, y_test
	


def create_model(X_train) :
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=X_train.shape[1:]))
	model.add(Conv2D(32, kernel_size=(5, 5),activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
	model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(classes, activation='softmax'))
	return model 


if __name__ == "__main__" :
	image_path = r'C:\Users\test\Desktop\ml\Python-Project-Traffic-Sign-Classification\Traffic sign classification\Data'
	data, labels = load_data(image_path)
	X_train, X_test, y_train, y_test = create_dataset(data, labels)
	y_train = to_categorical(y_train, classes)
	y_test = to_categorical(y_test, classes)
	# model = create_model(X_train)
	# model.compile(loss='categorical_crossentropy',
 #              optimizer='adam',
 #              metrics=['accuracy'])

	# model.fit(X_train, y_train,
 #          batch_size=128,
 #          epochs=10,
 #          validation_data=(X_test, y_test))

	# model.save('traffic_classifier.h5')
	classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }
            
	model = load_model('traffic_classifier.h5')
	y_test = pd.read_csv('Test.csv')

	labels = y_test["ClassId"].values
	imgs = y_test["Path"].values

	data=[]

	for img in imgs:
	    image = Image.open(os.path.join(image_path, img))
	    image = image.resize((30,30))
	    data.append(np.array(image))

	X_test=np.array(data)
	pred = model.predict_classes(X_test)
	print(accuracy_score(labels, pred))

	print('Given Image is ')
	image = Image.open(os.path.join(image_path, imgs[90]))
	image.show()
	print(f'The label is {labels[90]}')

	print(f'Predicted Label is {classes[pred[90]+1]}')

	


