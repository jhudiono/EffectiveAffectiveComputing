from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import logging

def binary_sigmoid(base_model, optimizer='adadelta', dropout=None):
	input = Input(shape=(224,224,3), name='image_input')
	for layer in base_model.layers:
    		layer.trainable=False

	output_model = base_model(input)
	x = Flatten(name='flatten')(output_model)
	if dropout:
		x = Dropout(rate=dropout)(x)
	x = Dense(1, activation='sigmoid', name='logistic')(x)  # 2 = binary

	my_model = Model(input=input, output=x)
	my_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	return my_model

def VGG16_Binary_Sigmoid(optimizer='adadelta', dropout=None):	
	base_model = VGG16(weights='imagenet', include_top=False)
	return binary_sigmoid(base_model, optimizer, dropout)


def RN50_Binary_Sigmoid(optimizer='adadelta', dropout=None):
	base_model = ResNet50(weights='imagenet', include_top=False)
	return binary_sigmoid(base_model, optimizer, dropout)

def IV3_Binary_Sigmoid(optimizer='adadelta', dropout=None):
	base_model = InceptionV3(weights='imagenet', include_top=False)
	return binary_sigmoid(base_model, optimizer, dropout)

def save(path="./models"):
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	models = {
		"vgg16_adadelta_dropout": VGG16_Binary_Sigmoid('adadelta', dropout=0.5),
		"vgg16_adam_dropout": VGG16_Binary_Sigmoid('adam', dropout=0.5),
		"vgg16_adam": VGG16_Binary_Sigmoid('adam', dropout=None),
		"vgg16_sgd": VGG16_Binary_Sigmoid('sgd', dropout=None),
		"rn50_adam_dropout": RN50_Binary_Sigmoid('adam', dropout=0.5),
		"rn50_adadelta_dropout": RN50_Binary_Sigmoid('adam', dropout=0.5),
		"iv3_adam_dropout": IV3_Binary_Sigmoid('adam', dropout=0.5),
		"iv3_adadelta_dropout": IV3_Binary_Sigmoid('adadelta', dropout=0.5)
	}
	path += "/{}.h5"
	for name, model in models.items():
		logging.info("Saving to", path.format(name))
		model.save(path.format(name))
		
		
