from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import logging

def multi_class(base_model, optimizer='adam', add_layers=[]):
	input = Input(shape=(224,224,3), name='image_input')
	for layer in base_model.layers:
		layer.trainable=False

	output_model = base_model(input)
	x = Flatten(name='flatten')(output_model)
	for layer in add_layers:
		x = layer(x)
	x = Dense(7, activation='softmax', name='predictions')(x)
	# Happy, sad, anger, disgust, surprise, fear, neutral

	my_model = Model(input=input, output=x)
	my_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return my_model

def single_class(base_model, optimizer='adadelta', layers=[], activation='sigmoid'):
	input = Input(shape=(224,224,3), name='image_input')
	for layer in base_model.layers:
    		layer.trainable=False

	output_model = base_model(input)
	x = Flatten(name='flatten')(output_model)
	for layer in layers:
		x = layer(x)
	x = Dense(1, activation='sigmoid', name='output')(x)  # 2 = binary

	my_model = Model(input=input, output=x)
	my_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	return my_model

def vgg16_top(base_model):
	x = Dense(4096, activation='relu', name='fc1')(base_model)
	x = Dense(4096, activation='relu', name='fc2')(x)
	return x

def dropout(base_model, rate=0.5):
	x = Dropout(rate=rate)(base_model)
	return x

def VGG16_single_class(optimizer='adadelta', activation='sigmoid', layers=[]):	
	base_model = VGG16(weights='imagenet', include_top=False)  # pooling?
	return single_class(base_model, optimizer, layers, activation)

def RN50_single_class(optimizer='adadelta', dropout=None):
	base_model = ResNet50(weights='imagenet', include_top=False)
	return single_class(base_model, optimizer, dropout)

def IV3_single_class(optimizer='adadelta', dropout=None):
	base_model = InceptionV3(weights='imagenet', include_top=False)
	return single_class(base_model, optimizer, dropout)

def save(path="./models/initial"):
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	models = {
		#"vgg16_adadelta_dropout": VGG16_single_class('adadelta', dropout=0.5),
		#"vgg16_adam_dropout": VGG16_single_class('adam', dropout=0.5),
		#"vgg16_adam": VGG16_single_class('adam', dropout=None),
		#"vgg16_sgd": VGG16_single_class('sgd', dropout=None),
		#"rn50_adam_dropout": RN50_single_class('adam', dropout=0.5),
		#"rn50_adadelta_dropout": RN50_single_class('adam', dropout=0.5),
		#"iv3_adam_dropout": IV3_single_class('adam', dropout=0.5),
		#"iv3_adadelta_dropout": IV3_single_class('adadelta', dropout=0.5)
		#"vgg16_adam_dropout_relu": VGG16_single_class('adam', dropout=0.5, activation='relu')
		#"vgg16_adam_dropout75": VGG16_single_class('adam', dropout=0.75),
		#"vgg16_adadelta_dropout75": VGG16_single_class('adadelta', dropout=0.75)
		#"vgg16_sgd_dropout": VGG16_single_class('sgd', dropout=0.5)
		"vgg16_morelayers_dropout": VGG16_single_class('adam', layers=[vgg16_top, dropout])
	}
	path += "/{}.h5"
	for name, model in models.items():
		logging.info("Saving to " + path.format(name))
		model.save(path.format(name))
		
		
