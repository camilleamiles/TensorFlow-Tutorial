import pandas as pd
import scipy as sp
import numpy as np
import tensorflow as tf 
import eagerpy as ep
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfDeepFoolAttack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

img_width, img_height =28, 28
batch_size=250
no_epochs=10
no_classes=10
validation_split=0.2
verbosity=1

#Load mnist dataset
(input_train, target_train),(input_test, target_test)=mnist.load_data()

#reshape data based on channels first/channels last

if K.image_data_format()=='channels_first':
	input_train=input_train.reshape(input_train.shape[0], 1, img_width, img_height)
	input_test=input_test.reshape(input_test.shape[0], 1, img_width, img_height)
	input_shape=(1, img_width, img_height)
else:
	input_train=input_train.reshape(input_train.shape[0], img_width, img_height, 1)
	input_test=input_test.reshape(input_test.shape[0], img_width, img_height, 1)
	input_shape=(img_width, img_height, 1)

#parse numbers as floats
input_train=input_train.astype('float32')
input_test=input_test.astype('float32')

#normalize the data
input_train=input_train/255
input_test=input_test/255

#convert target vectors to categorical targets
target_train=tf.keras.utils.to_categorical(target_train, no_classes)
target_test=tf.keras.utils.to_categorical(target_test, no_classes)


#create the model
model=Sequential()
model.add(Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)), name='maxpool')
model.add(Conv2D(10, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu', name='dense128'))
model.add(Dropout(0.2))
model.add(Dense(no_classes, activation='softmax'))

#compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

#fit data to model
model.fit(input_train, target_train, batch_size=batch_size, epochs=no_epochs, verbose=verbosity, validation_split=validation_split)

#generate generalization metrics
score=model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# means=pd.read_csv('mnist_mean_actv1000.csv')
# meanarraylist=[]
# meanarraylist.append(np.array(means['zero_mean_activation']))
# meanarraylist.append(np.array(means['one_mean_activation']))
# meanarraylist.append(np.array(means['two_mean_activation']))
# meanarraylist.append(np.array(means['three_mean_activation']))
# meanarraylist.append(np.array(means['four_mean_activation']))
# meanarraylist.append(np.array(means['five_mean_activation']))
# meanarraylist.append(np.array(means['six_mean_activation']))
# meanarraylist.append(np.array(means['seven_mean_activation']))
# meanarraylist.append(np.array(means['eight_mean_activation']))
# meanarraylist.append(np.array(means['nine_mean_activation']))

# data2=pd.read_csv('mnist_stdev_actv1000.csv')
# stddevlist=[]
# stddevlist.append(np.array(data2['zero_stdev_activation']))
# stddevlist.append(np.array(data2['one_stdev_activation']))
# stddevlist.append(np.array(data2['two_stdev_activation']))
# stddevlist.append(np.array(data2['three_stdev_activation']))
# stddevlist.append(np.array(data2['four_stdev_activation']))
# stddevlist.append(np.array(data2['five_stdev_activation']))
# stddevlist.append(np.array(data2['six_stdev_activation']))
# stddevlist.append(np.array(data2['seven_stdev_activation']))
# stddevlist.append(np.array(data2['eight_stdev_activation']))
# stddevlist.append(np.array(data2['nine_stdev_activation']))

from keract import get_activations, display_activations
keract_inputs=input_test[:1]
keract_target=target_test[:1]


#group mnist examples
#target_train[:5]
trainlabels=np.argmax(target_train, axis=1)

zero_indices=np.where(trainlabels==0)[0]
one_indices=np.where(trainlabels==1)[0]
two_indices=np.where(trainlabels==2)[0]
three_indices=np.where(trainlabels==3)[0]
four_indices=np.where(trainlabels==4)[0]
five_indices=np.where(trainlabels==5)[0]
six_indices=np.where(trainlabels==6)[0]
seven_indices=np.where(trainlabels==7)[0]
eight_indices=np.where(trainlabels==8)[0]
nine_indices=np.where(trainlabels==9)[0]

zero_indices=zero_indices[0:100]
one_indices=one_indices[0:100]
two_indices=two_indices[0:100]
three_indices=three_indices[0:100]
four_indices=four_indices[0:100]
five_indices=five_indices[0:100]
six_indices=six_indices[0:100]
seven_indices=seven_indices[0:100]
eight_indices=eight_indices[0:100]
nine_indices=nine_indices[0:100]

indices=[zero_indices,one_indices,two_indices,three_indices,four_indices,five_indices,six_indices,seven_indices,eight_indices,nine_indices]


images=[]
labels=[]
for indexarr in indices:
	for anindex in indexarr:
		images.append(input_train[anindex])

for anum in range(10):
	for i in range(100):
		labels.append(anum)

fmodel=TensorFlowModel(model, bounds=(0,1))
attack=LinfDeepFoolAttack()
#epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.1)
#_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.03)


advsinputs=np.array(advsDF) # 100, 28, 28, 1

avar=model.predict(tf.convert_to_tensor(images))
avar=tf.math.argmax(avar,1)
avar2=model.predict(advsinputs)
avar2=tf.math.argmax(avar2,1)

#avar
# # #([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 0, 9, 9, 9, 9, 9, 9], dtype=int64)>

                                                                                                     
#avar2
# # [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 4, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 7, 2,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 7, 2, 2, 2, 8, 2, 0, 1, 3, 3, 1, 2,
#        2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 2,
#        2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2,
#        2, 2, 3, 2, 7, 2, 2, 1, 2, 2, 2, 8, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2,
#        2, 7, 2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 2, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 8, 3, 3, 3, 3, 2, 9,
#        3, 3, 3, 3, 9, 3, 3, 9, 9, 3, 3, 7, 8, 3, 3, 3, 3, 3, 3, 3, 3, 9,
#        3, 3, 1, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 5, 9,
#        5, 3, 9, 2, 9, 4, 4, 4, 9, 4, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 9, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 9, 4, 4,
#        4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4,
#        8, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 9, 4, 4, 9, 4,
#        4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 8, 5, 5, 3, 9,
#        3, 8, 5, 5, 5, 3, 8, 5, 5, 3, 9, 3, 8, 3, 8, 3, 3, 5, 3, 9, 5, 5,
#        8, 5, 5, 5, 3, 9, 3, 3, 5, 9, 5, 5, 5, 5, 5, 9, 5, 5, 5, 9, 5, 9,
#        5, 9, 9, 9, 8, 5, 0, 5, 5, 5, 9, 9, 5, 5, 9, 5, 5, 8, 5, 6, 3, 5,
#        5, 5, 3, 9, 5, 5, 5, 8, 9, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9,
#        5, 9, 5, 5, 5, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 8, 6, 6, 6, 8, 6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 0, 6, 4, 6, 6, 4, 6, 6, 6, 6, 6, 0, 6, 8, 6, 6, 8,
#        0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1,
#        6, 0, 8, 0, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 8, 8, 6, 8, 7, 2, 2, 7,
#        9, 7, 7, 7, 9, 7, 7, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 1, 7, 7, 9, 7,
#        7, 9, 7, 9, 7, 9, 7, 9, 4, 9, 9, 1, 9, 7, 4, 9, 7, 7, 7, 4, 0, 9,
#        9, 7, 9, 2, 9, 7, 2, 9, 7, 7, 7, 7, 2, 2, 7, 2, 7, 7, 7, 2, 9, 7,
#        1, 9, 7, 7, 9, 9, 7, 7, 9, 7, 9, 7, 9, 9, 7, 7, 7, 9, 9, 7, 7, 9,
#        9, 7, 9, 1, 9, 7, 9, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 2, 8, 8,
#        8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 3, 8, 2, 8, 8, 8,
#        8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1,
#        8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 2, 8, 8, 8, 8, 8, 8, 4, 9,
#        4, 7, 9, 9, 3, 4, 9, 8, 9, 9, 9, 4, 9, 9, 9, 9, 9, 4, 9, 9, 9, 9,
#        8, 4, 4, 9, 4, 5, 4, 7, 4, 9, 9, 9, 4, 3, 9, 4, 7, 4, 9, 4, 4, 9,
#        9, 4, 9, 9, 9, 4, 9, 4, 7, 8, 4, 9, 4, 9, 4, 4, 4, 4, 4, 4, 8, 9,
#        9, 4, 4, 9, 0, 9, 9, 4, 4, 9, 9, 9, 9, 9, 4, 9, 9, 9, 4, 4, 9, 9,
#        7, 9, 4, 0, 9, 9, 9, 9, 8, 9]



#first trial of PCA...just looked at zero class
x=[]
for anindex in range(len(images[0:100])):
	#for i in range(len(10)):
	keract_inputs=images[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	x.append(valarray)


#just looking at zero class
# x0=np.array(x)
# pca=PCA(n_components=30)
# z=pca.fit(x0)
# howmuchvar=np.sum(z.explained_variance_ratio_)
# z0=pca.transform(x0)


#PCA for each class

#activations for benign 
b=0
a=0
x=[]
actdict={}
for anindex in range(len(images)):
	#for i in range(len(10)):
	keract_inputs=images[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	x.append(valarray)

	b+=1
	if b%100==0:
		# activations=get_activations(model, advsinputs[42].reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
		# valarray=activations['dense128'][0]
		# x.append(valarray)
		actdict[a]=x #all 100 zero example activations
		x=[]
		a+=1
	else:
		continue

#woohoo! actdict stores all activations for 10 examples of each class..

counter=0
advsindexlist=[]
for pos,aboolval in enumerate(success):
	if aboolval==True:
		counter+=1
		advsindexlist.append(pos) #gets index of all successful advs examples

#get activations for advs examples...

x=[]
advs_0=[]
advs_1=[]
advs_2=[]
advs_3=[]
advs_4=[]
advs_5=[]
advs_6=[]
advs_7=[]
advs_8=[]
advs_9=[]
advs_actdict={}
for anindex in advsindexlist:
	#for i in range(len(10)):
	keract_inputs=advsinputs[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	

	if anindex<100:
		advs_0.append(valarray)
	elif anindex>99 and anindex<200:
		advs_1.append(valarray)
	elif anindex>199 and anindex<300:
		advs_2.append(valarray)
	elif anindex>299 and anindex<400:
		advs_3.append(valarray)
	elif anindex>399 and anindex<500:
		advs_4.append(valarray)
	elif anindex>499 and anindex<600:
		advs_5.append(valarray)
	elif anindex>599 and anindex<700:
		advs_6.append(valarray)
	elif anindex>699 and anindex<800:
		advs_7.append(valarray)
	elif anindex>799 and anindex<900:
		advs_8.append(valarray)
	elif anindex>899 and anindex<1000:
		advs_9.append(valarray)


advs_actdict['advs_zero']=advs_0
the0advs=np.array(advs_actdict['advs_zero'])# shape (4,128)

advs_actdict['advs_one']=advs_1
the1advs=np.array(advs_actdict['advs_one'])# shape (8,128)

advs_actdict['advs_two']=advs_2
the2advs=np.array(advs_actdict['advs_two'])# shape (20,128)

advs_actdict['advs_three']=advs_3
the3advs=np.array(advs_actdict['advs_three'])# shape (20,128)

advs_actdict['advs_four']=advs_4
the4advs=np.array(advs_actdict['advs_four'])# shape (15,128)

advs_actdict['advs_five']=advs_5
the5advs=np.array(advs_actdict['advs_five'])# shape (47,128)

advs_actdict['advs_six']=advs_6
the6advs=np.array(advs_actdict['advs_six'])# shape (21,128)

advs_actdict['advs_seven']=advs_7
the7advs=np.array(advs_actdict['advs_seven'])# shape (57,128)

advs_actdict['advs_eight']=advs_8
the8advs=np.array(advs_actdict['advs_eight'])# shape (12,128)

advs_actdict['advs_nine']=advs_9
the9advs=np.array(advs_actdict['advs_nine'])# shape (49,128)



pca=PCA(n_components=40)
#howmuchvar=np.sum(z.explained_variance_ratio_) 


#could have done loop but just did manually

x0=np.array(actdict[0]) #x0.shape (100,128)
the0fit=pca.fit(x0)
#agh=thefit.transform(x0) this works!!!
z0=the0fit.transform(x0)  #100,40 shape
advz0=the0fit.transform(the0advs)  #(4,40)
#have to get just the advs zeros...


x1=np.array(actdict[1]) #x0.shape (100,128)
the1fit=pca.fit(x1)
z1=the1fit.transform(x1) #shape (100,40)
advz1=the1fit.transform(the1advs)


x2=np.array(actdict[2]) #x0.shape (100,128)
the2fit=pca.fit(x2)
z2=the2fit.transform(x2)
advz2=the2fit.transform(the2advs)


x3=np.array(actdict[3]) #x0.shape (100,128)
the3fit=pca.fit(x3)
z3=the3fit.transform(x3)
advz3=the3fit.transform(the3advs)

x4=np.array(actdict[4]) #x0.shape (100,128)
the4fit=pca.fit(x4)
z4=the4fit.transform(x4)
advz4=the4fit.transform(the4advs)

x5=np.array(actdict[5]) #x0.shape (100,128)
the5fit=pca.fit(x5)
z5=the5fit.transform(x5)
advz5=the5fit.transform(the5advs)

x6=np.array(actdict[6]) #x0.shape (100,128)
the6fit=pca.fit(x6)
z6=the6fit.transform(x6)
advz6=the6fit.transform(the6advs)

x7=np.array(actdict[7]) #x0.shape (100,128)
the7fit=pca.fit(x7)
z7=the7fit.transform(x7)
advz7=the7fit.transform(the7advs)

x8=np.array(actdict[8]) #x0.shape (100,128)
the8fit=pca.fit(x8)
z8=the8fit.transform(x8)
advz8=the8fit.transform(the8advs)

x9=np.array(actdict[9]) #x0.shape (100,128)
the9fit=pca.fit(x9)
z9=the9fit.transform(x9)
advz9=the9fit.transform(the9advs)

#zero means
z0_mean=np.zeros((40))
for i in range(0, len(z0)):
	z0_mean=z0_mean+z0[i]
z0_mean=z0_mean/100

z1_mean=np.zeros((40))
for i in range(0, len(z1)):
	z1_mean=z1_mean+z1[i]
z1_mean=z1_mean/100

z2_mean=np.zeros((40))
for i in range(0, len(z2)):
	z2_mean=z2_mean+z2[i]
z2_mean=z2_mean/100

z3_mean=np.zeros((40))
for i in range(0, len(z3)):
	z3_mean=z3_mean+z3[i]
z3_mean=z3_mean/100

z4_mean=np.zeros((40))
for i in range(0, len(z4)):
	z4_mean=z4_mean+z4[i]
z4_mean=z4_mean/100

z5_mean=np.zeros((40))
for i in range(0, len(z5)):
	z5_mean=z5_mean+z5[i]
z5_mean=z5_mean/100

z6_mean=np.zeros((40))
for i in range(0, len(z6)):
	z6_mean=z6_mean+z6[i]
z6_mean=z6_mean/100

z7_mean=np.zeros((40))
for i in range(0, len(z7)):
	z7_mean=z7_mean+z7[i]
z7_mean=z7_mean/100

z8_mean=np.zeros((40))
for i in range(0, len(z8)):
	z8_mean=z8_mean+z8[i]
z8_mean=z8_mean/100

z9_mean=np.zeros((40))
for i in range(0, len(z9)):
	z9_mean=z9_mean+z9[i]
z9_mean=z9_mean/100

#zero cov
cov0=np.cov(z0.T)
invcov0=sp.linalg.inv(cov0)

cov1=np.cov(z1.T)
invcov1=sp.linalg.inv(cov1)

cov2=np.cov(z2.T)
invcov2=sp.linalg.inv(cov2)

cov3=np.cov(z3.T)
invcov3=sp.linalg.inv(cov3)

cov4=np.cov(z4.T)
invcov4=sp.linalg.inv(cov4)

cov5=np.cov(z5.T)
invcov5=sp.linalg.inv(cov5)

cov6=np.cov(z6.T)
invcov6=sp.linalg.inv(cov6)

cov7=np.cov(z7.T)
invcov7=sp.linalg.inv(cov7)

cov8=np.cov(z8.T)
invcov8=sp.linalg.inv(cov8)

cov9=np.cov(z9.T)
invcov9=sp.linalg.inv(cov9)


#finding mahalanobis dist

#finding critval
from scipy.stats import chi2
critval=chi2.ppf((1-0.05),df=39)

#finding p value
p_val=1-chi2.cdf(mahal,39)
#p_valbenign=1-chi2.cdf(mahalbenign,39)

#adversarial
theadvz=[(advz0,z0_mean,invcov0),(advz1,z1_mean,invcov1),
	(advz2,z2_mean,invcov2),(advz3,z3_mean,invcov3), (advz4,z4_mean,invcov4), 
	(advz5,z5_mean,invcov5), (advz6,z6_mean,invcov6), (advz7,z7_mean,invcov7),
	(advz8,z8_mean,invcov8), (advz9,z9_mean,invcov9)]

mahaldict={}
dictcounter=0
for atuple in theadvz: #(anadvz,amean,aninvcov)
	anadvz=atuple[0]
	amean=atuple[1]
	aninvcov=atuple[2]
	mahalandplist=[]
	for anex in anadvz:
		x=anex
		x_minus_mu=x-amean

		left_term=np.dot(x_minus_mu, aninvcov)
		mahal=np.dot(left_term,x_minus_mu.T)
		p_val=1-chi2.cdf(mahal,39)
		mahalandplist.append((mahal,p_val))

	mahaldict[dictcounter]=mahalandplist
	dictcounter+=1

#viewing dict
# for akey in mahaldict:
# 	print(mahaldict[akey])

#checking how many are > critval/less than 0.05 p value
gtcritvalcount=0
pvalcount=0
smallestval=200

for aval in mahaldict.values():
	for atup in aval:
		if atup[0]<smallestval:
			smallestval=atup[0]
		if atup[0]>critval:
			gtcritvalcount+=1
		if atup[1]<0.05:
			pvalcount+=1


#compare to benign
# #not adversarial
newz0=[]
newz1=[]
newz2=[]
newz3=[]
newz4=[]
newz5=[]
newz6=[]
newz7=[]
newz8=[]
newz9=[]

for anindex in advsindexlist:
	if anindex<100:
		newz0.append(z0[anindex])
	elif anindex>99 and anindex<200:
		newz1.append(z1[anindex-100])
	elif anindex>199 and anindex<300:
		newz2.append(z2[anindex-200])
	elif anindex>299 and anindex<400:
		newz3.append(z3[anindex-300])
	elif anindex>399 and anindex<500:
		newz4.append(z4[anindex-400])
	elif anindex>499 and anindex<600:
		newz5.append(z5[anindex-500])
	elif anindex>599 and anindex<700:
		newz6.append(z6[anindex-600])
	elif anindex>699 and anindex<800:
		newz7.append(z7[anindex-700])
	elif anindex>799 and anindex<900:
		newz8.append(z8[anindex-800])
	elif anindex>899 and anindex<1000:
		newz9.append(z9[anindex-900])

newz0=np.array(newz0)
newz1=np.array(newz1)
newz2=np.array(newz2)
newz3=np.array(newz3)
newz4=np.array(newz4)
newz5=np.array(newz5)
newz6=np.array(newz6)
newz7=np.array(newz7)
newz8=np.array(newz8)
newz9=np.array(newz9)

thebenignz=[(newz0,z0_mean,invcov0),(newz1,z1_mean,invcov1),
	(newz2,z2_mean,invcov2),(newz3,z3_mean,invcov3), (newz4,z4_mean,invcov4), 
	(newz5,z5_mean,invcov5), (newz6,z6_mean,invcov6), (newz7,z7_mean,invcov7),
	(newz8,z8_mean,invcov8), (newz9,z9_mean,invcov9)]

#benign mahal dist
mahaldictb={}
dictcounter=0
for atuple in thebenignz: #(abenign,amean,aninvcov)
	abenign=atuple[0]
	amean=atuple[1]
	aninvcov=atuple[2]
	mahalandplistb=[]
	for anex in abenign:
		x=anex
		x_minus_mu=x-amean

		left_term=np.dot(x_minus_mu, aninvcov)
		mahal=np.dot(left_term,x_minus_mu.T)
		p_val=1-chi2.cdf(mahal,39)
		mahalandplistb.append((mahal,p_val))

	mahaldictb[dictcounter]=mahalandplistb
	dictcounter+=1



#checking how many are > critval/less than 0.05 p value
#want all less than critval so very small number
gtcritvalcountb=0
pvalcountb=0
smallestvalb=200

for aval in mahaldictb.values():
	for atup in aval:
		if atup[0]<smallestvalb:
			smallestvalb=atup[0]
		if atup[0]>critval:
			gtcritvalcountb+=1
		if atup[1]<0.05:
			pvalcountb+=1



