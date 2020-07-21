#visualize layer activations of a tensorflow.keras CNN with Keract (tutorial)

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

 

#configure model
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
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

#compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

#fit data to model
model.fit(input_train, target_train, batch_size=batch_size, epochs=no_epochs, verbose=verbosity, validation_split=validation_split)

#generate generalization metrics
score=model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')



#foolbox model
fmodel=TensorFlowModel(model, bounds=(0,1))

images, labels = samples(fmodel, dataset="mnist", batchsize=16)
images1, labels1=ep.astensors(*samples(fmodel, dataset="mnist", batchsize=16))

attack=LinfDeepFoolAttack()
epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]

_, advsDF, success = attack(fmodel, images, labels, epsilons=epsilons)


advs1=np.array(advsDF) #8, 16, 28, 28, 1
advsinputs=advs1[4] #eps val 0.1... 16, 28, 28, 1

# plot images after the attack

fig, axes =plt.subplots(1,16,figsize=(28,28))
axes=axes.flatten()
for img, ax in zip(advsinputs, axes):
	ax.imshow(np.squeeze(img), cmap="gray")
	ax.axis("off")
plt.tight_layout()
#fig.savefig('afterPGDattack.jpg',bbox_inches='tight', dpi=150)
plt.show()



#getting some adversarial samples for sarah

# picdict={}
# picdict['image1']=advsinputs[0].reshape(784,)
# picdict['image2']=advsinputs[1].reshape(784,)
# picdict['image3']=advsinputs[2].reshape(784,)
# picdict['image4']=advsinputs[3].reshape(784,)
# picdict['image5']=advsinputs[4].reshape(784,)
# picdict['image6']=advsinputs[5].reshape(784,)
# picdict['image7']=advsinputs[6].reshape(784,)
# picdict['image8']=advsinputs[7].reshape(784,)
# picdict['image9']=advsinputs[8].reshape(784,)
# picdict['image10']=advsinputs[9].reshape(784,)
# picdict['image11']=advsinputs[10].reshape(784,)
# picdict['image12']=advsinputs[11].reshape(784,)
# picdict['image13']=advsinputs[12].reshape(784,)
# picdict['image14']=advsinputs[13].reshape(784,)
# picdict['image15']=advsinputs[14].reshape(784,)
# picdict['image16']=advsinputs[15].reshape(784,)

# df=pd.DataFrame(picdict)
# df.to_csv('advspics2.csv')



#keract visualizations
from keract import get_activations, display_activations
keract_inputs=input_test[:1]
keract_target=target_test[:1]
activations=get_activations(model, keract_inputs)
display_activations(activations, cmap='gray', save=False)


#heatmaps
from keract import display_heatmaps
#display_heatmaps(activations, keract_inputs, save=False)


#display some more imgs heatmaps
keractlist=[]
for aninput in input_test[:2]:
	keractlist.append(aninput)

keract_inputs=keractlist
for i in range(len(keract_inputs)):
	activations=get_activations(model, keract_inputs[i].reshape(1, 28, 28, 1))
	display_heatmaps(activations, keract_inputs[i].reshape(1, 28, 28, 1), save=False)


#advs heatmaps
keractadvs=[]
for aninput in advsinputs[:2]:
	keractadvs.append(aninput)

keract_inputs=keractadvs
for i in range(len(keract_inputs)):
	activations=get_activations(model, keract_inputs[i].reshape(1,28,28,1))
	display_heatmaps(activations, keract_inputs[i].reshape(1,28,28,1), save=False)

