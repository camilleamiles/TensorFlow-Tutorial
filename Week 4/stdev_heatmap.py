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



#keract visualizations
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

zero_indices=zero_indices[0:1000]
one_indices=one_indices[0:1000]
two_indices=two_indices[0:1000]
three_indices=three_indices[0:1000]
four_indices=four_indices[0:1000]
five_indices=five_indices[0:1000]
six_indices=six_indices[0:1000]
seven_indices=seven_indices[0:1000]
eight_indices=eight_indices[0:1000]
nine_indices=nine_indices[0:1000]

indices=[zero_indices,one_indices,two_indices,three_indices,four_indices,five_indices,six_indices,seven_indices,eight_indices,nine_indices]


# #getting the mean activation for each nueron of 128 for each batch of numbers
# #keract_target=target_test[:1]
# #mean for 1000 from each class
# meanarraylist=[]
# for anum in indices:
# 	sumarray=np.zeros(128)
# 	for anindex in anum:
# 		keract_inputs=input_train[anindex]
# 		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
# 		sumarray=np.add(sumarray,activations['dense128'][0])

# 	meanarray=sumarray/len(anum)
# 	meanarraylist.append(meanarray)
# #print(meanarraylist)


# #getting the std dev
# stddevlist=[]
# i=0
# for anum in indices:
# 	sumarray2=np.zeros(128)
# 	for anindex in anum:
# 		keract_inputs=input_train[anindex]
# 		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
# 		valarray=activations['dense128'][0]
# 		subarray=np.subtract(valarray,meanarraylist[i]) #make counter i somewhere
# 		subarraysq=np.square(subarray)
# 		sumarray2=np.add(sumarray2, subarraysq)

# 	#meanarray2=sumarray2/len(anum)
# 	#sqrtarray=np.sqrt(meanarray2)
# 	stddevlist.append(sumarray2) 
# 	i+=1
# #changes with sarah

# for i in range(0, len(stddevlist)):
# 	stddevlist[i]=np.sqrt(stddevlist[i]/1000)



# # #print(stddevlist)



#getting data from csv
means=pd.read_csv('mnist_mean_actv1000.csv')
meanarraylist=[]
meanarraylist.append(np.array(means['zero_mean_activation']))
meanarraylist.append(np.array(means['one_mean_activation']))
meanarraylist.append(np.array(means['two_mean_activation']))
meanarraylist.append(np.array(means['three_mean_activation']))
meanarraylist.append(np.array(means['four_mean_activation']))
meanarraylist.append(np.array(means['five_mean_activation']))
meanarraylist.append(np.array(means['six_mean_activation']))
meanarraylist.append(np.array(means['seven_mean_activation']))
meanarraylist.append(np.array(means['eight_mean_activation']))
meanarraylist.append(np.array(means['nine_mean_activation']))

data2=pd.read_csv('mnist_stdev_actv1000.csv')
stddevlist=[]
stddevlist.append(np.array(data2['zero_stdev_activation']))
stddevlist.append(np.array(data2['one_stdev_activation']))
stddevlist.append(np.array(data2['two_stdev_activation']))
stddevlist.append(np.array(data2['three_stdev_activation']))
stddevlist.append(np.array(data2['four_stdev_activation']))
stddevlist.append(np.array(data2['five_stdev_activation']))
stddevlist.append(np.array(data2['six_stdev_activation']))
stddevlist.append(np.array(data2['seven_stdev_activation']))
stddevlist.append(np.array(data2['eight_stdev_activation']))
stddevlist.append(np.array(data2['nine_stdev_activation']))



#now that we have the mean and std dev, we have to find how many are more than one stdev away from mean
i=0
beyond1_stdev=0
for anum in indices:
	#sumarray2=np.zeros(128)
	for anindex in anum:
		keract_inputs=input_train[anindex]
		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
		valarray=activations['dense128'][0]
		subarray=np.subtract(meanarraylist[i],valarray) #make counter i somewhere
		for j in range(128):
			if abs(subarray[j])>stddevlist[i][j]:
				beyond1_stdev+=1
	#print("i before:", i)
	i+=1
	#print("i after: ", i)

#do this next	
perc_beyond1stdev=beyond1_stdev/1280000
#print(avg_beyond1stdev)





#making 100 advs ex...10 of each class
fmodel=TensorFlowModel(model, bounds=(0,1))
images=[]
labels=[]


zero_indices=zero_indices[0:10]
one_indices=one_indices[0:10]
two_indices=two_indices[0:10]
three_indices=three_indices[0:10]
four_indices=four_indices[0:10]
five_indices=five_indices[0:10]
six_indices=six_indices[0:10]
seven_indices=seven_indices[0:10]
eight_indices=eight_indices[0:10]
nine_indices=nine_indices[0:10]
indices=[zero_indices,one_indices,two_indices,three_indices,four_indices,five_indices,six_indices,seven_indices,eight_indices,nine_indices]


for indexarr in indices:
	for anindex in indexarr:
		images.append(input_train[anindex])

for anum in range(10):
	for i in range(10):
		labels.append(anum)




attack=LinfDeepFoolAttack()
#epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.1)
_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.03)


advsinputs=np.array(advsDF) # 100, 28, 28, 1
picdict={}

for i in range(len(advsinputs)):
	astr='image'+str(i)
	picdict[astr]=advsinputs[i].reshape(784,)

df=pd.DataFrame(picdict)
df.to_csv('dfadv_by_class.csv')


avar=model.predict(tf.convert_to_tensor(images))
avar=tf.math.argmax(avar,1)
avar2=model.predict(advsDF)
avar2=tf.math.argmax(avar2,1)

#predictions before...avar
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       # 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
       # 4, 4, 4, 4, 4, 4, 3, 5, 5, 5, 5, 5, 3, 5, 5, 5, 6, 6, 6, 6, 6, 6,
       # 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
       # 8, 8, 9, 9, 9, 9, 9, 9, 9, 4, 9, 8]

#predictions after attack...avar2
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 7, 2, 2, 2,
       # 3, 2, 2, 2, 2, 3, 7, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
       # 4, 4, 4, 4, 4, 4, 3, 5, 5, 5, 5, 5, 3, 3, 5, 5, 6, 6, 6, 6, 6, 6,
       # 6, 6, 6, 6, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 3, 8, 8, 8,
       # 8, 9, 4, 9, 4, 9, 9, 9, 3, 4, 9, 8]

#how many are out of bound for 100 benign examples
heatmapstdev=[]
a=0 #counters
b=0
for animg in range(len(advsinputs)):
	#for i in range(len(10)):
	keract_inputs=advsinputs[animg]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	subarray=np.subtract(meanarraylist[a],valarray) #make counter i somewhere
	for j in range(128):
		if abs(subarray[j])>(3*(stddevlist[a][j])):
			heatmapstdev.append(4)
		elif abs(subarray[j])>(2*(stddevlist[a][j])):
			heatmapstdev.append(3)
		elif abs(subarray[j])>(stddevlist[a][j]):
			heatmapstdev.append(2)
		else:
			heatmapstdev.append(1)
	break
	b+=1
	if b==10:
		a+=1
	else:
		continue

neurons=np.arange(0,128)
actv=['standard dev from mean']

harvest=np.array(heatmapstdev)

fig,ax=plt.subplots()
im=ax.imshow(harvest1)
ax.set_title('Activations of neurons in dense layer')
fig.tight_layout()
plt.show()

# activations={}
# activations['dense128']=np.array(heatmapstdev).reshape(1,128,1)
# display_heatmaps(activations, images[0].reshape(1, 28, 28, 1), save=False)




# activations=get_activations(model, images[0].reshape(1,28,28,1))
# display_heatmaps(activations, images[0].reshape(1,28,28,1), save=False)


#heatmapstdev
#6016 out of 12800...or 0.47 this seems high



#how many are out of bound for advs examples
avar=np.array(avar)
avar2=np.array(avar2)


beyond1_stdev=0
a=0 #counters
b=0
for anindex in range(len(advsinputs)):
	#for i in range(len(10)):
	keract_inputs=advsinputs[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	subarray=np.subtract(meanarraylist[a],valarray) #make counter i somewhere
	for j in range(128):
		if abs(subarray[j])>stddevlist[a][j]:
			beyond1_stdev+=1
	b+=1
	if b==10:
		a+=1
	else:
		continue

perc_beyond1stdev=beyond1_stdev/12800
#5853 neurons out of 12800 that are out of bounds...or .457265625 for DF 0.1
#5987 neurons out of 12800 that are out of bounds...or .467734375 for DF 0.03

#see if can go back and do for each


#how many are out of bounds for advs examples but given what they were misclassified as
beyond1_stdev=0
for anindex in range(len(advsinputs)):
	#for i in range(len(10)):
	keract_inputs=advsinputs[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')	
	avar2=model.predict(advsinputs[anindex].reshape(1,28,28,1))
	avar2=np.array(tf.math.argmax(avar2,1))	
	valarray=activations['dense128'][0]
	subarray=np.subtract(meanarraylist[avar2[0]],valarray) #make counter i somewhere
	for j in range(128):
		if abs(subarray[j])>stddevlist[avar2[0]][j]:
			beyond1_stdev+=1

#5169 neurons out of 12800 are out of bounds give class they were misclassified as
# or 0.403828125 for eps 0.1
#5307 neurons out of 12800 are out of bounds give class they were misclassified as
# or 0.414609375 for eps 0.03



#activations=get_activations(model, keract_inputs)
#activations=get_activations(model, keract_inputs, layer_names='dense128', output_format='simple')
#display_activations(activations, cmap='gray', save=False)


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





#comparing activations for benign vs advs
keract_inputs=images[0]

activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
valarray=activations['dense128'][0]

keract_inputs=advsinputs[0]

activationsadvs=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
valadvsarray=activationsadvs['dense128'][0]
diffarray=np.subtract(valarray,valadvsarray)



#mean and stv for maxpool
# meanarraylist=[]
# for anum in indices:
# 	sumarray=np.zeros(128)
# 	for anindex in anum:
# 		keract_inputs=input_train[anindex]
# 		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='maxpool', output_format='simple')
# 		sumarray=np.add(sumarray,activations['maxpool'][0])

# 	meanarray=sumarray/len(anum)
# 	meanarraylist.append(meanarray)
# #print(meanarraylist)