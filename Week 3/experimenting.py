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
model.add(Dense(116, activation='relu', name='dense116'))
model.add(Dropout(0.2))
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


# advs1=np.array(advsDF) #8, 16, 28, 28, 1
# advsinputs=advs1[4] #eps val 0.1... 16, 28, 28, 1

# plot images after the attack

# fig, axes =plt.subplots(1,16,figsize=(28,28))
# axes=axes.flatten()
# for img, ax in zip(advsinputs, axes):
# 	ax.imshow(np.squeeze(img), cmap="gray")
# 	ax.axis("off")
# plt.tight_layout()
# #fig.savefig('afterPGDattack.jpg',bbox_inches='tight', dpi=150)
# plt.show()



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

#mean to csv


#stddev to csv
# stdevdict1={}
# stdevdict1['zero_stdev_activation']=stddevlist[0].reshape(128,)
# stdevdict1['one_stdev_activation']=stddevlist[1].reshape(128,)
# stdevdict1['two_stdev_activation']=stddevlist[2].reshape(128,)
# stdevdict1['three_stdev_activation']=stddevlist[3].reshape(128,)
# stdevdict1['four_stdev_activation']=stddevlist[4].reshape(128,)
# stdevdict1['five_stdev_activation']=stddevlist[5].reshape(128,)
# stdevdict1['six_stdev_activation']=stddevlist[6].reshape(128,)
# stdevdict1['seven_stdev_activation']=stddevlist[7].reshape(128,)
# stdevdict1['eight_stdev_activation']=stddevlist[8].reshape(128,)
# stdevdict1['nine_stdev_activation']=stddevlist[9].reshape(128,)

# df=pd.DataFrame(stdevdict1)
# df.to_csv('mnist_stdev_actv1000.csv')



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


#getting the mean activation for each nueron of 128 for each batch of numbers
#keract_target=target_test[:1]
#mean for 1000 from each class
meanarraylist=[]
for anum in indices:
	sumarray=np.zeros(116)
	for anindex in anum:
		keract_inputs=input_train[anindex]
		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense116', output_format='simple')
		sumarray=np.add(sumarray,activations['dense116'][0])

	meanarray=sumarray/len(anum)
	meanarraylist.append(meanarray)
#print(meanarraylist)

meandict1={}
meandict1['zero_mean_activation']=meanarraylist[0].reshape(116,)
meandict1['one_mean_activation']=meanarraylist[1].reshape(116,)
meandict1['two_mean_activation']=meanarraylist[2].reshape(116,)
meandict1['three_mean_activation']=meanarraylist[3].reshape(116,)
meandict1['four_mean_activation']=meanarraylist[4].reshape(116,)
meandict1['five_mean_activation']=meanarraylist[5].reshape(116,)
meandict1['six_mean_activation']=meanarraylist[6].reshape(116,)
meandict1['seven_mean_activation']=meanarraylist[7].reshape(116,)
meandict1['eight_mean_activation']=meanarraylist[8].reshape(116,)
meandict1['nine_mean_activation']=meanarraylist[9].reshape(116,)

df=pd.DataFrame(meandict1)
df.to_csv('lessneurons_actv1000.csv')

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



# #getting data from csv
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



# #now that we have the mean and std dev, we have to find how many are more than one stdev away from mean
# i=0
# beyond1_stdev=0
# for anum in indices:
# 	#sumarray2=np.zeros(128)
# 	for anindex in anum:
# 		keract_inputs=input_train[anindex]
# 		activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')
# 		valarray=activations['dense128'][0]
# 		subarray=np.subtract(meanarraylist[i],valarray) #make counter i somewhere
# 		for j in range(128):
# 			if abs(subarray[j])>stddevlist[i][j]:
# 				beyond1_stdev+=1
# 	#print("i before:", i)
# 	i+=1
# 	#print("i after: ", i)

# #do this next	
# perc_beyond1stdev=beyond1_stdev/1280000
# #print(avg_beyond1stdev)





# #making 100 advs ex...10 of each class
# fmodel=TensorFlowModel(model, bounds=(0,1))
# images=[]
# labels=[]


# zero_indices=zero_indices[0:10]
# one_indices=one_indices[0:10]
# two_indices=two_indices[0:10]
# three_indices=three_indices[0:10]
# four_indices=four_indices[0:10]
# five_indices=five_indices[0:10]
# six_indices=six_indices[0:10]
# seven_indices=seven_indices[0:10]
# eight_indices=eight_indices[0:10]
# nine_indices=nine_indices[0:10]


# for indexarr in indices:
# 	for anindex in indexarr:
# 		images.append(input_train[anindex])

# for anum in range(10):
# 	for i in range(10):
# 		labels.append(anum)

# attack=LinfDeepFoolAttack()
# #epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
# _, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.1)
# _, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.03)


# advsinputs=np.array(advsDF) # 100, 28, 28, 1
# picdict={}

# for i in range(len(advsinputs)):
# 	astr='image'+str(i)
# 	picdict[astr]=advsinputs[i].reshape(784,)

# df=pd.DataFrame(picdict)
# df.to_csv('dfadv_by_class.csv')


# avar=model.predict(tf.convert_to_tensor(images))
# avar=tf.math.argmax(avar,1)
# avar2=model.predict(advsDF)
# avar2=tf.math.argmax(avar2,1)

# #predictions before...avar
# # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
#        # 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
#        # 4, 4, 4, 4, 4, 4, 3, 5, 5, 5, 5, 5, 3, 5, 5, 5, 6, 6, 6, 6, 6, 6,
#        # 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
#        # 8, 8, 9, 9, 9, 9, 9, 9, 9, 4, 9, 8]

# #predictions after attack...avar2
# # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 7, 2, 2, 2,
#        # 3, 2, 2, 2, 2, 3, 7, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
#        # 4, 4, 4, 4, 4, 4, 3, 5, 5, 5, 5, 5, 3, 3, 5, 5, 6, 6, 6, 6, 6, 6,
#        # 6, 6, 6, 6, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 3, 8, 8, 8,
#        # 8, 9, 4, 9, 4, 9, 9, 9, 3, 4, 9, 8]

# #how many are out of bound for 100 benign examples
# beyond1_stdev=0
# a=0 #counters
# b=0
# for animg in range(len(images)):
# 	#for i in range(len(10)):
# 	keract_inputs=images[animg]
# 	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
# 	valarray=activations['dense128'][0]
# 	subarray=np.subtract(meanarraylist[a],valarray) #make counter i somewhere
# 	for j in range(128):
# 		if abs(subarray[j])>stddevlist[a][j]:
# 			beyond1_stdev+=1
# 	b+=1
# 	if b==10:
# 		a+=1
# 	else:
# 		continue
# #6016 out of 12800...or 0.47 this seems high



# #how many are out of bound for advs examples
# avar=np.array(avar)
# avar2=np.array(avar2)


# beyond1_stdev=0
# a=0 #counters
# b=0
# for anindex in range(len(advsinputs)):
# 	#for i in range(len(10)):
# 	keract_inputs=advsinputs[anindex]
# 	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
# 	valarray=activations['dense128'][0]
# 	subarray=np.subtract(meanarraylist[a],valarray) #make counter i somewhere
# 	for j in range(128):
# 		if abs(subarray[j])>stddevlist[a][j]:
# 			beyond1_stdev+=1
# 	b+=1
# 	if b==10:
# 		a+=1
# 	else:
# 		continue

# perc_beyond1stdev=beyond1_stdev/12800
# #5853 neurons out of 12800 that are out of bounds...or .457265625 for DF 0.1
# #5987 neurons out of 12800 that are out of bounds...or .467734375 for DF 0.03

# #see if can go back and do for each


# #how many are out of bounds for advs examples but given what they were misclassified as
# beyond1_stdev=0
# for anindex in range(len(advsinputs)):
# 	#for i in range(len(10)):
# 	keract_inputs=advsinputs[anindex]
# 	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')	
# 	avar2=model.predict(advsinputs[anindex].reshape(1,28,28,1))
# 	avar2=np.array(tf.math.argmax(avar2,1))	
# 	valarray=activations['dense128'][0]
# 	subarray=np.subtract(meanarraylist[avar2[0]],valarray) #make counter i somewhere
# 	for j in range(128):
# 		if abs(subarray[j])>stddevlist[avar2[0]][j]:
# 			beyond1_stdev+=1

# #5169 neurons out of 12800 are out of bounds give class they were misclassified as
# # or 0.403828125 for eps 0.1
# #5307 neurons out of 12800 are out of bounds give class they were misclassified as
# # or 0.414609375 for eps 0.03



# #activations=get_activations(model, keract_inputs)
# #activations=get_activations(model, keract_inputs, layer_names='dense128', output_format='simple')
# #display_activations(activations, cmap='gray', save=False)


# #heatmaps
# from keract import display_heatmaps
# #display_heatmaps(activations, keract_inputs, save=False)


# #display some more imgs heatmaps
# keractlist=[]
# for aninput in input_test[:2]:
# 	keractlist.append(aninput)

# keract_inputs=keractlist
# for i in range(len(keract_inputs)):
# 	activations=get_activations(model, keract_inputs[i].reshape(1, 28, 28, 1))
# 	display_heatmaps(activations, keract_inputs[i].reshape(1, 28, 28, 1), save=False)


# #advs heatmaps
# keractadvs=[]
# for aninput in advsinputs[:2]:
# 	keractadvs.append(aninput)

# keract_inputs=keractadvs
# for i in range(len(keract_inputs)):
# 	activations=get_activations(model, keract_inputs[i].reshape(1,28,28,1))
# 	display_heatmaps(activations, keract_inputs[i].reshape(1,28,28,1), save=False)

